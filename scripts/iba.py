# do not alter parameter shape operations or dimension checks will fail

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scripts.utils import replace_layer, normalize, mySequential
from scripts.hooks import install_libragrad_hooks, remove_libragrad_hooks

# dev = "cpu"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Estimator:
    def __init__(self, layer_inst):
        self.layer = layer_inst
        self.M = None
        self.S = None
        self.N = None
        self.num_seen = 0
        self.eps = 1e-5

    def feed(self, z_arr: np.ndarray):
        if self.N is None:
            self.M = np.zeros_like(z_arr, dtype=float)
            self.S = np.zeros_like(z_arr, dtype=float)
            self.N = np.zeros_like(z_arr, dtype=float)

        self.num_seen += 1
        d_val = z_arr - self.M
        self.N += 1
        self.M += d_val / self.num_seen
        self.S += d_val * (z_arr - self.M)

    def feed_batch(self, b_data: np.ndarray):
        for itm in b_data:
            self.feed(itm)

    def shape(self):
        return self.M.shape

    def is_complete(self):
        return self.num_seen > 0

    def get_layer(self):
        return self.layer

    def mean(self):
        return self.M.squeeze()

    def p_zero(self):
        return 1 - self.N / (self.num_seen + 1)

    def std(self, stabilize=True):
        if stabilize:
            return np.sqrt(
                np.maximum(self.S, self.eps) / np.maximum(self.N, 1.0)
            )
        else:
            return np.sqrt(self.S / self.N)

    def normalize(self, z_arr):
        return (z_arr - self.mean()) / self.std()

    def load(self, src_obj):
        raw_st = src_obj if not isinstance(src_obj, str) else torch.load(src_obj)
        if self.__class__.__name__ != raw_st["class"]:
            raise RuntimeError("mismatch in estimator: " + str(self.__class__.__name__))
        if self.layer.__class__.__name__ != raw_st["layer_class"]:
            raise RuntimeError("mismatch in layer: " + str(self.layer.__class__.__name__))
        self.N = raw_st["N"]
        self.S = raw_st["S"]
        self.M = raw_st["M"]
        self.num_seen = raw_st["num_seen"]


class InformationBottleneck(nn.Module):
    def __init__(
        self,
        m_arr: np.ndarray,
        s_arr: np.ndarray,
        device=None,
    ):
        super().__init__()

        self.device = device
        self.initial_value = 5.0

        self.std = torch.tensor(
            s_arr, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.mean = torch.tensor(
            m_arr, dtype=torch.float, device=self.device, requires_grad=False
        )

        # setup alpha tensor
        self.alpha = nn.Parameter(
            torch.full(
                self.mean.shape,
                fill_value=self.initial_value,
                device=self.device,
            )
        )

        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None

        self.reset_alpha()

    @staticmethod
    def _sample_t(mu_v, v_noise):
        sd_val = v_noise.sqrt()
        eps_t = mu_v.data.new(mu_v.size()).normal_()
        return mu_v + sd_val * eps_t

    @staticmethod
    def _calc_capacity(mu_v, v_var):
        raw_cap = -0.5 * (1 + torch.log(v_var) - mu_v**2 - v_var)
        return torch.nan_to_num(raw_cap, nan=0.0, posinf=0.0, neginf=0.0)

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, x_in, **kwargs):
        g_val = self.sigmoid(self.alpha)

        # dimension match check
        if x_in.dim() == 3:
            g_val = g_val.unsqueeze(0).expand(x_in.shape[0], -1, -1)
        elif x_in.dim() == 2:
            g_val = g_val.unsqueeze(0).expand(x_in.shape[0], -1)

        m_mu = x_in * g_val
        m_v = (1 - g_val) ** 2
        self.buffer_capacity = self._calc_capacity(m_mu, m_v)

        t_out = self._sample_t(m_mu, m_v)
        return (t_out,)

    def libragrad_initialize_alpha(
        self,
        p_prior: torch.Tensor,
        strength: float = 3.0,
    ):
        with torch.no_grad():
            p_prior = p_prior.to(self.alpha.device).float()
            p_prior = p_prior.clamp(1e-4, 1 - 1e-4)

            # logit space conversion
            l_prior = torch.log(p_prior / (1 - p_prior))
            l_prior = l_prior * strength

            while l_prior.dim() > self.alpha.dim():
                l_prior = l_prior.squeeze(0)

            while l_prior.dim() < self.alpha.dim():
                l_prior = l_prior.unsqueeze(0)

            self.alpha.copy_(l_prior)

        self.alpha.data.clamp_(-5.0, 5.0)
        return self.alpha


class IBAInterpreter:
    def __init__(
        self,
        model,
        estim: Estimator,
        beta: float,
        steps: int = 10,
        lr: float = 1.0,
        batch_size: int = 10,
        progbar: bool = False,
        use_libragrad: bool = False,
        libragrad_init: bool = False,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.original_layer = estim.get_layer()
        self.shape = estim.shape()
        self.beta = beta
        self.batch_size = batch_size
        self.fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6)
        self.progbar = progbar
        self.lr = lr
        self.train_steps = steps
        self.use_libragrad = use_libragrad
        self.libragrad_init = libragrad_init

        self.bottleneck = InformationBottleneck(
            estim.mean(), estim.std(), device=self.device
        )
        self.sequential = mySequential(self.original_layer, self.bottleneck)

    def text_heatmap(
        self,
        text_t: torch.Tensor,
        image_t: torch.Tensor,
        prior: torch.Tensor = None,
    ) -> np.ndarray:
        raw_saliency, _, _, _ = self._run_text_training(text_t, image_t, prior)
        sal_out = torch.nansum(raw_saliency, -1).cpu().detach().numpy()
        return normalize(normalize(sal_out))

    def vision_heatmap(
        self,
        text_t: torch.Tensor,
        image_t: torch.Tensor,
        prior: torch.Tensor = None,
    ) -> np.ndarray:
        raw_saliency, _, _, _ = self._run_vision_training(text_t, image_t, prior)

        sal_v = torch.nansum(raw_saliency, -1)
        sal_v = torch.nan_to_num(sal_v, nan=0.0)

        # drop leading token
        sal_v = sal_v[1:]

        p_total = sal_v.numel()
        side_g = int(p_total**0.5)

        sal_v = sal_v.reshape(1, 1, side_g, side_g)
        sal_v = torch.nn.functional.interpolate(
            sal_v, size=224, mode="bilinear", align_corners=False
        )
        sal_arr = sal_v.squeeze().cpu().detach().numpy()
        return normalize(sal_arr)

    def _run_text_training(self, text_t, image_t, prior=None):
        h_mgr = None
        if self.use_libragrad:
            h_mgr = install_libragrad_hooks(self.model)

        replace_layer(self.model.text_model, self.original_layer, self.sequential)
        l_c, l_f, l_tot = self._train_bottleneck(text_t, image_t, prior)
        replace_layer(self.model.text_model, self.sequential, self.original_layer)

        if h_mgr is not None:
            remove_libragrad_hooks(h_mgr)

        return self.bottleneck.buffer_capacity.mean(axis=0), l_c, l_f, l_tot

    def _run_vision_training(self, text_t, image_t, prior=None):
        h_mgr = None
        if self.use_libragrad:
            h_mgr = install_libragrad_hooks(self.model)

        replace_layer(self.model.vision_model, self.original_layer, self.sequential)
        l_c, l_f, l_tot = self._train_bottleneck(text_t, image_t, prior)
        replace_layer(self.model.vision_model, self.sequential, self.original_layer)

        if h_mgr is not None:
            remove_libragrad_hooks(h_mgr)

        return self.bottleneck.buffer_capacity.mean(axis=0), l_c, l_f, l_tot

    def _train_bottleneck(self, text_t, image_t, prior=None):
        b_input = (
            text_t.expand(self.batch_size, -1),
            image_t.expand(self.batch_size, -1, -1, -1),
        )
        opt_b = torch.optim.Adam(
            lr=self.lr, params=self.bottleneck.parameters()
        )

        if self.libragrad_init and prior is not None:
            self.bottleneck.libragrad_initialize_alpha(prior)
        else:
            self.bottleneck.reset_alpha()

        self.model.eval()

        for _ in tqdm(
            range(self.train_steps),
            desc="Training LC-IBA" if self.use_libragrad else "Training IBA",
            disable=not self.progbar,
        ):
            opt_b.zero_grad()
            feats_out = (
                self.model.get_text_features(b_input[0]),
                self.model.get_image_features(b_input[1]),
            )
            l_comp, l_fit, l_all = self.calc_loss(
                outputs=feats_out[0], labels=feats_out[1]
            )
            l_all.backward()
            opt_b.step()

        return l_comp, l_fit, l_all

    def calc_loss(self, outputs, labels):
        c_part = self.bottleneck.buffer_capacity.mean()
        f_part = self.fitting_estimator(outputs, labels).mean()
        tot_loss = self.beta * c_part - f_part
        return c_part, f_part, tot_loss
