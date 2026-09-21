"""
scripts/iba.py

Information Bottleneck Attribution for CLIP.

Based on:
    https://github.com/bazingagin/IBA
    https://github.com/BioroboticsLab/IBA
    https://github.com/YingWANGG/M2IB

Changes vs original M2IB iba.py:
    [PATCH 1] Alpha shape fix:
              original had (1, *mean.shape) causing batch dim mismatch.
              Fixed to mean.shape directly.

    [PATCH 2] libragrad_initialize_alpha() added to InformationBottleneck:
              Initializes alpha from a LibraGrad saliency prior
              in logit space. Used by Option B.

    [PATCH 3] IBAInterpreter gains use_libragrad + libragrad_init flags.
              _train_bottleneck installs LibraGrad hooks around
              the optimization loop (Option A) and optionally
              initializes alpha from prior (Option B).

    [PATCH 4] lciba_train_bottleneck() added as explicit LC-IBA
              training method accepting a prior tensor.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from scripts.utils import replace_layer, normalize, mySequential
from scripts.hooks import install_libragrad_hooks, remove_libragrad_hooks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Estimator
# ============================================================

class Estimator:
    """
    Calculates empirical mean and variance of intermediate
    feature maps. Used to set up the IB prior distribution.
    """

    def __init__(self, layer):
        self.layer    = layer
        self.M        = None   # running mean
        self.S        = None   # running variance accumulator
        self.N        = None   # running count per entry
        self.num_seen = 0      # total samples seen
        self.eps      = 1e-5

    def feed(self, z: np.ndarray):
        if self.N is None:
            self.M = np.zeros_like(z, dtype=float)
            self.S = np.zeros_like(z, dtype=float)
            self.N = np.zeros_like(z, dtype=float)

        self.num_seen += 1
        diff   = z - self.M
        self.N += 1
        self.M += diff / self.num_seen
        self.S += diff * (z - self.M)

    def feed_batch(self, batch: np.ndarray):
        for point in batch:
            self.feed(point)

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

    def normalize(self, z):
        return (z - self.mean()) / self.std()

    def load(self, what):
        state = what if not isinstance(what, str) else torch.load(what)
        if self.__class__.__name__ != state["class"]:
            raise RuntimeError(
                f"Estimator mismatch: {self.__class__.__name__} vs {state['class']}"
            )
        if self.layer.__class__.__name__ != state["layer_class"]:
            raise RuntimeError(
                f"Layer mismatch: {self.layer.__class__.__name__} vs {state['layer_class']}"
            )
        self.N        = state["N"]
        self.S        = state["S"]
        self.M        = state["M"]
        self.num_seen = state["num_seen"]


# ============================================================
# Information Bottleneck
# ============================================================

class InformationBottleneck(nn.Module):
    """
    Learnable per-dimension bottleneck gate.

    Forward pass:
        lambda = sigmoid(alpha)
        masked_mu  = x * lambda
        masked_var = (1 - lambda)^2
        t ~ N(masked_mu, masked_var)

    The KL divergence KL[N(masked_mu, masked_var) || N(0,1)]
    is stored in buffer_capacity for loss computation.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        device=None
    ):
        super().__init__()

        self.device        = device
        self.initial_value = 5.0

        self.std  = torch.tensor(
            std,  dtype=torch.float, device=self.device, requires_grad=False
        )
        self.mean = torch.tensor(
            mean, dtype=torch.float, device=self.device, requires_grad=False
        )

        # -------------------------------------------------------
        # [PATCH 1] Alpha shape fix
        # Original: torch.full((1, *self.mean.shape), ...)
        #           caused extra batch dimension mismatch.
        # Fixed:    torch.full(self.mean.shape, ...)
        # -------------------------------------------------------
        self.alpha = nn.Parameter(
            torch.full(
                self.mean.shape,
                fill_value=self.initial_value,
                device=self.device
            )
        )

        self.sigmoid         = nn.Sigmoid()
        self.buffer_capacity = None

        self.reset_alpha()

    @staticmethod
    def _sample_t(mu, noise_var):
        noise_std = noise_var.sqrt()
        eps       = mu.data.new(mu.size()).normal_()
        return mu + noise_std * eps
    @staticmethod
    def _calc_capacity(mu, var):
        result = -0.5 * (1 + torch.log(var) - mu ** 2 - var)
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    #@staticmethod
    #def _calc_capacity(mu, var):
        # KL[ N(mu, var) || N(0,1) ]
      #  return -0.5 * (1 + torch.log(var) - mu ** 2 - var)

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)
        return self.alpha

    def forward(self, x, **kwargs):
        lamb = self.sigmoid(self.alpha)

        # Expand lambda to match batch dimension
        if x.dim() == 3:
            # x: [batch, seq, hidden]
            lamb = lamb.unsqueeze(0).expand(x.shape[0], -1, -1)
        elif x.dim() == 2:
            # x: [batch, hidden]
            lamb = lamb.unsqueeze(0).expand(x.shape[0], -1)

        masked_mu            = x * lamb
        masked_var           = (1 - lamb) ** 2
        self.buffer_capacity = self._calc_capacity(masked_mu, masked_var)

        t = self._sample_t(masked_mu, masked_var)
        return (t,)

    # -----------------------------------------------------------
    # [PATCH 2] LibraGrad alpha initialization
    # -----------------------------------------------------------
    def libragrad_initialize_alpha(
        self,
        prior: torch.Tensor,
        strength: float = 3.0
    ):
        """
        Initialize bottleneck alpha from a LibraGrad saliency prior.

        Converts a normalized saliency map in [0, 1] to logit space,
        scales by strength, and copies into alpha.

        High saliency  -> high alpha  -> lambda near 1 -> preserve info
        Low saliency   -> low alpha   -> lambda near 0 -> compress info

        Args:
            prior:    Tensor in [0, 1], shape compatible with alpha.
                      Typically [1, seq_len, hidden_dim] from
                      libragrad.prepare_vision_prior() or
                      libragrad.prepare_text_prior()
            strength: Scaling factor controlling initialization
                      confidence. Default 3.0.
                      Higher = more aggressive gating from prior.
        """

        with torch.no_grad():

            prior = prior.to(self.alpha.device).float()

            # Clamp for numerical stability before logit
            prior = prior.clamp(1e-4, 1 - 1e-4)

            # probability -> logit: log(p / (1-p))
            alpha_prior = torch.log(prior / (1 - prior))

            # Scale confidence
            alpha_prior = alpha_prior * strength

            # Match alpha dimensionality
            while alpha_prior.dim() > self.alpha.dim():
                alpha_prior = alpha_prior.squeeze(0)

            while alpha_prior.dim() < self.alpha.dim():
                alpha_prior = alpha_prior.unsqueeze(0)

            self.alpha.copy_(alpha_prior)

        self.alpha.data.clamp_(-5.0, 5.0)
        return self.alpha


# ============================================================
# IBA Interpreter
# ============================================================

class IBAInterpreter:
    """
    Information Bottleneck Attribution interpreter for CLIP.

    Supports three modes:
        M2IB (original):
            use_libragrad=False, libragrad_init=False

        LC-IBA Option A only (corrected backward):
            use_libragrad=True,  libragrad_init=False

        LC-IBA Option A + B (corrected backward + prior init):
            use_libragrad=True,  libragrad_init=True
            Requires passing prior to vision/text heatmap methods.
    """

    def __init__(
        self,
        model,
        estim: Estimator,
        beta: float,
        steps: int       = 10,
        lr: float        = 1.0,
        batch_size: int  = 10,
        progbar: bool    = False,
        # ---------------------------------------------------
        # [PATCH 3] LC-IBA flags
        # ---------------------------------------------------
        use_libragrad:   bool = False,
        libragrad_init:  bool = False,
    ):
        self.device          = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model           = model.to(self.device)
        self.original_layer  = estim.get_layer()
        self.shape           = estim.shape()
        self.beta            = beta
        self.batch_size      = batch_size
        self.fitting_estimator = torch.nn.CosineSimilarity(eps=1e-6)
        self.progbar         = progbar
        self.lr              = lr
        self.train_steps     = steps
        self.use_libragrad   = use_libragrad
        self.libragrad_init  = libragrad_init

        self.bottleneck = InformationBottleneck(
            estim.mean(), estim.std(), device=self.device
        )
        self.sequential = mySequential(self.original_layer, self.bottleneck)

    # -----------------------------------------------------------
    # Public heatmap methods
    # -----------------------------------------------------------

    def text_heatmap(
        self,
        text_t: torch.Tensor,
        image_t: torch.Tensor,
        prior: torch.Tensor = None,
    ) -> np.ndarray:
        """
        Compute text attribution heatmap.

        Args:
            text_t:  [1, seq_len] tokenized text
            image_t: [1, 3, 224, 224] preprocessed image
            prior:   Optional [1, seq_len, hidden_dim] LibraGrad prior
                     Required when libragrad_init=True

        Returns:
            np.ndarray [seq_len] normalized to [0, 1]
        """
        #saliency = torch.nan_to_num(saliency, nan=0.0)
        saliency, _, _, _ = self._run_text_training(text_t, image_t, prior)
        saliency = torch.nansum(saliency, -1).cpu().detach().numpy()
        return normalize(normalize(saliency))

    def vision_heatmap(
        self,
        text_t: torch.Tensor,
        image_t: torch.Tensor,
        prior: torch.Tensor = None,
    ) -> np.ndarray:
    
        saliency, _, _, _ = self._run_vision_training(text_t, image_t, prior)
    
        # saliency: [seq_len, hidden_dim] — sum over hidden dim first
        saliency = torch.nansum(saliency, -1)          # [seq_len]
        saliency = torch.nan_to_num(saliency, nan=0.0)
    
        # Discard CLS token (index 0)
        saliency = saliency[1:]                         # [num_patches]
    
        num_patches = saliency.numel()
        grid_size   = int(num_patches ** 0.5)
    
        #print(f"[DEBUG] num_patches={num_patches} grid_size={grid_size}")
    
        saliency = saliency.reshape(1, 1, grid_size, grid_size)
        saliency = torch.nn.functional.interpolate(
            saliency, size=224, mode="bilinear", align_corners=False
        )
        saliency = saliency.squeeze().cpu().detach().numpy()
        return normalize(saliency)

    # -----------------------------------------------------------
    # Internal training runners
    # -----------------------------------------------------------
    def _run_text_training(self, text_t, image_t, prior=None):
        # Install hooks BEFORE replace_layer
        libra_manager = None
        if self.use_libragrad:
            libra_manager = install_libragrad_hooks(self.model)
    
        replace_layer(self.model.text_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t, prior)
        replace_layer(self.model.text_model, self.sequential, self.original_layer)
    
        if libra_manager is not None:
            remove_libragrad_hooks(libra_manager)
    
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t


    def _run_vision_training(self, text_t, image_t, prior=None):
        # Install hooks BEFORE replace_layer
        libra_manager = None
        if self.use_libragrad:
            libra_manager = install_libragrad_hooks(self.model)
    
        replace_layer(self.model.vision_model, self.original_layer, self.sequential)
        loss_c, loss_f, loss_t = self._train_bottleneck(text_t, image_t, prior)
        replace_layer(self.model.vision_model, self.sequential, self.original_layer)
    
        if libra_manager is not None:
            remove_libragrad_hooks(libra_manager)
    
        return self.bottleneck.buffer_capacity.mean(axis=0), loss_c, loss_f, loss_t

    # Add this debug print inside vision_heatmap() in iba.py
    # after _run_vision_training returns:
    


    def _train_bottleneck(self, text_t, image_t, prior=None):
        batch = (
            text_t.expand(self.batch_size, -1),
            image_t.expand(self.batch_size, -1, -1, -1),
        )
        optimizer = torch.optim.Adam(
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
            optimizer.zero_grad()
            out = (
                self.model.get_text_features(batch[0]),
                self.model.get_image_features(batch[1]),
            )
            loss_c, loss_f, loss_t = self.calc_loss(outputs=out[0], labels=out[1])
            loss_t.backward()
            optimizer.step()
    
        return loss_c, loss_f, loss_t

    def calc_loss(self, outputs, labels):
        """
        Combined IB loss.

        L = beta * KL_compression - cosine_fitting
        """
        compression_term = self.bottleneck.buffer_capacity.mean()
        fitting_term     = self.fitting_estimator(outputs, labels).mean()
        total            = self.beta * compression_term - fitting_term
        return compression_term, fitting_term, total