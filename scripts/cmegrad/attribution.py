import torch
import torch.nn.functional as F

# from copy import deepcopy
# import numpy as np

from scripts.cmegrad.model import set_libra_mode
from scripts.cmegrad.metrics import entropy_weight


class CMEGradLXMERT:

    def __init__(
        self,
        model,
        tokenizer,
        id2label: dict,
        feat_index: dict,
        feat_tsv: str,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.feat_index = feat_index
        self.feat_tsv = feat_tsv
        self.device = device

    # vision attribution
    def _vision_attr(self, enc, feats, boxes):

        set_libra_mode(True)
        self.model.zero_grad()

        # clear old hooks
        # for m in self.model.modules():
        #     m._forward_hooks.clear()
        for _, m in self.model.named_modules():
            m._forward_hooks.clear()

        vis_ref = [None]

        def vis_hook(module, inp, out):

            # output can be tuple sometimes
            t = out[0] if isinstance(out, tuple) else out

            if isinstance(t, torch.Tensor):
                t.retain_grad()
                vis_ref[0] = t

        h = self.model.lxmert.encoder.visn_fc.register_forward_hook(
            vis_hook
        )

        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc["token_type_ids"],
            visual_feats=feats,
            visual_pos=boxes,
        )

        h.remove()

        pred_idx = (
            out.question_answering_score
            .argmax(-1)
            .item()
        )

        target_score = out.question_answering_score[
            0,
            pred_idx
        ]

        target_score.backward()

        with torch.no_grad():

            ve = vis_ref[0]

            if ve is not None and ve.grad is not None:

                vis_attr = (
                    ve * ve.grad
                ).sum(
                    dim=-1
                ).squeeze(0).cpu()

            else:
                print("  ⚠️  vis grad is None")
                vis_attr = torch.zeros(
                    feats.shape[1]
                )

        return vis_attr.detach(), pred_idx

    # text attribution
    def _text_attr(self, enc, feats, boxes):

        set_libra_mode(True)
        self.model.zero_grad()

        # remove hooks from previous pass
        for _, m in self.model.named_modules():
            m._forward_hooks.clear()

        T = enc["input_ids"].shape[1]
        ref = [None]

        def hook(module, inp_, out):

            t = (
                out[0]
                if isinstance(out, tuple)
                else out
            )

            if isinstance(t, torch.Tensor):
                t.retain_grad()
                ref[0] = t

        h = self.model.lxmert.embeddings.register_forward_hook(
            hook
        )

        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc["token_type_ids"],
            visual_feats=feats,
            visual_pos=boxes,
        )

        h.remove()

        pred_idx = (
            out.question_answering_score
            .argmax(-1)
            .item()
        )

        out.question_answering_score[
            0,
            pred_idx
        ].backward()

        with torch.no_grad():

            te = ref[0]

            if te is not None and te.grad is not None:

                ixg = (
                    te * te.grad
                ).sum(
                    dim=-1
                ).squeeze(0)

                text_attr = ixg[:T].cpu()

            else:
                text_attr = torch.zeros(T)

        return text_attr.detach(), pred_idx

    # bridge thing
    def _bridge_matrix(
        self,
        enc,
        feats,
        boxes,
        T: int,
        V: int = 36,
    ):

        with torch.no_grad():

            out = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc["token_type_ids"],
                visual_feats=feats,
                visual_pos=boxes,
                output_attentions=True,
            )

        layers = []

        if (
            hasattr(out, "cross_encoder_attentions")
            and out.cross_encoder_attentions is not None
        ):

            for layer_attn in out.cross_encoder_attentions:

                lang2vis = (
                    layer_attn[0]
                    if isinstance(layer_attn, tuple)
                    else layer_attn
                )

                avg = (
                    lang2vis
                    .squeeze(0)
                    .mean(0)
                )

                layers.append(avg.cpu())

        if layers:

            X = torch.stack(layers).mean(0)

        else:

            X = torch.ones(T, V) / V

            print(
                "  ⚠️  No cross-attention found, using uniform X"
            )

        return X

    def _ew(self, attr):
        return entropy_weight(attr)

    @staticmethod
    def _norm(x):

        s = x.abs().sum()

        # don't normalize very small stuff
        if s < 1e-10:
            return x

        return x / s

    def _refine(
        self,
        text_attr,
        vis_attr,
        w_lang,
        w_vis,
        X,
    ):

        X = X.to(text_attr.device)
        vis_attr = vis_attr.to(text_attr.device)

        T = text_attr.shape[0]
        V = vis_attr.shape[0]

        if X.shape != (T, V):
            X = X[:T, :V]

        cross_vis = self._norm(
            X.T @ text_attr
        )

        cross_lang = self._norm(
            X @ vis_attr
        )

        agree_vis = (
            cross_vis * vis_attr > 0
        ).float()

        agree_lang = (
            cross_lang * text_attr > 0
        ).float()

        # additive refinement
        vis_r = (
            vis_attr
            + 0.15 * agree_vis * cross_vis
        )

        lang_r = (
            text_attr
            + 0.15 * agree_lang * cross_lang
        )

        return lang_r, vis_r

    def generate(
        self,
        question: str,
        image_id: int
    ) -> tuple:

        from scripts.cmegrad.features import load_image_features_fast

        feats, boxes = load_image_features_fast(
            image_id,
            self.feat_index,
            self.feat_tsv
        )

        enc = self.tokenizer(
            question,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        feats = feats.to(self.device)
        boxes = boxes.to(self.device)

        # pass 1
        vis_attr, pred_idx = self._vision_attr(
            enc,
            feats,
            boxes
        )

        # pass 2
        text_attr, _ = self._text_attr(
            enc,
            feats,
            boxes
        )

        T = enc["input_ids"].shape[1]
        V = vis_attr.shape[0]

        w_vis, H_vis = self._ew(vis_attr)
        w_lang, H_lang = self._ew(text_attr)

        # bridge
        X = self._bridge_matrix(
            enc,
            feats,
            boxes,
            T,
            V
        )

        lang_r, vis_r = self._refine(
            text_attr,
            vis_attr,
            w_lang,
            w_vis,
            X
        )

        tokens = self.tokenizer.convert_ids_to_tokens(
            enc["input_ids"][0].tolist()
        )

        return lang_r, vis_r, {
            "question": question,
            "H_vis": H_vis,
            "H_lang": H_lang,
            "H_gap": abs(H_vis - H_lang),
            "w_vis": w_vis,
            "w_lang": w_lang,
            "pred_idx": pred_idx,
            "pred_answer": self.id2label[pred_idx],
            "tokens": tokens,
            "boxes": boxes.squeeze(0).cpu(),
            "vis_attr_raw": vis_attr.cpu(),
            "X_shape": list(X.shape),
            "X_std": X.std().item(),
        }
