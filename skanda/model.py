"""SkandaModel: Qwen3-VL (MoE) language-vision model + [SEG] projector + SAM 2 mask decoder, as one nn.Module.

Training forward (teacher forcing): the language model's loss on the report, plus, for every [SEG] token in the target,
the last-layer hidden state AT that token's position → projector → SAM 2 sparse prompt → mask logits, scored with
BCE + Dice against the radiologist mask. Inference uses the same position (see skanda/infer.py), so train and serve agree.

Run 2 additions (a checkpoint without them loads with them untrained and unused):
  score head   last-layer hidden state at the last prompt token (just before the reply) → 1 + 13 logits: "abnormal"
               and the 13 ChestX-Det classes. Trained with BCE on known labels (-1 = unknown). At inference it gives a
               calibrated abnormal score whose thresholds are chosen on dev films (skanda/infer.py: gate).
  object score SAM 2's own "is there an object" logit for each decoded mask, trained with BCE (1 = the target mask is
               non-empty); absent-structure prompts teach it 0. Reported per finding as a confidence.
  mask weights per-mask weight on BCE + Dice (0 = text-only supervision, e.g. a finding known only from a box).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

from . import SEG_TOKEN
from .sam2 import SAM2

# score head outputs: index 0 = abnormal, then these classes (ChestX-Det's 13, the app benchmark's classes)
SCORE_CLASSES = ["Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule", "Effusion", "Emphysema",
                 "Fibrosis", "Fracture", "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"]

SAM_MEAN = (0.485, 0.456, 0.406)
SAM_STD = (0.229, 0.224, 0.225)


def sam_preprocess(gray_or_rgb_uint8):
    """HxW or HxWx3 uint8 (numpy) → [3, 1024, 1024] float, ImageNet-normalised (SAM 2's input)."""
    t = torch.as_tensor(gray_or_rgb_uint8)
    if t.ndim == 2:
        t = t[None].expand(3, -1, -1)
    else:
        t = t.permute(2, 0, 1)
    t = F.interpolate(t[None].float(), size=(1024, 1024), mode="bilinear", align_corners=False)[0] / 255.0
    mean, std = torch.tensor(SAM_MEAN)[:, None, None], torch.tensor(SAM_STD)[:, None, None]
    return (t - mean) / std


def dice_loss(logits, target, eps=1.0, weight=None):
    p = logits.sigmoid().flatten(1)
    t = target.flatten(1)
    d = 1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return d.mean() if weight is None else (d * weight).sum() / weight.sum().clamp(min=1e-6)


def masked_asl(logits, target, gamma_neg=4.0, clip=0.05):
    """Run 4: asymmetric loss (Ridnik et al., ICCV 2021) on the class columns, BCE on column 0 (p(abnormal)).
    Easy negatives are down-weighted (focusing gamma_neg, probability shift clip), so the rare positives carry more of
    the loss; CXR-LT 2023/2024's top entries all re-weighted their loss this way. -1 = unknown, skipped."""
    known = target >= 0
    if not known.any():
        return logits.sum() * 0
    x, t = logits.float(), target.float()
    p = torch.sigmoid(x)
    pm = (p - clip).clamp(min=0)
    pos = t * F.logsigmoid(x)
    neg = (1 - t) * torch.log((1 - pm).clamp(min=1e-8)) * pm.pow(gamma_neg)
    loss = -(pos + neg)
    loss[:, 0] = F.binary_cross_entropy_with_logits(x[:, 0], t[:, 0].clamp(min=0), reduction="none")
    return loss[known].mean()


def masked_bce(logits, target):
    """BCE over entries with target >= 0 (-1 = unknown); 0 when nothing is known."""
    known = target >= 0
    if not known.any():
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[known].float(), target[known].float())


class SkandaModel(nn.Module):
    def __init__(self, vlm, seg_token_id, sam=None, bce_weight=2.0, dice_weight=0.5, score_weight=0.5, obj_weight=0.2):
        super().__init__()
        self.vlm = vlm
        self.seg_token_id = seg_token_id
        self.sam = sam if sam is not None else SAM2()
        hidden = vlm.config.text_config.hidden_size
        self.seg_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                      nn.Linear(hidden, self.sam.hidden_dim), nn.Dropout(0.0))
        self.score_head = nn.Linear(hidden, 1 + len(SCORE_CLASSES))
        self.has_score = False          # True once the checkpoint holds a trained score head
        self.thresholds = None          # abnormal-score gate {"low", "high"} chosen on dev films
        self.hints = None               # run 3: classifier-hint thresholds {class: {"likely", "possible"}}
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.score_weight, self.obj_weight = score_weight, obj_weight
        self.score_loss = "bce"         # run 4: "asl" (train.py --score-loss)

    # ---- SAM 2, single image (X-ray, or one slice): no memory, language prompt only ----
    def sam_features(self, sam_images):
        """Frozen SAM 2 encoder → (feats, sizes): 3 levels flattened HWxBxC, as SAM 2 uses them. Cacheable."""
        sm = self.sam.sam2_model
        _, feats, _, sizes = sm._prepare_backbone_features(sm.forward_image(sam_images))
        return feats, sizes

    def sam_masks(self, sam_images, prompts, sam_feats=None, return_obj=False):
        """sam_images [B,3,1024,1024] (or sam_feats=(feats, sizes) from the cache); prompts: list (len B) of [n_i, 256]
        → list of [n_i, 256, 256] mask logits (and, with return_obj, a list of [n_i] object-score logits)."""
        sm = self.sam.sam2_model
        if sam_feats is None and sam_images is None:  # no image given: only valid when there is nothing to decode
            assert all(p.shape[0] == 0 for p in prompts), "[SEG] tokens but no SAM image or cached features"
            dev = prompts[0].device if prompts else self.score_head.weight.device
            out = [torch.zeros(0, 256, 256, device=dev) for _ in prompts]
            return (out, [torch.zeros(0, device=dev) for _ in prompts]) if return_obj else out
        feats, sizes = sam_feats if sam_feats is not None else self.sam_features(sam_images)
        top = feats[-1] + sm.no_mem_embed if sm.directly_add_no_mem_embed else feats[-1]
        B = feats[-1].shape[1]
        pix = top.permute(1, 2, 0).reshape(B, sm.hidden_dim, *sizes[-1])
        high = [f.permute(1, 2, 0).reshape(B, -1, *s) for f, s in zip(feats[:-1], sizes[:-1])]
        out, objs = [], []
        for b, p in enumerate(prompts):
            n = p.shape[0]
            if n == 0:
                out.append(pix.new_zeros(0, 256, 256))
                objs.append(pix.new_zeros(0))
                continue
            res = sm._forward_sam_heads(
                backbone_features=pix[b:b + 1].expand(n, -1, -1, -1),
                high_res_features=[h[b:b + 1].expand(n, -1, -1, -1) for h in high],
                multimask_output=False,
                language_embd=p[:, None, :].to(pix.dtype),
            )
            out.append(res[0][:, 0])  # low-res mask logits [n, 256, 256]
            objs.append(res[-1].reshape(n, -1)[:, 0])  # object-score logit per mask [n]
        return (out, objs) if return_obj else out

    def volume_masks(self, slice_images, prompts, return_obj=False):
        """Stage 2: the same [SEG] prompts decoded on K slices of one volume. slice_images [K,3,1024,1024];
        prompts [n, 256] → [n, K, 256, 256] mask logits (and [n, K] object scores: is the lesion on this slice?).
        SAM 2's encoder is frozen, so it runs without gradients."""
        with torch.no_grad():
            feats, sizes = self.sam_features(slice_images)
        k = slice_images.shape[0]
        m, o = self.sam_masks(None, [prompts] * k, sam_feats=(feats, sizes), return_obj=True)
        return (torch.stack(m, 1), torch.stack(o, 1)) if return_obj else torch.stack(m, 1)

    def scores(self, hidden_last, pos):
        """Score-head logits [B, 1 + C] from last-layer hidden states [B, L, H] at positions pos [B]."""
        h = hidden_last[torch.arange(hidden_last.shape[0], device=hidden_last.device), pos]
        return self.score_head(h.to(self.score_head.weight.dtype))

    def forward(self, input_ids, attention_mask, labels, gt_masks, sam_images=None, sam_feats=None, loss_weights=None,
                vol_images=None, score_pos=None, score_target=None, mask_weights=None, **vision_inputs):
        """gt_masks: list (len B) of float tensors [n_i, 256, 256] in target order of the [SEG] tokens
        (stage 2: [n_i, K, 256, 256] over the K supervised slices of a volume, with vol_images a list of [K,3,1024,1024]).
        loss_weights: optional [B, L] per-token weights for the text loss (JSON field names are down-weighted).
        score_pos [B] / score_target [B, 1 + C]: last prompt position and score labels (-1 = unknown).
        mask_weights: optional list (len B) of [n_i] per-mask weights for BCE + Dice."""
        out = self.vlm(input_ids=input_ids, attention_mask=attention_mask, labels=None if loss_weights is not None else labels,
                       output_hidden_states=True, return_dict=True, **vision_inputs)
        if loss_weights is not None:
            lg, lb, lw = out.logits[:, :-1].float(), labels[:, 1:], loss_weights[:, 1:].float()
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lb.reshape(-1), ignore_index=-100, reduction="none")
            valid = (lb.reshape(-1) != -100).float()
            out.loss = (ce * lw.reshape(-1) * valid).sum() / (lw.reshape(-1) * valid).sum().clamp(min=1)
        last = out.hidden_states[-1]
        seg_pos = (input_ids == self.seg_token_id) & (labels != -100)  # only [SEG] tokens in the answer, not the prompt
        prompts = [self.seg_proj(last[b][seg_pos[b]]) for b in range(input_ids.shape[0])]
        if vol_images is not None:  # Dice over each lesion's K slices = a 3D Dice; empty slices teach where it ends
            res = [self.volume_masks(v, p, return_obj=True) if len(p) else
                   (p.new_zeros(0, v.shape[0], 256, 256), p.new_zeros(0, v.shape[0])) for v, p in zip(vol_images, prompts)]
            logits, objs = [r[0] for r in res], [r[1] for r in res]
        else:
            logits, objs = self.sam_masks(sam_images, prompts, sam_feats, return_obj=True)
        lg = torch.cat(logits) if logits else None
        zero = out.loss.new_zeros(())
        mask_bce = mask_dice = obj_loss = zero
        if lg is not None and lg.numel():
            gt = torch.cat(gt_masks).to(lg.dtype)
            w = torch.cat(mask_weights).to(lg.dtype) if mask_weights is not None else torch.ones(len(gt), device=lg.device, dtype=lg.dtype)
            if w.sum() > 0:
                per = F.binary_cross_entropy_with_logits(lg, gt, reduction="none").flatten(1).mean(1)
                mask_bce = (per * w).sum() / w.sum()
                mask_dice = dice_loss(lg, gt, weight=w)
            # object score: is the target mask (or, for volumes, this slice of it) non-empty? weight-0 masks are unknown
            ob = torch.cat(objs)
            present = (gt.flatten(2) if gt.ndim == 4 else gt.flatten(1)[:, None]).amax(-1) > 0.5
            ot = torch.where((w > 0).view(-1, *([1] * (present.ndim - 1))).expand_as(present), present.to(ob.dtype),
                             torch.full_like(present, -1, dtype=ob.dtype))
            obj_loss = masked_bce(ob.reshape(ot.shape), ot)
        score_loss = zero
        if score_pos is not None and score_target is not None:
            score_fn = masked_asl if self.score_loss == "asl" else masked_bce
            score_loss = score_fn(self.scores(last, score_pos), score_target)
        loss = (out.loss + self.bce_weight * mask_bce + self.dice_weight * mask_dice + self.obj_weight * obj_loss
                + self.score_weight * score_loss)
        return {"loss": loss, "text_loss": out.loss.detach(), "mask_bce": mask_bce.detach(),
                "mask_dice": mask_dice.detach(), "obj_loss": obj_loss.detach(), "score_loss": score_loss.detach(),
                "mask_logits": logits}

    # ---- assembly / save / load ----
    @staticmethod
    def load_sam_from_sa2va(sam, sa2va_dir):
        """Copy grounding_encoder.* from a Sa2VA checkpoint (its SAM 2 was trained to take language prompts)."""
        from safetensors import safe_open
        idx = json.load(open(os.path.join(sa2va_dir, "model.safetensors.index.json")))["weight_map"]
        state = {}
        for shard in sorted({v for k, v in idx.items() if k.startswith("grounding_encoder.")}):
            with safe_open(os.path.join(sa2va_dir, shard), "pt") as f:
                for k in f.keys():
                    if k.startswith("grounding_encoder."):
                        state[k[len("grounding_encoder."):]] = f.get_tensor(k)
        missing, unexpected = sam.load_state_dict(state, strict=False)
        return missing, unexpected

    def save(self, out_dir, processor=None, meta=None):
        os.makedirs(out_dir, exist_ok=True)
        # mid-training checkpoints of a LoRA run hold only the adapter (vlm_lora/); final ones are merged (vlm/)
        # small shards: each one is staged in host memory before it is written, and on the GB10's shared memory a
        # 50 GB shard next to the merged model ran out of memory (run 3 stage 2's final save)
        self.vlm.save_pretrained(os.path.join(out_dir, "vlm_lora" if hasattr(self.vlm, "peft_config") else "vlm"),
                                 max_shard_size="4GB")
        heads = {"sam": self.sam.state_dict(), "seg_proj": self.seg_proj.state_dict(),
                 "score_head": self.score_head.state_dict(), "has_score": self.has_score}
        if hasattr(self.vlm, "peft_config"):  # adapter-only checkpoint: keep the trained [SEG] rows with the heads
            i = self.seg_token_id
            heads["seg_rows"] = {"embed": self.vlm.get_input_embeddings().weight[i].detach().cpu().clone(),
                                 "lm_head": self.vlm.get_output_embeddings().weight[i].detach().cpu().clone()}
        torch.save(heads, os.path.join(out_dir, "skanda_heads.pt"))
        json.dump({"seg_token": SEG_TOKEN, "seg_token_id": self.seg_token_id, **(meta or {})},
                  open(os.path.join(out_dir, "skanda_config.json"), "w"), indent=1)
        if processor is not None:
            processor.save_pretrained(os.path.join(out_dir, "vlm"))

    @classmethod
    def load(cls, path, dtype=torch.bfloat16, device_map=None):
        cfg = json.load(open(os.path.join(path, "skanda_config.json")))
        if os.path.exists(os.path.join(path, "vlm", "skanda_quant.json")):  # int8/int4 build (skanda/quant.py)
            from .quant import load_vlm
            vlm = load_vlm(os.path.join(path, "vlm"), dtype)
        else:
            vlm = AutoModelForImageTextToText.from_pretrained(os.path.join(path, "vlm"), dtype=dtype, device_map=device_map)
        m = cls(vlm, cfg["seg_token_id"])
        heads = torch.load(os.path.join(path, "skanda_heads.pt"), map_location="cpu")
        m.sam.load_state_dict(heads["sam"])
        m.seg_proj.load_state_dict(heads["seg_proj"])
        if "score_head" in heads:  # run 1 checkpoints have none: the head stays untrained and unused
            m.score_head.load_state_dict(heads["score_head"])
            m.has_score = bool(heads.get("has_score", False))
        m.thresholds = cfg.get("thresholds")  # abnormal-score gate chosen on dev films (skanda/infer.py), if any
        m.hints = cfg.get("hints")  # run 3: per-class hint thresholds (skanda/calibrate_hints.py), if any
        m = m.to(dtype)
        hc = os.path.join(path, "hint_classifier.pt")  # hints from the separate classifier (skanda/hint_classifier.py)
        if os.path.exists(hc):  # kept off the module tree: inference only, never trained or saved with the heads
            from .hint_classifier import HintClassifier
            object.__setattr__(m, "hint_cls", HintClassifier.load(hc))
        return m
