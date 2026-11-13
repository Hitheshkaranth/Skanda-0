"""Skanda training records → model inputs (Qwen3-VL chat format + SAM 2 images + [SEG] masks).

Loss is on the assistant reply only. Masks are fed at SAM 2's low-res output size (256²).
Optional record fields (run 2):
  "target": null      label-only record (image-level labels, no report): no text loss, score head only
  "score": {...}      score-head labels: {"abnormal": 0/1, "<ChestX-Det class>": 0/1}; missing keys are unknown.
                      Without it, "abnormal" comes from the target's "normal" field (and, for ChestX-Det records,
                      the classes from "labels").
  "mask_weights"      per-mask weight for BCE + Dice (default 1; 0 = the mask is not trusted, text only)
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..model import SCORE_CLASSES, sam_preprocess
from ..hints import hint_from_labels


class SkandaDataset(Dataset):
    def __init__(self, *paths, limit=0):
        self.rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _messages(r, with_answer, hint=""):
    content = [{"type": "image"} for _ in r["images"]] + [{"type": "text", "text": r["prompt"]}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": r["system"]}]},
            {"role": "user", "content": content}]
    if with_answer and r.get("target") is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": hint + r["target"]}]})
    return msgs


# JSON structure of skanda-report/2; tokens that only spell these get a lower text-loss weight
BOILERPLATE = ['{"normal": ', ', "findings": [', '{"title": "', '", "evidence": "', '", "location": "',
               '", "severity": "', '", "mask": "', '"}', '"}, ', '], "impression": "', '"}', '[]', '{', '}', '"', ': ', ', ']


def score_target(r):
    """[1 + C] score-head labels for a record, -1 where unknown (connector records: all unknown)."""
    t = torch.full((1 + len(SCORE_CLASSES),), -1.0)
    if r.get("stage") == "connector":
        return t
    sc = dict(r.get("score") or {})
    if "abnormal" not in sc and r.get("target") is not None:
        try:
            sc["abnormal"] = int(not json.loads(r["target"])["normal"])
        except (ValueError, KeyError, TypeError):
            pass
    if r.get("source") == "chestxdet" and "labels" in r:  # radiologist labels: every class is known
        for c in SCORE_CLASSES:
            sc.setdefault(c, int(c in r["labels"]))
    if "abnormal" in sc:
        t[0] = float(sc["abnormal"])
    for i, c in enumerate(SCORE_CLASSES, 1):
        if c in sc:
            t[i] = float(sc[c])
    return t


def sam_cache_path(cache_dir, image_path):
    import hashlib
    return f"{cache_dir}/{hashlib.sha1(image_path.encode()).hexdigest()}.pt"


class SkandaCollator:
    def __init__(self, processor, max_pixels=1024 * 1024, min_pixels=256 * 256, max_len=8192, boiler_weight=0.3,
                 sam_cache=None, hints=False):
        self.p, self.max_len, self.hints = processor, max_len, hints
        self.kw = {"max_pixels": max_pixels, "min_pixels": min_pixels}
        self.boiler_weight, self.sam_cache = boiler_weight, sam_cache
        tok = processor.tokenizer
        self.boiler_ids = torch.tensor(sorted({i for frag in BOILERPLATE for i in tok(frag, add_special_tokens=False)["input_ids"]}))

    def __call__(self, rows):
        images = [[Image.open(x).convert("RGB") for x in r["images"]] for r in rows]
        prompts = [self.p.apply_chat_template(_messages(r, False), tokenize=False, add_generation_prompt=True) for r in rows]
        hints = [self.hint(r) for r in rows]
        # label-only records (target null) are just the prompt: no text loss, the score head reads its last token
        full = [self.p.apply_chat_template(_messages(r, True, h), tokenize=False) if r.get("target") is not None else pr
                for r, pr, h in zip(rows, prompts, hints)]
        batch = self.p(text=full, images=images, return_tensors="pt", padding=True, images_kwargs=self.kw)
        plen = [len(x) for x in self.p(text=prompts, images=images, images_kwargs=self.kw)["input_ids"]]
        labels = batch["input_ids"].clone()
        for i, (n, h) in enumerate(zip(plen, hints)):  # the hint is input, not something to learn to write
            labels[i, :n + (len(self.p.tokenizer(h, add_special_tokens=False)["input_ids"]) if h else 0)] = -100
        labels[batch["attention_mask"] == 0] = -100
        if labels.shape[1] > self.max_len:
            raise ValueError(f"{labels.shape[1]} tokens > max_len {self.max_len}")
        batch["labels"] = labels
        w = torch.ones_like(labels, dtype=torch.float32)
        w[torch.isin(batch["input_ids"], self.boiler_ids)] = self.boiler_weight
        batch["loss_weights"] = w
        # SAM 2 sees the first image of each record (X-ray, or the key slice of a series): cached features if present
        if not any(r["masks"] for r in rows):  # nothing to outline (normal films, label-only records): no SAM inputs
            pass
        elif self.sam_cache:
            cached = [torch.load(sam_cache_path(self.sam_cache, r["images"][0])) for r in rows]
            sizes = cached[0]["sizes"]
            batch["sam_feats"] = ([torch.cat([c["feats"][k] for c in cached], 1).float() for k in range(3)], sizes)
        else:
            batch["sam_images"] = torch.stack([sam_preprocess(np.array(im[0])) for im in images])
        batch["gt_masks"] = [
            torch.stack([F.interpolate(torch.from_numpy(np.array(Image.open(m).convert("L")) > 127)[None, None].float(),
                                       size=(256, 256), mode="bilinear", align_corners=False)[0, 0] for m in r["masks"]])
            if r["masks"] else torch.zeros(0, 256, 256) for r in rows]
        batch["mask_weights"] = [torch.tensor(r.get("mask_weights") or [1.0] * len(r["masks"]), dtype=torch.float32)
                                 for r in rows]
        batch["score_pos"] = torch.tensor([n - 1 for n in plen])  # before the hint: the head never sees its own hint
        batch["score_target"] = torch.stack([score_target(r) for r in rows])
        return batch

    def hint(self, r):
        """Run 3: a noisy classifier hint for X-ray records with a report (skanda/hints.py), else ""."""
        if not self.hints or r.get("modality") != "cxr" or r.get("target") is None:
            return ""
        t = score_target(r)[1:]
        pos = [c for c, v in zip(SCORE_CLASSES, t) if v == 1]
        neg = [c for c, v in zip(SCORE_CLASSES, t) if v == 0]
        return hint_from_labels(pos, neg)


class SkandaVolumeCollator(SkandaCollator):
    """Stage 2 (3D stacks): all slices of a volume go to Qwen3-VL at low resolution (32 slices + key slice, ~100 tokens
    each at the default 320² cap); SAM 2 decodes each [SEG] on up to `max_slices` slices: the key slice, slices with
    lesion (evenly spread) and `empty_slices` lesion-free slices, so the Dice over them is a 3D Dice."""

    def __init__(self, processor, max_pixels=320 * 320, min_pixels=128 * 128, max_slices=10, empty_slices=2, **kw):
        super().__init__(processor, max_pixels=max_pixels, min_pixels=min_pixels, **{**kw, "sam_cache": None})
        self.max_slices, self.empty_slices = max_slices, empty_slices

    def pick_slices(self, masks, key):
        """masks [n, S, H, W] bool → sorted slice indices to supervise."""
        has = masks.reshape(masks.shape[0], masks.shape[1], -1).any(-1).any(0) if masks.shape[0] else np.zeros(masks.shape[1], bool)
        les = [int(i) for i in np.flatnonzero(has) if i != key]
        empty = [int(i) for i in np.flatnonzero(~has) if i != key]
        n_les = self.max_slices - 1 - min(self.empty_slices, len(empty))
        pick = {key} | {les[int(i)] for i in np.linspace(0, len(les) - 1, min(n_les, len(les))).round()} if les else {key}
        if empty:
            pick |= {empty[int(i)] for i in np.linspace(0, len(empty) - 1, min(self.empty_slices, len(empty))).round()}
        return sorted(pick)

    def __call__(self, rows):
        assert len(rows) == 1, "stage 2 uses batch size 1"
        r = rows[0]
        z = np.load(r["masks_npz"])
        masks = z["masks"] if z["masks"].ndim == 4 else np.zeros((0, len(r["images"]), 1, 1), bool)
        pick = self.pick_slices(masks, r["key_slice"])
        batch = super().__call__([{**r, "masks": []}])  # text, labels and loss weights from the 2D path (no SAM inputs)
        batch.pop("sam_feats", None); batch.pop("sam_images", None)
        batch["vol_images"] = [torch.stack([sam_preprocess(np.array(Image.open(r["images"][i]).convert("RGB"))) for i in pick])]
        gt = torch.from_numpy(masks[:, pick].astype(np.float32)) if masks.shape[0] else torch.zeros(0, len(pick), 256, 256)
        if gt.shape[0]:
            gt = F.interpolate(gt, size=(256, 256), mode="bilinear", align_corners=False)
        batch["gt_masks"] = [gt]
        batch["mask_weights"] = [torch.ones(gt.shape[0])]
        self.last_slices = pick
        return batch


class SkandaMixedCollator:
    """Stage 2 in run 2: 3D stacks (records with masks_npz) go through the volume collator, every other record through
    the 2D collator (with the SAM cache), so 2D replay, X-rays included, can be mixed into stage 2. Batch size 1."""

    def __init__(self, processor, stack_pixels=320 * 320, **kw2d):
        self.vol = SkandaVolumeCollator(processor, max_pixels=stack_pixels)  # volumes: no hints (CT/MRI have no classes)
        self.flat = SkandaCollator(processor, **kw2d)

    def __call__(self, rows):
        assert len(rows) == 1, "stage 2 uses batch size 1"
        return self.vol(rows) if "masks_npz" in rows[0] else self.flat(rows)
