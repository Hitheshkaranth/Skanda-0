"""One Skanda read, shared by the benchmark (skanda/evaluate.py) and the app server (skanda/serve.py).

  generate the report → teacher-forced pass over prompt + reply → hidden state AT each [SEG] → projector → SAM 2
Run 2 adds the abnormal-score gate. When the checkpoint has a trained score head and thresholds
(skanda_config.json "thresholds": {modality: {"low", "high"}}, chosen on validation records by skanda/calibrate.py):
  p(abnormal) < low                      → the normal report, no outlines
  p(abnormal) > high and the reply says normal → generate again from the forced start of an abnormal report
Without them (run 1 checkpoints) a read is exactly what it was.
Run 3 adds classifier hints (skanda/hints.py): when the checkpoint has hint thresholds, an X-ray read starts its reply
with the score head's likely / possible classes; the returned text is the report after that line.
fill_seg (off unless asked): run 3 learned to leave the outline token off some X-ray findings (21% of them on the test
films, mostly effusion and cardiomegaly; the NIH label-only targets have no [SEG]). With fill_seg, a finding without one
gets "mask": "[SEG]" before the teacher-forced pass, so every finding is outlined.
Hint classifier: when the checkpoint has hint_classifier.pt (skanda/hint_classifier.py), the hint is written from its
probabilities instead of the score head's; the abnormal gate still uses the score head.
"""
import json

import numpy as np
import torch

from .model import sam_preprocess

ABNORMAL_PREFIX = '{"normal": false, "findings": [{"title": "'


def pad_vis(inp, total_len):
    """Vision kwargs for a forward over prompt + reply: per-token tensors (mm_token_type_ids) get text (0) for the reply."""
    plen = inp["input_ids"].shape[1]
    vis = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
    for k, v in vis.items():
        if torch.is_tensor(v) and v.ndim == 2 and v.shape == (1, plen):
            vis[k] = torch.cat([v, v.new_zeros(1, total_len - plen)], 1)
    return vis


def eos_ids(proc):
    return [proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]


def decode(proc, ids):
    return proc.tokenizer.decode(ids, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")


@torch.no_grad()
def abnormal_score(model, inp):
    """p(abnormal) and the class probabilities from the score head at the last prompt token, or None (no trained head)."""
    if not getattr(model, "has_score", False):
        return None
    out = model.vlm(**inp, output_hidden_states=True, return_dict=True)
    last = out.hidden_states[-1]
    s = model.scores(last, torch.tensor([last.shape[1] - 1], device=last.device))[0].float().sigmoid()
    return s.cpu()


def normal_text(stack=False):
    from .templates import report
    if stack:
        return report([], normal_text="No focal abnormality on these slices.")
    return report([])


SIDE_PHRASES = [("right cerebral hemisphere", "right"), ("left cerebral hemisphere", "left"),
                ("both cerebral hemispheres", "both")]


def mask_side(m):
    """Side of an outline (bool [H, W] or [S, H, W]) on a radiologically displayed image: image left = patient right.
    Volumes are weighted by voxel, not projected onto one slice: a projection lets a few stray voxels on the other side
    turn a one-sided lesion into "both" (run 2 MRI volumes: sides right 11 → 19 of 33 on validation, 46 → 88 of 145 on test)."""
    m = np.asarray(m)
    if not m.any():
        return None
    left_img = (np.nonzero(m)[-1] < m.shape[-1] / 2).mean()
    return "right" if left_img >= 0.8 else "left" if left_img <= 0.2 else "both"


def side_from_outlines(text, masks):
    """Brain MRI: take each finding's side from its outline, not from the words. In run 1 the outlines were in the right
    place while the text named the wrong hemisphere in ~35% of reads (the brain is nearly symmetric); outline-derived
    sides lifted reads with every side right from 52% to 64%. Returns (text, number of findings changed)."""
    try:
        rep = json.loads(text)
    except ValueError:
        return text, 0
    changed = 0
    for f, m in zip([f for f in rep.get("findings", []) if isinstance(f, dict)], masks):
        side = mask_side(m)
        if side is None:
            continue
        new = dict(SIDE_PHRASES_BY_SIDE)[side]
        for key in ("title", "location", "evidence"):
            v = f.get(key)
            if isinstance(v, str):
                for phrase, _ in SIDE_PHRASES:
                    if phrase in v and phrase != new:
                        f[key] = v = v.replace(phrase, new).replace("the patient's both", "both")
                        changed += key == "location"
    if changed:  # the impression lists the finding titles (skanda/templates.report): rebuild it from the corrected ones
        titles = [f.get("title") for f in rep.get("findings", []) if isinstance(f, dict) and f.get("title")]
        if titles:
            rep["impression"] = "; ".join(titles) + "."
    return (json.dumps(rep, ensure_ascii=False), changed) if changed else (text, 0)


SIDE_PHRASES_BY_SIDE = [(s, p) for p, s in SIDE_PHRASES]


@torch.no_grad()
def extend(inp, ids):
    """Processor output for the prompt + extra tokens (a forced start of the reply)."""
    out = dict(inp)
    out["input_ids"] = torch.cat([inp["input_ids"], ids], 1)
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    out.update({k: v for k, v in pad_vis(inp, out["input_ids"].shape[1]).items() if k in inp})
    return out


def read(model, proc, inp, images, stack=False, max_new_tokens=1024, temperature=0.0, normal_report=None,
         thresholds="model", modality="cxr", fill_seg=False):
    """inp: processor output for the prompt (batch of 1, on the model's device); images: PIL images (the X-ray or slice
    first; for a stack, every slice). Returns a dict: text, finished, new_tokens, seq, masks (bool [n, 256, 256] or
    [n, S, 256, 256]), obj (object-score probabilities per finding), p_abnormal, gate ('low' | 'high' | None)."""
    dev = inp["input_ids"].device
    eos = eos_ids(proc)
    th = model.thresholds if thresholds == "model" else thresholds
    if th and "low" not in th:  # per modality: {"cxr": {...}, "ct": {...}, "mri": {...}}
        th = th.get(modality)
    sc = abnormal_score(model, inp)
    p_abn = float(sc[0]) if sc is not None else None
    sample = temperature > 0.05
    gkw = {"do_sample": sample, "eos_token_id": eos, "max_new_tokens": max_new_tokens,
           **({"temperature": temperature, "top_p": 0.95} if sample else {})}
    hint = ""
    hc = getattr(model, "hint_cls", None)
    if getattr(model, "hints", None) and (sc is not None or hc is not None) and modality == "cxr" and not stack:
        from .hints import hint_from_scores
        if hc is not None:  # the hint classifier's probabilities (its thresholds are in model.hints)
            hc.to(dev)
            hs = hc.predict(images[:1])[0, 1:].tolist()
        else:
            hs = [float(x) for x in sc[1:]]
        hint = hint_from_scores(hs, model.hints)
        inp = extend(inp, proc.tokenizer(hint, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev))
    plen = inp["input_ids"].shape[1]  # the reply starts after the hint
    gate = None
    if th and p_abn is not None and p_abn < th["low"]:
        gate = "low"
        text = normal_report or normal_text(stack)
        ids = proc.tokenizer(text + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        seq = torch.cat([inp["input_ids"], ids], 1)
        return {"text": text, "finished": True, "new_tokens": 0, "seq": seq, "plen": plen,
                "masks": torch.zeros(0, *((len(images),) if stack else ()), 256, 256, dtype=torch.bool),
                "obj": [], "p_abnormal": p_abn, "scores": sc, "gate": gate}
    seq = model.vlm.generate(**inp, **gkw)
    gen = seq[0, plen:]
    text = decode(proc, gen)
    if th and p_abn is not None and p_abn > th["high"]:
        try:
            said_normal = bool(json.loads(text).get("normal"))
        except (ValueError, AttributeError):
            said_normal = False
        if said_normal:  # the score is sure it is abnormal: make the model write the findings it would write
            gate = "high"
            pre = proc.tokenizer(ABNORMAL_PREFIX, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
            seq = model.vlm.generate(**extend(inp, pre), **gkw)
            gen = seq[0, plen:]
            text = decode(proc, gen)
    finished = bool(len(gen) and int(gen[-1]) in eos)
    if fill_seg and not stack:
        try:
            rep = json.loads(text)
        except ValueError:
            rep = None
        fs = rep.get("findings") if isinstance(rep, dict) else None
        if isinstance(fs, list) and any(isinstance(f, dict) and f.get("mask") != "[SEG]" for f in fs):
            for f in fs:
                if isinstance(f, dict):
                    f["mask"] = "[SEG]"
            text = json.dumps(rep)
            ids = proc.tokenizer(text + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
            seq = torch.cat([seq[:, :plen], ids], 1)
    # hidden state AT each generated [SEG] (teacher-forced pass over prompt + reply, as in training)
    out = model.vlm(input_ids=seq, attention_mask=torch.ones_like(seq), output_hidden_states=True, return_dict=True,
                    **pad_vis(inp, seq.shape[1]))
    pos = (seq[0] == model.seg_token_id).nonzero().flatten()
    pos = pos[pos >= plen]
    prompts = model.seg_proj(out.hidden_states[-1][0, pos])
    dt = next(model.sam.parameters()).dtype
    if stack:  # each [SEG] on every slice, 8 slices at a time → [n, S, 256, 256]
        sl = torch.stack([sam_preprocess(np.array(im)) for im in images]).to(dev, dt)
        if len(pos):
            parts = [model.volume_masks(sl[i:i + 8], prompts, return_obj=True) for i in range(0, len(sl), 8)]
            logits = torch.cat([p[0] for p in parts], 1)
            obj = torch.cat([p[1] for p in parts], 1).float().sigmoid().amax(1)
        else:
            logits, obj = torch.zeros(0, len(images), 256, 256, device=dev), torch.zeros(0)
    else:
        sam_img = sam_preprocess(np.array(images[0]))[None].to(dev, dt)
        if len(pos):
            m, o = model.sam_masks(sam_img, [prompts], return_obj=True)
            logits, obj = m[0], o[0].float().sigmoid()
        else:
            logits, obj = torch.zeros(0, 256, 256, device=dev), torch.zeros(0)
    masks = (logits.float() > 0).cpu()
    side_fixed = 0
    if modality == "mri" and len(masks):
        text, side_fixed = side_from_outlines(text, list(masks.numpy()))
    return {"text": text, "finished": finished, "new_tokens": int(len(gen)), "seq": seq, "plen": plen,
            "masks": masks, "obj": [round(float(x), 3) for x in obj.cpu()],
            "p_abnormal": p_abn, "scores": sc, "gate": gate, "side_fixed": side_fixed, "hint": hint}
