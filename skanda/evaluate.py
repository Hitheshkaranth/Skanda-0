"""Benchmark a Skanda checkpoint on the held-out test sets: written report + one outline per finding.

  python -I skanda/evaluate.py --model runs/stage1/final --data data/skanda/cxr/sft_test.jsonl data/skanda/msd/*/sft_test.jsonl \
      --out runs/eval
Greedy generation, then one forward pass over prompt + reply with the hidden state read AT each [SEG] token (as in
training) → projector → SAM 2 mask. Per record it writes one line to <out>/results.jsonl (resumable: done ids are
skipped) and the predicted masks to <out>/masks/. Records are taken round-robin across modalities so a partial run
still covers X-ray, CT and MRI. Scoring: a radiologist mask is found when a
predicted mask overlaps it with IoU >= 0.3 (greedy one-to-one by IoU); for found pairs we compare the finding type
(title before the first comma) and the side (right / left / bilateral / midline) of the location.
3D stacks (stack_test.jsonl, records with masks_npz): all slices go in at --stack-pixels, each [SEG] is decoded on
every slice, and IoU / Dice are computed over the whole volume (modality reported as "ct-3d" / "mri-3d").
--summary prints the table from an existing results.jsonl without loading the model.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

IOU_HIT = 0.3
SIDES = ("bilateral", "right", "left", "midline")


def finding_type(title):
    return (title or "").split(",")[0].strip().lower()


def side(location):
    loc = (location or "").lower()
    if "both" in loc or "bilateral" in loc:
        return "bilateral"
    return next((s for s in SIDES if re.search(rf"\b{s}\b", loc)), "none")


def parse_report(text):
    """Model reply → dict, or None. Tolerates text around the JSON object."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("findings", []), list) else None


def match(iou):
    """iou: list (gt) of lists (pred) → [(gt_i, pred_j, iou)] greedy one-to-one, IoU >= IOU_HIT."""
    pairs = sorted(((v, i, j) for i, row in enumerate(iou) for j, v in enumerate(row)), reverse=True)
    used_g, used_p, out = set(), set(), []
    for v, i, j in pairs:
        if v < IOU_HIT:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); out.append((i, j, v))
    return out


def score_record(gt, pred, iou, dice):
    """gt/pred: report dicts (pred may be None); iou/dice: [n_gt][n_pred]. → flat per-record numbers."""
    gf = gt.get("findings", [])
    pf = (pred or {}).get("findings", []) if pred else []
    pf = [f for f in pf if isinstance(f, dict)]
    pairs = match(iou) if gf and pf else []
    return {
        "parse_ok": pred is not None,
        "gt_normal": bool(gt.get("normal")), "pred_normal": bool(pred.get("normal")) if pred else None,
        "n_gt": len(gf), "n_pred": len(pf), "n_found": len(pairs),
        "type_ok": sum(finding_type(gf[i].get("title")) == finding_type(pf[j].get("title")) for i, j, _ in pairs),
        "side_ok": sum(side(gf[i].get("location")) == side(pf[j].get("location")) for i, j, _ in pairs),
        "iou_found": [round(v, 4) for _, _, v in pairs],
        "dice_found": [round(dice[i][j], 4) for i, j, _ in pairs],
        "best_iou_per_gt": [round(max(r), 4) if r else 0.0 for r in iou],
    }


def summarise(rows):
    """rows → {modality: metrics}; 'all' included. Shared with the exporter."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["modality"]].append(r); by["all"].append(r)
    out = {}
    for mod, rs in sorted(by.items()):
        parsed = [r for r in rs if r["parse_ok"]]
        abn = [r for r in rs if not r["gt_normal"]]
        nor = [r for r in rs if r["gt_normal"]]
        n_gt, n_pred, n_found = (sum(r[k] for r in rs) for k in ("n_gt", "n_pred", "n_found"))
        found_iou = [v for r in rs for v in r["iou_found"]]
        found_dice = [v for r in rs for v in r["dice_found"]]
        best = [v for r in rs for v in r["best_iou_per_gt"]]
        out[mod] = {
            "records": len(rs),
            "parse_rate": len(parsed) / len(rs),
            "abnormal_flagged": sum(r["pred_normal"] is False for r in abn) / len(abn) if abn else None,
            "normals_clear": sum(r["pred_normal"] is True and r["n_pred"] == 0 for r in nor) / len(nor) if nor else None,
            "outline_recall": n_found / n_gt if n_gt else None,
            "outline_precision": n_found / n_pred if n_pred else None,
            "mean_iou_found": sum(found_iou) / len(found_iou) if found_iou else None,
            "mean_dice_found": sum(found_dice) / len(found_dice) if found_dice else None,
            "mean_best_iou": sum(best) / len(best) if best else None,
            "type_agreement": sum(r["type_ok"] for r in rs) / n_found if n_found else None,
            "side_agreement": sum(r["side_ok"] for r in rs) / n_found if n_found else None,
            "sec_per_record": sum(r["seconds"] for r in rs) / len(rs),
        }
    return out


def round_robin(rows):
    by = collections.OrderedDict()
    for r in rows:
        by.setdefault(r["modality"], []).append(r)
    lists, out = list(by.values()), []
    for k in range(max(map(len, lists))):
        out += [l[k] for l in lists if k < len(l)]
    return out


def load_rows(paths):
    return [json.loads(l) for p in paths for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--stack-pixels", type=int, default=320 * 320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deadline", help="stop starting new records after this local time, 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fill-seg", action="store_true", help="outline every X-ray/slice finding (skanda/infer.py fill_seg)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--no-gate", action="store_true", help="ignore the checkpoint's abnormal-score thresholds (raw reads, "
                                                           "used to calibrate them)")
    ap.add_argument("--no-hint", action="store_true", help="no classifier hint (ablation)")
    ap.add_argument("--no-fallback", action="store_true", help="SAM 2's dynamic multimask fallback off: always mask 0 "
                                                               "(white-paper mechanism check; training uses mask 0)")
    args = ap.parse_args()
    res_path = os.path.join(args.out, "results.jsonl")
    if args.summary:
        print(json.dumps(summarise(load_rows([res_path])), indent=1))
        return

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from skanda.data.dataset import _messages  # noqa: E402
    from skanda.infer import read  # noqa: E402
    from skanda.model import SkandaModel  # noqa: E402

    deadline = time.mktime(time.strptime(args.deadline, "%Y-%m-%d %H:%M")) if args.deadline else None
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    done = {json.loads(l)["id"] for l in open(res_path)} if os.path.exists(res_path) else set()
    rows = round_robin([r for r in load_rows(args.data) if r.get("target") is not None])  # label-only records: no report
    if args.limit:
        rows = rows[:args.limit]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} records, {len(todo)} to run", flush=True)
    model = SkandaModel.load(args.model).to(args.device).eval()
    if args.no_hint:
        model.hints = None
    if args.no_fallback:
        model.sam.sam2_model.sam_mask_decoder.dynamic_multimask_via_stability = False
    proc = AutoProcessor.from_pretrained(os.path.join(args.model, "vlm"))
    kw = {"max_pixels": args.max_pixels, "min_pixels": 256 * 256}
    thresholds = None if args.no_gate else "model"
    log = open(res_path, "a")
    t_start = time.time()
    for n, r in enumerate(todo, 1):
        if deadline and time.time() > deadline:
            print("deadline reached, stopping", flush=True)
            break
        t0 = time.time()
        stack = "masks_npz" in r
        images = [Image.open(x).convert("RGB") for x in r["images"]]
        prompt = proc.apply_chat_template(_messages(r, False), tokenize=False, add_generation_prompt=True)
        inp = proc(text=[prompt], images=[images], return_tensors="pt",
                   images_kwargs={**kw, "max_pixels": args.stack_pixels, "min_pixels": 128 * 128} if stack else kw).to(args.device)
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
            o = read(model, proc, inp, images, stack=stack, max_new_tokens=args.max_new_tokens,
                     normal_report=None, thresholds=thresholds, modality=r["modality"],
                     fill_seg=args.fill_seg)
        text, finished, pm = o["text"], o["finished"], o["masks"]
        gen, pos = range(o["new_tokens"]), range(len(pm))
        if stack:
            z = np.load(r["masks_npz"])["masks"]
            gm = [F.interpolate(torch.from_numpy(m.astype(np.float32))[:, None], size=(256, 256), mode="bilinear",
                                align_corners=False)[:, 0] > 0.5 for m in z] if z.ndim == 4 else []
        else:
            gm = [F.interpolate(torch.from_numpy(np.array(Image.open(m).convert("L")) > 127)[None, None].float(),
                                size=(256, 256), mode="bilinear", align_corners=False)[0, 0] > 0.5 for m in r["masks"]]
        iou = [[float((g & p).sum() / max((g | p).sum(), 1)) for p in pm] for g in gm]
        dice = [[float(2 * (g & p).sum() / max(g.sum() + p.sum(), 1)) for p in pm] for g in gm]
        mask_files = []
        for j, p in enumerate(pm):
            if stack:  # one compressed volume per predicted finding
                f = os.path.join(args.out, "masks", f"{r['id']}_p{j}.npz")
                np.savez_compressed(f, mask=p.numpy())
            else:
                f = os.path.join(args.out, "masks", f"{r['id']}_p{j}.png")
                Image.fromarray(p.numpy().astype(np.uint8) * 255).save(f, optimize=True)
            mask_files.append(f)
        pred = parse_report(text)
        rec = {"id": r["id"], "modality": r["modality"] + ("-3d" if stack else ""), "source": r.get("source"), "text": text, "finished": finished,
               "new_tokens": int(len(gen)), "n_seg": int(len(pos)), "pred_masks": mask_files,
               "p_abnormal": o["p_abnormal"], "gate": o["gate"], "obj": o["obj"], "side_fixed": o.get("side_fixed", 0), "hint": o.get("hint", ""),
               "scores": [round(float(x), 4) for x in o["scores"]] if o["scores"] is not None else None,
               "seconds": round(time.time() - t0, 2),
               **score_record(json.loads(r["target"]), pred, iou, dice)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"{n}/{len(todo)} {(time.time() - t_start) / n:.1f}s/rec {r['id']} found {rec['n_found']}/{rec['n_gt']}",
              flush=True)
    log.close()
    if os.path.exists(res_path):
        s = summarise(load_rows([res_path]))
        json.dump(s, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
        print(json.dumps(s, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
