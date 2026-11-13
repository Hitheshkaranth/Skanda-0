"""Choose the abnormal-score gate (skanda/infer.py) per modality from validation reads, and write it into a checkpoint.

  python -I skanda/evaluate.py --model runs/stage2/final --data <val records> --out runs/calib_stage2 --no-gate
  python -I skanda/calibrate.py --results runs/calib_stage2/results.jsonl --model runs/stage2/final [--min-sens 0.91]
For every modality, replays the raw reads under each (low, high) pair on a grid:
  p(abnormal) < low → called normal;  p(abnormal) > high → called abnormal;  otherwise the model's own call.
Among pairs keeping specificity >= --min-spec (MedGemma's 72%), picks the highest specificity whose sensitivity reaches
--min-sens (ties: higher sensitivity, then the narrowest override band); if none reaches it, the highest sensitivity.
(low, high) = (0, 1) leaves the model's own calls untouched, so the gate can only help on these records.
Writes {"thresholds": {modality: {"low", "high", "sens", "spec", "n"}}} into <model>/skanda_config.json.
"""
import argparse
import json
import os


def replay(rows, low, high):
    tp = fn = tn = fp = 0
    for r in rows:
        p = r["p_abnormal"]
        abnormal = False if p < low else True if p > high else r["pred_normal"] is False
        if r["gt_normal"]:
            tn += not abnormal; fp += abnormal
        else:
            tp += abnormal; fn += not abnormal
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def choose(rows, min_sens, min_spec):
    grid_low = [0.0] + [i / 100 for i in range(1, 51)]
    grid_high = [i / 100 for i in range(50, 100)] + [1.0]
    best, key = None, None
    for lo in grid_low:
        for hi in grid_high:
            if hi <= lo:
                continue
            sens, spec = replay(rows, lo, hi)
            k = (spec >= min_spec, sens >= min_sens, spec if sens >= min_sens else sens, sens, -(lo + (1 - hi)))
            if key is None or k > key:
                best, key = (lo, hi, sens, spec), k
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-sens", type=float, default=0.91)
    ap.add_argument("--min-spec", type=float, default=0.72, help="never trade specificity below this for sensitivity")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for p in args.results for l in open(p) if l.strip()]
    rows = [r for r in rows if r.get("p_abnormal") is not None and r.get("parse_ok")]
    out = {}
    for mod in sorted({r["modality"].replace("-3d", "") for r in rows}):
        rs = [r for r in rows if r["modality"].replace("-3d", "") == mod]
        if not any(r["gt_normal"] for r in rs) or all(r["gt_normal"] for r in rs):
            continue  # needs both normal and abnormal records
        base = replay(rs, 0.0, 1.0)
        lo, hi, sens, spec = choose(rs, args.min_sens, args.min_spec)
        out[mod] = {"low": lo, "high": hi, "sens": round(sens, 3), "spec": round(spec, 3), "n": len(rs),
                    "raw_sens": round(base[0], 3), "raw_spec": round(base[1], 3)}
        print(f"{mod}: n={len(rs)} raw sens {base[0]:.3f} spec {base[1]:.3f} → gate low {lo} high {hi}: "
              f"sens {sens:.3f} spec {spec:.3f}", flush=True)
    if not args.dry_run:
        p = os.path.join(args.model, "skanda_config.json")
        cfg = json.load(open(p))
        cfg["thresholds"] = out
        json.dump(cfg, open(p, "w"), indent=1)
        print("wrote", p)


if __name__ == "__main__":
    main()
