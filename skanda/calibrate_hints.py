"""Choose the classifier-hint thresholds (skanda/hints.py) per chest X-ray class, and write them into a checkpoint.

  python -I skanda/calibrate_hints.py --model runs/stage1/final --data data/skanda/cxr3/sft_val.jsonl \
      data/skanda/nih3/sft_val.jsonl data/skanda/cxr3/sft_dev.jsonl
One prompt-only forward per film (no generation) gives the score head's class probabilities. Per class, over the films
whose label for that class is known:
  likely    the lowest threshold whose precision is >= --likely (default 0.6) with >= 3 true positives
  possible  the lowest threshold whose precision is >= --possible (default 0.3), never above "likely"
A class with no threshold reaching the precision gets no hint. Validation and dev films only (dev = the app
benchmark's 106 dev films), never the test films. Writes {"hints": {class: {"likely", "possible", "n_pos"}}} into
<model>/skanda_config.json and the scores to <model>/hint_scores.jsonl.
With <model>/hint_classifier.pt (skanda/hint_classifier.py) the scores come from that classifier instead (no language
model is loaded; scores in <model>/hint_scores_cls.jsonl), and the config records "hint_source": "classifier".
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pick(pairs, precision, min_tp=3):
    """pairs: [(p, label)] → lowest threshold t with precision(p >= t) >= precision and >= min_tp hits, or None."""
    best = None
    for t in sorted({round(p, 3) for p, _ in pairs}, reverse=True):
        sel = [y for p, y in pairs if p >= t]
        tp = sum(sel)
        if tp >= min_tp and tp / len(sel) >= precision:
            best = t
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--likely", type=float, default=0.6)
    ap.add_argument("--possible", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from skanda.data.dataset import _messages, score_target
    from skanda.infer import abnormal_score
    from skanda.model import SCORE_CLASSES, SkandaModel
    rows = [json.loads(l) for p in args.data for l in open(p) if l.strip()]
    rows = [r for r in rows if r.get("modality") == "cxr"]
    use_cls = os.path.exists(os.path.join(args.model, "hint_classifier.pt"))
    out_path = os.path.join(args.model, "hint_scores_cls.jsonl" if use_cls else "hint_scores.jsonl")
    done = {json.loads(l)["id"]: json.loads(l) for l in open(out_path)} if os.path.exists(out_path) else {}
    todo = [r for r in rows if r["id"] not in done]
    if todo and use_cls:
        from skanda.hint_classifier import HintClassifier
        hc = HintClassifier.load(os.path.join(args.model, "hint_classifier.pt")).to(args.device)
        with open(out_path, "a") as f:
            for r in todo:
                if r["id"] in done:
                    continue
                sc = hc.predict([Image.open(r["images"][0])])[0]
                row = {"id": r["id"], "scores": [round(float(x), 4) for x in sc], "target": score_target(r).tolist()}
                f.write(json.dumps(row) + "\n")
                done[r["id"]] = row
    elif todo:
        model = SkandaModel.load(args.model).to(args.device).eval()
        proc = AutoProcessor.from_pretrained(os.path.join(args.model, "vlm"))
        with open(out_path, "a") as f:
            for n, r in enumerate(todo, 1):
                prompt = proc.apply_chat_template(_messages(r, False), tokenize=False, add_generation_prompt=True)
                inp = proc(text=[prompt], images=[[Image.open(x).convert("RGB") for x in r["images"]]],
                           return_tensors="pt", images_kwargs={"max_pixels": 1024 * 1024, "min_pixels": 256 * 256}).to(args.device)
                with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                    sc = abnormal_score(model, inp)
                row = {"id": r["id"], "scores": [round(float(x), 4) for x in sc], "target": score_target(r).tolist()}
                f.write(json.dumps(row) + "\n"); f.flush()
                done[r["id"]] = row
                if n % 100 == 0:
                    print(f"{n}/{len(todo)}", flush=True)
    hints = {}
    for i, c in enumerate(SCORE_CLASSES, 1):
        pairs = [(d["scores"][i], int(d["target"][i])) for d in done.values() if d["target"][i] >= 0]
        n_pos = sum(y for _, y in pairs)
        lk, ps = pick(pairs, args.likely), pick(pairs, args.possible)
        if lk is None and ps is None:
            print(f"{c:20s} positives {n_pos:4d}: no hint", flush=True)
            continue
        lk = lk if lk is not None else 1.01
        ps = min(ps if ps is not None else lk, lk)
        hints[c] = {"likely": lk, "possible": ps, "n_pos": n_pos}
        print(f"{c:20s} positives {n_pos:4d}: likely >= {lk}, possible >= {ps}", flush=True)
    p = os.path.join(args.model, "skanda_config.json")
    cfg = json.load(open(p))
    cfg["hints"] = hints
    cfg["hint_source"] = "classifier" if use_cls else "score_head"
    json.dump(cfg, open(p, "w"), indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
