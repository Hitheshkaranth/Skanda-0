"""Pre-compute SAM 2 encoder features for every training image (the encoder is frozen, so this is done once).

  python -I skanda/cache_sam.py --sa2va <Sa2VA-Qwen3-VL-4B dir> --data data/skanda/cxr/sft_train.jsonl ... --out cache/sam
Stores fp16 tensors (~8 MB per image): 3 feature levels flattened as SAM 2 uses them, plus their sizes.
Runs on GPU if available, otherwise CPU (~1.5 s per image). Resumable: existing files are skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skanda.data.dataset import sam_cache_path  # noqa: E402
from skanda.model import sam_preprocess  # noqa: E402
from skanda.sam2 import SAM2  # noqa: E402


class _Heads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sam = SAM2()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa2va", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from skanda.model import SkandaModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    h = _Heads()
    SkandaModel.load_sam_from_sa2va(h.sam, args.sa2va)
    sm = h.sam.sam2_model.to(dev).eval()
    os.makedirs(args.out, exist_ok=True)
    # only images with something to outline: records without masks (normal films, label-only records) never use SAM
    recs = [json.loads(l) for p in args.data for l in open(p) if l.strip()]
    imgs = sorted({r["images"][0] for r in recs if r.get("masks")})
    todo = [i for i in imgs if not os.path.exists(sam_cache_path(args.out, i))]
    print(f"{len(imgs)} images, {len(todo)} to encode on {dev}", flush=True)
    t0 = time.time()
    for n, ip in enumerate(todo, 1):
        x = sam_preprocess(np.array(Image.open(ip).convert("RGB")))[None].to(dev)
        with torch.no_grad():
            _, feats, _, sizes = sm._prepare_backbone_features(sm.forward_image(x))
        torch.save({"feats": [f.half().cpu() for f in feats], "sizes": sizes}, sam_cache_path(args.out, ip))
        if n % 50 == 0:
            print(f"{n}/{len(todo)} {(time.time() - t0) / n:.2f}s/img", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
