"""Assemble the initial Skanda checkpoint: Qwen3-VL-30B-A3B + SAM 2 (from Sa2VA) + a fresh [SEG] projector.

  python -I skanda/assemble.py --backbone Qwen/Qwen3-VL-30B-A3B-Instruct --sa2va <Sa2VA-Qwen3-VL-4B dir> --out models/skanda-init
Needs ~70 GB of free memory for the bf16 backbone (stop other GPU jobs first). The [SEG] id is taken from Sa2VA's tokenizer
(151669), which fits inside Qwen3-VL's padded vocabulary, so no embedding resize is needed; its row starts as the mean
of the existing embeddings.
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skanda import SEG_TOKEN  # noqa: E402
from skanda.model import SkandaModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--sa2va", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    processor = AutoProcessor.from_pretrained(args.backbone)
    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN]})
    seg_id = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    vlm = AutoModelForImageTextToText.from_pretrained(args.backbone, dtype=torch.bfloat16)
    emb = vlm.get_input_embeddings().weight
    if seg_id >= emb.shape[0]:
        vlm.resize_token_embeddings(len(processor.tokenizer))
        emb = vlm.get_input_embeddings().weight
    with torch.no_grad():
        emb[seg_id] = emb[:151643].float().mean(0).to(emb.dtype)
    model = SkandaModel(vlm, seg_id)
    missing, unexpected = SkandaModel.load_sam_from_sa2va(model.sam, args.sa2va)
    print(f"[SEG] id {seg_id} (added {added}); SAM 2 missing {len(missing)} unexpected {len(unexpected)}")
    model.to(torch.bfloat16).save(args.out, processor,
                                  meta={"backbone": args.backbone, "sam_from": "ByteDance/Sa2VA-Qwen3-VL-4B",
                                        "licence": "Apache-2.0 (Qwen3-VL, SAM 2.1, Sa2VA)"})
    print("saved", args.out)


if __name__ == "__main__":
    main()
