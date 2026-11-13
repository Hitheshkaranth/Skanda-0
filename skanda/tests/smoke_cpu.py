"""CPU smoke test: a tiny random Qwen3-VL MoE + the real SAM 2 weights from Sa2VA, one training step on real Skanda data.

  python -I skanda/tests/smoke_cpu.py --sa2va <Sa2VA-Qwen3-VL-4B dir> --data data/skanda/cxr/sft_train.jsonl
Checks: processor + chat template + [SEG] labels, SAM 2 weight loading, forward, losses, backward into projector/decoder.
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoProcessor, Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skanda.data.dataset import SkandaCollator, SkandaDataset  # noqa: E402
from skanda.model import SkandaModel  # noqa: E402


def tiny_config():
    return Qwen3VLMoeConfig(
        text_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         intermediate_size=128, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                         decoder_sparse_step=1, vocab_size=151936, max_position_embeddings=32768,
                         rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=64, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0],
                           num_position_embeddings=2304),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa2va", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    proc = AutoProcessor.from_pretrained(args.sa2va)
    seg_id = proc.tokenizer.convert_tokens_to_ids("[SEG]")
    vlm = Qwen3VLMoeForConditionalGeneration(tiny_config())
    model = SkandaModel(vlm, seg_id)
    missing, unexpected = SkandaModel.load_sam_from_sa2va(model.sam, args.sa2va)
    print(f"SAM 2 loaded: missing {len(missing)} unexpected {len(unexpected)}")
    for p in model.sam.sam2_model.image_encoder.parameters():
        p.requires_grad_(False)
    ds = SkandaDataset(args.data)
    row = next(r for r in ds.rows if len(r["masks"]) >= 2)
    batch = SkandaCollator(proc, max_pixels=256 * 256, min_pixels=128 * 128)([row])
    n_seg = int(((batch["input_ids"] == seg_id) & (batch["labels"] != -100)).sum())
    n_lab = int((batch["labels"] != -100).sum())
    print(f"record {row['id']}: tokens {batch['input_ids'].shape[1]}, supervised {n_lab}, [SEG] {n_seg}, masks {len(row['masks'])}")
    assert n_seg == len(row["masks"]), "one [SEG] per mask"
    t = time.time()
    out = model(**{k: v for k, v in batch.items()})
    out["loss"].backward()
    g = sum(p.grad.abs().sum().item() for p in model.seg_proj.parameters())
    gd = sum(p.grad.abs().sum().item() for p in model.sam.sam2_model.sam_mask_decoder.parameters() if p.grad is not None)
    print(f"loss {out['loss'].item():.3f} (text {out['text_loss'].item():.3f}, bce {out['mask_bce'].item():.3f}, "
          f"dice {out['mask_dice'].item():.3f}); grad projector {g:.3e}, decoder {gd:.3e}; {time.time() - t:.1f}s")
    assert g > 0 and gd > 0
    w = batch["loss_weights"][batch["labels"] != -100]
    print(f"text-loss weights: {(w < 1).float().mean():.0%} of answer tokens down-weighted (JSON structure)")
    assert 0 < (w < 1).float().mean() < 0.9
    # cached-feature path must give the same masks as the image path
    model.zero_grad()
    with torch.no_grad():
        feats = model.sam_features(batch["sam_images"])
        p = [torch.randn(2, 256)]
        a = model.sam_masks(batch["sam_images"], p)[0]
        b = model.sam_masks(None, p, sam_feats=feats)[0]
    print(f"cached vs live SAM features: max diff {float((a - b).abs().max()):.2e}")
    assert torch.allclose(a, b, atol=1e-4)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
