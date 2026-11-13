"""Weight-only quantized Skanda builds (int8 / int4) for GPUs smaller than bf16 needs.

  python -I skanda/quant.py --model checkpoints/skanda-0 --out checkpoints/skanda-0-int8 --bits 8
  python -I skanda/quant.py --model checkpoints/skanda-0 --out checkpoints/skanda-0-int4 --bits 4 --group 128

What is quantized: the MoE experts (gate_up_proj / down_proj, ~93% of the weights; transformers keeps them as fused 3D
tensors, so bitsandbytes and torchao's nn.Linear filters skip them) and the language model's attention projections.
What stays bf16: the vision encoder, the MoE routers, embeddings and lm_head (the [SEG] rows), the projector, score head
and SAM 2.1. Weights stay quantized in memory and each expert is dequantized only when a token is routed to it.
  int8: symmetric, one scale per output row.   int4: symmetric (-8..7), one scale per --group input columns, 2 per byte.
The output is an ordinary Skanda checkpoint directory (SkandaModel.load reads it) with vlm/skanda_quant.json, the
quantized vlm/ weights as safetensors shards and every other file copied. The abnormal gate and hint thresholds are
tuned to bf16: re-run skanda/calibrate.py and skanda/calibrate_hints.py on the build.
"""
import argparse
import json
import os
import re
import shutil
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUANT_FILE = "skanda_quant.json"
ATTN = re.compile(r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def quantize(w, bits, group):
    """w [..., O, I] float → (q, scale). int8: q int8 [..., O, I], scale [..., O, 1]; int4: q uint8 [..., O, I/2]
    (low nibble = even column), scale [..., O, I/group]."""
    w = w.float()
    if bits == 8:
        s = (w.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        return torch.round(w / s).clamp_(-127, 127).to(torch.int8), s.to(torch.bfloat16)
    *lead, o, i = w.shape
    g = w.reshape(*lead, o, i // group, group)
    s = (g.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-12)
    q = (torch.round(g / s).clamp_(-8, 7) + 8).to(torch.uint8).reshape(*lead, o, i)
    return (q[..., 0::2] | (q[..., 1::2] << 4)), s.squeeze(-1).to(torch.bfloat16)


def dequantize(q, s, bits, group, dtype=torch.bfloat16):
    if bits == 8:
        return q.to(dtype) * s.to(dtype)
    lo, hi = (q & 15).to(torch.int8) - 8, (q >> 4).to(torch.int8) - 8
    w = torch.stack((lo, hi), -1).reshape(*q.shape[:-1], q.shape[-1] * 2).to(dtype)
    return (w.reshape(*w.shape[:-1], -1, group) * s.to(dtype)[..., None]).reshape(w.shape)


class QExperts(nn.Module):
    """Qwen3VLMoeTextExperts with quantized weights; same call signature and routing as transformers' eager loop."""

    def __init__(self, experts, bits, group, empty=False):
        super().__init__()
        self.num_experts, self.act_fn, self.bits, self.group = experts.num_experts, experts.act_fn, bits, group
        for name in ("gate_up_proj", "down_proj"):
            w = getattr(experts, name)
            if empty:  # shapes only (loading): q/scale tensors are filled from the checkpoint
                e, o, i = w.shape
                q = torch.empty(e, o, i if bits == 8 else i // 2, dtype=torch.int8 if bits == 8 else torch.uint8)
                s = torch.empty(e, o, 1 if bits == 8 else i // group, dtype=torch.bfloat16)
            else:
                q, s = quantize(w.data, bits, group)
            self.register_buffer(f"{name}_q", q)
            self.register_buffer(f"{name}_s", s)

    def w(self, name, e, dtype):
        return dequantize(getattr(self, f"{name}_q")[e], getattr(self, f"{name}_s")[e], self.bits, self.group, dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            gate, up = nn.functional.linear(x, self.w("gate_up_proj", e, x.dtype)).chunk(2, dim=-1)
            y = nn.functional.linear(self.act_fn(gate) * up, self.w("down_proj", e, x.dtype))
            final.index_add_(0, tok, (y * top_k_weights[tok, pos, None]).to(final.dtype))
        return final


class QLinear(nn.Module):
    def __init__(self, lin, bits, group, empty=False):
        super().__init__()
        self.bits, self.group, self.in_features, self.out_features = bits, group, lin.in_features, lin.out_features
        o, i = lin.out_features, lin.in_features
        if empty:
            q = torch.empty(o, i if bits == 8 else i // 2, dtype=torch.int8 if bits == 8 else torch.uint8)
            s = torch.empty(o, 1 if bits == 8 else i // group, dtype=torch.bfloat16)
        else:
            q, s = quantize(lin.weight.data, bits, group)
        self.register_buffer("weight_q", q)
        self.register_buffer("weight_s", s)
        self.bias = None if lin.bias is None else nn.Parameter(torch.empty_like(lin.bias) if empty else lin.bias.data.clone())

    def forward(self, x):
        return nn.functional.linear(x, dequantize(self.weight_q, self.weight_s, self.bits, self.group, x.dtype), self.bias)


def swap(vlm, bits, group, empty=False):
    """Replace the experts and attention projections in place → number of modules swapped."""
    n = 0
    for name, mod in list(vlm.named_modules()):
        if type(mod).__name__ == "Qwen3VLMoeTextExperts":
            new = QExperts(mod, bits, group, empty)
        elif isinstance(mod, nn.Linear) and ATTN.match(name):
            new = QLinear(mod, bits, group, empty)
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(vlm.get_submodule(parent), child, new)
        n += 1
    return n


def _tied(cfg):
    return bool(getattr(cfg, "tie_word_embeddings", False) or getattr(cfg.get_text_config(), "tie_word_embeddings", False))


def load_vlm(vlm_dir, dtype=torch.bfloat16):
    """A quantized vlm/ directory → model (built on the meta device, then filled from the shards)."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig
    q = json.load(open(os.path.join(vlm_dir, QUANT_FILE)))
    cfg = AutoConfig.from_pretrained(vlm_dir)
    with torch.device("meta"):
        vlm = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
        swap(vlm, q["bits"], q["group"], empty=True)
    sd = {}
    for f in sorted(os.listdir(vlm_dir)):
        if f.startswith("model") and f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(vlm_dir, f)))
    missing, unexpected = vlm.load_state_dict(sd, strict=False, assign=True)
    tied = {k for k in missing if k.endswith("lm_head.weight") and _tied(cfg)}
    if set(missing) - tied or unexpected:
        raise RuntimeError(f"quantized checkpoint mismatch: missing {sorted(set(missing) - tied)[:5]} "
                           f"unexpected {sorted(unexpected)[:5]}")
    if tied:
        vlm.tie_weights()
    for k, t in load_file(os.path.join(vlm_dir, "buffers.safetensors")).items():  # rotary inv_freq etc.
        mod, _, name = k.rpartition(".")
        vlm.get_submodule(mod)._buffers[name] = t
    still = [n for n, t in list(vlm.named_parameters()) + list(vlm.named_buffers()) if t.is_meta]
    if still:
        raise RuntimeError(f"tensors left on meta: {still[:5]}")
    if os.path.exists(os.path.join(vlm_dir, "generation_config.json")):
        vlm.generation_config = GenerationConfig.from_pretrained(vlm_dir)
    return vlm.eval()


def save_vlm(vlm, src_dir, out_dir, bits, group):
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_dir):  # config, processor, tokenizer, chat template, generation config
        if not (f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".index.json")):
            shutil.copy(os.path.join(src_dir, f), out_dir)
    sd = {k: v.contiguous() for k, v in vlm.state_dict().items()}
    if _tied(vlm.config) and vlm.lm_head.weight.data_ptr() == vlm.get_input_embeddings().weight.data_ptr():
        sd.pop("lm_head.weight")  # truly shared (Skanda's are separate: the [SEG] rows train apart)
    split = split_torch_state_dict_into_shards(sd, filename_pattern="model{suffix}.safetensors", max_shard_size="4GB")
    for fname, keys in split.filename_to_tensors.items():
        save_file({k: sd[k] for k in keys}, os.path.join(out_dir, fname), metadata={"format": "pt"})
    pers = set(sd)
    save_file({k: b.contiguous() for k, b in vlm.named_buffers() if k not in pers and not k.endswith(("_q", "_s"))},
              os.path.join(out_dir, "buffers.safetensors"))
    size = sum(t.numel() * t.element_size() for t in sd.values())
    json.dump({"bits": bits, "group": group, "format": "skanda-weight-only-v1", "bytes": size,
               "quantized": "MoE experts + language-model attention projections",
               "kept_bf16": "vision encoder, routers, embeddings, lm_head, projector, score head, SAM 2.1"},
              open(os.path.join(out_dir, QUANT_FILE), "w"), indent=1)
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bf16 Skanda checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, choices=[4, 8], required=True)
    ap.add_argument("--group", type=int, default=128, help="int4 group size (input columns per scale)")
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText
    vlm = AutoModelForImageTextToText.from_pretrained(os.path.join(args.model, "vlm"), dtype=torch.bfloat16)
    before = sum(p.numel() * p.element_size() for p in vlm.parameters())
    n = swap(vlm, args.bits, args.group)
    tmp = args.out.rstrip("/") + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    size = save_vlm(vlm, os.path.join(args.model, "vlm"), os.path.join(tmp, "vlm"), args.bits, args.group)
    for f in os.listdir(args.model):  # heads, config (gate + hint thresholds, re-calibrated afterwards), hint classifier
        p = os.path.join(args.model, f)
        if f != "vlm" and os.path.isfile(p):
            shutil.copy(p, tmp)
    cfg = json.load(open(os.path.join(tmp, "skanda_config.json")))
    cfg["quant"] = {"bits": args.bits, "group": args.group, "from": os.path.realpath(args.model)}
    json.dump(cfg, open(os.path.join(tmp, "skanda_config.json"), "w"), indent=1)
    shutil.rmtree(args.out, ignore_errors=True)
    os.rename(tmp, args.out)
    print(f"int{args.bits}: {n} modules quantized, vlm {before / 1e9:.1f} GB → {size / 1e9:.1f} GB → {args.out}", flush=True)


if __name__ == "__main__":
    main()
