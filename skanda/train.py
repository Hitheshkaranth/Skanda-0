"""Train Skanda (stages 0 and 1; stage 2 adds CT/MRI records, stage 3 GRPO lives in grpo.py).

  # Stage 0: connector warm-up. Language model frozen; trains the [SEG] projector + SAM 2 mask decoder.
  python -I skanda/train.py --init models/skanda-init --stage 0 --train data/skanda/cxr/connector_train.jsonl \
      --val data/skanda/cxr/connector_val.jsonl --out runs/stage0
  # Stage 1: SFT. LoRA on the language model's attention + projector + mask decoder.
  python -I skanda/train.py --init runs/stage0/final --stage 1 --train data/skanda/cxr/sft_train.jsonl \
      --val data/skanda/cxr/sft_val.jsonl --out runs/stage1
  # Stage 2: CT/MRI volumes (stack_*.jsonl): 33 slices in, each [SEG] decoded on up to 10 slices with a 3D Dice.
  python -I skanda/train.py --init runs/stage1/final --stage 2 --train data/skanda/msd/*/stack_train.jsonl \
      --val data/skanda/msd/*/stack_val.jsonl --out runs/stage2
Resumable: every --ckpt-minutes (and at each validation) the trainable weights, optimiser, scheduler and step go to
<out>/ckpt (written atomically); a restarted run with the same --out continues from there (power cut, reboot).
From stage 1 on, the [SEG] rows of the input embedding and lm_head train too (gradient-masked to that one row, 20x LR):
the lm_head row starts as an untrained padding row, and LoRA on attention alone would struggle to make the model write [SEG].
Needs a GPU with ~80 GB free. SAM 2's image encoder and the vision encoder stay frozen.
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skanda.data.dataset import SkandaCollator, SkandaDataset, SkandaMixedCollator, SkandaVolumeCollator  # noqa: E402
from skanda.model import SkandaModel  # noqa: E402

LORA_TARGETS = r".*language_model.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
# run 2: also the vision encoder's last 11 of 27 blocks and its mergers, so X-ray detail can adapt (--vision-lora)
VISION_TARGETS = (r".*visual\.blocks\.(1[6-9]|2[0-6])\.(attn\.(qkv|proj)|mlp\.linear_fc[12])"
                  r"|.*visual\.(merger|deepstack_merger_list\.\d+)\.linear_fc[12]")


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif k == "sam_feats":
            out[k] = ([t.to(device) for t in v[0]], v[1])
        elif isinstance(v, list):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def trainable(model, stage, rank, vision_lora=False):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in (model.seg_proj, model.sam.sam2_model.sam_mask_decoder, model.score_head):
        for p in m.parameters():
            p.requires_grad_(True)
    if stage >= 1:
        from peft import LoraConfig, get_peft_model
        targets = f"(?:{LORA_TARGETS})|(?:{VISION_TARGETS})" if vision_lora else LORA_TARGETS
        model.vlm = get_peft_model(model.vlm, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.05,
                                                         target_modules=targets, task_type="CAUSAL_LM"))
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"stage {stage}: {n / 1e6:.1f}M trainable parameters (+ 2 [SEG] rows from stage 1)")


def train_seg_rows(model):
    """Stage 1+: let the [SEG] row of the input embedding and of lm_head train; every other row's gradient is zeroed
    (AdamW with weight_decay 0 then leaves those rows exactly unchanged). Returns the two weight tensors."""
    ws = [model.vlm.get_input_embeddings().weight, model.vlm.get_output_embeddings().weight]
    for w in ws:
        w.requires_grad_(True)
        keep = torch.zeros(w.shape[0], 1, dtype=w.dtype, device=w.device)
        keep[model.seg_token_id] = 1
        w.register_hook(lambda g, keep=keep: g * keep)
    return ws


@torch.no_grad()
def evaluate(model, loader, device, limit=50):
    model.eval()
    tot, n = {"loss": 0.0, "text_loss": 0.0, "mask_dice": 0.0}, 0
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(**to_device(batch, device))
        for k in tot:
            tot[k] += float(out[k])
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def save_ckpt(model, opt, sched, step, it, out):
    """Trainable weights only (LoRA, [SEG] rows, projector, mask decoder) + optimiser + scheduler, written atomically."""
    tmp, dst = os.path.join(out, "ckpt.tmp"), os.path.join(out, "ckpt")
    os.makedirs(tmp, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, os.path.join(tmp, "weights.pt"))
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "it": it}, os.path.join(tmp, "state.pt"))
    if os.path.isdir(dst):
        os.rename(dst, dst + ".old")
    os.rename(tmp, dst)
    if os.path.isdir(dst + ".old"):
        import shutil
        shutil.rmtree(dst + ".old")


def load_ckpt(model, opt, sched, out):
    """→ (step, it) from <out>/ckpt, or (0, 0) when there is none."""
    dst = os.path.join(out, "ckpt")
    if not os.path.exists(os.path.join(dst, "state.pt")) and os.path.isdir(dst + ".old"):
        dst += ".old"  # power cut between the two renames
    if not os.path.exists(os.path.join(dst, "state.pt")):
        return 0, 0
    w = torch.load(os.path.join(dst, "weights.pt"), map_location="cpu")
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in w.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
    st = torch.load(os.path.join(dst, "state.pt"), map_location="cpu", weights_only=False)
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    print(f"resumed from {dst} at step {st['step']}", flush=True)
    return st["step"], st["it"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--stage", type=int, choices=[0, 1, 2], required=True)
    ap.add_argument("--stack-pixels", type=int, default=320 * 320, help="stage 2: per-slice pixel cap for Qwen3-VL")
    ap.add_argument("--seg-row-lr-mult", type=float, default=20.0)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--replay", nargs="*", default=[], help="stage 2: 2D records mixed in (X-rays included)")
    ap.add_argument("--replay-frac", type=float, default=0.4, help="stage 2: share of draws that come from --replay")
    ap.add_argument("--vision-lora", action="store_true", help="stage 1+: LoRA on the vision encoder's last blocks too")
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 (stage 0) / 5e-5 (stage 1+)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sam-cache", help="dir of cached SAM 2 features (skanda/cache_sam.py); SAM's encoder is frozen")
    ap.add_argument("--no-balance", action="store_true", help="disable class-balanced sampling (sample_weight)")
    ap.add_argument("--score-loss", choices=["bce", "asl"], default="bce", help="run 4: asl = asymmetric loss on the 13 classes")
    ap.add_argument("--hints", action="store_true", help="run 3: noisy classifier hints before X-ray reports (skanda/hints.py)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-minutes", type=float, default=20.0)
    ap.add_argument("--max-draws", type=int, default=0, help="cap on draws per epoch (same sampling mix, shorter epoch)")
    args = ap.parse_args()
    device = args.device
    os.makedirs(args.out, exist_ok=True)
    model = SkandaModel.load(args.init)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(os.path.join(args.init, "vlm"))
    model.score_loss = args.score_loss
    trainable(model, args.stage, args.rank, args.vision_lora)
    model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    seg_rows = train_seg_rows(model) if args.stage >= 1 else []
    if args.stage == 2 and args.replay:
        col = SkandaMixedCollator(proc, stack_pixels=args.stack_pixels, max_pixels=args.max_pixels, sam_cache=args.sam_cache,
                                  hints=args.hints)
    elif args.stage == 2:
        col = SkandaVolumeCollator(proc, max_pixels=args.stack_pixels)
    else:
        col = SkandaCollator(proc, max_pixels=args.max_pixels, sam_cache=args.sam_cache, hints=args.hints)
    tds = SkandaDataset(*args.train, limit=args.limit)
    weights = [r.get("sample_weight", 1.0) for r in tds.rows]
    n_draws = len(weights)  # draws per epoch; with --replay, the stacks' count scaled up by the replay share
    cap = args.max_draws
    if cap and n_draws > cap:  # same sampling mix, fewer draws per epoch
        print(f"stage {args.stage}: {n_draws} records, capped at {cap} draws per epoch", flush=True)
        n_draws = cap
    if args.replay:  # replay rows keep their relative weights but get replay_frac of the total draw probability
        rep = SkandaDataset(*args.replay, limit=args.limit)
        rw = [r.get("sample_weight", 1.0) for r in rep.rows]
        scale = args.replay_frac / (1 - args.replay_frac) * sum(weights) / max(sum(rw), 1e-9)
        tds.rows += rep.rows
        weights += [w * scale for w in rw]
        n_draws = int(n_draws / (1 - args.replay_frac))
        print(f"stage 2 mix: {len(tds.rows) - len(rep.rows)} stacks + {len(rep.rows)} replay rows "
              f"({args.replay_frac:.0%} of draws)", flush=True)

    def loader(seed):  # reseeded per pass and on resume, so a restarted run does not replay the same order
        g = torch.Generator().manual_seed(seed)
        sampler = None if args.no_balance else torch.utils.data.WeightedRandomSampler(weights, n_draws, replacement=True,
                                                                                    generator=g)
        return DataLoader(tds, batch_size=1, shuffle=sampler is None, sampler=sampler, collate_fn=col, num_workers=4,
                          generator=g if sampler is None else None)

    va = DataLoader(SkandaDataset(*args.val), batch_size=1, shuffle=False, collate_fn=col, num_workers=2)
    lr = args.lr or (1e-4 if args.stage == 0 else 5e-5)
    seg_ids = {id(w) for w in seg_rows}
    groups = [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in seg_ids], "lr": lr}]
    if seg_rows:
        groups.append({"params": seg_rows, "lr": lr * args.seg_row_lr_mult})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    steps = int(n_draws * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=max(steps, 1),
                                                pct_start=0.03)
    step, it = load_ckpt(model, opt, sched, args.out)
    mpath = os.path.join(args.out, "metrics.jsonl")
    if os.path.exists(mpath):  # drop lines logged after the checkpoint (they will be redone)
        keep = [l for l in open(mpath) if l.strip() and json.loads(l)["step"] <= step]
        open(mpath, "w").writelines(keep)
    log = open(mpath, "a")
    t0, start, last_ckpt, epoch = time.time(), step, time.time(), 0
    model.train()
    while step < steps:
        epoch += 1
        for batch in loader(1000 * epoch + step):
            batch = to_device(batch, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                out = model(**batch)
            (out["loss"] / args.accum).backward()
            it += 1
            if it % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if float(out["score_loss"]) > 0:
                model.has_score = True
            rec = {"step": step, "steps": steps, "loss": float(out["loss"]), "text": float(out["text_loss"]),
                   "bce": float(out["mask_bce"]), "dice": float(out["mask_dice"]), "obj": float(out["obj_loss"]),
                   "score": float(out["score_loss"]), "lr": sched.get_last_lr()[0],
                   "sec_per_step": round((time.time() - t0) / (step - start), 2)}
            if step % args.eval_every == 0 or step == steps:
                rec["val"] = evaluate(model, va, device)
                if args.stage >= 1:  # adapter-only snapshot; stage 0 would write the whole 60 GB backbone each time
                    model.save(os.path.join(args.out, f"step{step}"), proc, meta={"stage": args.stage, "step": step})
            log.write(json.dumps(rec) + "\n"); log.flush()
            if "val" in rec or time.time() - last_ckpt > args.ckpt_minutes * 60:
                save_ckpt(model, opt, sched, step, it, args.out); last_ckpt = time.time()
            if step >= steps:
                break
    if args.stage >= 1:
        model.vlm = model.vlm.merge_and_unload()
    model.save(os.path.join(args.out, "final"), proc, meta={"stage": args.stage, "args": vars(args)})
    print("done", args.out)


if __name__ == "__main__":
    main()
