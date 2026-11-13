"""ChestX-Det → Skanda X-ray records, with measured evidence and CheXmask anatomy.

  python -I skanda/data/build_cxr.py --chestxdet <dir with parquet/> --out data/skanda/cxr \
      [--nih-map data/skanda/cxr/nih_map.json --chexmask data/raw/chexmask/ChestX-Ray8.csv]
Writes images/<id>.png, masks/<id>_<k>.png (finding outlines) and anat/<id>_{rl,ll,heart}.png, plus
  sft_{train,val,test}.jsonl     report per film; each finding has measured evidence and one [SEG]; `sample_weight`
                                 up-weights films with rare classes (inverse square-root frequency)
  connector_{train,val}.jsonl    Stage 0: "Outline the <finding | right lung | left lung | heart>" → [SEG] + mask
Findings: one per radiologist instance (run 3): each outline is its own finding with its own [SEG], except that
overlapping outlines of one class are merged (one lesion drawn twice) and diffuse classes (cardiomegaly, emphysema,
diffuse nodules) stay one finding per film. Run 2 merged every outline of a class on the same side (and zone) into one
finding, so the model learned to stop at about one finding per side and wrote 164 boxes for 307 radiologist boxes.
Split: official ChestX-Det test → test; official train → train/val (5% by id hash). Train/val films whose NIH
patient also appears in the test split are dropped (patient-level leakage).
"""
import argparse
import collections
import glob
import hashlib
import io
import json
import math
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skanda import SEG_TOKEN  # noqa: E402
from skanda.measure import anatomy, cxr_measures, load_chexmask  # noqa: E402
from skanda.registry import require  # noqa: E402
from skanda.templates import CXR, PROMPT_CXR, SYSTEM, cxr_finding, report  # noqa: E402

SEV = {"critical": 0, "significant": 1, "moderate": 2, "minor": 3}


def split_of(fid, official):
    if official == "test":
        return "test"
    return "val" if int(hashlib.sha1(fid.encode()).hexdigest(), 16) % 20 == 0 else "train"


def poly_mask(points, w, h):
    m = Image.new("L", (w, h), 0)
    if len(points) >= 3:
        ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=1)
    return np.array(m, bool)


def save_mask(m, path):
    Image.fromarray((m * 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chestxdet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nih-map")
    ap.add_argument("--chexmask")
    ap.add_argument("--images", help="image dir (default <out>/images); point it at an earlier build to reuse its images "
                    "and their SAM 2 feature cache, which is keyed by image path")
    args = ap.parse_args()
    img_dir = args.images or f"{args.out}/images"
    lic = require("chestxdet", "template")
    if args.chexmask:
        require("chexmask", "template")
    for d in ("masks", "anat"):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    nih = json.load(open(args.nih_map)) if args.nih_map else {}
    cm = load_chexmask(args.chexmask, {v["nih"] for v in nih.values()}) if args.chexmask else {}
    test_patients = {v["patient"] for k, v in nih.items() if k.startswith("cxd_")}
    print(f"NIH matches {len(nih)}, CheXmask anatomy {len(cm)}, test patients {len(test_patients)}")

    recs, conn = collections.defaultdict(list), collections.defaultdict(list)
    stats = collections.defaultdict(collections.Counter)
    for official in ("train", "test"):
        for pf in sorted(glob.glob(f"{args.chestxdet}/parquet/{official}-*.parquet")):
            for r in pq.read_table(pf, columns=["image_id", "image", "annotation_json", "width", "height"]).to_pylist():
                fid = f"cxd{'tr' if official == 'train' else ''}_{r['image_id']}"
                split = split_of(fid, official)
                src = nih.get(fid)
                if split != "test" and src and src["patient"] in test_patients:
                    stats[split]["dropped: patient in test"] += 1
                    continue
                a = json.loads(r["annotation_json"])
                w, h = r["width"], r["height"]
                ipath = f"{img_dir}/{fid}.png"
                if not os.path.exists(ipath):
                    Image.open(io.BytesIO(r["image"]["bytes"])).convert("L").save(ipath)
                anat = anatomy(cm[src["nih"]], (h, w)) if src and src["nih"] in cm else None
                stats[split]["films with anatomy"] += anat is not None
                groups = []   # [label, mask]: one per instance; overlapping outlines of a class merged; diffuse: one
                for lab, poly in zip(a["syms"], a["polygons"]):
                    if lab not in CXR:
                        continue
                    m = poly_mask(poly, w, h)
                    if not m.any():
                        continue
                    same = [g for g in groups if g[0] == lab and (CXR[lab][3] or (g[1] & m).any())]
                    if same:
                        same[0][1] = same[0][1] | m
                    else:
                        groups.append([lab, m])
                found = [(cxr_finding(lab, cxr_measures(m, anat, (h, w))), m, lab) for lab, m in groups]
                found.sort(key=lambda t: (SEV[t[0]["severity"]], t[0]["title"]))
                masks = []
                for k, (f, m, lab) in enumerate(found):
                    mp = f"{args.out}/masks/{fid}_{k}.png"
                    save_mask(m, mp)
                    masks.append(mp)
                    stats[split][lab] += 1
                stats[split]["films"] += 1
                stats[split]["normal films"] += not found
                recs[split].append({"id": fid, "stage": "sft", "modality": "cxr", "images": [ipath], "system": SYSTEM,
                                    "prompt": PROMPT_CXR, "target": report([f for f, _, _ in found]), "masks": masks,
                                    "labels": sorted({lab for _, _, lab in found}), "nih": src["nih"] if src else None,
                                    "source": "chestxdet", "licence": lic, "text_source": "template", "split": split})
                if split == "test":
                    continue
                for (f, _, _), mp in zip(found, masks):
                    conn[split].append({"id": f"{fid}#{os.path.basename(mp)}", "stage": "connector", "modality": "cxr",
                                        "images": [ipath], "system": SYSTEM,
                                        "prompt": f"Outline the {f['title'].lower()} on this chest radiograph.",
                                        "target": SEG_TOKEN, "masks": [mp], "source": "chestxdet", "licence": lic,
                                        "text_source": "template", "split": split})
                if anat is not None:
                    for name, m in zip(("right lung", "left lung", "heart"), anat):
                        ap_ = f"{args.out}/anat/{fid}_{name.replace(' ', '_')}.png"
                        save_mask(m, ap_)
                        conn[split].append({"id": f"{fid}#{name}", "stage": "connector", "modality": "cxr",
                                            "images": [ipath], "system": SYSTEM,
                                            "prompt": f"Outline the patient's {name} on this chest radiograph.",
                                            "target": SEG_TOKEN, "masks": [ap_], "source": "chexmask",
                                            "licence": "CC BY 4.0", "text_source": "template", "split": split})
    # class-balance weights from training frequencies
    freq = collections.Counter(l for r in recs["train"] for l in r["labels"])
    top = max(freq.values()) if freq else 1
    for split, rows in recs.items():
        with open(f"{args.out}/sft_{split}.jsonl", "w") as f:
            for r in rows:
                r["sample_weight"] = round(max([math.sqrt(top / freq[l]) for l in r["labels"] if l in freq] or [1.0]), 3)
                f.write(json.dumps(r) + "\n")
    for split, rows in conn.items():
        with open(f"{args.out}/connector_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    summary = {s: dict(c) for s, c in stats.items()}
    summary["connector"] = {s: len(v) for s, v in conn.items()}
    json.dump(summary, open(f"{args.out}/stats.json", "w"), indent=1)
    for s, c in summary.items():
        print(s, c)


if __name__ == "__main__":
    main()
