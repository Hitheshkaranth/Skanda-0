"""Medical Segmentation Decathlon task → Skanda CT/MRI records.

  python -I skanda/data/build_msd.py --task data/raw/msd/Task01_BrainTumour --out data/skanda/msd
Per case:
  sft  (stage 1/2, 2D): the axial slice with the most lesion, report JSON with one [SEG] per lesion class on it
  normal (2D): one lesion-free slice from the same volume, normal report (teaches restraint)
  connector (stage 0): "Outline the <structure>" → [SEG], for lesion and organ classes on the key slice
  stack (stage 2, 3D): 32 evenly spaced slices + key-slice index + per-slice masks (npz) for slice-memory training
Display: reoriented to RAS, then shown radiologically (patient RIGHT on image LEFT, anterior at top).
CT → RGB windows: lung (-1350..150), soft tissue (-160..240), bone (-500..1300). MRI → up to three sequences as RGB.
Split by case: 10% val, 10% test (MSD's own test labels are not public).
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from skanda import SEG_TOKEN  # noqa: E402
from skanda.measure import volume_measures  # noqa: E402
from skanda.registry import require  # noqa: E402
from skanda.templates import SYSTEM, report as compact_report  # noqa: E402

CT_WINDOWS = [(-1350, 150), (-160, 240), (-500, 1300)]
# task → (modality, body part, organ word, {label value: (name, kind, evidence)}); kind: lesion | organ
TASKS = {
    "Task01_BrainTumour": ("MRI", "Brain", "cerebral hemisphere", {
        1: ("peritumoral oedema", "lesion", "high FLAIR signal surrounding the tumour"),
        2: ("non-enhancing tumour", "lesion", "solid tumour component without contrast enhancement"),
        3: ("enhancing tumour", "lesion", "contrast-enhancing tumour rim on T1 post-contrast")}),
    "Task03_Liver": ("CT", "Abdomen", "liver", {
        1: ("liver", "organ", ""),
        2: ("liver lesion", "lesion", "focal hypoattenuating lesion within the liver parenchyma")}),
    "Task05_Prostate": ("MRI", "Pelvis", "prostate", {
        1: ("prostate peripheral zone", "organ", ""), 2: ("prostate transition zone", "organ", "")}),
    "Task06_Lung": ("CT", "Chest", "lung", {
        1: ("lung tumour", "lesion", "soft-tissue mass within the lung parenchyma")}),
    "Task07_Pancreas": ("CT", "Abdomen", "pancreas", {
        1: ("pancreas", "organ", ""),
        2: ("pancreatic lesion", "lesion", "focal lesion within the pancreas, distinct from normal parenchyma")}),
    "Task10_Colon": ("CT", "Abdomen", "colon", {
        1: ("colonic tumour", "lesion", "focal wall thickening and mass of the colon")}),
    # added 2026-10-24
    "Task08_HepaticVessel": ("CT", "Abdomen", "liver", {
        1: ("hepatic vessels", "organ", ""),
        2: ("liver tumour", "lesion", "focal hypoattenuating mass within the liver, next to the hepatic vessels")}),
    "Task02_Heart": ("MRI", "Chest", "heart", {1: ("left atrium", "organ", "")}),
    "Task09_Spleen": ("CT", "Abdomen", "spleen", {1: ("spleen", "organ", "")}),
    # ISLES 2022 (CC BY 4.0), converted to this layout by skanda/data/convert_isles.py: channels DWI (b=1000), ADC
    "ISLES2022": ("MRI", "Brain", "cerebral hemisphere", {
        1: ("acute ischaemic infarct", "lesion", "restricted diffusion: high DWI signal with matching low ADC")}),
    # run 4 (2026-11-01), each converted to this layout by its skanda/data/convert_*.py
    "SOOP": ("MRI", "Brain", "cerebral hemisphere", {
        1: ("acute ischaemic infarct", "lesion", "restricted diffusion: high DWI signal with matching low ADC")}),
    "CRLM": ("CT", "Abdomen", "liver", {
        1: ("liver", "organ", ""),
        2: ("liver metastasis", "lesion", "focal hypoattenuating lesion within the liver parenchyma")}),
    "HCCTACE": ("CT", "Abdomen", "liver", {
        1: ("liver", "organ", ""),
        2: ("liver tumour", "lesion", "focal mass within the liver parenchyma")}),
    "LUNA16": ("CT", "Chest", "lung", {
        1: ("lung", "organ", ""),
        2: ("lung nodule", "lesion", "round soft-tissue nodule within the lung parenchyma")}),
}
STACK = 32


def split_of(cid):
    h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 10
    return "val" if h == 0 else "test" if h == 1 else "train"


def radiological(sl):
    """RAS axial slice [x, y] → display [rows=anterior..posterior, cols=patient right..left]."""
    return np.flipud(np.fliplr(sl.T))


def to_rgb(vols, modality, z, norms):
    if modality == "CT":
        v = radiological(vols[0][:, :, z])
        ch = [np.clip((v - lo) / (hi - lo), 0, 1) for lo, hi in CT_WINDOWS]
    else:
        ch = []
        for c in range(3):
            vv = vols[min(c, len(vols) - 1)]
            lo, hi = norms[min(c, len(vols) - 1)]
            ch.append(np.clip((radiological(vv[:, :, z]) - lo) / (hi - lo + 1e-6), 0, 1))
    return (np.stack(ch, -1) * 255).astype(np.uint8)


def side_of(mask):
    xs = np.nonzero(mask)[1]
    w = mask.shape[1]
    left = (xs < w / 2).mean()          # image left = patient right
    return "bilateral" if 0.2 < left < 0.8 else ("right" if left >= 0.8 else "left")


def finding(name, ev, mask, organ, m, sev="significant"):
    side = side_of(mask)
    both = "both cerebral hemispheres" if organ == "cerebral hemisphere" else f"both sides of the {organ}"
    loc = f"{side} {organ}" if side != "bilateral" else both
    facts = []
    if m.get("diameter_mm"):
        facts.append(f"{m.get('shape', 'irregular')}, about {m['diameter_mm']:.0f} mm across on this slice")
    if m.get("volume_ml") is not None and m.get("slices"):
        facts.append(f"{m['volume_ml']:.1f} mL in total over {m['slices']} slices ({m['extent_mm']:.0f} mm)")
    if "mean_hu" in m:
        facts.append(f"mean attenuation {m['mean_hu']} HU")
    if m.get("parts", 1) > 1:
        facts.append(f"{m['parts']} separate parts")
    evidence = f"{ev[0].upper()}{ev[1:]} in the patient's {loc}" + (", " + ", ".join(facts) if facts else "") + "."
    return {"title": f"{name[0].upper()}{name[1:]}, {loc}", "evidence": evidence, "location": loc, "severity": sev,
            "mask": SEG_TOKEN}


def report(modality, body, findings):
    return compact_report(findings, normal_text="No focal abnormality on this slice.")


def prompt_for(modality, body, stack=False):
    what = f"{'these axial slices of a' if stack else 'this axial slice of a'} {body.lower()} {modality}"
    tail = " Give the key slice number in each finding's location." if stack else ""
    return f"Read {what}. Reply as JSON: normal, findings (title, evidence, location, severity, mask), impression.{tail}"


def load_case(task_dir, c, task, modality):
    """One MSD case → (case id, display volumes, label volume, voxel spacing, per-volume [1, 99] percentiles)."""
    cid = os.path.basename(c["image"]).replace(".nii.gz", "")
    img = nib.as_closest_canonical(nib.load(os.path.join(task_dir, c["image"])))
    lab = nib.as_closest_canonical(nib.load(os.path.join(task_dir, c["label"])))
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    data = np.asanyarray(img.dataobj).astype(np.float32)
    vols = [data[..., k] for k in range(data.shape[-1])] if data.ndim == 4 else [data]
    if modality == "MRI" and task == "Task01_BrainTumour":
        vols = [vols[0], vols[2], vols[3]]          # FLAIR, T1 post-contrast, T2
    if task in ("ISLES2022", "SOOP"):
        vols = [vols[0], vols[1], vols[0]]          # DWI, ADC, DWI → RGB (stroke: bright R/B, dark G)
    seg = np.asanyarray(lab.dataobj).astype(np.int16)
    norms = [tuple(np.percentile(v[v > 0] if (v > 0).any() else v, [1, 99])) for v in vols]
    return cid, vols, seg, spacing, norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--source", default="msd", help="registry id written into every record (isles2022 for ISLES)")
    args = ap.parse_args()
    source = args.source
    lic = require(source, "template")
    task = os.path.basename(args.task.rstrip("/"))
    modality, body, organ, labels = TASKS[task]
    meta = json.load(open(os.path.join(args.task, "dataset.json")))
    out = os.path.join(args.out, task)
    for d in ("images", "masks", "stacks"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    files = collections.defaultdict(lambda: None)
    stats = collections.defaultdict(collections.Counter)

    def write(kind, split, rec):
        key = f"{kind}_{split}"
        if files[key] is None:
            files[key] = open(os.path.join(out, f"{key}.jsonl"), "w")
        files[key].write(json.dumps(rec) + "\n")

    cases = meta["training"][: args.limit or None]
    for c in cases:
        cid, vols, seg, spacing, norms = load_case(args.task, c, task, modality)
        split = split_of(cid)
        lesion_vals = [v for v, (_, k, _) in labels.items() if k == "lesion"]
        area = np.isin(seg, lesion_vals).sum((0, 1)) if lesion_vals else (seg > 0).sum((0, 1))
        if area.max() == 0:
            continue
        # up to 3 key slices with the most lesion, at least 5 slices apart
        keys = []
        for z in np.argsort(-area):
            if area[z] == 0 or len(keys) == 3:
                break
            if all(abs(int(z) - k) >= 5 for k in keys):
                keys.append(int(z))
        main_key = keys[0]
        for key in keys:
            ip = os.path.join(out, "images", f"{cid}_z{key}.png")
            Image.fromarray(to_rgb(vols, modality, key, norms)).save(ip)
            found, masks = [], []
            for v, (name, kind, ev) in labels.items():
                m = radiological(seg[:, :, key] == v)
                if not m.any():
                    continue
                mp = os.path.join(out, "masks", f"{cid}_z{key}_l{v}.png")
                Image.fromarray((m * 255).astype(np.uint8)).save(mp)
                write("connector", split, {"id": f"{cid}_z{key}_l{v}", "stage": "connector", "modality": modality.lower(),
                                           "images": [ip], "system": SYSTEM,
                                           "prompt": f"Outline the {name} on this axial {modality} slice.",
                                           "target": SEG_TOKEN, "masks": [mp], "source": source, "licence": lic,
                                           "text_source": "template", "split": split})
                if kind == "lesion":
                    meas = volume_measures(seg == v, spacing, key, intensity=vols[0], ct=modality == "CT")
                    found.append(finding(name, ev, m, organ, meas))
                    masks.append(mp)
                    stats[split][name] += 1
            if not lesion_vals:      # anatomy-only task (prostate zones): connector records only, never a report
                continue
            write("sft", split, {"id": f"{cid}_z{key}", "stage": "sft", "modality": modality.lower(), "images": [ip],
                                 "system": SYSTEM, "prompt": prompt_for(modality, body),
                                 "target": report(modality, body, found), "masks": masks, "source": source,
                                 "licence": lic, "text_source": "template", "split": split})
            stats[split]["lesion slices"] += 1
        stats[split]["cases"] += 1
        if not lesion_vals:
            continue
        # two lesion-free slices inside the body, away from every key slice
        free = [z for z in range(seg.shape[2]) if area[z] == 0 and min(abs(z - k) for k in keys) > 5
                and vols[0][:, :, z].std() > 0]
        for z in ([free[len(free) // 3], free[2 * len(free) // 3]] if len(free) >= 3 else free[:1]):
            np_ = os.path.join(out, "images", f"{cid}_z{z}.png")
            Image.fromarray(to_rgb(vols, modality, z, norms)).save(np_)
            write("sft", split, {"id": f"{cid}_z{z}", "stage": "sft", "modality": modality.lower(), "images": [np_],
                                 "system": SYSTEM, "prompt": prompt_for(modality, body), "target": report(modality, body, []),
                                 "masks": [], "source": source, "licence": lic, "text_source": "template", "split": split})
            stats[split]["normal slices"] += 1
        # 3D stack for stage 2: evenly spaced slices, main key slice always included
        zs = sorted(set(np.linspace(0, seg.shape[2] - 1, STACK).round().astype(int).tolist()) | {main_key})
        sdir = os.path.join(out, "stacks", cid)
        os.makedirs(sdir, exist_ok=True)
        paths = []
        for z in zs:
            p = os.path.join(sdir, f"z{z:04d}.png")
            Image.fromarray(to_rgb(vols, modality, z, norms)).save(p)
            paths.append(p)
        lv = [v for v in lesion_vals if (seg[:, :, main_key] == v).any()]
        m3 = np.stack([np.stack([radiological(seg[:, :, z] == v) for z in zs]) for v in lv]) if lv else np.zeros((0,))
        npz = os.path.join(sdir, "masks.npz")
        np.savez_compressed(npz, masks=m3.astype(bool), slices=np.array(zs), label_values=np.array(lv))
        sf = [finding(labels[v][0], labels[v][2], radiological(seg[:, :, main_key] == v), organ,
                      volume_measures(seg == v, spacing, main_key, intensity=vols[0], ct=modality == "CT")) for v in lv]
        for f_ in sf:
            f_["location"] += f", key slice {zs.index(main_key) + 1} of {len(zs)}"
        write("stack", split, {"id": cid, "stage": "stack", "modality": modality.lower(), "images": paths,
                               "key_slice": zs.index(main_key), "slice_z": zs, "system": SYSTEM,
                               "prompt": prompt_for(modality, body, stack=True), "target": report(modality, body, sf),
                               "masks_npz": npz, "spacing_mm": spacing, "source": source, "licence": lic,
                               "text_source": "template", "split": split})
    for f in files.values():
        if f:
            f.close()
    json.dump({s: dict(c) for s, c in stats.items()}, open(os.path.join(out, "stats.json"), "w"), indent=1)
    for s, c in stats.items():
        print(task, s, dict(c))


if __name__ == "__main__":
    main()
