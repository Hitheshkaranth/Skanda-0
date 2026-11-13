"""CT and MRI reads through the app server (skanda/serve.py): Skanda's own prompt, and numbers measured from the outline.

Why: a client that sends CT/MRI slices with a generic box_2d prompt makes the model write coordinates and confidences it was never trained to produce; and every number
in Skanda's evidence text ("about 23 mm across", "mean attenuation 54 HU") is the language model's prediction of a value
that was measured from radiologist masks in training, not a measurement of the image in front of it.
Here:
  prompt()     the training prompt for an axial slice (skanda/data/build_msd.py prompt_for)
  render_ct()  the training rendering of a CT slice from HU: lung / soft-tissue / bone windows as R / G / B
  clean()      removes the model's own numeric claims (size, volume, HU, ADC, percentages, pixel counts, part counts)
  measure()    measures each outline: largest diameter and area (pixel spacing), parts, side, mean HU (CT, raw HU only)
  apply()      clean() + measure() on one finding; the facts are appended as "Measured from the outline: …" and also
               returned as structured fields, so the app can show where each number came from
"""
import re

import numpy as np
from PIL import Image

CT_WINDOWS = [(-1350, 150), (-160, 240), (-500, 1300)]   # = build_msd.CT_WINDOWS (lung, soft tissue, bone)
BODIES = {"ct": ("abdomen", "chest", "body"), "mri": ("brain",)}

_NUM = r"-?\d+(?:[.,]\d+)?"
_CLAIMS = [
    rf"(?:about |approximately |~)?{_NUM} ?(?:mm|cm)(?: across| in (?:size|diameter|length))?(?: on (?:this|the) (?:slice|image))?(?: across its longest axis)?",
    rf"(?:lesion )?size (?:on this slice )?(?:is )?(?:about |approximately )?{_NUM} ?(?:mm|cm)[^,.;]*",
    rf"(?:under |about |approximately )?{_NUM} ?mL(?: in total)?(?: over {_NUM} slices)?(?: \({_NUM} ?mm\))?",
    rf"(?:with )?(?:a )?mean attenuation (?:of )?(?:about )?{_NUM} ?HU",
    rf"{_NUM} ?HU",
    rf"(?:involving |occupying )?(?:about |approximately )?{_NUM} ?% of [^,.;()]*",
    rf"\(?{_NUM} of {_NUM}[^,.;()]*pixels?[^,.;()]*\)?",
    rf"(?:involving )?(?:approximately |about )?{_NUM} separate (?:parts|areas)(?: within [^,.;]*)?",
]
_CLAIM_RE = re.compile("|".join(f"(?:{c})" for c in _CLAIMS), re.I)


def prompt(modality, body):
    """modality "ct" | "mri" → the training prompt for one axial slice."""
    return (f"Read this axial slice of a {body.lower()} {'CT' if modality == 'ct' else 'MRI'}. Reply as JSON: normal, "
            "findings (title, evidence, location, severity, mask), impression.")


def body_for(modality, hint=""):
    """Body word Skanda was trained with, from the app's body-part text."""
    h = (hint or "").lower()
    if modality == "mri":
        return "brain"
    if any(w in h for w in ("liver", "abdom", "pancrea", "kidney", "colon", "pelvi", "hepat")):
        return "abdomen"
    if any(w in h for w in ("chest", "lung", "thora", "pulmon", "mediast")):
        return "chest"
    return "body"


def render_ct(hu):
    """HU slice [H, W] (display orientation) → the RGB image Skanda's CT records were rendered as."""
    hu = np.asarray(hu, dtype=np.float32)
    ch = [np.clip((hu - lo) / (hi - lo), 0, 1) for lo, hi in CT_WINDOWS]
    return Image.fromarray((np.stack(ch, -1) * 255).astype(np.uint8))


def clean(text):
    """Remove numeric claims the model wrote itself; tidy the punctuation they leave."""
    if not isinstance(text, str) or not text:
        return text, 0
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keep = [x for x in sentences if not re.search(r"\bADC\b|diffusion coefficient", x, re.I)]
    n0 = len(sentences) - len(keep)   # no ADC map is ever part of an app request: drop sentences that cite one
    out, n = _CLAIM_RE.subn("", " ".join(keep))
    n += n0
    out = re.sub(r"\(\s*\)", "", out)
    out = re.sub(r"\s+([,.;])", r"\1", out)
    out = re.sub(r"([,;])\s*(?=[,;.])", "", out)
    out = re.sub(r"\b(?:and|with|of|at|is|was)\s*([,.;])", r"\1", out)
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\.\s*,", ".", out)
    out = re.sub(r"(?:,\s*)?\b(?:and(?: the)?|the|with)\s*\.", ".", out)
    out = re.sub(r"\.{2,}", ".", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,;")
    if out and out[-1] not in ".!?":
        out += "."
    return out, n


def _side(m):
    """Image left = patient right (radiological display, as in training and in the app's axial view)."""
    xs = np.nonzero(m)[1]
    if not len(xs):
        return None
    left = (xs < m.shape[1] / 2).mean()
    return "right" if left >= 0.8 else "left" if left <= 0.2 else "both"


def measure(mask, size, spacing=None, hu=None):
    """mask: bool [h, w] over the whole image (SAM's 256² output); size: (W, H) of the image the app sent;
    spacing: (column mm, row mm) or None; hu: raw HU [H, W] or None → facts dict (only what can be measured)."""
    from scipy import ndimage as ndi
    from skimage import measure as skm
    W, H = size
    m = np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    if not m.any():
        return {}
    facts = {"side": _side(m), "pixels": int(m.sum())}
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        facts["parts"] = int((sizes >= 0.05 * sizes.max()).sum())
    else:
        facts["parts"] = 1
    if spacing and all(s and s > 0 for s in spacing):
        sx, sy = float(spacing[0]), float(spacing[1])
        props = max(skm.regionprops(skm.label(m)), key=lambda p: p.area)
        facts["diameter_mm"] = round(float(props.feret_diameter_max) * float(np.sqrt(sx * sy)), 1)
        facts["area_mm2"] = round(float(m.sum()) * sx * sy, 1)
    if hu is not None and np.shape(hu) == (H, W):
        a = np.asarray(hu, dtype=np.float32)
        v = a[m]
        facts["mean_hu"] = int(round(float(v.mean())))
        facts["hu_sd"] = int(round(float(v.std())))
        # the soft tissue in an ~8 mm band around the outline: what "hypo-" or "hyperattenuating" is relative to
        r = max(2, int(round(8 / float(np.sqrt(spacing[0] * spacing[1]))))) if spacing else 10
        band = ndi.binary_dilation(m, iterations=r) & ~m
        sv = a[band]
        sv = sv[(sv > -100) & (sv < 300)]
        if sv.size >= 20:
            facts["surround_hu"] = int(round(float(sv.mean())))
    return facts


def measured_text(facts):
    bits = []
    if "diameter_mm" in facts:
        bits.append(f"about {facts['diameter_mm']:.0f} mm across on this slice (area {facts['area_mm2']:.0f} mm²)")
    if "mean_hu" in facts:
        bits.append(f"mean attenuation {facts['mean_hu']} HU (SD {facts['hu_sd']})"
                    + (f", surrounding tissue {facts['surround_hu']} HU" if "surround_hu" in facts else ""))
    if facts.get("parts", 1) > 1:
        bits.append(f"{facts['parts']} separate parts")
    return ("Measured from the outline: " + ", ".join(bits) + ".") if bits else ""


_SIDE_WORD = re.compile(r"\b(left|right)\b", re.I)


def fix_side(text, side):
    """One-sided outline, text naming only the other side → swap left/right (laterality from the outline, as
    skanda/infer.py side_from_outlines does for brain MRI)."""
    if not isinstance(text, str) or side not in ("left", "right"):
        return text, False
    words = {w.lower() for w in _SIDE_WORD.findall(text)}
    other = "left" if side == "right" else "right"
    if words != {other}:
        return text, False
    return _SIDE_WORD.sub(lambda mo: side if mo.group(0).islower() else side.capitalize(), text), True


_HYPO = re.compile(r"\b(hypo-?(?:attenuat|dens)\w*|low(?:er)?[- ]attenuation|low(?:er)?[- ]density)", re.I)
_HYPER = re.compile(r"\b(hyper-?(?:attenuat|dens)\w*|high(?:er)?[- ]attenuation|high(?:er)?[- ]density|enhancing)", re.I)


def consistency(text, facts):
    """Checks of the outline against the finding's own words (CT with raw HU). The labeled-case re-run (bench/
    labeled_cases.py): the text said "hypoattenuating" while the outline followed enhanced vessels (133 vs 115 HU around
    it; 135 vs 124). Returns the issues; an outline with any is withheld."""
    issues = []
    mu, sur = facts.get("mean_hu"), facts.get("surround_hu")
    if mu is not None and sur is not None and isinstance(text, str):
        if _HYPO.search(text) and not _HYPER.search(text) and mu >= sur:
            issues.append(f"Outline not darker than its surroundings ({mu} vs {sur} HU) although the finding is described "
                          "as low attenuation: the outline probably follows other tissue (for example enhanced vessels).")
        elif _HYPER.search(text) and not _HYPO.search(text) and mu <= sur:
            issues.append(f"Outline not brighter than its surroundings ({mu} vs {sur} HU) although the finding is "
                          "described as high attenuation or enhancing.")
    if facts.get("parts", 1) >= 5:
        issues.append(f"Outline split into {facts['parts']} separate parts for one finding: not a single lesion outline.")
    return issues


_SEQ = {  # MRI sequence → wording that claims it (replaced when that sequence was not supplied)
    "flair": re.compile(r"\b(?:high |increased |bright )?FLAIR(?:[- ](?:signal|hyperintensity|hyperintense))?", re.I),
    "t2": re.compile(r"\b(?:high |increased |bright )?T2(?:-weighted)?(?:[- ](?:signal|hyperintensity|hyperintense))?(?![-\w])", re.I),
    "dwi": re.compile(r"\b(?:restricted diffusion|diffusion restriction|(?:high |bright )?DWI(?:[- ](?:signal|hyperintensity))?)", re.I),
}


def supplied_sequences(sequence, composite):
    """None = every sequence (Skanda's training composite); else the set the app said it sent."""
    if composite:
        return None
    s = (sequence or "").lower()
    return {k for k, keys in {"flair": ("flair",), "t2": ("t2",), "dwi": ("dwi", "diffusion", "adc")}.items()
            if any(x in s for x in keys)}


def clean_sequences(text, supplied):
    """Sequence claims the input cannot support (e.g. "High FLAIR signal" on a T1 post-contrast slice) → "signal change"."""
    if supplied is None or not isinstance(text, str):
        return text, 0
    n = 0
    for k, rx in _SEQ.items():
        if k not in supplied:
            text, c = rx.subn("signal change", text)
            n += c
    text = re.sub(r"(^|[.!?]\s+)signal change", lambda mo: mo.group(1) + "Signal change", text)
    return re.sub(r"signal change signal change", "signal change", text), n


def apply(item, mask, size, spacing=None, hu=None, supplied=None, modality="ct"):
    """One app-schema finding (title, description, location, …) + its outline → cleaned text, measured facts, and
    validationIssues (an outline that fails consistency() is withheld: no mask_png, no box_2d)."""
    facts = measure(mask, size, spacing, hu) if mask is not None else {}
    issues = consistency(item.get("description", ""), facts) if modality == "ct" else []
    desc, removed = clean(item.get("description", ""))
    seq_fixed = 0
    if modality == "mri":
        for k in ("title", "location"):
            item[k], c = clean_sequences(item.get(k, ""), supplied)
            seq_fixed += c
        desc, c = clean_sequences(desc, supplied)
        seq_fixed += c
    side = facts.get("side")
    swapped = False
    for k in ("title", "location"):
        item[k], s = fix_side(item.get(k, ""), side)
        swapped |= s
    desc, s = fix_side(desc, side)
    swapped |= s
    mt = measured_text(facts)
    item["description"] = (desc + (" " + mt if mt else "")).strip()
    item["measurements"] = {**{k: v for k, v in facts.items() if k != "pixels"}, "source": "outline",
                            "removed_model_numbers": removed, "side_from_outline": swapped,
                            "unsupported_sequence_claims": seq_fixed}
    if issues:
        item["validationIssues"] = issues
        item.pop("mask_png", None)
        item.pop("box_2d", None)
    return item
