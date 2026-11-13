"""Skanda report format and evidence text built from measured facts (no model-generated text, see skanda/registry.py).

Format skanda-report/2 (compact; the serving layer adds modality, body part and app fields):
  {"normal": bool, "findings": [{"title", "evidence", "location", "severity", "mask": "[SEG]"}], "impression": str}
Every number in `evidence` is measured from the radiologist mask and CheXmask anatomy (X-ray) or the NIfTI volume (CT/MRI).
"""
import json

from . import SEG_TOKEN
from .measure import zone_text

SYSTEM = (
    "You are Skanda, a radiology assistant that drafts findings for a radiologist to review. For each abnormal finding, "
    "state the visible evidence with its measured extent and its location from the patient's perspective, and put the "
    "mask token in its \"mask\" field so its outline can be drawn. Report only what the image supports. "
    "Reply with one JSON object and nothing else."
)

PROMPT_CXR = ("Read this chest radiograph. Reply as JSON: normal, findings (title, evidence, location, severity, mask), "
              "impression.")

# label → (term, sign, severity, diffuse)
CXR = {
    "Atelectasis": ("atelectasis", "linear or wedge-shaped opacity with volume loss", "moderate", False),
    "Calcification": ("calcification", "dense, sharply marginated calcified focus", "minor", False),
    "Cardiomegaly": ("cardiomegaly", "enlarged cardiac silhouette", "moderate", True),
    "Consolidation": ("consolidation", "airspace opacity with ill-defined margins", "significant", False),
    "Diffuse Nodule": ("diffuse nodular opacities", "multiple small nodular opacities", "moderate", True),
    "Effusion": ("pleural effusion", "dependent opacity with a meniscus", "significant", False),
    "Emphysema": ("emphysema", "hyperlucent lung with attenuated vascular markings", "moderate", True),
    "Fibrosis": ("fibrosis", "coarse reticular opacities with architectural distortion", "moderate", False),
    "Fracture": ("rib fracture", "cortical discontinuity of a rib", "moderate", False),
    "Mass": ("mass", "opacity larger than 3 cm", "significant", False),
    "Nodule": ("nodule", "opacity smaller than 3 cm", "moderate", False),
    "Pleural Thickening": ("pleural thickening", "soft-tissue thickening along the pleural surface", "minor", False),
    "Pneumothorax": ("pneumothorax", "visible pleural line with absent lung markings beyond it", "critical", False),
}
PARENCHYMAL = {"Atelectasis", "Consolidation", "Fibrosis", "Emphysema", "Diffuse Nodule"}
FOCAL = {"Nodule", "Mass", "Calcification"}
NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
# focal findings: (singular phrase, plural phrase with {n}), shape word inserted where {shape} appears
FOCAL_TEXT = {
    "Nodule": ("{shape} opacity smaller than 3 cm", "{n} {shape} opacities, each smaller than 3 cm"),
    "Mass": ("{shape} opacity larger than 3 cm", "{n} {shape} opacities larger than 3 cm"),
    "Calcification": ("dense, sharply marginated calcified focus", "{n} dense calcified foci"),
}
PLURAL = {"nodule": "nodules", "mass": "masses", "calcification": "calcifications"}


def cxr_finding(label, m):
    term, sign, sev, diffuse = CXR[label]
    side = m["side"]
    zt = zone_text(m["zones"])
    where = f"both lungs, {zt}" if side == "bilateral" else f"{side} {zt}"
    field = "both lung fields" if side == "bilateral" else f"the {side} lung field"
    if label == "Cardiomegaly":
        ev = f"Enlarged cardiac silhouette, cardiothoracic ratio about {m['ctr_mask']:.2f}." if "ctr_mask" in m else \
             "Enlarged cardiac silhouette."
        return {"title": "Cardiomegaly", "evidence": ev, "location": "heart", "severity": sev, "mask": SEG_TOKEN}
    parts = []
    if label in FOCAL:
        one, many = FOCAL_TEXT[label]
        txt = one.format(shape=m["shape"]) if m["parts"] <= 1 else many.format(n=NUM.get(m["parts"], "multiple"),
                                                                               shape=m["shape"])
        parts.append(txt[0].upper() + txt[1:])
    elif m["parts"] > 1 and not diffuse:
        parts.append(f"{sign.capitalize()} in {NUM.get(m['parts'], str(m['parts']))} separate areas")
    else:
        parts.append(sign.capitalize())
    if label == "Effusion" and m.get("basal"):
        parts.append(f"blunting the {'costophrenic angles' if side == 'bilateral' else side + ' costophrenic angle'}")
    if label == "Pneumothorax" and m.get("apical"):
        parts.append("at the apex")
    parts.append(f"in {where}" if side == "bilateral" else f"in the patient's {where}")
    if "field_pct" in m and (label in PARENCHYMAL or label in {"Effusion", "Pneumothorax", "Pleural Thickening"}):
        parts.append(f"involving about {max(m['field_pct'], 1)}% of {field}")
    if label in FOCAL and m["parts"] > 1:
        term = PLURAL.get(term, term)
    if diffuse:
        title = f"Bilateral {term}"
    elif side == "bilateral":
        title = f"Bilateral {term}"
    elif label in FOCAL:
        title = f"{side.capitalize()} lung {term}, {zt}"
    else:
        title = f"{side.capitalize()} {m['zones'][0]} zone {term}"
    if label in PARENCHYMAL | {"Effusion"} and m.get("field_pct", 0) >= 40:
        sev = "critical" if label == "Effusion" else "significant"
    return {"title": title, "evidence": ", ".join(parts) + ".", "location": where, "severity": sev, "mask": SEG_TOKEN}


def report(findings, normal_text="No acute cardiopulmonary abnormality."):
    return json.dumps({"normal": not findings, "findings": findings,
                       "impression": "; ".join(f["title"] for f in findings) + "." if findings else normal_text},
                      ensure_ascii=False)
