"""Classifier hints (run 3): the score head's chest X-ray classes, written at the start of the reply before the report.

  Classifier: likely pleural effusion, consolidation; possible nodule.
  Classifier: no finding likely.
Why: the score head ranks findings the text almost never names (pneumothorax AUC 0.76, emphysema 0.77; the text names
them 20% and 0-12% of the time). Feeding a classifier's per-disease predictions to the report writer as
positive / uncertain / absent prompts was the largest single gain in PromptMRG (AAAI 2024, clinical F1 0.370 → 0.444).
The hint sits in the assistant turn but outside the text loss, and the score head reads the token before it, so the
head never sees its own hint.
Training (hint_from_labels): built from the record's labels with noise, so the model learns to check the hint against
the image rather than copy it: 15% no hint; otherwise each true class is "likely" (75%), "possible" (15%) or left out
(10%), and each known-absent class is added as "possible" 5% of the time; in a third of the hinted records the noise is
doubled (a stand-in for the score head's own mistakes). The report always follows the radiologist labels, not the hint.
Reading (hint_from_scores): per-class thresholds chosen on validation by skanda/calibrate_hints.py.
"""
import random

from .model import SCORE_CLASSES

NAMES = {"Atelectasis": "atelectasis", "Calcification": "calcification", "Cardiomegaly": "cardiomegaly",
         "Consolidation": "consolidation", "Diffuse Nodule": "diffuse nodular opacities", "Effusion": "pleural effusion",
         "Emphysema": "emphysema", "Fibrosis": "fibrosis", "Fracture": "rib fracture", "Mass": "mass",
         "Nodule": "nodule", "Pleural Thickening": "pleural thickening", "Pneumothorax": "pneumothorax"}


def hint_text(likely, possible):
    """Class lists → the hint line (ends with a newline), in SCORE_CLASSES order."""
    lk = [NAMES[c] for c in SCORE_CLASSES if c in likely]
    ps = [NAMES[c] for c in SCORE_CLASSES if c in possible and c not in likely]
    if not lk and not ps:
        return "Classifier: no finding likely.\n"
    parts = []
    if lk:
        parts.append("likely " + ", ".join(lk))
    if ps:
        parts.append("possible " + ", ".join(ps))
    return "Classifier: " + "; ".join(parts) + ".\n"


def hint_from_labels(positive, known_negative, rng=random):
    """Noisy training hint from labels, or "" (no hint) 15% of the time."""
    if rng.random() < 0.15:
        return ""
    k = 2.0 if rng.random() < 1 / 3 else 1.0
    likely, possible = set(), set()
    for c in positive:
        u = rng.random()
        if u < 0.10 * k:
            continue
        (possible if u < 0.25 * k else likely).add(c)
    for c in known_negative:
        if rng.random() < 0.05 * k:
            possible.add(c)
    return hint_text(likely, possible)


def hint_from_scores(scores, thresholds):
    """scores: class probabilities in SCORE_CLASSES order; thresholds: {class: {"likely": t, "possible": t}}."""
    likely, possible = set(), set()
    for c, p in zip(SCORE_CLASSES, scores):
        t = thresholds.get(c)
        if not t:
            continue
        if p >= t["likely"]:
            likely.add(c)
        elif p >= t["possible"]:
            possible.add(c)
    return hint_text(likely, possible)
