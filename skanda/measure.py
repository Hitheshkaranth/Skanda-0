"""Measurements taken from radiologist masks, so each finding's evidence states facts about THIS image.

X-ray: lung and heart outlines come from CheXmask (CC BY 4.0, RCA Dice >= 0.7), matched to each film via its NIH source.
Zones are thirds of the lung-field height, sides split at the midline between the lungs, and the cardiothoracic ratio is
heart width over thoracic width. Without CheXmask anatomy, image thirds are used and `anatomy` is False. No absolute
sizes are given for X-ray (no reliable pixel spacing); extent is a share of the lung field instead.
CT / MRI: in-plane spacing and slice thickness come from the NIfTI header, so sizes are in mm and volumes in mL.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure


# ---------- X-ray ----------
def rle_decode(rle, h, w):
    """CheXmask RLE: 1-indexed (start, length) pairs over the row-major flattened mask."""
    r = np.array(rle.split(), dtype=np.int64)
    m = np.zeros(h * w, bool)
    for st, ln in zip(r[::2], r[1::2]):
        m[st - 1:st - 1 + ln] = True
    return m.reshape(h, w)


def load_chexmask(csv_path, wanted, min_dice=0.7):
    """Stream the CheXmask ChestX-Ray8 CSV, keeping rows for `wanted` NIH files with RCA Dice >= min_dice.
    Returns {nih_file: (right_lung_rle, left_lung_rle, heart_rle, h, w)}."""
    import csv
    csv.field_size_limit(10 ** 9)
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            n = row["Image Index"]
            if n in wanted and float(row["Dice RCA (Mean)"]) >= min_dice:
                out[n] = (row["Right Lung"], row["Left Lung"], row["Heart"], int(row["Height"]), int(row["Width"]))
    return out


def anatomy(entry, shape):
    """CheXmask entry → (right_lung, left_lung, heart) at the film's resolution (patient right = image left)."""
    from PIL import Image
    rr, lr, hr, h, w = entry
    ms = [rle_decode(x, h, w) for x in (rr, lr, hr)]
    if (h, w) != tuple(shape):
        ms = [np.array(Image.fromarray(m).resize((shape[1], shape[0]), Image.NEAREST)) for m in ms]
    return ms


def shape_word(mask):
    props = max(measure.regionprops(measure.label(mask)), key=lambda p: p.area)
    circ = 4 * np.pi * props.area / max(props.perimeter, 1) ** 2
    if circ > 0.72:
        return "rounded"
    if props.minor_axis_length > 0 and props.major_axis_length / props.minor_axis_length > 3.5:
        return "linear"
    return "irregular"


def n_parts(mask, min_frac=0.05):
    lab, n = ndi.label(mask)
    if n <= 1:
        return int(n)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return int((sizes >= min_frac * sizes.max()).sum())


def cxr_measures(mask, anat, shape):
    """Facts about one finding mask on a radiograph. anat = (right_lung, left_lung, heart) from CheXmask, or None."""
    h, w = shape
    ys, xs = np.nonzero(mask)
    m = {"parts": n_parts(mask), "shape": shape_word(mask), "anatomy": anat is not None}
    if anat is not None:
        rl, ll, heart = anat
        both = rl | ll
        rows = np.nonzero(both.any(1))[0]
        top, bot = rows.min(), rows.max()
        mid_x = (np.nonzero(rl.any(0))[0].max() + np.nonzero(ll.any(0))[0].min()) / 2
        cols = np.nonzero(both.any(0))[0]
        thorax = max(cols.max() - cols.min(), 1)
        if heart.any():
            hc = np.nonzero(heart.any(0))[0]
            m["ctr_chexmask"] = round((hc.max() - hc.min()) / thorax, 2)
        m["ctr_mask"] = round((xs.max() - xs.min()) / thorax, 2)
    else:
        rl = ll = None
        top, bot, mid_x = 0, h - 1, w / 2
    in_right = (xs < mid_x).mean()
    m["side"] = "bilateral" if 0.2 < in_right < 0.8 else ("right" if in_right >= 0.8 else "left")
    rel = (ys - top) / max(bot - top, 1)
    zones = [name for name, lo, hi in (("upper", -9, 1 / 3), ("mid", 1 / 3, 2 / 3), ("lower", 2 / 3, 9))
             if ((rel >= lo) & (rel < hi)).mean() > 0.2]
    m["zones"] = zones or ["lower" if rel.mean() > 0.5 else "upper"]
    if rl is not None:
        field = rl if m["side"] == "right" else ll if m["side"] == "left" else (rl | ll)
        m["field_pct"] = int(round(100 * (mask & field).sum() / max(field.sum(), 1)))
        m["apical"] = bool(rel.min() < 0.1)
        m["basal"] = bool(rel.max() > 0.85)
    return m


def zone_text(zones):
    if len(zones) == 3:
        return "upper to lower zones"
    if len(zones) == 2:
        return f"{zones[0]} and {zones[1]} zones"
    return f"{zones[0]} zone"


# ---------- CT / MRI ----------
def volume_measures(seg3d, spacing, key_z, intensity=None, ct=False):
    """seg3d: XxYxZ bool (canonical RAS); spacing: (sx, sy, sz) mm. Facts for one lesion class."""
    sx, sy, sz = spacing
    m = {}
    sl = seg3d[:, :, key_z]
    if sl.any():
        props = max(measure.regionprops(measure.label(sl)), key=lambda p: p.area)
        m["diameter_mm"] = round(float(props.feret_diameter_max) * float(np.sqrt(sx * sy)), 1)
        m["shape"] = shape_word(sl)
    m["volume_ml"] = round(float(seg3d.sum() * sx * sy * sz / 1000.0), 1)
    zs = np.nonzero(seg3d.any((0, 1)))[0]
    m["slices"] = int(len(zs))
    m["extent_mm"] = round(float((zs.max() - zs.min() + 1) * sz), 1) if len(zs) else 0.0
    m["parts"] = n_parts(seg3d)
    if ct and intensity is not None:
        m["mean_hu"] = int(round(float(intensity[seg3d].mean())))
    return m
