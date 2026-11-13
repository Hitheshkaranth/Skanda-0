"""Licence registry and lineage gate: only datasets cleared for commercial use may enter Skanda training data.

Every training record carries `source`; build scripts call `require(source)` before writing anything.
"""

DATASETS = {
    # id: (licence, commercial use allowed, note)
    "chestxdet": ("Apache-2.0 annotations; NIH CXR14 images (no use restrictions, citation required)", True, ""),
    "nih-cxr14": ("NIH public release (no use restrictions, citation required)", True, "not used yet: patient-level overlap with the ChestX-Det test split is unmapped"),
    "chexmask": ("CC BY 4.0", True, "anatomy masks only"),
    "lidc-idri": ("CC BY 3.0", True, ""),
    "totalsegmentator": ("CC BY 4.0 (dataset only)", True, "pretrained models not used"),
    "amos22": ("CC BY 4.0", True, ""),
    "isles2022": ("CC BY 4.0", True, ""),
    "msd": ("CC BY-SA 4.0", True, "counsel: does share-alike reach model weights?"),
    # added 2026-10-26 (licences read on the official pages)
    "deeplesion": ("NIH release: 'The usage of the data set is unrestricted' (FAQ Q7), citation requested", True, "boxes only, no masks"),
    "autopet": ("CC BY 4.0 (TCIA FDG-PET-CT-Lesions, defaced release)", True, "lesion masks drawn on PET, co-registered with CT"),
    # added 2026-11-01 for run 4 (licences checked on the official pages 30 Oct 2026; scripts/download_run4.py)
    "ptx498": ("CC BY 4.0 (Zenodo 8266529)", True, "pneumothorax masks by 2 senior radiologists"),
    "luna16": ("CC BY 4.0 (Zenodo 3723295 / 4121926)", True, "nodule centres + diameters; masks are spheres"),
    "soop": ("CC0 (OpenNeuro ds004889)", True, "acute stroke, manual lesion masks"),
    "crlm": ("CC BY 4.0 (TCIA Colorectal-Liver-Metastases)", True, "DICOM SEG liver + tumour masks"),
    "hcc-tace": ("CC BY 4.0 (TCIA HCC-TACE-Seg)", True, "DICOM SEG liver + tumour masks"),
    "partner": ("data agreement + consent", True, "per-hospital agreement id required"),
    # refused
    "vindr-cxr": ("PhysioNet Credentialed Health Data License 1.5.0 (research only)", False, ""),
    "ms-cxr": ("PhysioNet credentialed", False, ""),
    "uls23": ("CC BY-NC-SA 4.0", False, ""),
    "kits23": ("CC BY-NC-SA 4.0", False, ""),
    "brats": ("CC BY-NC", False, ""),
    "ct-rate": ("CC BY-NC-SA 4.0", False, ""),
    "radgenome": ("non-commercial terms", False, ""),
    "abdomenatlas": ("CC BY-NC-SA 4.0", False, ""),
    "flare": ("CC BY-NC 4.0", False, ""),
    "siim-acr": ("competition rules", False, ""),
    "padchest": ("BIMCV research use agreement: 'research purposes only'", False, "commercial use case by case"),
    "node21": ("CC BY-NC-ND 4.0", False, ""),
    "ribfrac": ("CC BY-NC 4.0", False, ""),
    "chaos": ("CC BY-NC-SA 4.0", False, ""),
    "nsclc-radiomics": ("CC BY-NC 3.0", False, ""),
}

# Text sources for evidence sentences. Model outputs whose terms reach downstream models are refused.
TEXT_SOURCES = {
    "template": True,          # hand-written per-class templates in skanda/templates.py
    "radiologist": True,       # written or edited by a radiologist under agreement
    "nv-reason-ct": True,      # OpenMDW-1.1 (NVIDIA lists it as commercial)
    "backbone-replay": True,   # answers written by the Apache-2.0 backbone itself to general prompts, mixed in to keep
                               # its language skills during fine-tuning (generated on GPU before stage 1)
    "medgemma": False,         # HAI-DEF s.1.1(h): models trained on its outputs become Model Derivatives
}


def require(source, text_source="template"):
    if source not in DATASETS:
        raise SystemExit(f"lineage gate: unknown dataset '{source}' — add it to skanda/registry.py with its licence first")
    lic, ok, _ = DATASETS[source]
    if not ok:
        raise SystemExit(f"lineage gate: '{source}' is refused ({lic})")
    if not TEXT_SOURCES.get(text_source, False):
        raise SystemExit(f"lineage gate: text source '{text_source}' is refused")
    return lic
