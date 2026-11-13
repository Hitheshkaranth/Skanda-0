"""Skanda: one model that writes, explains and outlines radiology findings (X-ray, CT, MRI).

Backbone: Qwen3-VL-30B-A3B-Instruct (Apache-2.0). Outlines: SAM 2.1 Hiera-L, taken from Sa2VA-Qwen3-VL-4B (Apache-2.0),
prompted by the hidden state of a [SEG] token through a projector. See README.md and docs/TRAINING.md.
"""
SEG_TOKEN = "[SEG]"
SCHEMA = "skanda-report/2"
