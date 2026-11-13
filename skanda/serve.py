"""Serve Skanda behind an OpenAI-compatible API, so the Skanda Radiology app and its benchmark use it unchanged.

  python -I skanda/serve.py --model runs/grpo/final --port 1234 --name skanda
GET  /v1/models, /api/v0/models → {"data": [{"id": <name>, "state": "loaded"}]}  (the app checks the exact id)
POST /v1/chat/completions  → stream or not, like LM Studio / vLLM. Routing by request:
  - an image + the app's lung-pattern system prompt → {"grade": 0, ...} (Skanda has no separate grading pass; its main
    read already reports diffuse findings)
  - an image (main read) → Skanda's own prompt for that image (skanda/infer.py, with the run 2 abnormal-score gate when
    the checkpoint has one), then its report in the app's schema: findings with
    box_2d [ymin, xmin, ymax, xmax] on 0-1000 (top-left origin, the extent of that finding's outline) and the outline
    itself as "mask_png" (256×256 data URL; the app ignores unknown fields today); "confidence" is SAM 2's object score
    for that outline (0.8 for run 1 checkpoints, whose object score was never trained)
  - an image + a "skanda" request field (the app's CT/MRI slice reads): {"modality": "ct" | "mri", "body": <text>,
    "pixel_spacing": [column mm, row mm], "hu": {"b64", "width", "height"} (raw HU, little-endian int16, CT only),
    "sequence": <MRI sequence name>} → Skanda's own CT/MRI prompt (skanda/ctmri.py), the per-modality gate, outlines;
    sizes, mean HU and sides are measured from each outline and the model's own numbers are removed. With "hu", the
    CT image is re-rendered from HU exactly as in training (lung / soft-tissue / bone windows as R / G / B).
  - text only (report writing) → plain chat with the backbone
One request at a time (a lock); CORS is open so the app can point at this box directly (base URL http://<host>:1234/v1).
"""
import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skanda.evaluate import parse_report  # noqa: E402
from skanda.infer import read as infer_read  # noqa: E402
from skanda.model import SkandaModel  # noqa: E402
from skanda import ctmri  # noqa: E402
from skanda.templates import PROMPT_CXR, SYSTEM, report  # noqa: E402

LUNG_PATTERN_MARK = "Grade the image objectively"
SEVERITY = {"critical", "significant", "moderate", "minor", "normal"}


class Skanda:
    def __init__(self, path, device="cuda", max_pixels=1024 * 1024):
        from transformers import AutoProcessor
        self.model = SkandaModel.load(path).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(os.path.join(path, "vlm"))
        self.dev, self.kw = device, {"max_pixels": max_pixels, "min_pixels": 256 * 256}
        self.eos = [self.proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.lock = threading.Lock()

    def _generate(self, msgs, images, max_new_tokens, temperature):
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[images] if images else None, return_tensors="pt",
                        **({"images_kwargs": self.kw} if images else {})).to(self.dev)
        sample = temperature > 0.05
        seq = self.model.vlm.generate(**inp, max_new_tokens=max_new_tokens, do_sample=sample,
                                      **({"temperature": temperature, "top_p": 0.95} if sample else {}), eos_token_id=self.eos)
        gen = seq[0, inp["input_ids"].shape[1]:]
        finished = bool(len(gen) and int(gen[-1]) in self.eos)
        text = self.proc.tokenizer.decode(gen, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")
        return inp, seq, text, finished, len(gen)

    @torch.no_grad()
    def read(self, image, max_new_tokens=1024, temperature=0.0, ctx=None):
        """One image → (app-schema dict, raw Skanda text, finished, tokens). The abnormal-score gate (run 2 checkpoints)
        is applied inside skanda.infer.read. ctx (CT/MRI, see the module docstring) switches the prompt and modality."""
        mod = ctx["modality"] if ctx else "cxr"
        text_prompt = ctmri.prompt(mod, ctx["body"]) if ctx else PROMPT_CXR
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_prompt}]}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[[image]], return_tensors="pt", images_kwargs=self.kw).to(self.dev)
        with torch.autocast(self.dev, dtype=torch.bfloat16):
            o = infer_read(self.model, self.proc, inp, [image], max_new_tokens=max_new_tokens, temperature=temperature,
                           fill_seg=os.environ.get("SKANDA_FILL_SEG") == "1", modality=mod,
                           normal_report=report([], normal_text="No focal abnormality on this slice.") if ctx else None)
        obj = o["obj"] if self.model.has_score else None  # run 2 trains the object score; run 1 never did
        app = to_app(parse_report(o["text"]), list(o["masks"].numpy()), obj, ctx=ctx, size=image.size)
        if app is not None and ctx:
            app["gate"] = {"p_abnormal": o.get("p_abnormal"), "decision": o.get("gate")}
        return app, o["text"], o["finished"], o["new_tokens"]

    @torch.no_grad()
    def chat_text(self, messages, max_new_tokens=1024, temperature=0.2):
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": content_text(m["content"])}]} for m in messages]
        with torch.autocast(self.dev, dtype=torch.bfloat16):
            _, _, text, finished, n = self._generate(msgs, None, max_new_tokens, temperature)
        return text, finished, n


def content_text(c):
    return c if isinstance(c, str) else "\n".join(p.get("text", "") for p in c if p.get("type") == "text")


def content_image(c):
    if isinstance(c, str):
        return None
    for p in c:
        if p.get("type") == "image_url":
            url = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
            return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    return None


def to_app(rep, masks, obj=None, ctx=None, size=None):
    """Skanda report + outlines → the app's AnalysisResult JSON.
    SAM 2 sees the whole image resized to 1024², so the 256² mask maps linearly onto the full image.
    ctx (CT/MRI): each finding's numbers are measured from its outline (skanda/ctmri.apply)."""
    if rep is None:
        return None
    findings = []
    for k, f in enumerate([f for f in rep.get("findings", []) if isinstance(f, dict)]):
        sev = str(f.get("severity", "moderate")).lower()
        item = {"title": f.get("title", "Finding"), "description": f.get("evidence", ""),
                "severity": sev if sev in SEVERITY else "moderate",
                "confidence": obj[k] if obj and k < len(obj) else 0.8,
                "confidenceSource": "SAM 2 outline object score (not a calibrated probability of disease)",
                "location": f.get("location", ""), "category": "", "recommendation": ""}
        if k < len(masks) and masks[k].any():
            m = masks[k]
            ys, xs = np.nonzero(m)
            h, w = m.shape
            item["box_2d"] = [round(1000 * ys.min() / h), round(1000 * xs.min() / w),
                              round(1000 * (ys.max() + 1) / h), round(1000 * (xs.max() + 1) / w)]
            buf = io.BytesIO()
            Image.fromarray(m.astype(np.uint8) * 255).save(buf, "PNG", optimize=True)
            item["mask_png"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        if ctx and size:
            item = ctmri.apply(item, masks[k] if k < len(masks) else None, size, ctx.get("pixel_spacing"), ctx.get("hu"),
                               ctmri.supplied_sequences(ctx.get("sequence"), ctx.get("composite")), ctx["modality"])
        findings.append(item)
    if not ctx:
        return {"modality": "X-ray", "bodyPart": "Chest", "view": "", "imageQuality": "adequate", "findings": findings,
                "pertinentNegatives": [], "impression": rep.get("impression", ""), "recommendations": [],
                "differential": [], "model": "skanda"}
    notes = []
    if ctx["modality"] == "ct" and ctx.get("hu") is None:
        notes.append("No raw HU supplied: the slice was read as displayed (Skanda trained on lung, soft-tissue and bone "
                     "windows together) and no attenuation was measured.")
    if ctx["modality"] == "mri" and not ctx.get("composite"):
        notes.append(f"Single MRI sequence ({ctx.get('sequence') or 'unknown'}): Skanda was trained on FLAIR, T1 "
                     "post-contrast and T2 together; single-sequence reads are outside its training data.")
    if not ctx.get("pixel_spacing"):
        notes.append("No pixel spacing supplied: sizes were not measured.")
    return {"modality": "CT" if ctx["modality"] == "ct" else "MR", "bodyPart": ctx["body"], "view": "axial",
            "imageQuality": " ".join(notes) or "adequate", "findings": findings, "pertinentNegatives": [],
            "impression": rep.get("impression", ""), "recommendations": [], "differential": [], "model": "skanda"}


def request_ctx(req, image):
    """The app's "skanda" request field → read context (None for a plain X-ray read)."""
    sk = req.get("skanda")
    if not isinstance(sk, dict) or sk.get("modality") not in ("ct", "mri"):
        return None
    ctx = {"modality": sk["modality"], "body": ctmri.body_for(sk["modality"], sk.get("body", "")),
           "sequence": sk.get("sequence"), "composite": bool(sk.get("composite"))}
    ps = sk.get("pixel_spacing")
    if isinstance(ps, (list, tuple)) and len(ps) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in ps):
        ctx["pixel_spacing"] = (float(ps[0]), float(ps[1]))
    hu = sk.get("hu")
    if ctx["modality"] == "ct" and isinstance(hu, dict):
        try:
            a = np.frombuffer(base64.b64decode(hu["b64"]), dtype="<i2").reshape(int(hu["height"]), int(hu["width"]))
            if a.shape == (image.height, image.width):
                ctx["hu"] = a.astype(np.float32)
        except (KeyError, ValueError, TypeError):
            pass
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--name", default="skanda")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sk = Skanda(args.model, device=args.device)
    print(f"Skanda ({args.model}) serving as '{args.name}' on :{args.port}", flush=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, b"")

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                # "state": LM Studio's /api/v0/models field, checked by LM Studio-style clients
                self._send(200, {"object": "list", "data": [{"id": args.name, "object": "model", "owned_by": "skanda",
                                                             "state": "loaded"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            msgs = req.get("messages", [])
            system = " ".join(content_text(m["content"]) for m in msgs if m.get("role") == "system")
            image = next((im for im in (content_image(m["content"]) for m in msgs if m.get("role") == "user") if im), None)
            t0 = time.time()
            try:
                with sk.lock:
                    if image is not None and LUNG_PATTERN_MARK in system:
                        text, finished, n = json.dumps({"grade": 0, "evidence": "Not graded separately: Skanda reports "
                                                        "diffuse findings in its main read."}), True, 0
                    elif image is not None:
                        ctx = request_ctx(req, image)
                        if ctx and ctx.get("hu") is not None:  # the training rendering, from HU
                            image = ctmri.render_ct(ctx["hu"])
                        app, raw, finished, n = sk.read(image, min(int(req.get("max_tokens") or 1024), 1536),
                                                        float(req.get("temperature") or 0.0), ctx=ctx)
                        text = json.dumps(app) if app is not None else raw  # unparsable → let the app's parser fail/retry
                    else:
                        text, finished, n = sk.chat_text(msgs, int(req.get("max_tokens") or 1024),
                                                         float(req.get("temperature") or 0.2))
            except Exception as e:  # report, don't kill the server
                return self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            fin = "stop" if finished else "length"
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:12], int(t0)
            usage = {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n}
            if req.get("stream"):
                chunks = [{"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.name,
                           "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                          {"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.name,
                           "choices": [{"index": 0, "delta": {}, "finish_reason": fin}], "usage": usage}]
                body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                return self._send(200, body.encode(), "text/event-stream")
            self._send(200, {"id": cid, "object": "chat.completion", "created": created, "model": args.name,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": fin}], "usage": usage})

        def log_message(self, fmt, *a):
            print(f"[{time.strftime('%T')}] {self.command} {self.path} {a[1] if len(a) > 1 else ''}", flush=True)

    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
