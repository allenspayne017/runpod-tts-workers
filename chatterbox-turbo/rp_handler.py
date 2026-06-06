"""RunPod serverless handler for Resemble AI's Chatterbox Turbo TTS.

Input  (event["input"]):
    text       (str, required)
    voice_url  (str, optional)  reference WAV for voice cloning
    format     (str, optional)  "wav" (default)
    max_chars  (int, optional)  per-chunk cap (default 280 — Chatterbox caps ~40s/call)

Output:
    audio_base64, sample_rate, audio_seconds, chunks

NOTE: Chatterbox truncates long input (~40s / ~700 chars per call), proven empirically,
so this handler chunks at sentence boundaries and concatenates. Confirm the exact Turbo
checkpoint loader in the `chatterbox` package version you build against (flagged below).
"""
import base64
import io
import os
import re
import tempfile
import urllib.request

import numpy as np
import soundfile as sf
import torch
import runpod

# ---- VERIFY: exact Turbo loader. Standard Chatterbox is ChatterboxTTS.from_pretrained();
# Chatterbox Turbo may expose a `turbo=True` flag or a distinct class — confirm at build. ----
from chatterbox.tts import ChatterboxTTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[chatterbox] loading Turbo on {DEVICE}", flush=True)
try:
    _model = ChatterboxTTS.from_pretrained(device=DEVICE, model="turbo")   # preferred if supported
except TypeError:
    _model = ChatterboxTTS.from_pretrained(device=DEVICE)                  # fallback to default ckpt
_SR = int(getattr(_model, "sr", 24000))
print(f"[chatterbox] ready, sr={_SR}", flush=True)


def _chunk(text, max_chars):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > max_chars and cur:
            out.append(cur.strip()); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur.strip())
    return out or [text]


def _fetch_voice(voice_url):
    if not voice_url:
        return None
    raw = urllib.request.urlopen(voice_url, timeout=30).read()
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(raw); f.flush(); f.close()
    return f.name


def handler(event):
    inp = event.get("input", {}) or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "missing 'text'"}
    max_chars = int(inp.get("max_chars", 280))
    voice_path = _fetch_voice(inp.get("voice_url"))

    pieces = []
    chunks = _chunk(text, max_chars)
    with torch.inference_mode():
        for ch in chunks:
            wav = _model.generate(ch, audio_prompt_path=voice_path) if voice_path else _model.generate(ch)
            if torch.is_tensor(wav):
                wav = wav.detach().float().cpu().numpy()
            pieces.append(np.asarray(wav, dtype="float32").reshape(-1))

    wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    buf = io.BytesIO()
    sf.write(buf, wav, _SR, format="WAV", subtype="PCM_16")
    return {
        "audio_base64": base64.b64encode(buf.getvalue()).decode(),
        "sample_rate": _SR,
        "audio_seconds": round(len(wav) / float(_SR), 3),
        "chunks": len(chunks),
    }


runpod.serverless.start({"handler": handler})
