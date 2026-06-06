"""RunPod serverless handler for Hume AI's TADA-1B text-to-speech.

Input  (event["input"]):
    text        (str, required)  text to synthesize
    voice_url   (str, optional)  URL to a reference voice WAV (overrides the baked default)
    voice_text  (str, optional)  transcript of the reference voice audio
    format      (str, optional)  "wav" (default) — only wav is implemented
    max_chars   (int, optional)  per-chunk char cap for long input (default 600)

Output:
    audio_base64 (str)   base64-encoded WAV of the full synthesized audio
    sample_rate  (int)
    audio_seconds(float) duration of the returned audio (for RTF/cost math)
    chunks       (int)   how many generate() calls were stitched

Model API (per HumeAI/tada README):
    encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder")
    model   = TadaForCausalLM.from_pretrained("HumeAI/tada-1b")
    prompt  = encoder(ref_audio, text=[ref_text], sample_rate=sr)
    output  = model.generate(prompt=prompt, text="...")
The decode of `output` -> waveform is NOT documented; _to_waveform() probes the likely
paths and logs the real type on first run so it can be pinned. See README verify-flags.
"""
import base64
import io
import os
import re
import urllib.request

import numpy as np
import soundfile as sf
import torch
import torchaudio
import runpod

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = os.environ.get("MODEL_ID", "HumeAI/tada-1b")
CODEC_ID = os.environ.get("CODEC_ID", "HumeAI/tada-codec")
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "24000"))
REF_AUDIO_PATH = os.environ.get("REFERENCE_AUDIO", "/app/samples/reference.wav")
REF_TEXT = os.environ.get(
    "REFERENCE_TEXT",
    "The examination and testimony of the experts enabled the committee to reach a clear conclusion.",
)

# ---- load once at cold start ----
print(f"[tada] loading codec={CODEC_ID} model={MODEL_ID} on {DEVICE}", flush=True)
_encoder = Encoder.from_pretrained(CODEC_ID, subfolder="encoder").to(DEVICE)
_model = TadaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
_model.eval()
print("[tada] ready", flush=True)


def _load_ref(voice_url, voice_text):
    if voice_url:
        raw = urllib.request.urlopen(voice_url, timeout=30).read()
        audio, sr = torchaudio.load(io.BytesIO(raw))
        return audio.to(DEVICE), sr, (voice_text or REF_TEXT)
    audio, sr = torchaudio.load(REF_AUDIO_PATH)
    return audio.to(DEVICE), sr, REF_TEXT


def _chunk(text, max_chars):
    """Split on sentence boundaries into <= max_chars pieces (TADA handles long text but
    chunking keeps memory bounded and matches the production stitching pattern)."""
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


def _to_waveform(output):
    """Best-effort decode of model.generate() output to a 1-D float32 numpy waveform.
    Logs the real type on first call so the correct path can be pinned (see README)."""
    # 1) already a tensor/array of audio samples
    if torch.is_tensor(output):
        return output.detach().float().cpu().numpy().reshape(-1)
    if isinstance(output, np.ndarray):
        return output.astype("float32").reshape(-1)
    # 2) common attribute names on a result object
    for attr in ("audio", "waveform", "wav", "audio_values", "values"):
        if hasattr(output, attr):
            v = getattr(output, attr)
            if torch.is_tensor(v):
                return v.detach().float().cpu().numpy().reshape(-1)
            if isinstance(v, np.ndarray):
                return v.astype("float32").reshape(-1)
    # 3) codec decode of token ids
    for attr in ("audio_codes", "codes", "tokens", "sequences"):
        if hasattr(output, attr) and hasattr(_encoder, "decode"):
            wav = _encoder.decode(getattr(output, attr))
            return wav.detach().float().cpu().numpy().reshape(-1)
    raise RuntimeError(
        f"[tada] cannot decode generate() output type={type(output)} "
        f"attrs={[a for a in dir(output) if not a.startswith('_')][:40]} — pin the decode path."
    )


def handler(event):
    inp = event.get("input", {}) or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "missing 'text'"}
    max_chars = int(inp.get("max_chars", 600))

    ref_audio, ref_sr, ref_text = _load_ref(inp.get("voice_url"), inp.get("voice_text"))
    prompt = _encoder(ref_audio, text=[ref_text], sample_rate=ref_sr)

    pieces = []
    chunks = _chunk(text, max_chars)
    with torch.inference_mode():
        for ch in chunks:
            out = _model.generate(prompt=prompt, text=ch)
            pieces.append(_to_waveform(out))

    wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    audio_b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "audio_base64": audio_b64,
        "sample_rate": SAMPLE_RATE,
        "audio_seconds": round(len(wav) / float(SAMPLE_RATE), 3),
        "chunks": len(chunks),
    }


runpod.serverless.start({"handler": handler})
