"""RunPod serverless handler for Resemble AI's Chatterbox Turbo TTS.

Mirrors the audio-generator-allen contract. Model loads LAZILY on first request so the
worker starts healthy and any GPU/load error is RETURNED in the response (API-readable)
instead of crash-looping. Chatterbox truncates ~40s/call, so long text is chunked.

Input (job["input"]): text, voice_name|prompt_audio, audio_path/app/file_title, max_chars
"""
import base64
import json
import os
import re
import tempfile
import traceback

import numpy as np
import soundfile as sf
import torch
import runpod
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

load_dotenv()

VOICE_DIRS = [d for d in [os.getenv("VOICES_DIR"), "/runpod-volume/voices", "/app/voices"] if d]
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "media.apparentgroup.co")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_M = {"loaded": False, "model": None, "sr": 24000, "error": None}


def _ensure_model():
    if _M["loaded"]:
        return _M
    try:
        from chatterbox.tts import ChatterboxTTS  # VERIFY Turbo loader for the installed version
        try:
            _M["model"] = ChatterboxTTS.from_pretrained(device=DEVICE, model="turbo")
        except TypeError:
            _M["model"] = ChatterboxTTS.from_pretrained(device=DEVICE)
        _M["sr"] = int(getattr(_M["model"], "sr", 24000))
        print("Chatterbox loaded, sr=%s" % _M["sr"], flush=True)
    except Exception as e:  # noqa: BLE001
        _M["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
        print("Chatterbox load FAILED:", _M["error"], flush=True)
    _M["loaded"] = True
    return _M


gcs_bucket = None
_gcs_json = os.getenv("GCS_SERVICE_ACCOUNT_JSON")
if _gcs_json:
    try:
        info = json.loads(_gcs_json)
        creds = service_account.Credentials.from_service_account_info(info)
        gcs_bucket = storage.Client(credentials=creds, project=info.get("project_id")).bucket(GCS_BUCKET_NAME)
    except Exception as e:  # noqa: BLE001
        print("GCS init failed:", e, flush=True)


def base64_to_temp_file(b64, suffix=".wav"):
    if "," in b64:
        b64 = b64.split(",")[1]
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(base64.b64decode(b64)); f.close()
    return f.name


def get_voice_file_path(voice_name):
    if not voice_name.endswith(".wav"):
        voice_name += ".wav"
    for d in VOICE_DIRS:
        p = os.path.join(d, voice_name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Voice '{voice_name}' not found in {VOICE_DIRS}")


def upload_audio_to_gcs(fp, audio_path, app, file_title):
    if not gcs_bucket:
        return None
    blob = f"{audio_path}/{app}/{file_title}.wav"
    gcs_bucket.blob(blob).upload_from_filename(fp, content_type="audio/wav")
    return f"https://{GCS_BUCKET_NAME}/{blob}"


def _chunk(text, mx):
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        if len(cur) + len(s) + 1 > mx and cur:
            out.append(cur.strip()); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur.strip())
    return out or [text]


def handler(job):
    ji = job.get("input", {}) or {}
    text = (ji.get("text") or "").strip()
    if not text:
        return {"error": "missing 'text'"}

    st = _ensure_model()
    if st["error"]:
        return {"error": "model_load_failed", "detail": st["error"], "device": DEVICE,
                "cuda": torch.cuda.is_available(),
                "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}
    model, sr = st["model"], st["sr"]

    voice_path, is_temp, out_path = None, False, f"/tmp/out_{job['id']}.wav"
    try:
        if ji.get("voice_name"):
            voice_path = get_voice_file_path(ji["voice_name"])
        elif ji.get("prompt_audio"):
            voice_path = base64_to_temp_file(ji["prompt_audio"]); is_temp = True
        pieces = []
        with torch.inference_mode():
            for ch in _chunk(text, int(ji.get("max_chars", 280))):
                wav = model.generate(ch, audio_prompt_path=voice_path) if voice_path else model.generate(ch)
                if torch.is_tensor(wav):
                    wav = wav.detach().float().cpu().numpy()
                pieces.append(np.asarray(wav, dtype="float32").reshape(-1))
        wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        sf.write(out_path, wav, sr, subtype="PCM_16")
        asec = round(len(wav) / float(sr), 3)
        ap, app, ft = ji.get("audio_path"), ji.get("app"), ji.get("file_title")
        if gcs_bucket and ap and app and ft:
            url = upload_audio_to_gcs(out_path, ap, app, ft)
            return {"audio_url": url, "audio_seconds": asec} if url else {"error": "gcs upload failed"}
        with open(out_path, "rb") as f:
            return {"audio_base64": base64.b64encode(f.read()).decode(), "audio_seconds": asec}
    except Exception as e:  # noqa: BLE001
        return {"error": "inference_failed", "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1000:]}"}
    finally:
        if is_temp and voice_path and os.path.exists(voice_path):
            os.remove(voice_path)
        if os.path.exists(out_path):
            os.remove(out_path)


runpod.serverless.start({"handler": handler})
