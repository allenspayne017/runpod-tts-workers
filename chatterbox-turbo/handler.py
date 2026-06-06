"""RunPod serverless handler for Resemble AI's Chatterbox Turbo TTS.

Mirrors the proven `audio-generator-allen` (IndexTTS2) contract: same input keys, voice
from the network volume, GCS upload with base64 fallback.

Input (job["input"]):
    text        (str, required)
    voice_name  (str)   reference voice on volume: /runpod-volume/voices/{name}.wav (clone)
    prompt_audio(str)   base64 reference WAV (alternative to voice_name)
    audio_path/app/file_title  -> upload to GCS and return {"audio_url"}, else {"audio_base64"}
    max_chars   (int)   per-chunk cap (default 280; Chatterbox truncates ~40s/call)

Output: {"audio_url"} or {"audio_base64"}  (+ "audio_seconds")
"""
import base64
import json
import os
import re
import tempfile

import numpy as np
import soundfile as sf
import torch
import runpod
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

# VERIFY at build: standard loader is ChatterboxTTS.from_pretrained(); confirm the Turbo
# checkpoint/flag for the chatterbox-tts version you build against.
from chatterbox.tts import ChatterboxTTS

load_dotenv()

VOICES_DIR = os.getenv("VOICES_DIR", "/runpod-volume/voices")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "media.apparentgroup.co")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Initializing Chatterbox Turbo on {DEVICE}...", flush=True)
model = None
try:
    try:
        model = ChatterboxTTS.from_pretrained(device=DEVICE, model="turbo")
    except TypeError:
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
    print("Model loaded successfully!", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"WARNING: model load deferred/failed (expected during build): {e}", flush=True)
_SR = int(getattr(model, "sr", 24000)) if model is not None else 24000

gcs_bucket = None
_gcs_json = os.getenv("GCS_SERVICE_ACCOUNT_JSON")
if _gcs_json:
    try:
        info = json.loads(_gcs_json)
        creds = service_account.Credentials.from_service_account_info(info)
        gcs_bucket = storage.Client(credentials=creds, project=info.get("project_id")).bucket(GCS_BUCKET_NAME)
        print(f"GCS client initialized for bucket: {GCS_BUCKET_NAME}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: Failed to initialize GCS client: {e}", flush=True)
else:
    print("WARNING: GCS_SERVICE_ACCOUNT_JSON not set. Audio will be returned as base64.", flush=True)


def base64_to_temp_file(b64: str, suffix: str = ".wav") -> str:
    if "," in b64:
        b64 = b64.split(",")[1]
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(base64.b64decode(b64))
    f.close()
    return f.name


def get_voice_file_path(voice_name: str):
    if not voice_name.endswith(".wav"):
        voice_name += ".wav"
    path = os.path.join(VOICES_DIR, voice_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Voice file '{voice_name}' not found in {VOICES_DIR}")
    return path


def upload_audio_to_gcs(file_path, audio_path, app, file_title):
    if not gcs_bucket:
        return None
    blob_name = f"{audio_path}/{app}/{file_title}.wav"
    gcs_bucket.blob(blob_name).upload_from_filename(file_path, content_type="audio/wav")
    return f"https://{GCS_BUCKET_NAME}/{blob_name}"


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


def handler(job):
    job_input = job["input"]
    if model is None:
        return {"error": "Model not initialized."}

    text = job_input.get("text")
    if not text:
        return {"error": "Missing required parameter: 'text'"}
    max_chars = int(job_input.get("max_chars", 280))
    audio_path = job_input.get("audio_path")
    app = job_input.get("app")
    file_title = job_input.get("file_title")

    voice_path, is_temp = None, False
    output_path = f"/tmp/out_{job['id']}.wav"
    try:
        if job_input.get("voice_name"):
            voice_path = get_voice_file_path(job_input["voice_name"])
        elif job_input.get("prompt_audio"):
            voice_path = base64_to_temp_file(job_input["prompt_audio"]); is_temp = True

        pieces = []
        with torch.inference_mode():
            for ch in _chunk(text, max_chars):
                wav = model.generate(ch, audio_prompt_path=voice_path) if voice_path else model.generate(ch)
                if torch.is_tensor(wav):
                    wav = wav.detach().float().cpu().numpy()
                pieces.append(np.asarray(wav, dtype="float32").reshape(-1))
        wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        sf.write(output_path, wav, _SR, subtype="PCM_16")
        audio_seconds = round(len(wav) / float(_SR), 3)

        if gcs_bucket and audio_path and app and file_title:
            url = upload_audio_to_gcs(output_path, audio_path, app, file_title)
            return {"audio_url": url, "audio_seconds": audio_seconds} if url else {"error": "GCS upload failed"}
        with open(output_path, "rb") as f:
            return {"audio_base64": base64.b64encode(f.read()).decode(), "audio_seconds": audio_seconds}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Inference failed: {str(e)}"}
    finally:
        if is_temp and voice_path and os.path.exists(voice_path):
            os.remove(voice_path)
        if os.path.exists(output_path):
            os.remove(output_path)


runpod.serverless.start({"handler": handler})
