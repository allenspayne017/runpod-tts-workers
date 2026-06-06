"""RunPod serverless handler for Hume AI's TADA-1B TTS.

Mirrors the proven `audio-generator-allen` (IndexTTS2) contract so it drops straight into
the existing n8n pipeline + GCS storage.

Input (job["input"]):
    text        (str, required)
    voice_name  (str)   reference voice on the network volume: /runpod-volume/voices/{name}.wav
    prompt_audio(str)   base64 reference WAV (alternative to voice_name)
    voice_text  (str)   transcript of the reference voice (TADA is zero-shot and REQUIRES it).
                        Falls back to /runpod-volume/voices/{name}.txt then REFERENCE_TEXT env.
    audio_path  (str)   GCS base path, e.g. "audio"      ┐ all three present -> upload to GCS,
    app         (str)   project, e.g. "daily-grace"      ├ return {"audio_url"}
    file_title  (str)   filename w/o ext                 ┘ else return {"audio_base64"}
    max_chars   (int)   per-chunk cap (default 600)

Output: {"audio_url": ...} or {"audio_base64": ...}  (+ "audio_seconds")
"""
import base64
import io
import json
import os
import re
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
import runpod
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

load_dotenv()

# --- Config (matches audio-generator-allen layout) ---
VOICES_DIR = os.getenv("VOICES_DIR", "/runpod-volume/voices")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "media.apparentgroup.co")
MODEL_ID = os.getenv("MODEL_ID", "HumeAI/tada-1b")
CODEC_ID = os.getenv("CODEC_ID", "HumeAI/tada-codec")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
REFERENCE_TEXT = os.getenv(
    "REFERENCE_TEXT",
    "The examination and testimony of the experts enabled the committee to reach a clear conclusion.",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model init (weights baked into image; guarded so docker build doesn't fail) ---
print(f"Initializing TADA-1B on {DEVICE}...", flush=True)
encoder = None
model = None
try:
    encoder = Encoder.from_pretrained(CODEC_ID, subfolder="encoder").to(DEVICE)
    model = TadaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    print("Model loaded successfully!", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"WARNING: model load deferred/failed (expected during build): {e}", flush=True)

# --- GCS client (same pattern as the reference worker) ---
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


def get_voice_file_path(voice_name: str) -> str:
    if not voice_name.endswith(".wav"):
        voice_name += ".wav"
    path = os.path.join(VOICES_DIR, voice_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Voice file '{voice_name}' not found in {VOICES_DIR}")
    return path


def resolve_reference(job_input):
    """Return (ref_wav_path, ref_transcript, is_temp)."""
    voice_name = job_input.get("voice_name")
    prompt_audio = job_input.get("prompt_audio")
    voice_text = job_input.get("voice_text")
    if voice_name:
        path = get_voice_file_path(voice_name)
        if not voice_text:
            sidecar = os.path.splitext(path)[0] + ".txt"
            voice_text = open(sidecar).read().strip() if os.path.exists(sidecar) else REFERENCE_TEXT
        return path, voice_text, False
    if prompt_audio:
        return base64_to_temp_file(prompt_audio), (voice_text or REFERENCE_TEXT), True
    raise ValueError("Missing 'voice_name' or 'prompt_audio'.")


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


def _to_waveform(output):
    """Decode model.generate() output to 1-D float32 audio. Logs real type on first run
    so the correct path can be pinned (HumeAI README doesn't document decode)."""
    if torch.is_tensor(output):
        return output.detach().float().cpu().numpy().reshape(-1)
    if isinstance(output, np.ndarray):
        return output.astype("float32").reshape(-1)
    for attr in ("audio", "waveform", "wav", "audio_values", "values"):
        if hasattr(output, attr):
            v = getattr(output, attr)
            return (v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v, "float32")).reshape(-1)
    for attr in ("audio_codes", "codes", "tokens", "sequences"):
        if hasattr(output, attr) and hasattr(encoder, "decode"):
            return encoder.decode(getattr(output, attr)).detach().float().cpu().numpy().reshape(-1)
    raise RuntimeError(
        f"cannot decode generate() output type={type(output)} "
        f"attrs={[a for a in dir(output) if not a.startswith('_')][:40]}"
    )


def handler(job):
    job_input = job["input"]
    if model is None or encoder is None:
        return {"error": "Model not initialized."}

    text = job_input.get("text")
    if not text:
        return {"error": "Missing required parameter: 'text'"}
    max_chars = int(job_input.get("max_chars", 600))
    audio_path = job_input.get("audio_path")
    app = job_input.get("app")
    file_title = job_input.get("file_title")

    ref_path, ref_text, is_temp = None, None, False
    output_path = f"/tmp/out_{job['id']}.wav"
    try:
        ref_path, ref_text, is_temp = resolve_reference(job_input)
        ref_audio, ref_sr = torchaudio.load(ref_path)
        prompt = encoder(ref_audio.to(DEVICE), text=[ref_text], sample_rate=ref_sr)

        pieces = []
        with torch.inference_mode():
            for ch in _chunk(text, max_chars):
                pieces.append(_to_waveform(model.generate(prompt=prompt, text=ch)))
        wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        sf.write(output_path, wav, SAMPLE_RATE, subtype="PCM_16")
        audio_seconds = round(len(wav) / float(SAMPLE_RATE), 3)

        if gcs_bucket and audio_path and app and file_title:
            url = upload_audio_to_gcs(output_path, audio_path, app, file_title)
            return {"audio_url": url, "audio_seconds": audio_seconds} if url else {"error": "GCS upload failed"}
        with open(output_path, "rb") as f:
            return {"audio_base64": base64.b64encode(f.read()).decode(), "audio_seconds": audio_seconds}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Inference failed: {str(e)}"}
    finally:
        if is_temp and ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(output_path):
            os.remove(output_path)


runpod.serverless.start({"handler": handler})
