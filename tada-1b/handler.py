"""RunPod serverless handler for Hume AI's TADA-1B TTS.

Mirrors the audio-generator-allen contract. Model loads LAZILY on first request (not at
import) so the worker starts healthy and any GPU/load error is RETURNED in the response
(readable via the API) instead of crash-looping the worker.

Input (job["input"]):
    text, voice_name|prompt_audio, voice_text (TADA needs the ref transcript),
    audio_path/app/file_title (-> GCS url, else base64), max_chars
"""
import base64
import io
import json
import os
import re
import tempfile
import traceback

import numpy as np
import soundfile as sf
import torch
import torchaudio
import runpod
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account

load_dotenv()

VOICE_DIRS = [d for d in [os.getenv("VOICES_DIR"), "/runpod-volume/voices", "/app/voices"] if d]
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "media.apparentgroup.co")
MODEL_ID = os.getenv("MODEL_ID", "HumeAI/tada-1b")
CODEC_ID = os.getenv("CODEC_ID", "HumeAI/tada-codec")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
REFERENCE_TEXT = os.getenv("REFERENCE_TEXT",
                           "The examination and testimony of the experts enabled the committee to reach a clear conclusion.")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- lazy model load (first request) so import never touches the GPU ---
_M = {"loaded": False, "encoder": None, "model": None, "error": None}


def _ensure_model():
    if _M["loaded"]:
        return _M
    try:
        from tada.modules.encoder import Encoder
        from tada.modules.tada import TadaForCausalLM, InferenceOptions
        _M["InferenceOptions"] = InferenceOptions
        _M["encoder"] = Encoder.from_pretrained(CODEC_ID, subfolder="encoder").to(DEVICE)
        m = TadaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
        m.eval()
        _M["model"] = m
        print("TADA-1B loaded.", flush=True)
    except Exception as e:  # noqa: BLE001
        _M["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
        print("TADA-1B load FAILED:", _M["error"], flush=True)
    _M["loaded"] = True
    return _M


# --- GCS (optional; base64 fallback) ---
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


def resolve_reference(ji):
    if ji.get("voice_name"):
        path = get_voice_file_path(ji["voice_name"])
        vt = ji.get("voice_text")
        if not vt:
            side = os.path.splitext(path)[0] + ".txt"
            vt = open(side).read().strip() if os.path.exists(side) else REFERENCE_TEXT
        return path, vt, False
    if ji.get("prompt_audio"):
        return base64_to_temp_file(ji["prompt_audio"]), (ji.get("voice_text") or REFERENCE_TEXT), True
    raise ValueError("Missing 'voice_name' or 'prompt_audio'.")


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


def _one(x):
    if torch.is_tensor(x):
        if x.dim() > 1:
            x = x.reshape(-1)
        return x.detach().to("cpu", torch.float32).numpy().reshape(-1)
    if isinstance(x, np.ndarray):
        return x.astype("float32").reshape(-1)
    return None


def _to_waveform(output, encoder):
    """TADA: output.audio is a list of per-item waveform tensors (24kHz). Notebook uses
    output.audio[0]; we concat if there are several."""
    aud = getattr(output, "audio", None)
    if aud is None and hasattr(output, "get"):
        try:
            aud = output.get("audio")
        except Exception:
            aud = None
    if isinstance(aud, (list, tuple)):
        parts = [p for p in (_one(a) for a in aud) if p is not None]
        if parts:
            return np.concatenate(parts) if len(parts) > 1 else parts[0]
    w = _one(aud)
    if w is not None:
        return w
    raise RuntimeError(f"output.audio not decodable: type={type(aud)} repr={repr(aud)[:160]}")


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
    encoder, model = st["encoder"], st["model"]

    ref_path, is_temp, out_path = None, False, f"/tmp/out_{job['id']}.wav"
    try:
        ref_path, ref_text, is_temp = resolve_reference(ji)
        ref_audio, ref_sr = torchaudio.load(ref_path)
        ref_audio = ref_audio.mean(dim=0, keepdim=True)                    # mono (notebook)
        ref_audio = ref_audio / ref_audio.abs().max().clamp(min=1e-8)      # normalize (notebook)
        ref_audio = ref_audio.to(DEVICE)
        try:
            prompt = encoder(ref_audio, sample_rate=ref_sr)                # current TADA API (no transcript)
        except TypeError:
            prompt = encoder(ref_audio, text=[ref_text], sample_rate=ref_sr)  # older API fallback
        # tone/quality controls: pass inference_options (speed_up_factor, acoustic_cfg_scale,
        # num_acoustic_candidates+scorer, etc.) + num_transition_steps from the request.
        IO = st.get("InferenceOptions")
        io_kwargs = ji.get("inference_options") or {}
        gen_kwargs = {"num_transition_steps": int(ji.get("num_transition_steps", 5))}
        if io_kwargs and IO is not None:
            gen_kwargs["inference_options"] = IO(**io_kwargs)
        pieces = []
        with torch.inference_mode():
            for ch in _chunk(text, int(ji.get("max_chars", 600))):
                pieces.append(_to_waveform(model.generate(prompt=prompt, text=ch, **gen_kwargs), encoder))
        wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        sf.write(out_path, wav, SAMPLE_RATE, subtype="PCM_16")
        asec = round(len(wav) / float(SAMPLE_RATE), 3)
        ap, app, ft = ji.get("audio_path"), ji.get("app"), ji.get("file_title")
        if gcs_bucket and ap and app and ft:
            url = upload_audio_to_gcs(out_path, ap, app, ft)
            return {"audio_url": url, "audio_seconds": asec} if url else {"error": "gcs upload failed"}
        with open(out_path, "rb") as f:
            return {"audio_base64": base64.b64encode(f.read()).decode(), "audio_seconds": asec}
    except Exception as e:  # noqa: BLE001
        return {"error": "inference_failed", "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1000:]}"}
    finally:
        if is_temp and ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(out_path):
            os.remove(out_path)


runpod.serverless.start({"handler": handler})
