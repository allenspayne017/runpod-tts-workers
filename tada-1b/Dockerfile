# TADA-1B RunPod serverless worker (mirrors audio-generator-allen build pattern).
# hume-tada requires torch>=2.7.0,<2.8.0 + torchvision/torchaudio — use a matching base
# so pip does NOT reinstall torch (the cause of the earlier build failure).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf \
    SAMPLE_RATE=24000 \
    MODEL_ID=HumeAI/tada-1b \
    CODEC_ID=HumeAI/tada-codec \
    VOICES_DIR=/runpod-volume/voices \
    GCS_BUCKET_NAME=media.apparentgroup.co \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Bake model weights so cold starts don't pull from HF. Non-fatal: if the repo is gated
# (Llama-3.2 license) or HF is flaky, build still succeeds and weights pull at runtime
# (set an HF_TOKEN endpoint env if they turn out to be gated).
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('HumeAI/tada-codec'); snapshot_download('HumeAI/tada-1b')" || \
    echo "WARN: weight pre-download failed (gated/offline) — will download on first run"

COPY handler.py .

# Reference voices + GCS creds come from the Network Volume / env at runtime
# (mount /runpod-volume with the shared voices, set GCS_SERVICE_ACCOUNT_JSON).
CMD ["python", "-u", "handler.py"]
