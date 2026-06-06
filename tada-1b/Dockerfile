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

COPY handler.py .

# Model weights (HumeAI/tada-codec + HumeAI/tada-1b) download on the FIRST cold start via
# from_pretrained — keeps the image small and CI-buildable. If the repo is gated, set an
# HF_TOKEN endpoint env. Subsequent (warm) requests reuse the loaded model.

# Reference voices + GCS creds come from the Network Volume / env at runtime
# (mount /runpod-volume with the shared voices, set GCS_SERVICE_ACCOUNT_JSON).
CMD ["python", "-u", "handler.py"]
