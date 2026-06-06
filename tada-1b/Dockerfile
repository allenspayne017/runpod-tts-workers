# TADA-1B RunPod serverless worker (mirrors audio-generator-allen build pattern).
# torch + torchaudio ship with the base image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

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

# Bake model weights so cold starts don't pull from HF.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('HumeAI/tada-codec'); snapshot_download('HumeAI/tada-1b')"

COPY handler.py .

# Reference voices + GCS creds come from the Network Volume / env at runtime
# (mount /runpod-volume with the shared voices, set GCS_SERVICE_ACCOUNT_JSON).
CMD ["python", "-u", "handler.py"]
