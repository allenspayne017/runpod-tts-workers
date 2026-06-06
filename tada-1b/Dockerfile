# TADA-1B RunPod serverless worker (mirrors audio-generator-allen build pattern).
# hume-tada needs torch>=2.7,<2.8. CUDA 12.8 build so it also runs on Blackwell GPUs
# (sm_120) that RunPod may assign in the 24GB/MIG pools — cu126 lacks sm_120 kernels.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

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
# Staged install avoids pip 'resolution-too-deep': resolve hume-tada (heavy transformers
# graph) on its own first, then the lightweight extras separately.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir hume-tada && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Model weights (HumeAI/tada-codec + HumeAI/tada-1b) download on the FIRST cold start via
# from_pretrained — keeps the image small and CI-buildable. If the repo is gated, set an
# HF_TOKEN endpoint env. Subsequent (warm) requests reuse the loaded model.

# Reference voices + GCS creds come from the Network Volume / env at runtime
# (mount /runpod-volume with the shared voices, set GCS_SERVICE_ACCOUNT_JSON).
CMD ["python", "-u", "handler.py"]
