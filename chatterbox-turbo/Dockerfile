# Chatterbox Turbo RunPod serverless worker (mirrors audio-generator-allen build pattern).
# RunPod substitutes Blackwell GPUs (sm_120) when Ada/Ampere supply is low, and chatterbox-tts
# pins torch==2.6.0 (CUDA 12.6, no sm_120). So: install chatterbox normally, then FORCE torch
# 2.7.1+cu128 on top so it runs on Blackwell. (handler.py lazy-loads + surfaces any error if
# chatterbox misbehaves on torch 2.7.)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf \
    VOICES_DIR=/runpod-volume/voices \
    GCS_BUCKET_NAME=media.apparentgroup.co \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install chatterbox (pulls torch 2.6), THEN force torch/torchaudio 2.7.1+cu128 so it runs
# on Blackwell GPUs. transformers 5.2.0 from chatterbox stays.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall --no-deps \
        torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

COPY handler.py .

# Chatterbox Turbo weights download on the FIRST cold start via from_pretrained — keeps the
# image small and CI-buildable. Warm requests reuse the loaded model.

CMD ["python", "-u", "handler.py"]
