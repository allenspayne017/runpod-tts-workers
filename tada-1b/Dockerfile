# TADA-1B RunPod serverless worker.
# Base image already ships torch + torchaudio with CUDA, so we don't reinstall them.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf \
    SAMPLE_RATE=24000 \
    MODEL_ID=HumeAI/tada-1b \
    CODEC_ID=HumeAI/tada-codec

WORKDIR /app

# system deps for soundfile/torchaudio (libsndfile, ffmpeg) + git for pip VCS installs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake a default reference voice (zero-shot TADA needs reference audio + transcript).
# REFERENCE_TEXT in rp_handler.py must match this clip's transcript.
RUN mkdir -p /app/samples && \
    curl -fsSL -o /app/samples/reference.wav \
      https://raw.githubusercontent.com/HumeAI/tada/main/samples/ljspeech.wav || \
    echo "WARN: bake your own /app/samples/reference.wav + set REFERENCE_TEXT"

# Pre-download model weights into the image so cold starts don't pull from HF.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('HumeAI/tada-codec'); snapshot_download('HumeAI/tada-1b')"

COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
