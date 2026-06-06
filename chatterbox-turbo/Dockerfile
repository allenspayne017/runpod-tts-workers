# Chatterbox Turbo RunPod serverless worker (mirrors audio-generator-allen build pattern).
# chatterbox-tts pins torch==2.6.0 + torchaudio==2.6.0 — use a matching base so pip does
# NOT reinstall torch.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

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
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Chatterbox Turbo weights download on the FIRST cold start via from_pretrained — keeps the
# image small and CI-buildable. Warm requests reuse the loaded model.

CMD ["python", "-u", "handler.py"]
