# Chatterbox Turbo RunPod serverless worker (mirrors audio-generator-allen build pattern).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

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

# Bake Chatterbox Turbo weights so cold starts don't pull from HF.
# VERIFY the Turbo checkpoint id matches handler.py's loader.
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')" || \
    echo "WARN: pre-download skipped — weights pull on first cold start"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
