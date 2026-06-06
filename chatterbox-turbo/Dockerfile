# Chatterbox Turbo RunPod serverless worker.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Chatterbox Turbo weights so cold starts don't pull from HF.
# VERIFY the repo id for the Turbo checkpoint matches the one rp_handler.py loads.
RUN python -c "from chatterbox.tts import ChatterboxTTS; \
    import torch; ChatterboxTTS.from_pretrained(device='cpu')" || \
    echo "WARN: pre-download skipped — weights will pull on first cold start"

COPY rp_handler.py .

CMD ["python", "-u", "rp_handler.py"]
