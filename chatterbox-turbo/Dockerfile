# Chatterbox Turbo RunPod serverless worker (mirrors audio-generator-allen build pattern).
# RunPod substitutes Blackwell GPUs (sm_120) when Ada/Ampere supply is low, and chatterbox-tts
# pins torch==2.6.0 (CUDA 12.6, no sm_120). So: install chatterbox normally, then FORCE torch
# 2.7.1+cu128 on top so it runs on Blackwell. (handler.py lazy-loads + surfaces any error if
# chatterbox misbehaves on torch 2.7.)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf \
    VOICES_DIR=/app/voices \
    GCS_BUCKET_NAME=media.apparentgroup.co \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Keep the base image's torch 2.7.1+cu128 (Blackwell-capable). Install chatterbox-tts with
# --no-deps so its hard torch==2.6.0 pin can't downgrade torch, then install its real deps
# (none of which pin torch to 2.6), then the worker extras.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-deps chatterbox-tts && \
    pip install --no-cache-dir "numpy<2" librosa==0.11.0 s3tokenizer transformers==5.2.0 \
        diffusers==0.29.0 "resemble-perth>=1.0.0" conformer==0.3.2 safetensors==0.5.3 \
        spacy-pkuseg pykakasi==2.3.0 pyloudnorm omegaconf && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Bake the app reference voices INTO the image so voice_name (jenny/mark/nayan/Lisa/nico/
# evelyn) works on any GPU in any datacenter — no Network Volume dependency. A mounted
# /runpod-volume/voices still overrides (handler checks $VOICES_DIR -> volume -> /app/voices).
COPY voices/ /app/voices/

# Chatterbox Turbo weights download on the FIRST cold start via from_pretrained — keeps the
# image small and CI-buildable. Warm requests reuse the loaded model.
CMD ["python", "-u", "handler.py"]
