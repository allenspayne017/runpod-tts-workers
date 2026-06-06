# RunPod TTS Workers — TADA-1B & Chatterbox Turbo

Self-hosted RunPod **serverless** workers for benchmarking new open-source TTS models
against the existing IndexTTS2 service, on the **same GPU (RTX 3090)** so the cost
comparison is apples-to-apples (GPU-time billing, not per-audio-second).

## Why these exist
The existing IndexTTS2 endpoint (`indextts-api-service`, id `7axelvd332ctge`) runs the
public image `mob124/indextts-xtts:latest` as a serverless template (`63kiqawgvm`) on an
RTX 3090. We replicate that exact pattern for two new models so we can measure real
`executionTime` (GPU-seconds) per request and compute true cost.

| Model | Image source | Build needed? |
|---|---|---|
| IndexTTS2 (baseline) | `mob124/indextts-xtts:latest` (public) | already deployed |
| Chatterbox Turbo | `chatterbox-turbo/` (this repo) | yes — build + push |
| TADA-1B | `tada-1b/` (this repo) | yes — build + push |

> Chatterbox Turbo also exists as a RunPod **public** endpoint
> (`api.runpod.ai/v2/chatterbox-turbo`) but it bills **$0.001 per second of audio**
> (~$3.60/audio-hour) — pricier than Inworld and NOT the cheap path. Self-hosting on a
> 3090 is ~17× cheaper, which is what these workers measure.

## Layout
```
runpod-workers/
├── README.md              ← this file
├── deploy_endpoint.py     ← create RunPod template+endpoint via REST (mirrors IndexTTS2)
├── benchmark.py           ← run the same paragraph through all 3 endpoints, print RTF+cost
├── tada-1b/
│   ├── rp_handler.py
│   ├── requirements.txt
│   └── Dockerfile
└── chatterbox-turbo/
    ├── rp_handler.py
    ├── requirements.txt
    └── Dockerfile
```

## Build + push (the one manual step — needs Docker)
Run on any machine with Docker + a Docker Hub (or RunPod registry) login:
```bash
# TADA-1B
cd tada-1b
docker build -t YOURUSER/tada-1b-runpod:latest .
docker push YOURUSER/tada-1b-runpod:latest

# Chatterbox Turbo
cd ../chatterbox-turbo
docker build -t YOURUSER/chatterbox-turbo-runpod:latest .
docker push YOURUSER/chatterbox-turbo-runpod:latest
```
No local Docker? Use RunPod's **GitHub build**: push this folder to a GitHub repo and point
a RunPod serverless endpoint at it (RunPod builds the image for you).

## Deploy (after the image is pushed)
```bash
export RUNPOD_API_KEY=rpa_...
python deploy_endpoint.py --name tada-1b --image YOURUSER/tada-1b-runpod:latest --gpu "NVIDIA GeForce RTX 3090"
python deploy_endpoint.py --name chatterbox-turbo --image YOURUSER/chatterbox-turbo-runpod:latest --gpu "NVIDIA GeForce RTX 3090"
```
This mirrors the IndexTTS2 endpoint config: 20GB disk, flashboot on, idle 5s,
exec timeout 600s, workersMax 1, scaler REQUEST_COUNT.

## Benchmark
```bash
export RUNPOD_API_KEY=rpa_...
python benchmark.py
```
Submits the same test paragraph(s) to each endpoint, reads RunPod `executionTime` +
output audio duration, computes RTF and GPU-time cost at the 3090 rate, and prints a
comparison table vs IndexTTS2.

## ⚠️ Verify-at-build flags
- **TADA-1B output decode**: the HumeAI/tada README documents `model.generate(...)` but not
  how `output` becomes a waveform. `rp_handler.py` tries several decode paths and logs the
  exact output type on first run — confirm the right one then pin it.
- **TADA-1B reference voice**: TADA is zero-shot and REQUIRES a reference audio + its
  transcript. The Dockerfile bakes a default sample; override per-request via `voice_url` +
  `voice_text`.
- **Chatterbox Turbo model id**: confirm the exact Turbo checkpoint name in the
  `chatterbox` package (flagged in `rp_handler.py`). Chatterbox caps ~40s/call, so the
  handler chunks long text internally.
