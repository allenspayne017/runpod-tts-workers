# RunPod Deploy — Step by Step

Goal: get **TADA-1B** and **Chatterbox Turbo** running as RunPod serverless endpoints (on
the same RTX 3090 as IndexTTS2) so we can benchmark real cost. Repo is **private**, so we
use RunPod's GitHub build (no local Docker needed).

You only do the build + endpoint creation. Then send me the two endpoint IDs and I run the
benchmark.

---

## A. One-time: connect GitHub to RunPod
1. Go to **console.runpod.io** → log in.
2. Left sidebar → **Settings** → **Connections** (or **Integrations**).
3. Click **Connect** next to **GitHub** → authorize the **RunPod** GitHub app.
4. When GitHub asks which repos: choose **Only select repositories** →
   pick **`allenspayne017/runpod-tts-workers`** → **Install/Authorize**.

(That grants RunPod read access to just this one private repo.)

---

## B. Create the TADA-1B endpoint
1. Left sidebar → **Serverless** → **New Endpoint**.
2. Source: choose **GitHub Repo** (a.k.a. "Import Git Repository").
3. Repository: **`allenspayne017/runpod-tts-workers`** · Branch: **`main`**.
4. Build settings:
   - **Dockerfile Path:** `tada-1b/Dockerfile`
   - **Build Context:** `tada-1b`  ← important (so `COPY requirements.txt` resolves).
     If you don't see a Build Context field, set Dockerfile Path to `tada-1b/Dockerfile`
     and leave context as repo root — then tell me and I'll switch the COPY lines.
5. GPU: pick **24 GB → "RTX 3090"** (same as IndexTTS2; if the volume's datacenter doesn't
   offer 3090, pick any 24 GB option it shows).
6. **Container Disk:** `20 GB`.
7. **Network Volume:** attach **`ae8uking8o`** (this is the shared volume that holds the
   `voices/jenny.wav` reference — required so `voice_name: jenny` works).
8. Workers: **Max Workers `1`**, **Idle Timeout `5`**, **FlashBoot ON**,
   **Execution Timeout `600`** seconds.
9. Environment variables: **none required** for the benchmark (the worker falls back to
   returning base64 audio). *Optional for production:* add
   `GCS_SERVICE_ACCOUNT_JSON` = (paste a service-account JSON with write access to the
   `media.apparentgroup.co` bucket) to make it upload + return a URL instead.
10. Click **Create / Deploy**. The first build takes ~10–20 min (it bakes the model
    weights into the image). Wait until status shows the endpoint is ready.
11. Copy the **Endpoint ID** (looks like `abcd1234xyz`).

---

## C. Create the Chatterbox Turbo endpoint
Repeat section B with only these changes:
- **Dockerfile Path:** `chatterbox-turbo/Dockerfile`
- **Build Context:** `chatterbox-turbo`
- **Name:** `chatterbox-turbo`
Everything else (RTX 3090, 20 GB disk, volume `ae8uking8o`, workers/timeout) is the same.
Copy this **Endpoint ID** too.

---

## D. Hand back to me
Send me the two endpoint IDs (or just say "done" — I can read them from your RunPod account
with the API key). I will:
1. Drop them into `benchmark.py`.
2. Run the same paragraph through TADA-1B, Chatterbox Turbo, and IndexTTS2 on the 3090.
3. Report real RTF + GPU-time cost, and how much cheaper each is vs IndexTTS2.

---

## Notes / gotchas
- **TADA-1B voice transcript:** TADA is zero-shot and needs the transcript of the reference
  voice. For best clone quality add a `jenny.txt` (its transcript) next to `jenny.wav` on the
  volume, or pass `voice_text` per request. Timing/cost is unaffected — fine to skip for the
  benchmark.
- **Chatterbox checkpoint:** if the build logs complain about the Turbo model id, tell me and
  I'll adjust the loader line in `chatterbox-turbo/handler.py`.
- **TADA output decode:** the handler logs the model's output type on first real run; if it
  errors with "cannot decode", paste me that log line and I'll pin the exact decode call.
- **Cost during this:** building is free-ish; each test generation is a few cents. We can
  delete both endpoints after benchmarking if you don't want them kept warm.

## Alternative: local Docker (only if you prefer)
```bash
docker login
cd tada-1b        && docker build -t allenspayne017/tada-1b-runpod:latest . && docker push allenspayne017/tada-1b-runpod:latest
cd ../chatterbox-turbo && docker build -t allenspayne017/chatterbox-turbo-runpod:latest . && docker push allenspayne017/chatterbox-turbo-runpod:latest
```
Then in RunPod → New Endpoint → **Docker Image** (instead of GitHub), paste the image name,
and use the same GPU / disk / volume settings from section B.
