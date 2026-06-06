# RunPod Deploy — Step by Step

Goal: get **TADA-1B** and **Chatterbox Turbo** running as RunPod serverless endpoints so we
can benchmark real cost vs IndexTTS2.

> ⚠️ RunPod's **direct GitHub build is broken** for this account — its builder is missing
> `git`, so it fails with `exec: "git": executable file not found` / "handler.py not found".
> We bypass it: **GitHub Actions builds the images** → pushes to GHCR → **RunPod pulls the
> finished image**. No local Docker needed.

---

## A. Let GitHub Actions build the images (automatic)
A workflow (`.github/workflows/build.yml`) builds both images on every push to `main`.
1. Go to the repo on GitHub → **Actions** tab → watch **build-runpod-images** run
   (or click **Run workflow** to start it manually).
2. Wait for both matrix jobs (`tada-1b`, `chatterbox-turbo`) to go green (~10–20 min each).
3. The images are now at:
   - `ghcr.io/allenspayne017/tada-1b-runpod:latest`
   - `ghcr.io/allenspayne017/chatterbox-turbo-runpod:latest`

## B. Make the two GHCR packages public (one-time, so RunPod can pull)
1. GitHub → your profile → **Packages** → click **tada-1b-runpod**.
2. **Package settings** → **Danger Zone** → **Change visibility** → **Public**.
3. Repeat for **chatterbox-turbo-runpod**.

(The images contain only the worker code + public deps — no secrets, no model weights — so
public is safe. Your source repo stays private. Alternatively keep them private and add a
GitHub PAT as registry credentials in RunPod.)

## C. Deploy each on RunPod from the image
1. **Serverless → New Endpoint → Custom deployment → "Deploy from Docker registry or a template"**.
2. **Container Image:** `ghcr.io/allenspayne017/tada-1b-runpod:latest`
3. GPU: **24 GB** (RTX 3090 / A5000 / L4) · **Container Disk: 20 GB**
4. Workers: Max **1** · Idle **5s** · FlashBoot **ON** · Execution Timeout **600s**
5. Env vars: none needed for the benchmark (worker returns base64).
6. **Create.** First request will be slow (it downloads the model weights once), then warm.
7. Repeat with image `ghcr.io/allenspayne017/chatterbox-turbo-runpod:latest`.

## D. Hand back to me
Send the two endpoint IDs (or say "done" — I'll read them from your account) and I run the
benchmark across TADA-1B + Chatterbox Turbo + IndexTTS2.

---

## Notes
- **First request per endpoint is slow** (downloads weights on cold start). Send one warm-up
  request, then the benchmark numbers (warm `executionTime`) are accurate.
- **TADA-1B voice transcript:** zero-shot, needs `voice_text` (the reference voice's
  transcript) for good cloning. Timing/cost is unaffected.
- **Production multi-voice:** attach the shared volume `ae8uking8o` (all app voices) or bake a
  `voices/` folder — see README. The benchmark passes a reference voice in the request, so it
  doesn't need the volume.
- If a GitHub Actions build fails, open the failed job log and paste me the error line.
