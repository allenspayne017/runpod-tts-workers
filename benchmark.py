"""Benchmark TTS endpoints on the SAME GPU and print RTF + GPU-time cost vs IndexTTS2.

For each model: submit the same paragraph, poll, read RunPod executionTime (GPU-seconds)
and the output audio duration, then compute RTF (= exec / audio) and cost.

    export RUNPOD_API_KEY=rpa_...
    # fill the endpoint ids below (from deploy_endpoint.py output), then:
    python benchmark.py

Cost basis: RTX 3090 serverless flex = $0.00019/GPU-sec ($0.684/hr). Community pods are
cheaper (~$0.22/hr); pass --rate to override.
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.request
import wave

KEY = os.environ.get("RUNPOD_API_KEY", "")
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

# ~250-word meditative paragraph (representative of one deep-session section).
TEXT = (
    "Take a slow, deep breath in through your nose, and let it go gently through your mouth. "
    "Feel your shoulders soften as the tension of the day begins to melt away. There is nowhere "
    "you need to be right now, nothing you need to do, except to be here in this quiet moment with "
    "yourself. Notice the gentle rhythm of your breathing, the rise and fall of your chest. With "
    "each breath you are becoming more present, more grounded, more at peace. Imagine a warm golden "
    "light glowing at the center of your chest, growing brighter with every inhale, spreading slowly "
    "through your entire body, down your arms to your fingertips, through your stomach and hips and "
    "legs, into the soles of your feet. Rest here, held in stillness, knowing you are safe and whole, "
    "and exactly where you need to be."
)

# Same voice ("jenny") across all three for a fair comparison — it lives on the shared
# /runpod-volume/voices network volume. TADA also needs the voice transcript (voice_text);
# set VOICE_TEXT to jenny.wav's actual transcript for best quality (timing is unaffected).
VOICE = "jenny"
VOICE_TEXT = os.environ.get("VOICE_TEXT", "")  # transcript of jenny.wav (TADA only)

# Fill in endpoint ids after deploying. `inp` builds the RunPod input for that model.
CONFIG = {
    "IndexTTS2":  dict(endpoint="7axelvd332ctge",
                       inp=lambda t: {"input": {"text": t, "voice_name": VOICE}}),
    "ChatterboxTurbo": dict(endpoint="REPLACE_ME",
                            inp=lambda t: {"input": {"text": t, "voice_name": VOICE}}),
    "TADA-1B":    dict(endpoint="REPLACE_ME",
                       inp=lambda t: {"input": {"text": t, "voice_name": VOICE, "voice_text": VOICE_TEXT}}),
    # Reference only: RunPod-hosted PUBLIC Chatterbox (billed per audio-sec, not GPU-time):
    "ChatterboxTurbo(public)": dict(endpoint="chatterbox-turbo",
                                    inp=lambda t: {"input": {"prompt": t, "voice": "lucy", "format": "wav"}}),
}


def _req(url, body=None):
    r = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                               headers=HDR, method="POST" if body else "GET")
    return json.load(urllib.request.urlopen(r, timeout=180))


def _audio_seconds(out):
    if isinstance(out, dict):
        if out.get("audio_seconds"):
            return float(out["audio_seconds"])
        b64 = out.get("audio_base64")
        url = out.get("audio_url") or out.get("url") or out.get("audio")
        data = base64.b64decode(b64) if b64 else (urllib.request.urlopen(url, timeout=60).read() if url else None)
        if data:
            w = wave.open(io.BytesIO(data))
            return w.getnframes() / float(w.getframerate())
    return None


def run(name, cfg, rate_per_sec):
    ep = cfg["endpoint"]
    if ep in ("REPLACE_ME", ""):
        print(f"{name:26s} — endpoint not set, skipping"); return
    sub = _req(f"https://api.runpod.ai/v2/{ep}/run", cfg["inp"](TEXT))
    jid = sub.get("id")
    st = sub
    for _ in range(180):
        if st.get("status") in ("COMPLETED", "FAILED"):
            break
        time.sleep(2)
        st = _req(f"https://api.runpod.ai/v2/{ep}/status/{jid}")
    if st.get("status") != "COMPLETED":
        print(f"{name:26s} — {st.get('status')} {str(st.get('error'))[:80]}"); return
    exec_s = (st.get("executionTime") or 0) / 1000.0
    delay_s = (st.get("delayTime") or 0) / 1000.0
    audio_s = _audio_seconds(st.get("output"))
    if not audio_s:
        print(f"{name:26s} exec={exec_s:.1f}s (no audio duration)"); return
    rtf = exec_s / audio_s
    cost = exec_s * rate_per_sec
    print(f"{name:26s} audio={audio_s:6.1f}s  exec={exec_s:6.1f}s  cold={delay_s:5.1f}s  "
          f"RTF={rtf:5.3f}  $/audio-hr={rtf*rate_per_sec*3600:6.3f}  thisrun=${cost:.4f}")
    return rtf


def main():
    if not KEY:
        raise SystemExit("set RUNPOD_API_KEY")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.00019, help="$/GPU-sec (3090 flex=0.00019; community 0.22/hr=0.0000611)")
    a = ap.parse_args()
    print(f"input: {len(TEXT)} chars / {len(TEXT.split())} words | rate ${a.rate}/GPU-sec\n")
    results = {}
    for name, cfg in CONFIG.items():
        try:
            r = run(name, cfg, a.rate)
            if r:
                results[name] = r
        except Exception as e:
            print(f"{name:26s} ERROR {str(e)[:80]}")
    base = results.get("IndexTTS2")
    if base:
        print("\nvs IndexTTS2:")
        for k, v in results.items():
            if k != "IndexTTS2":
                print(f"  {k:26s} {base/v:.2f}x cheaper (RTF {v:.3f} vs {base:.3f})")


if __name__ == "__main__":
    main()
