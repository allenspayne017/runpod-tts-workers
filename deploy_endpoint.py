"""Create a RunPod serverless template + endpoint that mirrors the IndexTTS2
(audio-generator-allen) config: RTX 3090, 20GB disk, flashboot, shared voices network
volume, and GCS upload env.

Usage:
    export RUNPOD_API_KEY=rpa_...
    python deploy_endpoint.py --name tada-1b \
        --image YOURUSER/tada-1b-runpod:latest \
        --network-volume ae8uking8o \
        --gcs-key-file /path/to/media-bucket-sa.json   # optional; omit -> base64 output

REST API: https://rest.runpod.io/v1  (verify field names against current RunPod docs).
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
KEY = os.environ.get("RUNPOD_API_KEY", "")


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {path}: {e.read().decode()[:600]}")


def main() -> None:
    if not KEY:
        sys.exit("set RUNPOD_API_KEY")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 3090")
    ap.add_argument("--disk", type=int, default=20)
    ap.add_argument("--exec-timeout-ms", type=int, default=600000)
    ap.add_argument("--workers-max", type=int, default=1)
    ap.add_argument("--network-volume", default="ae8uking8o",
                    help="shared volume holding /runpod-volume/voices (default: audio-generator's)")
    ap.add_argument("--gcs-key-file", default="", help="path to GCS service-account JSON (optional)")
    a = ap.parse_args()

    env = {
        "SAMPLE_RATE": "24000",
        "VOICES_DIR": "/runpod-volume/voices",
        "GCS_BUCKET_NAME": "media.apparentgroup.co",
    }
    if a.gcs_key_file:
        with open(a.gcs_key_file) as f:
            env["GCS_SERVICE_ACCOUNT_JSON"] = f.read()

    print(f"creating template '{a.name}-tpl' -> {a.image}")
    tpl = _post("/templates", {
        "name": f"{a.name}-tpl",
        "imageName": a.image,
        "containerDiskInGb": a.disk,
        "isServerless": True,
        "env": env,
    })
    tpl_id = tpl.get("id")
    print("  templateId:", tpl_id)

    ep_body = {
        "name": a.name,
        "templateId": tpl_id,
        "gpuTypeIds": [a.gpu],
        "computeType": "GPU",
        "workersMin": 0,
        "workersMax": a.workers_max,
        "idleTimeout": 5,
        "flashboot": True,
        "executionTimeoutMs": a.exec_timeout_ms,
        "scalerType": "REQUEST_COUNT",
        "scalerValue": 1,
    }
    if a.network_volume:
        ep_body["networkVolumeId"] = a.network_volume  # endpoint pins to the volume's datacenter

    print(f"creating endpoint '{a.name}' on {a.gpu} (volume {a.network_volume or 'none'})")
    ep = _post("/endpoints", ep_body)
    ep_id = ep.get("id")
    print("  endpointId:", ep_id)
    print(f"\n  run URL:    https://api.runpod.ai/v2/{ep_id}/run")
    print(f"  add to benchmark.py CONFIG: '{a.name}': dict(endpoint='{ep_id}', ...)")


if __name__ == "__main__":
    main()
