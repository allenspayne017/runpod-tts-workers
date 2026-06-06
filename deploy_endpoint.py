"""Create a RunPod serverless template + endpoint that mirrors the IndexTTS2 config.

Usage:
    export RUNPOD_API_KEY=rpa_...
    python deploy_endpoint.py --name tada-1b \
        --image YOURUSER/tada-1b-runpod:latest \
        --gpu "NVIDIA GeForce RTX 3090"

Mirrors endpoint `indextts-api-service` (id 7axelvd332ctge): 20GB disk, flashboot on,
idle 5s, exec timeout 600s, workersMax 1, scaler REQUEST_COUNT.
REST API: https://rest.runpod.io/v1  (verify field names against current RunPod docs).
"""
import argparse
import json
import os
import sys
import urllib.request

API = "https://rest.runpod.io/v1"
KEY = os.environ.get("RUNPOD_API_KEY", "")


def _post(path, body):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()[:500]}")
        sys.exit(1)


def main():
    if not KEY:
        sys.exit("set RUNPOD_API_KEY")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 3090")
    ap.add_argument("--disk", type=int, default=20)
    ap.add_argument("--exec-timeout-ms", type=int, default=600000)
    ap.add_argument("--workers-max", type=int, default=1)
    a = ap.parse_args()

    print(f"creating template '{a.name}-tpl' -> {a.image}")
    tpl = _post("/templates", {
        "name": f"{a.name}-tpl",
        "imageName": a.image,
        "containerDiskInGb": a.disk,
        "isServerless": True,
        "env": {"SAMPLE_RATE": "24000"},
    })
    tpl_id = tpl.get("id")
    print("  templateId:", tpl_id)

    print(f"creating endpoint '{a.name}' on {a.gpu}")
    ep = _post("/endpoints", {
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
    })
    ep_id = ep.get("id")
    print("  endpointId:", ep_id)
    print(f"\n  run URL:    https://api.runpod.ai/v2/{ep_id}/run")
    print(f"  status URL: https://api.runpod.ai/v2/{ep_id}/status/<jobId>")
    print(f"\nAdd to benchmark.py CONFIG: '{a.name}': dict(endpoint='{ep_id}', ...)")


if __name__ == "__main__":
    main()
