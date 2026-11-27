"""Start, drive and terminate a RunPod GPU pod from this machine.

The API key is read from the environment variable RUNPOD_API_KEY and from nowhere else.
The pod clones this repository's branch, installs the training requirements and starts
experiments/pod_server.py; all later traffic goes through https://<pod>-8000.proxy.runpod.net
with a random token that this script generates and keeps in the state file.

    python experiments/runpod_driver.py gpus                     # prices, no key needed
    python experiments/runpod_driver.py up [--gpu "NVIDIA RTX A6000" ...]
    python experiments/runpod_driver.py wait                     # until the job server answers
    python experiments/runpod_driver.py run NAME "shell command"
    python experiments/runpod_driver.py jobs | log NAME | ls [PATH] | get REMOTE LOCAL
    python experiments/runpod_driver.py pods                     # every pod on the account
    python experiments/runpod_driver.py down                     # terminate the pod (stops billing)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.runpod.io/graphql"
REPO = "https://github.com/lsshepard/scalable-oversight.git"
IMAGE = "runpod/pytorch:2.2.0-cu1290-torch291-ubuntu2404"
GPUS = ["NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
        "NVIDIA L40S", "NVIDIA GeForce RTX 5090"]
STATE = pathlib.Path(os.environ.get("RUNPOD_STATE", pathlib.Path.home() / ".runpod_state.json"))


def gql(query: str, auth: bool = True) -> dict:
    url = API
    if auth:
        key = os.environ.get("RUNPOD_API_KEY")
        if not key:
            sys.exit("RUNPOD_API_KEY is not set in the environment")
        url += "?api_key=" + urllib.parse.quote(key)
    req = urllib.request.Request(url, data=json.dumps({"query": query}).encode(),
                                 headers={"content-type": "application/json", "user-agent": "runpod-driver"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API error {e.code}: {e.read()[:300]!r}")
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"])[:600])
    return out["data"]


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def pod_call(path: str, body: dict | None = None, raw: bool = False, timeout: int = 120):
    st = state()
    if "pod" not in st:
        sys.exit("no pod in the state file; run `up` first")
    req = urllib.request.Request(f"https://{st['pod']}-8000.proxy.runpod.net{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"x-token": st["token"], "content-type": "application/json",
                                          "user-agent": "runpod-driver"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if raw else json.loads(data)


def cmd_gpus(a):
    for g in sorted(gql("query { gpuTypes { id memoryInGb securePrice communityPrice } }", auth=False)["gpuTypes"],
                    key=lambda g: (g["memoryInGb"] or 0, g["securePrice"] or 0)):
        if 24 <= (g["memoryInGb"] or 0) <= 96:
            print(f"{g['id']:44s} {g['memoryInGb']:4d} GB   secure ${g['securePrice']}/h   community ${g['communityPrice']}/h")


def cmd_up(a):
    if state().get("pod"):
        sys.exit(f"state file already holds pod {state()['pod']}; run `down` first")
    token = secrets.token_hex(24)
    branch = a.branch or subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    boot = (f"cd /workspace && rm -rf so && git clone --depth 1 -b {branch} {REPO} so && cd so && "
            f"pip install -q transformers peft accelerate scipy safetensors && python experiments/pod_server.py 8000")
    args = "bash -c '" + boot + "'"
    last = None
    for gpu in ([a.gpu] if a.gpu else GPUS):
        q = ('mutation { podFindAndDeployOnDemand(input: { cloudType: %s, gpuCount: 1, volumeInGb: %d, '
             'containerDiskInGb: %d, minVcpuCount: 4, minMemoryInGb: 24, gpuTypeId: %s, name: %s, imageName: %s, '
             'dockerArgs: %s, ports: "8000/http", volumeMountPath: "/workspace", '
             'env: [{ key: "POD_TOKEN", value: %s }, { key: "HF_HUB_ENABLE_HF_TRANSFER", value: "0" }] }) '
             '{ id imageName machineId costPerHr machine { podHostId gpuDisplayName } } }') % (
            a.cloud, a.volume, a.disk, json.dumps(gpu), json.dumps(a.name), json.dumps(a.image), json.dumps(args),
            json.dumps(token))
        try:
            pod = gql(q)["podFindAndDeployOnDemand"]
        except RuntimeError as e:
            last = e
            print(f"  {gpu}: not available ({str(e)[:140]})")
            continue
        if pod:
            STATE.write_text(json.dumps(dict(pod=pod["id"], token=token, gpu=gpu, cost_per_hr=pod.get("costPerHr"),
                                             started=time.time(), branch=branch)))
            STATE.chmod(0o600)
            print(f"pod {pod['id']} on {gpu}, ${pod.get('costPerHr')}/h. Next: wait")
            return
    sys.exit(f"no GPU could be rented: {last}")


def cmd_wait(a):
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        try:
            if pod_call("/health", timeout=15).get("ok"):
                print(f"job server is up after {time.time() - t0:.0f}s")
                return
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(10)
    sys.exit(f"job server did not answer within {a.timeout}s ({last!r})")


def cmd_run(a):
    print(pod_call("/run", dict(name=a.name, cmd=a.command)))


def cmd_jobs(a):
    for n, j in pod_call("/jobs").items():
        print(f"{n:24s} pid {j['pid']:6d} exit {j['exit']}   {j['cmd'][:90]}")


def cmd_log(a):
    sys.stdout.write(pod_call(f"/log?name={urllib.parse.quote(a.name)}&tail={a.tail}", raw=True).decode(errors="replace"))


def cmd_ls(a):
    for path, size in pod_call(f"/ls?path={urllib.parse.quote(a.path)}"):
        print(f"{size:12d}  {path}")


def cmd_get(a):
    data = pod_call(f"/file?path={urllib.parse.quote(a.remote)}", raw=True, timeout=600)
    out = pathlib.Path(a.local)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"{len(data)} bytes -> {out}")


def cmd_pods(a):
    for p in gql("query { myself { clientBalance pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }")["myself"]["pods"]:
        up = (p.get("runtime") or {}).get("uptimeInSeconds")
        print(f"{p['id']}  {p['name']:20s} {p['desiredStatus']:10s} ${p['costPerHr']}/h  up {up}s")


def cmd_balance(a):
    print(gql("query { myself { clientBalance currentSpendPerHr } }")["myself"])


def cmd_down(a):
    st = state()
    pod = a.pod or st.get("pod")
    if not pod:
        sys.exit("no pod to terminate")
    gql('mutation { podTerminate(input: { podId: %s }) }' % json.dumps(pod))
    if st.get("pod") == pod:
        hours = (time.time() - st["started"]) / 3600
        print(f"terminated {pod} after {hours:.2f} h, about ${hours * (st.get('cost_per_hr') or 0):.2f}")
        STATE.unlink()
    else:
        print(f"terminated {pod}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus")
    u = sub.add_parser("up")
    u.add_argument("--gpu", default=""); u.add_argument("--cloud", default="ALL", choices=["ALL", "SECURE", "COMMUNITY"])
    u.add_argument("--name", default="hpo-trained"); u.add_argument("--image", default=IMAGE)
    u.add_argument("--branch", default=""); u.add_argument("--volume", type=int, default=40)
    u.add_argument("--disk", type=int, default=40)
    w = sub.add_parser("wait"); w.add_argument("--timeout", type=int, default=1200)
    r = sub.add_parser("run"); r.add_argument("name"); r.add_argument("command")
    sub.add_parser("jobs")
    lg = sub.add_parser("log"); lg.add_argument("name"); lg.add_argument("--tail", type=int, default=6000)
    ls = sub.add_parser("ls"); ls.add_argument("path", nargs="?", default="results")
    g = sub.add_parser("get"); g.add_argument("remote"); g.add_argument("local")
    sub.add_parser("pods"); sub.add_parser("balance")
    d = sub.add_parser("down"); d.add_argument("--pod", default="")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
