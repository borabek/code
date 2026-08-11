"""Watches for v8 training to finish, then runs sweep_decode + tta_eval automatically.

Start in a new terminal and leave:
    python watch_and_eval.py
"""
import json
import os
import subprocess
import sys
import time

HISTORY  = "checkpoints/cp_knn_v8_history.json"
CKPT     = "checkpoints/cp_knn_v8_best.ckpt"
CORPUS   = r"C:\Users\DE00024082\Desktop\JSON"

POLL_SEC      = 60   # check every minute
STABLE_NEEDED = 3    # 3 unchanged polls in a row = training stopped


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def _last_epoch():
    try:
        with open(HISTORY, encoding="utf-8") as f:
            rows = json.load(f)
        return rows[-1]["epoch"] if rows else "?"
    except Exception:
        return "?"


print("Watching for cp_knn_v8 training to finish...", flush=True)
print(f"  polling {HISTORY} every {POLL_SEC}s", flush=True)

prev_mtime  = 0
stable      = 0

while True:
    mtime = _mtime(HISTORY)
    if mtime == 0:
        print("  waiting for checkpoint to appear...", flush=True)
        stable = 0
    elif mtime != prev_mtime:
        stable = 0
        print(f"  training active  (last epoch={_last_epoch()})", flush=True)
    else:
        stable += 1
        print(f"  history unchanged {stable}/{STABLE_NEEDED}  (last epoch={_last_epoch()})",
              flush=True)
    prev_mtime = mtime

    if stable >= STABLE_NEEDED:
        break
    time.sleep(POLL_SEC)

print(f"\nTraining done. Best checkpoint: {CKPT}", flush=True)

# --- sweep_decode -------------------------------------------------------
print("\n=== sweep_decode ===", flush=True)
subprocess.run([sys.executable, "sweep_decode.py", CKPT, CORPUS,
                "--split-group", "prefix", "--val-frac", "0.2", "--seed", "0",
                "--holdout-frac", "0.3", "--min-precision", "0.5",
                "--device", "cpu", "--max-gpu-verts", "7000",
                "--cache-arrs", "sweep_v8_arrs.npz", "--top", "15"],
               check=True)

# --- tta_eval -----------------------------------------------------------
print("\n=== tta_eval ===", flush=True)
subprocess.run([sys.executable, "tta_eval.py", CKPT, CORPUS,
                "--k", "8", "--device", "cpu", "--max-gpu-verts", "7000",
                "--split-group", "prefix", "--test-frac", "0.15"],
               check=True)

print("\nAll done.", flush=True)
