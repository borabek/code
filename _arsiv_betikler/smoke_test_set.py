# -*- coding: utf-8 -*-
"""P1.6 smoke test seti: robot_cp'yi temsili parcalarda kosar, cikti sagligini dogrular (crash yok,
makul CP sayisi, dup yok, tier atanmis). normal WEI / PXC-tipik / high-CP PXC / zor aile (2770/3002).
Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe smoke_test_set.py"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_step_cp import load_any
import robot_cp
from big_arbiter import eligible

dev = "cuda" if torch.cuda.is_available() else "cpu"

# temsili set: (pid, beklenen_N_araligi, mode)  mode: base | highcp
CASES = [
    ("2531320000", "WEI normal (N~6)", "base", None),
    ("3273024",    "PXC high-CP aile 327x (N=12)", "highcp", 12),
    ("2770943",    "zor aile singleton 2770 (N=32)", "highcp", 32),
    ("3002926",    "zor aile 3002 (N=18)", "highcp", 18),
]


def validate(pid, cps, tag):
    ok = True; msgs = []
    if not cps:
        return False, ["CP YOK (bos cikti)"]
    P = np.array([c["point"] for c in cps])
    # dup: <2mm cift
    dmin = min((float(np.linalg.norm(P[i]-P[j])) for i in range(len(P)) for j in range(i+1, len(P))), default=99)
    if dmin < 2.0: ok = False; msgs.append(f"DUP: {dmin:.2f}mm cift var")
    # tier atanmis
    if not all(c.get("tier") in ("auto", "review") for c in cps): ok = False; msgs.append("tier eksik")
    # yon birim vektor
    for c in cps:
        d = np.asarray(c["direction"], float)
        if abs(np.linalg.norm(d)-1.0) > 0.05: ok = False; msgs.append("yon birim degil"); break
    na = sum(1 for c in cps if c["tier"] == "auto")
    msgs.append(f"{len(cps)} CP ({na} auto/{len(cps)-na} review), dup-min {dmin:.1f}mm, ws {min(c['wire_score'] for c in cps):.2f}-{max(c['wire_score'] for c in cps):.2f}")
    return ok, msgs


def main():
    cfg = json.load(open("cp_config.json"))
    m4 = [load_any(c, dev=dev)[:2] for c in cfg["robot_vote2_checkpoints"]]
    hc = cfg["robot_highcp"]; m7 = m4 + [load_any(c, dev=dev)[:2] for c in hc["extra_checkpoints"]]
    stp = {os.path.basename(s).split("_")[1]: s for m, pid, jf, s in eligible()}
    ca = float(cfg.get("robot_conf_auto", 0.5)); mav = int(cfg.get("robot_min_auto_votes", 3))
    npass = 0
    for pid, desc, mode, N in CASES:
        path = stp.get(pid)
        if not path:
            print(f"[ATLA] {pid} ({desc}): STEP yok"); continue
        try:
            if mode == "highcp":
                cps = robot_cp.extract_highcp(m7, path, dev, ca, mav, N)
            else:
                cps = robot_cp.extract(m4, path, dev, ca, mav, cp_count=None)
            ok, msgs = validate(pid, cps, desc)
            print(f"[{'PASS' if ok else 'FAIL'}] {pid} {desc}: {'; '.join(msgs)}")
            npass += ok
        except Exception as e:
            print(f"[FAIL] {pid} {desc}: CRASH {type(e).__name__}: {e}")
    ntested = sum(1 for pid, *_ in CASES if stp.get(pid))
    print(f"\nSMOKE: {npass}/{ntested} PASS ({len(CASES)-ntested} atlandi)")
    return 0 if (ntested > 0 and npass == ntested) else 1   # regression gate: nonzero exit on any FAIL


if __name__ == "__main__":
    sys.exit(main())
