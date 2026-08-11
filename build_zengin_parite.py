# -*- coding: utf-8 -*-
"""ZENGIN GATE BLOKLARI -- parite-sadik uretim (denetimin P1 maddesi).

DENETIM TAVSIYESI: "konum-9 ve cok-yaricap-24 bloklarini guncel 22 ozellige yeniden uret.
Taper'i ekleme, olculmus olu." Kanit: results/rich_gate_receipt.json --
PART_out gain_ALL +0.0351, FAMILY_out_STRICT gain_ALL +0.0489,
AUC 0.9275 -> 0.9549 (part-out), 0.8956 -> 0.9373 (family-out).

O olcum 13 sutunlu gate zamanindan; gate o gunden beri 22 sutuna cikti (B-rep fiziksel +
icbukey topoloji). Zengin bloklarin KATTIGI SEY hala var mi, YENIDEN olculmeli.

BU BETIK NE YAPAR: parite korpusunun AYNI adaylarini uretir (robot_cp.adaylari_uret) ve her
aday icin HEM 22 temel sutunu HEM 33 zengin sutunu yazar. Boylece A/B'de TEK DEGISKEN ozellik
kumesi olur -- ayri kosularda uretilmis iki npz'yi karsilastirmak bu projede daha once
yaniltmisti.

ZENGIN BLOKLAR (33):
    A1 konum          (3)  mesh-bbox'a normalize edilmis konum
    A2 yuz uzakligi   (6)  bbox yuzlerine kosegen-normalize uzakliklar
    B  cok-yaricap   (24)  r=3/6/10/15mm kurelerde sinif-olasilik profili (5) + yogunluk (1)
    D  normal-degisim (3)  r=3/6/10mm'de normallerin eksene izdusumunun std'si
TAPER (5) BILEREK YOK -- olculmus olu.

Cikti: results/zengin_parite.npz (X22, XR, y, pids, mfg, votes, ...)
"""
import io
import json
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ.setdefault("WG_FIZ_FEATS", "1")
os.environ.setdefault("WG_TOPO", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cp_openings                      # noqa: E402
import diffusionnet as D                # noqa: E402
import robot_cp as RC                   # noqa: E402
import thesis_remesh                    # noqa: E402
import wire_gate                        # noqa: E402
from big_arbiter import eligible        # noqa: E402
from cad_eval import align_frames       # noqa: E402
from connector_constants import CABLE_ENTRY as CE, CONTACT as CT   # noqa: E402
from infer_step_cp import load_any, step_to_mesh  # noqa: E402

RAD = (3.0, 6.0, 10.0, 15.0)
OUT = "results/zengin_parite.npz"
ZENGIN_AD = (["kon_x", "kon_y", "kon_z"]
             + [f"yuz_{a}{b}" for b in "xyz" for a in ("lo", "hi")]
             + [f"r{int(r)}_{k}" for r in RAD for k in ("c0", "c1", "c2", "c3", "c4", "yog")]
             + [f"nstd_{int(r)}" for r in (3.0, 6.0, 10.0)])


def zengin(V, F, probs, cps, Nrm):
    lo, hi = V.min(0), V.max(0)
    diag = float(np.linalg.norm(hi - lo)) + 1e-9
    out = []
    for c in cps:
        p = np.asarray(c["point"], float)
        d = np.asarray(c["direction"], float)
        d = d / (np.linalg.norm(d) + 1e-9)
        f = list((p - lo) / np.maximum(hi - lo, 1e-6))          # A1 konum (3)
        f += list((p - lo) / diag) + list((hi - p) / diag)      # A2 yuz uzakligi (6)
        rel = V - p
        dist = np.linalg.norm(rel, axis=1)
        for r in RAD:                                           # B cok-yaricap (24)
            m = dist <= r
            if m.any():
                f += list(probs[m].mean(0)); f.append(float(m.sum()) / len(V))
            else:
                f += [0.0] * probs.shape[1]; f.append(0.0)
        for r in (3.0, 6.0, 10.0):                              # D normal-degisim (3)
            m = dist <= r
            f.append(float(np.std(Nrm[m] @ d)) if m.sum() >= 4 else 0.0)
        out.append(f)
    return np.array(out, float)


def _normaller(V, F):
    """Vertex normalleri (yuz normallerinin alan-agirlikli toplami)."""
    V = np.asarray(V, float); F = np.asarray(F, int)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(n, 1e-12)


def main():
    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cks = cfg["current_product"]["checkpoints"] if "checkpoints" in cfg.get("current_product", {}) \
        else cfg["robot_vote2_checkpoints"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cks]
    print(f"{len(models)} model | cihaz {dev}", flush=True)

    parts = list(eligible())
    print(f"{len(parts)} parca", flush=True)
    X22, XR, YY, PID, MFG, VOT, NGT = [], [], [], [], [], [], {}
    t0 = time.time(); atlanan = 0
    for k, (mfg, pid, jf, stp) in enumerate(parts, 1):
        if k % 50 == 0:
            print(f"  {k}/{len(parts)}  {time.time()-t0:.0f}s  (atlanan {atlanan})", flush=True)
        try:
            with io.open(jf, encoding="utf-8-sig") as f:
                j = json.load(f)
            G = np.array([[c["Point"][q] for q in "XYZ"]
                          for c in (j.get("ConnectionPoints") or [])], float)
            if not len(G):
                atlanan += 1; continue
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"]
                           for c in j["ConnectionPoints"]], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            R, t, _ = align_frames(Vr, Vj)
            Gm = (G - t) @ R; Gdm = Gd @ R

            pbs = []
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=dev,
                                  op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig',64))}",
                                  return_probs=True)
                pbs.append(np.asarray(pb, float))
            # PARITE: urunun kendi aday ureticisi
            cps, probs, _, _u = RC.adaylari_uret(V, F, pbs, stp, cfg=cfg)
            if not cps:
                atlanan += 1; continue
            xb = wire_gate.feats_for(V, F, probs, cps, CE, CT, step_path=stp)
            xr = zengin(V, F, probs, cps, _normaller(V, F))

            P = np.array([c["point"] for c in cps], float)
            Pd = np.array([c["direction"] for c in cps], float)
            tol = max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0))))
            diff = P[:, None, :] - Gm[None, :, :]
            al = (diff * Gdm[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gdm[None, :, :], axis=-1)
            pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
            yy = np.zeros(len(P), int); up, ug = set(), set()
            for dd, a_, b_ in sorted((pe[a, b], a, b)
                                     for a in range(len(P)) for b in range(len(Gm))):
                if dd > tol or a_ in up or b_ in ug:
                    continue
                up.add(a_); ug.add(b_); yy[a_] = 1
            X22.append(xb); XR.append(xr); YY.append(yy)
            PID += [pid] * len(cps); MFG += [mfg] * len(cps)
            VOT += [int(c.get("_votes", 1)) for c in cps]
            NGT[pid] = len(Gm)
        except Exception as e:
            atlanan += 1
            if atlanan <= 5:
                print(f"    atlandi {pid}: {type(e).__name__}: {e}", flush=True)
        if k % 200 == 0 and X22:
            np.savez(OUT, X22=np.vstack(X22), XR=np.vstack(XR), y=np.concatenate(YY),
                     pids=np.array(PID), mfg=np.array(MFG), votes=np.array(VOT),
                     zengin_ad=np.array(ZENGIN_AD))
    np.savez(OUT, X22=np.vstack(X22), XR=np.vstack(XR), y=np.concatenate(YY),
             pids=np.array(PID), mfg=np.array(MFG), votes=np.array(VOT),
             zengin_ad=np.array(ZENGIN_AD))
    n = sum(len(a) for a in X22)
    print(f"\n-> {OUT}  ({n} aday, {len(set(PID))} parca, atlanan {atlanan})")
    print(f"   X22 {np.vstack(X22).shape} | XR {np.vstack(XR).shape} | "
          f"votes maks {max(VOT)} (urun tavani {len(models)})")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
