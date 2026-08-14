# -*- coding: utf-8 -*-
"""R8: UYE-YON verisi (tam korpus) -- ogrenilmis uye secici icin.

R7 olcumu: dort modelin ATILAN yonlerinde robot-haziri 0.5523 -> 0.6059 yapacak bilgi VAR
(+0.0536). Konum tarafinda YOK (-0.0022; pose head zaten almis). Yani kol ACI'da canli.

Bu betik her BIRLESTIRILMIS CP icin, ona katkida bulunan UYE adaylarini ve her uyenin
ozelliklerini yazar. Hedef: "bu uyenin yonu GT'ye 10 derece icinde mi".

UYE OZELLIKLERI (calisma aninda hesaplanabilir, GT'ye BAKMAZ):
    uye_conf      uyenin kendi guveni
    uye_mesafe    birlestirilmis noktaya uzakligi (mm)
    uye_aci_ort   uyenin yonunun DIGER uyelerin ortalamasindan sapmasi
    uye_aci_bir   birlestirilmis (temsilci) yona sapmasi
    n_uye         bu CP'ye katkida bulunan uye sayisi
    yayilim       uye yonlerinin birbirine gore ortalama sapmasi
    uye_sira      guvene gore sirasi
    + CP'nin 58 gate sutunu
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
os.environ.setdefault("WG_ZENGIN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cp_openings                       # noqa: E402
import diffusionnet as D                 # noqa: E402
import robot_cp as RC                    # noqa: E402
import thesis_remesh                     # noqa: E402
import wire_gate                         # noqa: E402
from big_arbiter import eligible         # noqa: E402
from cad_eval import align_frames        # noqa: E402
from connector_constants import CABLE_ENTRY as CE, CONTACT as CT   # noqa: E402
from infer_step_cp import load_any, step_to_mesh                   # noqa: E402

OUT = "results/uye_veri.npz"
YAKIN = 5.0


def main():
    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"])
    CL = float(pp["cluster_mm"])
    cks = cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cks]
    with io.open("results/_strict_geometry_keys.json", encoding="utf-8") as f:
        gk = json.load(f)
    parts = list(eligible())
    print(f"{len(models)} model | {len(parts)} parca", flush=True)

    X, Y, PID, GEO, CPI = [], [], [], [], []
    t0 = time.time(); atlanan = 0
    for k, (mfg, pid, jf, stp) in enumerate(parts, 1):
        if k % 50 == 0:
            print(f"  {k}/{len(parts)}  {time.time()-t0:.0f}s (atlanan {atlanan})", flush=True)
        try:
            with io.open(jf, encoding="utf-8-sig") as f:
                j = json.load(f)
            cps_gt = j.get("ConnectionPoints") or []
            if not cps_gt:
                atlanan += 1; continue
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in cps_gt], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in cps_gt], float)
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
            per = [cp_openings.connection_points(
                V, F, q.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
                probs=q, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
                step_path=stp) for q in pbs]
            cps, probs, _, _u = RC.adaylari_uret(V, F, pbs, stp, cfg=cfg)
            if not cps:
                atlanan += 1; continue
            X58 = wire_gate.feats_for(V, F, probs, cps, CE, CT, step_path=stp)
            for ci, c in enumerate(cps):
                p0 = np.asarray(c["point"], float); d0 = np.asarray(c["direction"], float)
                uy = [(d0, float(c.get("confidence", 1.0)), 0.0)]
                for lst in per:
                    for m in lst:
                        q = np.asarray(m["point"], float)
                        dd = float(np.linalg.norm(q - p0))
                        if dd <= YAKIN:
                            uy.append((np.asarray(m["direction"], float),
                                       float(m.get("confidence", 1.0)), dd))
                if len(uy) < 2:
                    continue
                DIR = np.array([u[0] for u in uy]); DIR /= (np.linalg.norm(DIR, axis=1, keepdims=True) + 1e-9)
                CONF = np.array([u[1] for u in uy]); MES = np.array([u[2] for u in uy])
                ort = DIR.mean(0); ort /= np.linalg.norm(ort) + 1e-9
                a_ort = np.degrees(np.arccos(np.clip(np.abs(DIR @ ort), 0, 1)))
                a_bir = np.degrees(np.arccos(np.clip(np.abs(DIR @ d0), 0, 1)))
                yay = float(np.mean(a_ort))
                sira = np.argsort(np.argsort(-CONF))
                b = int(np.argmin(np.linalg.norm(Gm - p0, axis=1)))
                a_gt = np.degrees(np.arccos(np.clip(np.abs(DIR @ Gdm[b]), 0, 1)))
                for u_ in range(len(DIR)):
                    X.append([CONF[u_], MES[u_], a_ort[u_], a_bir[u_], float(len(DIR)),
                              yay, float(sira[u_])] + X58[ci].tolist())
                    Y.append(1 if a_gt[u_] <= 10.0 else 0)
                    PID.append(pid); GEO.append(gk.get(pid, "yok:" + pid)); CPI.append(f"{pid}#{ci}")
        except Exception as e:
            atlanan += 1
            if atlanan <= 5:
                print(f"    atlandi {pid}: {type(e).__name__}: {e}", flush=True)
        if k % 200 == 0 and X:
            np.savez(OUT, X=np.array(X, float), y=np.array(Y), pid=np.array(PID),
                     geo=np.array(GEO), cp=np.array(CPI))
    np.savez(OUT, X=np.array(X, float), y=np.array(Y), pid=np.array(PID),
             geo=np.array(GEO), cp=np.array(CPI))
    Y = np.array(Y)
    print(f"\n-> {OUT} | {len(Y)} uye satiri / {len(set(CPI))} CP / {len(set(PID))} parca")
    print(f"   'yon dogru' orani {Y.mean():.1%} | atlanan {atlanan}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
