# -*- coding: utf-8 -*-
"""W1: YENI WEI VERISI -- korpusta hic olmayan parcalari isle (denetimin P2 maddesi).

DOGRULANDI: 842 WEI parcasinin 264'u zengin korpusta YOK. LOCKED, mevcut olcum kumesi ve
GEOMETRI CAKISMALARI cikarilinca **198 parca / 154 grup** gercekten yeni.

Bu, hafizadaki "WSCAD veri kaldiraci OLDU" kaydindan FARKLI: o kayit yeni parca INDIRMEKLE
ilgiliydi (325 deneme -> 7 dosya). Bunlar ZATEN ELIMIZDEKI ama korpusa hic konmamis parcalar.

TEZ CIZGISI: ayni boru hatti, ayni ag, ayni remesh, ayni aday ureticisi. Degisen TEK sey
egitim korpusunun buyuklugu -- yani en temiz "yalniz veri" deneyi.

BOLME (denetimin P2.2 tavsiyesi):
    154 grubun YARISI  -> PROSPEKTIF WEI TESTI (hic dokunulmaz, egitime girmez)
    diger yarisi       -> egitim korpusuna eklenir
Boylece veri etkisi HEM havuzlanmis olcumde HEM de gorulmemis-parca testinde olculur.

Cikti: results/yeni_wei.npz (X22+XR ozellikleri, etiketler, bolme etiketi)
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

OUT = "results/yeni_wei.npz"


def main():
    from build_zengin_parite import _normaller, zengin
    import olcum_kumesi

    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    gk = olcum_kumesi.geo_anahtarlari()
    sat, rap = olcum_kumesi.kume("results/_der_tam.pkl")
    kul = {r["geo"] for r in sat}
    s3 = json.load(io.open("results/split3.json", encoding="utf-8"))
    lock = {str(p) for p in s3["locked"]["parts"]}
    lock_geo = {gk.get(p, "yok:" + p) for p in lock}

    zen = np.load("results/zengin_parite.npz", allow_pickle=True)
    var = {str(x) for x in zen["pids"]}
    E = list(eligible())
    yeni = [(m, p, jf, s) for m, p, jf, s in E
            if m == "WEI" and p not in var and p not in lock
            and gk.get(p, "yok:" + p) not in kul
            and gk.get(p, "yok:" + p) not in lock_geo]
    gruplar = sorted({gk.get(p, "yok:" + p) for _, p, _, _ in yeni})
    print(f"GERCEKTEN YENI: {len(yeni)} parca / {len(gruplar)} grup", flush=True)

    # GRUP bazinda ikiye bol (parca degil grup -- ikizler ayni tarafta kalsin)
    rng = np.random.default_rng(0)
    kar = list(gruplar); rng.shuffle(kar)
    test_grup = set(kar[:len(kar) // 2])
    print(f"  PROSPEKTIF TEST: {len(test_grup)} grup | EGITIME EK: {len(kar)-len(test_grup)} grup",
          flush=True)

    cks = cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cks]

    X22, XR, YY, PID, BOL, PTS, DIRS, NGT = [], [], [], [], [], [], [], {}
    t0 = time.time(); atlanan = 0

    # ZEHIRLI PARCA KORUMASI + ARA KAYIT (2026-08-02: ilk kosu 100/222'de takildi ve
    # ara kayit OLMADIGI icin 100 parcalik is kayboldu -- gate_regrow'da bu zaten cozulmustu).
    ATLA = "results/yeni_wei_atla.txt"
    ISLENEN = "results/yeni_wei_islenen.txt"
    atla = set()
    if os.path.exists(ATLA):
        with io.open(ATLA, encoding="utf-8") as f:
            atla = {x.strip() for x in f if x.strip()}
    if atla:
        print(f"  kalici atlama listesi: {sorted(atla)}", flush=True)
    # DEVAM: onceki kosuda islenmis parcalari YENIDEN ISLEME (ara kayittan oku).
    if os.path.exists(OUT):
        try:
            _o = np.load(OUT, allow_pickle=True)
            _bit = {str(x) for x in _o["pids"]}
            for _k in ("X22", "XR"):
                pass
            X22.append(np.asarray(_o["X22"], float)); XR.append(np.asarray(_o["XR"], float))
            YY.append(np.asarray(_o["y"])); PID.extend([str(x) for x in _o["pids"]])
            BOL.extend([str(x) for x in _o["bolme"]])
            PTS.append(np.asarray(_o["pts"], float)); DIRS.append(np.asarray(_o["dirs"], float))
            for _p, _n in zip([str(x) for x in _o["pids"]], np.asarray(_o["ngt"])):
                NGT[_p] = int(_n)
            atla |= _bit
            print(f"  DEVAM: {len(_bit)} parca onceki kosudan geldi", flush=True)
        except Exception as _e:
            print(f"  devam okunamadi ({type(_e).__name__})", flush=True)

    def _ara_kayit():
        if not X22:
            return
        np.savez(OUT, X22=np.vstack(X22), XR=np.vstack(XR), y=np.concatenate(YY),
                 pids=np.array(PID), bolme=np.array(BOL), pts=np.vstack(PTS),
                 dirs=np.vstack(DIRS),
                 ngt=np.array([NGT.get(p, 0) for p in np.array(PID)]))
    for k, (mfg, pid, jf, stp) in enumerate(yeni, 1):
        if pid in atla:
            atlanan += 1; continue
        if k % 25 == 0:
            print(f"  {k}/{len(yeni)}  {time.time()-t0:.0f}s (atlanan {atlanan})", flush=True)
            _ara_kayit()
        # SU AN ISLENEN parcayi diske yaz: kosu takilip oldurulurse bir sonraki calistirmada
        # bu parca kalici atlama listesine girer ve is bir daha ayni yerde durmaz.
        with io.open(ISLENEN, "w", encoding="utf-8") as _f:
            _f.write(pid)
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
            cps, probs, _, _u = RC.adaylari_uret(V, F, pbs, stp, cfg=cfg)
            if not cps:
                atlanan += 1; continue
            xb = wire_gate.feats_for(V, F, probs, cps, CE, CT, step_path=stp)
            # zengin sutunlar TEK KAYNAKTAN (build_zengin_parite.zengin)
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
            g = gk.get(pid, "yok:" + pid)
            X22.append(xb); XR.append(xr); YY.append(yy)
            PID += [pid] * len(cps)
            BOL += ["test" if g in test_grup else "egitim"] * len(cps)
            PTS.append(P); DIRS.append(Pd); NGT[pid] = len(Gm)
        except Exception as e:
            atlanan += 1
            if atlanan <= 5:
                print(f"    atlandi {pid}: {type(e).__name__}: {e}", flush=True)
    _ara_kayit()
    if os.path.exists(ISLENEN):
        os.remove(ISLENEN)
    B = np.array(BOL)
    print(f"\n-> {OUT} | {len(B)} aday / {len(set(PID))} parca (atlanan {atlanan})")
    print(f"   egitim {int((B=='egitim').sum())} aday | PROSPEKTIF TEST {int((B=='test').sum())} aday")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
