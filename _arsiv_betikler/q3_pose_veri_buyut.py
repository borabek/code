# -*- coding: utf-8 -*-
"""Q3: POSE egitim verisini TAM KORPUSTAN cikar (ag cikarimi YOK).

Q2'de pose head robot-hazira +0.0222 getirdi ama GA sifiri iceriyor -- yalnizca 1068 satirla
egitiliyordu (olcum kumesinin 194 parcasindan). Ogrenilmis bir kafayi guclendirmenin en
guvenilir yolu daha cok veri.

TAM KORPUS icin gereken parcalar ZATEN diskte:
    ozellikler (X22+XR)  -> results/zengin_parite.npz
    aday nokta/yonleri   -> results/gate_regrow_data_parite.npz  (pts, dirs)
    GT                   -> uretici JSON'lari
Eksik olan tek sey GT'nin MESH CERCEVESINE tasinmasi (align_frames). O da ham STEP mesh'i
yerine ONBELLEKTEKI REMESH ile yapilabiliyor -- DOGRULANDI: 18 parcada medyan fark 0.0000mm,
maks 0.0191mm. Yani gmsh cagrisi ve ag cikarimi GEREKMIYOR.

Cikti: results/pose_veri.npz (X58, hedef4, pid, geo) -- yalniz TESPITTE ESLESEN adaylar.
"""
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("Q3_OUT", "results/pose_veri_graf.npz")


def yerel_cerceve(d):
    d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-9)
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(d @ a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u) + 1e-9
    return d, u, np.cross(d, u)


def main():
    from big_arbiter import eligible
    from cad_eval import align_frames

    jf_of = {p: jf for m, p, jf, s in eligible()}
    zen = np.load("results/zengin_parite.npz", allow_pickle=True)
    X58 = np.hstack([np.asarray(zen["X22"], float), np.asarray(zen["XR"], float)])
    # B-rep GRAF sutunlari: gate'te uctan uca OLU cikti (WEI -0.031) ama POSE tahmini AYRI bir
    # gorev -- kanal kor mu/gecis mi, eseksenli zincir var mi bilgisi konum artigi icin
    # bilgilendirici olabilir. Ayni satir hizasinda ekleniyor.
    try:
        _g = np.load("results/brep_graf.npz", allow_pickle=True)
        _ad = [str(x) for x in _g["ad"]]
        _tut = [i for i, a in enumerate(_ad) if a not in ("g_agiz_cev", "g_yuz_alan")]
        GX = np.asarray(_g["X"], float)[:, _tut]
        assert len(GX) == len(X58)
        X58 = np.hstack([X58, GX])
        print(f"  graf sutunlari eklendi -> {X58.shape[1]} sutun")
    except Exception as _e:
        print(f"  graf sutunlari EKLENMEDI ({type(_e).__name__})")
    zpid = np.array([str(x) for x in zen["pids"]])
    par = np.load("results/gate_regrow_data_parite.npz", allow_pickle=True)
    ppid = np.array([str(x) for x in par["pids"]])
    PTS = np.asarray(par["pts"], float); DIRS = np.asarray(par["dirs"], float)
    with io.open("results/_strict_geometry_keys.json", encoding="utf-8") as f:
        gk = json.load(f)

    # Iki npz AYNI aday ureticisinden geldi; parca basina aday SAYISI eslesmeli.
    ortak = [p for p in np.unique(zpid) if (ppid == p).sum() == (zpid == p).sum()]
    print(f"zengin {len(np.unique(zpid))} parca | parite {len(np.unique(ppid))} | "
          f"aday sayisi ESLESEN {len(ortak)}", flush=True)

    RX, RY, RP, RG = [], [], [], []
    atlanan = 0
    for k, pid in enumerate(ortak, 1):
        if k % 200 == 0:
            print(f"  {k}/{len(ortak)} (atlanan {atlanan})", flush=True)
        mc = f"results/mesh_cache/{pid}.npz"
        if not os.path.exists(mc) or pid not in jf_of:
            atlanan += 1; continue
        try:
            V = np.load(mc)["V"].astype(float)
            with io.open(jf_of[pid], encoding="utf-8-sig") as f:
                j = json.load(f)
            cps = j.get("ConnectionPoints") or []
            if not cps:
                atlanan += 1; continue
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in cps], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in cps], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vj = np.array([[q["X"], q["Y"], q["Z"]] for q in j["Graphic3d"]["Points"]], float)
            R, t, _ = align_frames(V, Vj)
            Gm = (G - t) @ R; Gdm = Gd @ R
        except Exception:
            atlanan += 1; continue

        zi = np.where(zpid == pid)[0]; pi = np.where(ppid == pid)[0]
        P = PTS[pi]; Pd = DIRS[pi]
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        diff = P[:, None, :] - Gm[None, :, :]
        al = (diff * Gdm[None, :, :]).sum(-1)
        pe = np.linalg.norm(diff - al[..., None] * Gdm[None, :, :], axis=-1)
        pe = np.where(np.abs(al) > 40, np.inf, pe)
        # ESLESME TOLERANSI (2026-08-14: cevreden ayarlanabilir).
        # Bolum 21.53: pose head buyuk duzeltme ONERMIYOR cunku egitim
        # verisi yalnizca ZATEN ESLESMIS adaylardan kuruluyor -- eslesmis
        # adayin artigi tanim geregi kucuktur. Bu bir SECIM YANLILIGI.
        # Toleransi acmak, modele BUYUK artiklari gosterir.
        tt = float(os.environ.get("Q3_TOL", "0")) or max(3.0, 0.06 * diag)
        used, hit = set(), set()
        for dd, a_, b_ in sorted((pe[a, b], a, b)
                                 for a in range(len(P)) for b in range(len(Gm))):
            if dd > tt or a_ in used or b_ in hit:
                continue
            used.add(a_); hit.add(b_)
            d, u, v = yerel_cerceve(Pd[a_])
            w = Gm[b_] - P[a_]
            w_perp = w - float(w @ Gdm[b_]) * Gdm[b_]
            g = Gdm[b_] * (1.0 if float(Gdm[b_] @ d) >= 0 else -1.0)
            RX.append(X58[zi[a_]])
            RY.append([float(w_perp @ u), float(w_perp @ v), float(g @ u), float(g @ v)])
            RP.append(pid); RG.append(gk.get(pid, "yok:" + pid))
    RX = np.array(RX, float); RY = np.array(RY, float)
    np.savez(OUT, X=RX, Y=RY, pid=np.array(RP), geo=np.array(RG))
    yan = np.linalg.norm(RY[:, :2], axis=1)
    aci = np.degrees(np.arcsin(np.clip(np.linalg.norm(RY[:, 2:], axis=1), 0, 1)))
    print(f"\n-> {OUT} | {len(RY)} eslesen aday / {len(set(RP))} parca / {len(set(RG))} grup")
    print(f"   yanal sapma: medyan {np.median(yan):.2f}mm | <=2mm {np.mean(yan <= 2):.1%}")
    print(f"   yon sapma  : medyan {np.median(aci):.2f} deg | <=10 deg {np.mean(aci <= 10):.1%}")
    print(f"   (Q1'de 1068 satir vardi -> {len(RY)/1068:.1f} kat)")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
