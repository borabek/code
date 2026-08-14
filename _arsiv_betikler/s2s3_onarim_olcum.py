# -*- coding: utf-8 -*-
"""S2+S3: ONARIM OPERATORLERINI OLC -- kusur gercekten gitti mi, METRIGE ne oldu?

S1 uc operator yazdi (govde ici -> agza tasi, duvara yapisik -> merkeze cek,
onu kapali -> yonu cevir). Bu betik onlari 194 parcada uygular ve IKI SORUYU ayirir:

  S2  KUSUR GITTI MI? Onarim sonrasi bayraklar YENIDEN olculur. Bir kusuru duzeltirken
      BASKASINI yakmak mumkun (govde disina tasirken agiz duvarina yapismak gibi).
  S3  METRIGE NE OLDU? Kusur oraninin dusmesi TEK BASINA yetmez. GECIS SAYILIR:
          TP -> FP  (onarim dogru CP'yi bozdu)   -> ZARAR
          FP -> TP  (onarim yanlis CP'yi kurtardi) -> FAYDA
      Net etki tespit ve robot F1'de raporlanir.

C1 BULGUSU (bu betigin cercevesini belirledi): kusurlar SON ISLEMDE URETILMIYOR.
    HAM 249 -> POZ 218 (-31) -> TAM 222
Poz kafasi kusuru AZALTIYOR. Yani onarim, zincirin kendi urettigi bir hatayi degil,
ADAY/GATE duzeyinden MIRAS ALINAN bir hatayi hedefliyor.

KILL (onceden yazildi): FP->TP >= TP->FP VE tespit >= -0.005.
"""
import io
import json
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
from sina_kume import esle_detay, f1w
from a1a3_metrik_cerrahi import urun_ciktisi
import s1_onarim as S1

ONB = "results/_s2_onarilmis.pkl"


def onar_hepsi(mesh, p, d):
    """Uc operatoru SIRAYLA uygula. (yeni_p, yeni_d, uygulanan_listesi)"""
    uyg = []
    p2, ok, _ = S1.onar_govde_ici(mesh, p, d)
    if ok:
        p = p2; uyg.append("govde")
    p2, ok, _ = S1.onar_merkezle(mesh, p, d)
    if ok:
        p = p2; uyg.append("merkez")
    d2, ok, msj = S1.onar_yon_cevir(mesh, p, d)
    if ok:
        d = d2; uyg.append("yon")
    elif msj == "iki taraf kapali":
        uyg.append("KANAL_DEGIL")
    return p, d, uyg


def bayrakla(mesh, p, d):
    from cp_geometry import is_inside, mouth_width, ray_hits
    try:
        ic = bool(is_inside(mesh, p))
    except Exception:
        ic = False
    try:
        h = ray_hits(mesh, p + 1e-3 * d, d, 60.0)
        il = float(min(h)) if len(h) else float("inf")
    except Exception:
        il = float("inf")
    try:
        cp = float(mouth_width(mesh, p, d)[0])
    except Exception:
        cp = float("nan")
    return (ic, bool(np.isfinite(il) and il < S1.ILERI_MIN),
            bool(np.isfinite(cp) and 0 < cp < S1.IC_CAP_MIN))


def main():
    import protokol
    protokol.tez_dogrula()
    import trimesh
    import thesis_remesh
    from big_arbiter import eligible
    from infer_step_cp import step_to_mesh

    DER, gate, ek = T2.yukle()
    CIKTI = {r["pid"]: urun_ciktisi(r, gate) for r in DER}
    stp = {p: s for m, p, jf, s in eligible()}

    if os.path.exists(ONB):
        with open(ONB, "rb") as f:
            R = pickle.load(f)
        print(f"onbellekten: {len(R)} parca", flush=True)
    else:
        R = {}
        t0 = time.time()
        for k, r in enumerate(DER, 1):
            if k % 20 == 0:
                print(f"  {k}/{len(DER)}  {time.time()-t0:.0f}s", flush=True)
                with open(ONB, "wb") as f:
                    pickle.dump(R, f)
            P, D = CIKTI[r["pid"]]
            if not len(P) or r["pid"] not in stp:
                continue
            try:
                Vr, Fr = step_to_mesh(stp[r["pid"]])
                V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
                mesh = trimesh.Trimesh(vertices=np.ascontiguousarray(V, float),
                                       faces=np.ascontiguousarray(F, np.int64), process=False)
            except Exception:
                continue
            P2 = P.copy(); D2 = D.copy(); UY = []; B0 = []; B1 = []
            for i in range(len(P)):
                B0.append(bayrakla(mesh, P[i], D[i]))
                p2, d2, uyg = onar_hepsi(mesh, P[i], D[i])
                P2[i] = p2; D2[i] = d2; UY.append(uyg)
                B1.append(bayrakla(mesh, p2, d2))
            R[r["pid"]] = {"P0": P, "D0": D, "P1": P2, "D1": D2,
                           "B0": B0, "B1": B1, "UY": UY}
        with open(ONB, "wb") as f:
            pickle.dump(R, f)
        print(f"-> {ONB}", flush=True)

    ADLAR = ("govde_ici", "onu_kapali", "duvara_yapisik")
    n = sum(len(d["B0"]) for d in R.values())
    print(f"\n{'='*76}\nS2 -- ONARIM SONRASI BAYRAKLAR ({len(R)} parca / {n} CP)\n{'='*76}")
    print(f"{'bayrak':<18}{'once':>8}{'sonra':>8}{'fark':>8}")
    for q, ad in enumerate(ADLAR):
        a = sum(1 for d in R.values() for b in d["B0"] if b[q])
        b_ = sum(1 for d in R.values() for b in d["B1"] if b[q])
        print(f"{ad:<18}{a:>8}{b_:>8}{b_-a:>+8}")
    ha = sum(1 for d in R.values() for b in d["B0"] if any(b))
    hb = sum(1 for d in R.values() for b in d["B1"] if any(b))
    print(f"{'HERHANGI':<18}{ha:>8}{hb:>8}{hb-ha:>+8}   (%{100*ha/max(n,1):.1f} -> %{100*hb/max(n,1):.1f})")

    uyg = {}
    for d in R.values():
        for u in d["UY"]:
            uyg[tuple(u)] = uyg.get(tuple(u), 0) + 1
    print(f"\nuygulanan operator dagilimi:")
    for k, v in sorted(uyg.items(), key=lambda x: -x[1])[:8]:
        print(f"   {('hicbiri' if not k else '+'.join(k)):<28}{v:>5}")

    print(f"\n{'='*76}\nS3 -- METRIGE ETKI (gecis sayimi)\n{'='*76}")
    for ad, robot in (("TESPIT", False), ("ROBOT(FIZ)", True)):
        r0, r1 = [], []
        t2f = f2t = 0
        for r in DER:
            d = R.get(r["pid"])
            G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
            rj = "cok" if r["n"] >= 8 else "dusuk"
            if d is None:
                P0 = P1 = np.zeros((0, 3)); D0 = D1 = np.zeros((0, 3))
            else:
                P0, D0, P1, D1 = d["P0"], d["D0"], d["P1"], d["D1"]
            kw = dict(isaretli=True) if robot else {}
            a = (2.0, 10.0, False) if robot else (0.0, 180.0, True)
            tp0, fp0, fn0, b0 = esle_detay(P0, D0, G, Gd, r["diag"], *a, **kw)
            tp1, fp1, fn1, b1 = esle_detay(P1, D1, G, Gd, r["diag"], *a, **kw)
            r0.append((rj, tp0, fp0, fn0)); r1.append((rj, tp1, fp1, fn1))
            e0 = {e[0] for e in b0["eslesme"]}; e1 = {e[0] for e in b1["eslesme"]}
            t2f += len(e0 - e1); f2t += len(e1 - e0)
        print(f"  {ad:<12}{f1w(r0):>9.4f} -> {f1w(r1):>9.4f}  ({f1w(r1)-f1w(r0):+.4f})"
              f"   TP->FP {t2f:>4} | FP->TP {f2t:>4}")
        if not robot:
            gecti = f2t >= t2f and (f1w(r1) - f1w(r0)) >= -0.005
            KILL = (f2t, t2f, gecti)
    print(f"\nKILL: FP->TP ({KILL[0]}) >= TP->FP ({KILL[1]}) VE tespit >= -0.005 -> "
          f"{'GECTI' if KILL[2] else 'GECMEDI'}")

    with io.open("results/s2s3_onarim.json", "w", encoding="utf-8") as f:
        json.dump({"n_cp": n, "kusur_once": ha, "kusur_sonra": hb,
                   "fp2tp": KILL[0], "tp2fp": KILL[1], "gecti": bool(KILL[2]),
                   "operator_dagilimi": {"+".join(k) or "hicbiri": v for k, v in uyg.items()}},
                  f, indent=1, ensure_ascii=False)
    print("makbuz -> results/s2s3_onarim.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
