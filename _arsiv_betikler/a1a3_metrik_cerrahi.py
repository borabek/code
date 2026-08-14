# -*- coding: utf-8 -*-
"""A1+A2+A3: OLCUM CERRAHISI -- eksenel tolerans, belirsiz eslesme, aci gorunurlugu.

OTOPSI BULGUSU (2026-08-04): metrik uc yerden kordu.
  A1  EKSENEL TOLERANS 40mm -- kodda gomulu, gerekcesiz. Olculen seat->agiz farki
      medyan 7.5-10.6mm (bazi parcalarda 0.0). 40mm o farkin ~4 kati.
  A2  BELIRSIZ ESLESME -- GT'lerin %29.3'unun kabul kutusunda BASKA bir GT var
      (dusuk-CP %20.8, cok-CP %33.0). "Dogru" sayilan eslesme komsu delige ait olabilir.
  A3  ACI -- tespit olcutu aciya TAMAMEN KOR (am=180). 180 derece TERS bir CP tabloda
      "dogru" gorunuyor.

BU BETIK HICBIR SEYI DAGITMAZ. Dort sayiyi YAN YANA koyar:
      F1(eksen 40)      = bugunku manset (sisik)
      F1(eksen 15)      = eksenel toleransi olcuye dayandirilmis hali
      F1_kesin(15)      = belirsiz eslesmeler KREDILENDIRILMEZ (alt sinir)
      + aci dagilimi
MANSET DUSECEK. Bu bir kayip degil, sisintinin geri alinmasidir.
"""
import io
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
import wire_gate
from sina_kume import esle_detay, f1w, f1_rejim


def urun_ciktisi(r, gate):
    """Urunun karar yolu -> (P, Pd). tezgah2.puanla ile AYNI zincir."""
    P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
    if r["X"] is not None and r.get("XR") is not None:
        X = np.hstack([r["X"], r["XR"]])
        if X.shape[1] * 2 == gate["n_feat"]:
            k = wire_gate.karar_maskesi(wire_gate.karar_skoru(gate, X))
            if k.any():
                P = np.asarray(r["P"], float)[k].copy()
                Pd = np.asarray(r["Pd"], float)[k].copy()
                c = [{"point": P[i], "direction": Pd[i]} for i in range(len(P))]
                c = wire_gate.pose_duzelt(X[k], c)
                c = wire_gate.aci_duzelt(X[k], c)
                if r.get("UYE"):
                    c = wire_gate.uye_yonu_sec(X[k], c, r["UYE"])
                P = np.array([x["point"] for x in c], float)
                Pd = np.array([x["direction"] for x in c], float)
    return P, Pd


def main():
    import protokol
    protokol.tez_dogrula()
    DER, gate, ek = T2.yukle()

    CIKTI = {}
    for r in DER:
        CIKTI[r["pid"]] = urun_ciktisi(r, gate)

    def olc(eksen_tol, kesin=False, robot=False):
        """kesin=True: belirsiz eslesmeler TP sayilmaz (alt sinir)."""
        rows, aci, belirsiz, toplam = [], [], 0, 0
        for r in DER:
            P, Pd = CIKTI[r["pid"]]
            G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
            rj = "cok" if r["n"] >= 8 else "dusuk"
            if robot:
                tp, fp, fn, b = esle_detay(P, Pd, G, Gd, r["diag"], 2.0, 10.0, False,
                                           isaretli=True, eksen_tol=eksen_tol)
            else:
                tp, fp, fn, b = esle_detay(P, Pd, G, Gd, r["diag"], 0.0, 180.0, True,
                                           eksen_tol=eksen_tol)
            nb = b["belirsiz"]; belirsiz += nb; toplam += tp
            if kesin and nb:
                tp -= nb; fp += nb; fn += nb
            rows.append((rj, tp, fp, fn))
            aci += [e[4] for e in b["eslesme"]]
        return rows, np.array(aci), belirsiz, toplam

    print("=" * 78)
    print("A1 -- EKSENEL TOLERANS (tespit)")
    print("=" * 78)
    R = {}
    for et in (40.0, 25.0, 15.0, 10.0):
        rows, aci, bel, top = olc(et)
        R[et] = {"f1": f1w(rows), "belirsiz": bel, "tp": top}
        rj = f1_rejim(rows) if callable(globals().get("f1_rejim")) else {}
        print(f"  eksen_tol {et:>5.1f}mm -> tespit F1 {f1w(rows):.4f} | TP {top} | "
              f"belirsiz {bel} (%{100*bel/max(top,1):.1f})")
    d = R[15.0]["f1"] - R[40.0]["f1"]
    print(f"\n  40mm -> 15mm FARK: {d:+.4f}  (manset {R[40.0]['f1']:.4f} -> {R[15.0]['f1']:.4f})")

    print("\n" + "=" * 78)
    print("A2 -- BELIRSIZ ESLESME (F1 vs F1_kesin)")
    print("=" * 78)
    for et in (40.0, 15.0):
        r1, _, b1, t1 = olc(et, kesin=False)
        r2, _, _, _ = olc(et, kesin=True)
        print(f"  eksen_tol {et:>5.1f}mm  F1 {f1w(r1):.4f}  |  F1_kesin {f1w(r2):.4f}"
              f"  |  fark {f1w(r2)-f1w(r1):+.4f}  (belirsiz {b1}/{t1})")

    print("\n" + "=" * 78)
    print("A3 -- ACI GORUNURLUGU (tespit eslesmelerinin ACI dagilimi, ISARETSIZ eksen)")
    print("=" * 78)
    _, aci, _, _ = olc(15.0)
    if len(aci):
        for q in (50, 75, 90, 95, 99):
            print(f"  %{q:<3} {np.percentile(aci, q):>7.1f} derece")
        for esik in (10, 30, 60, 90):
            print(f"  aci > {esik:>2} derece olan eslesme: {int((aci>esik).sum())}/{len(aci)}"
                  f" (%{100*(aci>esik).mean():.1f})")

    print("\n" + "=" * 78)
    print("ROBOT (FIZIKSEL) metrigi ayni cerrahi ile")
    print("=" * 78)
    for et in (40.0, 15.0):
        rows, _, bel, top = olc(et, robot=True)
        rows2, _, _, _ = olc(et, kesin=True, robot=True)
        print(f"  eksen_tol {et:>5.1f}mm  robot {f1w(rows):.4f} | kesin {f1w(rows2):.4f}"
              f" | belirsiz {bel}/{top}")

    with io.open("results/a1a3_metrik_cerrahi.json", "w", encoding="utf-8") as f:
        json.dump({"tespit": {str(k): v for k, v in R.items()},
                   "aci_p90": float(np.percentile(aci, 90)) if len(aci) else None,
                   "aci_60_ustu": int((aci > 60).sum()) if len(aci) else 0,
                   "not": "A1 eksen 40->15, A2 belirsiz bayragi, A3 aci gorunurlugu"},
                  f, indent=1, ensure_ascii=False)
    print("\nmakbuz -> results/a1a3_metrik_cerrahi.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
