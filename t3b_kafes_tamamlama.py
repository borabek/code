# -*- coding: utf-8 -*-
"""T3b: KACIRILAN CP'ler KAFES DUGUMLERINDE MI? (B kolunun IKINCI, ayri kapisi)

NEDEN AYRI OLCUM: T3 yapisal oznitelikleri gate'in KABUL ETTIGI adaylar uzerinde sinadi
ve gecmedi. Ama o popülasyonda **FN HIC YOK** -- kacirilan CP'ler ya reddedildi ya hic
aday olmadi. Kafes TAMAMLAMA fikri tam da onlari hedefliyor, dolayisiyla T3 onu OLCEMEZ.
Kendi sondamin kor noktasi; ayri kapatiliyor.

SORU: bir parcada eslesmis (TP) adaylar bir dizi olusturuyorsa, KACIRILAN GT noktalari
o dizinin BOS DUGUMLERINDE mi duruyor?

  EVET ise -> tamamlama gercek bir FN ilaci; B kolu bu haliyle kurulur.
  HAYIR ise -> kacirmalar duzensiz; B tamamen kapanir.

TEZ DEGISMEZ: olcum; hicbir sey egitilmez, `v_o` ve aday ureticisi aynen kalir.

GO: kacirilan GT'lerin >=%40'i kafes dugumune <=1.0mm otursun VE bu oran rastgele
konumlardan ACIK ARA yuksek olsun (sahte-kafes kontrolu: ayni parcada rastgele
noktalarin kafese oturma orani KARSILASTIRILIR -- kafes cok sikysa her sey oturur).
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
from t3_yapisal_tavan import _baskin_adim

ESIK = 1.0      # mm -- kafes dugumune oturma toleransi


def kafes_kalinti(nokta, TP, e0, ort, adim):
    """Noktanin dizi ekseni boyunca en yakin kafes dugumune uzakligi (mm)."""
    if adim <= 1e-6:
        return np.inf
    t = (nokta - ort) @ e0
    return abs(t / adim - round(t / adim)) * adim


def main():
    import protokol
    protokol.tez_dogrula()
    DER, gate, ek = T2.yukle()

    kalinti_fn, kalinti_rast, kalinti_tp = [], [], []
    parca_ok = 0
    rng = np.random.RandomState(0)
    for r in DER:
        if r["X"] is None or r.get("XR") is None:
            continue
        M = np.hstack([r["X"], r["XR"]])
        if M.shape[1] * 2 != gate["n_feat"]:
            continue
        k = wire_gate.karar_maskesi(wire_gate.karar_skoru(gate, M))
        P = np.asarray(r["P"], float)[k]
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        if len(G) < 4 or len(P) < 3:
            continue                       # dizi konusabilmek icin en az 4 GT / 3 aday
        tol = max(3.0, 0.06 * float(r["diag"]))
        d = P[:, None, :] - G[None, :, :]
        al = (d * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(d - al[..., None] * Gd[None, :, :], axis=-1)
        pe = np.where(np.abs(al) > 40, np.inf, pe)
        # tekil eslesme (metrikle AYNI kural)
        up, ug = set(), set()
        for dd, a_, b_ in sorted((pe[a, b], a, b)
                                 for a in range(len(P)) for b in range(len(G))):
            if dd > tol or a_ in up or b_ in ug:
                continue
            up.add(a_); ug.add(b_)
        fn = [b for b in range(len(G)) if b not in ug]
        tp = sorted(up)
        if not fn or len(tp) < 3:
            continue
        TP = P[tp]
        ort = TP.mean(0)
        _, _, Vt = np.linalg.svd(TP - ort, full_matrices=False)
        e0 = Vt[0]
        adim = _baskin_adim((TP - ort) @ e0)
        if adim <= 1e-6:
            continue
        parca_ok += 1
        for b in fn:
            kalinti_fn.append(kafes_kalinti(G[b], TP, e0, ort, adim))
        for a in tp:
            kalinti_tp.append(kafes_kalinti(P[a], TP, e0, ort, adim))
        # SAHTE-KAFES KONTROLU: ayni parcanin sinir kutusunda rastgele noktalar
        lo, hi = G.min(0), G.max(0)
        for _ in range(len(fn)):
            q = lo + rng.rand(3) * (hi - lo)
            kalinti_rast.append(kafes_kalinti(q, TP, e0, ort, adim))

    kf = np.array(kalinti_fn); kr = np.array(kalinti_rast); kt = np.array(kalinti_tp)
    print(f"dizi konusulabilen parca: {parca_ok} | FN {len(kf)} | TP {len(kt)}")
    if not len(kf):
        print("olculecek FN yok"); return
    of = float((kf <= ESIK).mean()); orst = float((kr <= ESIK).mean())
    ot = float((kt <= ESIK).mean())
    print(f"\n{'kume':<22}{'kafese oturan':>16}{'ortanca kalinti':>18}")
    for ad, v in (("KACIRILAN (FN)", kf), ("rastgele nokta", kr), ("eslesen (TP)", kt)):
        print(f"{ad:<22}{100*(v<=ESIK).mean():>15.1f}%{np.median(v):>17.2f}mm")
    kazanc = of - orst
    print(f"\nFN kafese oturma %{100*of:.1f} | rastgele %{100*orst:.1f} | "
          f"FARK {100*kazanc:+.1f} puan")
    gecti = of >= 0.40 and kazanc >= 0.15
    print(f"GO (FN >=%40 VE rastgeleden >=15 puan yuksek) -> "
          f"{'GECTI -- kafes TAMAMLAMA gercek FN ilaci' if gecti else 'GECMEDI -- B TAMAMEN KAPANIR'}")
    if not gecti and of >= 0.40:
        print("  NOT: FN orani yuksek ama rastgele de yuksek -> kafes cok sik, AYIRT ETMIYOR")
    with io.open("results/t3b_kafes_tamamlama.json", "w", encoding="utf-8") as f:
        json.dump({"parca": parca_ok, "n_fn": len(kf), "esik_mm": ESIK,
                   "fn_oturan": of, "rastgele_oturan": orst, "tp_oturan": ot,
                   "fark": kazanc, "gecti": bool(gecti)}, f, indent=1)
    print("makbuz -> results/t3b_kafes_tamamlama.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
