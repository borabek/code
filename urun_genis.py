# -*- coding: utf-8 -*-
"""GENISLETILMIS URUN YOLU: B-rep havuzu + agiz tanimlayicilari + isaret duzeltme.

OLCULDU (D7 = 835 parca marka-disi, TAM ZINCIR, MIKRO, esik D6'da secildi):
| yigin | robot | tespit | makro |
|---|---|---|---|
| dagitilan urun (v6 + NMS) | 0.2029 | 0.4523 | 0.2146 |
| **bu yol** | **0.3090** | **0.4813** | **0.3142** |

10/12 markada artis; gercek kayip YOK (CCD -0.007 duz, C3 -0.009 bes parcada).
Makbuzlar: `results/secici_ailesi.json`, `results/b2a_isaret.json`,
`results/tam_havuz_gate.json`.

UC KALDIRAC (hepsi tek tek olculdu):
 1. ETIKET TANIMI duzeltmesi -- gate korpusu `build_zengin_parite.py`'nin
    tanimiyla (yanal + eksenel 40mm + acgozlu bire-bir). Tek basina 0.1159 -> 0.2009.
 2. B-REP HAVUZU (silindir agizlari + duzlemsel aciklik merkezleri, 3mm dedupe):
    0.2213 -> 0.2563. Mesh tepeleri EKLENMEZ -- uc bagimsiz olcumde ZARAR verdi.
 3. AGIZ TANIMLAYICILARI (9 sutun) + HGB-derin + ISARET DUZELTME: -> 0.3090.

TEZE SADIK: DiffusionNet 5 sinif, ~6000 uniform izotropik remesh ve `v_o`
agiz-ortasi turetmesi DEGISMEDI. B-rep onerileri tezin adaylarinin YANINA
eklenen ikinci bir kaynaktir; tezin cevabi her zaman havuzda ve `kaynak==0`
ile isaretlidir. Sonuclar "tez sonucu" degil "tez-omurgali genisletme" olarak
raporlanir.

KAPATMA: `cp_config.json` icinde `robot_genis_havuz=false` ya da
`URUN_GENIS=0` cevre degiskeni -> urun eski yoluna doner.
"""
import os

import numpy as np

import agiz_tanimlayici
import brep_havuz
import wire_gate

GIRME = agiz_tanimlayici.AD.index("girme")
ERISIM = agiz_tanimlayici.AD.index("erisim")


def _cfg(ad, cevre, vars_):
    v = os.environ.get(cevre)
    if v is not None:
        return v not in ("0", "", "false", "False")
    try:
        import json
        return bool(json.load(open("cp_config.json", encoding="utf-8"))
                    .get(ad, vars_))
    except Exception:
        return vars_


ACIK = _cfg("robot_genis_havuz", "URUN_GENIS", True)
ESIK = 0.05          # D6'da secildi (`results/secici_ailesi.json`), D7'de taranmadi


def havuz(P_seg, D_seg, cyl, acik):
    """Tezin `v_o` adaylari + B-rep agizlari. Mesh tepeleri KASTEN YOK.

    Mesh tepeleri havuz recall'unu 0.6654 -> 0.8465 acar ama uctan uca UC
    bagimsiz olcumde ZARAR verdi (0.2972 vs 0.3095): FP neredeyse ikiye
    katlaniyor. Tavan acmak yetmiyor, secici kullanamiyor.
    """
    return brep_havuz.birlesik_havuz(P_seg, D_seg, cyl, acik)


def tanimlayici(P, D, cyl, mesh, diag):
    """Agiz olculeri. `cyl` yoksa yalnizca isin olculeri dolar."""
    met = []
    if cyl:
        C = np.asarray([c["center"] for c in cyl], float)
        A = np.asarray([c["axis"] for c in cyl], float)
        n = np.linalg.norm(A, axis=1, keepdims=True)
        A = A / np.maximum(n, 1e-12)
        for p in np.asarray(P, float).reshape(-1, 3):
            w = p[None] - C
            e = np.einsum("ij,ij->i", w, A)
            d = np.linalg.norm(w - e[:, None] * A, axis=1)
            i = int(np.argmin(d))
            m = {"radius": float(cyl[i].get("radius", 0.0))}
            if cyl[i].get("mouth_a") is not None:
                m["mouth_a"] = cyl[i]["mouth_a"]
                m["mouth_b"] = cyl[i].get("mouth_b")
            met.append(m)
    else:
        met = [{} for _ in range(len(P))]
    return agiz_tanimlayici.tanimla(P, D, met, mesh, diag)


def isaret_duzelt(D, T):
    """Tel DISARIDAN girer: disari yolu iceriden kisaysa yon TERS cevrilir.

    Olculdu (`results/b2a_isaret.json`): +0.0316 robot, tespit DEGISMEDI.
    Saf fiziksel kural ogrenilmis siniflandiricinin %94'unu veriyor; urunde
    OGRENME YOK, kural var -- daha az hareketli parca.
    """
    D = np.asarray(D, float).reshape(-1, 3)
    T = np.asarray(T, float)
    if not len(D):
        return D
    return np.where((T[:, ERISIM] < T[:, GIRME])[:, None], -D, D)


def sec(P, D, X58, cyl, mesh, diag, model, esik=ESIK):
    """Genisletilmis yolun KARAR fonksiyonu. Doner: (P, D) secilmis adaylar.

    Sira: oznitelik -> gate -> esik -> NMS -> ISARET DUZELTME.
    (Poz kafasi bu fonksiyonun DISINDA, urun zincirinde kalir.)
    """
    P = np.asarray(P, float).reshape(-1, 3)
    D = np.asarray(D, float).reshape(-1, 3)
    if len(P) < 2:
        return P, D
    T = tanimlayici(P, D, cyl, mesh, diag)
    X = np.hstack([np.asarray(X58, float), T])
    s = np.asarray(model.predict_proba(
        wire_gate.parca_ici(X, "zskor"))[:, 1], float)
    k = s >= esik
    if not k.any():
        return P[:0], D[:0]
    P, D, T, s = P[k], D[k], T[k], s[k]
    if len(P) > 1:
        nm = wire_gate.kalabalik_maskesi(P, s)
        P, D, T = P[nm], D[nm], T[nm]
    return P, isaret_duzelt(D, T)


MODEL_YOL = "results/kazanan_hgb_derin.pkl"
_MODEL = None


def model_yukle(yol=MODEL_YOL):
    """HGB-derin gate. Yoksa None doner -> cagiran ESKI yola duser."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(yol):
            return None
        import pickle
        _MODEL = pickle.load(open(yol, "rb"))["HGB-derin"]
    return _MODEL


def brep_cikar(step_path):
    """Silindir + duzlemsel aciklik. Cikarim aninda ~0.7 s/parca.

    Onbellek YOK -- urun yeni bir parca gorur ve STEP'ten hesaplamak zorundadir.
    STEP verilmezse (None) kol devre disi kalir ve cagiran ESKI yola duser.
    """
    if not step_path or not os.path.exists(step_path):
        return None, None
    import brep_aciklik
    import brep_snap
    try:
        cyl = brep_snap.exact_cylinders(step_path)
    except Exception:
        cyl = []
    try:
        acik = brep_aciklik.acikliklar(step_path)
    except Exception:
        acik = []
    return cyl, acik


def cikti(V, F, probs, cps_seg, step_path, CE, CT):
    """URUNUN genisletilmis ciktisi. Doner: cps listesi (point/direction).

    Girdi `cps_seg` tezin `v_o` adaylaridir ve HAVUZDA KALIR (kaynak 0).
    Kol calisamiyorsa (model yok / STEP yok / B-rep bos) None doner ve cagiran
    ESKI yola duser -- sessizce bozuk cikti URETILMEZ.
    """
    model = model_yukle()
    if model is None or not cps_seg:
        return None
    cyl, acik = brep_cikar(step_path)
    if cyl is None:
        return None
    import trimesh
    Ps = np.asarray([c["point"] for c in cps_seg], float)
    Ds = np.asarray([c["direction"] for c in cps_seg], float)
    P, D, _kay = havuz(Ps, Ds, cyl, acik)
    if len(P) < 2:
        return None
    diag = float(np.linalg.norm(np.asarray(V).max(0) - np.asarray(V).min(0)))
    mesh = trimesh.Trimesh(np.asarray(V, float), np.asarray(F, np.int64),
                           process=False)
    X58 = np.asarray(wire_gate.feats_for(
        V, F, probs, [{"point": P[i], "direction": D[i]}
                      for i in range(len(P))], CE, CT, step_path=step_path),
        float)
    P2, D2 = sec(P, D, X58, cyl, mesh, diag, model)
    return [{"point": P2[i], "direction": D2[i], "wire_score": 1.0}
            for i in range(len(P2))]
