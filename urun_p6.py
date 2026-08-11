# -*- coding: utf-8 -*-
"""URUN P6 YOLU: (konum x yon) ortak siralayici.

`urun_genis.cikti` ile AYNI imza -- `kanonik_zincir.urun_cikti` bunu bir config
anahtariyla cagirabilsin diye. Kol calisamazsa (model/STEP/B-rep yok) None doner
ve cagiran BIR ONCEKI yola duser; sessizce bozuk cikti URETILMEZ.

FARKI (tek cumleyle): `urun_genis` her adayi havuzun verdigi TEK yonle puanlar;
bu modul her adaya `yon_bankasi` seceneklerini takar ve (konum, yon) ciftini TEK
skorla siralar. Yon artik SECILIR.

ISARET DUZELTME YOK -- ve bu kasten. `urun_genis.isaret_duzelt` fiziksel bir
kuralla yonu ters cevirir (+0.0316 olculmustu); burada +u ve -u ZATEN ayri iki
secenek olarak bankada ve siralayici hangisinin dogru oldugunu ogrenir. Kurali
ustune koymak, ogrenilen karari eziyor. `P6_ISARET=1` ile acilir (ablasyon).
"""
import os

import numpy as np

import p6_karar
import urun_genis
import yon_bankasi as YB

MODEL_YOL = "results/p6_ortak_model.pkl"
_MODEL = None
ISARET = os.environ.get("P6_ISARET", "0") == "1"


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


ACIK = _cfg("robot_p6_ortak", "URUN_P6", False)


def model_yukle(yol=MODEL_YOL):
    """Doner: (model, esik, zskor) ya da None."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(yol):
            return None
        import pickle
        d = pickle.load(open(yol, "rb"))
        _MODEL = (d["model"], float(d["esik"]), d.get("zskor", "ab"))
    return _MODEL


def secenek_tablosu(V, F, probs, cps_seg, step_path, CE, CT):
    """Havuz + yon bankasi + 92 sutunluk oznitelik. Doner: (P, idx, YD, X) ya da None.

    Egitim betikleri de bunu cagirabilsin diye ayri: boylece egitimdeki oznitelik
    ile urundeki oznitelik AYNI koddan cikar.
    """
    import trimesh

    import wire_gate
    cyl, acik = urun_genis.brep_cikar(step_path)
    if cyl is None:
        return None
    Ps = np.asarray([c["point"] for c in cps_seg], float)
    Ds = np.asarray([c["direction"] for c in cps_seg], float)
    P, D, _kay = urun_genis.havuz(Ps, Ds, cyl, acik)
    if len(P) < 2:
        return None
    V = np.asarray(V, float)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    mesh = trimesh.Trimesh(V, np.asarray(F, np.int64), process=False)
    A = np.asarray(wire_gate.feats_for(
        V, F, probs, [{"point": P[i], "direction": D[i]}
                      for i in range(len(P))], CE, CT, step_path=step_path),
        float)
    B = urun_genis.tanimlayici(P, D, cyl, mesh, diag)
    idx, YD, C = YB.secenekler(P, D, cyl, V)
    if not len(idx):
        return None
    Dblok = urun_genis.tanimlayici(P[idx], YD, cyl, mesh, diag)
    return P, idx, YD, np.hstack([A[idx], B[idx], C, Dblok])


def cikti(V, F, probs, cps_seg, step_path, CE, CT):
    m = model_yukle()
    if m is None or not cps_seg:
        return None
    model, esik, zskor = m
    tab = secenek_tablosu(V, F, probs, cps_seg, step_path, CE, CT)
    if tab is None:
        return None
    P, idx, YD, X = tab
    s = np.asarray(model.predict_proba(
        p6_karar.donustur(X, zskor))[:, 1], float)
    P2, D2 = p6_karar.sec(P, idx, YD, s, esik)
    if ISARET and len(P2):
        T2 = urun_genis.tanimlayici(P2, D2, *_mesh_arg(V, F))
        D2 = urun_genis.isaret_duzelt(D2, T2)
    return [{"point": P2[i], "direction": D2[i], "wire_score": 1.0}
            for i in range(len(P2))]


def _mesh_arg(V, F):
    import trimesh
    V = np.asarray(V, float)
    return (None, trimesh.Trimesh(V, np.asarray(F, np.int64), process=False),
            float(np.linalg.norm(V.max(0) - V.min(0))))
