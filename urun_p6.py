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

MODEL_YOL = os.environ.get("P6_MODEL", "results/p6_kademe2_model.pkl")
_MODEL = None
ISARET = os.environ.get("P6_ISARET", "0") == "1"
MESH_HAVUZ = os.environ.get("P6_MESH_HAVUZ", "1") == "1"


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
    """Egitimin yazdigi PAKETI oku. Doner: sozluk ya da None.

    Paket: kademe1 (+ istege bagli kademe2), karar kurali, NMS, tohum kurali.
    Tek bir yerden okunur ki urun ile egitim AYNI kurali kullansin.
    """
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(yol):
            return None
        import pickle
        _MODEL = pickle.load(open(yol, "rb"))
    return _MODEL


def secenek_tablosu(V, F, probs, cps_seg, step_path, CE, CT):
    """Havuz + yon bankasi + 92 sutunluk oznitelik. Doner: (P, idx, YD, X) ya da None.

    Egitim betikleri de bunu cagirabilsin diye ayri: boylece egitimdeki oznitelik
    ile urundeki oznitelik AYNI koddan cikar.
    """
    import trimesh

    import brep_havuz
    import havuz_seyrelt
    import wire_gate
    cyl, acik = urun_genis.brep_cikar(step_path)
    if cyl is None:
        return None
    Ps = np.asarray([c["point"] for c in cps_seg], float)
    Ds = np.asarray([c["direction"] for c in cps_seg], float)
    P, D, kaynak = urun_genis.havuz(Ps, Ds, cyl, acik)
    if len(P) < 2:
        return None
    if MESH_HAVUZ:
        # MESH TEPESI HAVUZU. Olculdu (D6): yalniz-konum recall 0.5371 -> 0.9768.
        # Tarihte UC kez zarar vermisti cunku yon bankasi yoktu; tek basina
        # yalniz FP uretiyor. Burada yonu ortak siralayici seciyor.
        # Seyreltme kurali `havuz_seyrelt` -- EGITIMDEKIYLE AYNI FONKSIYON.
        pp = havuz_seyrelt.ppos(probs, CE, CT)
        Pm, Dm = brep_havuz.mesh_adaylari(V, F, pp, brep_havuz.MESH_ESIK,
                                          brep_havuz.MESH_DEDUPE_MM)
        if len(Pm):
            uz = np.linalg.norm(Pm[:, None] - P[None], axis=-1).min(1)
            k = uz >= brep_havuz.MESH_DEDUPE_MM
            Pm, Dm = Pm[k], Dm[k]
        if len(Pm):
            sm = pp[np.argmin(np.linalg.norm(
                Pm[:, None, :] - np.asarray(V, float)[None, :, :],
                axis=-1), axis=1)] if len(Pm) * len(V) < 6e7 \
                else np.zeros(len(Pm))
            s_ = havuz_seyrelt.seyrelt(Pm, sm, len(P))
            if len(s_):
                P = np.vstack([P, Pm[s_]])
                D = np.vstack([D, Dm[s_]])
                kaynak = np.concatenate([kaynak, np.full(len(s_), 2, int)])
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
    return P, idx, YD, np.hstack([A[idx], B[idx], C, Dblok]), kaynak


def cikti(V, F, probs, cps_seg, step_path, CE, CT):
    pk = model_yukle()
    if pk is None or not cps_seg:
        return None
    tab = secenek_tablosu(V, F, probs, cps_seg, step_path, CE, CT)
    if tab is None:
        return None
    P, idx, YD, X, kaynak = tab
    zskor = pk.get("zskor", "ab")
    # EGITIMDEKI SUTUN SIRASI: [donusturulmus 92] + [kaynak gostergesi 3]
    Xd = np.hstack([p6_karar.donustur(X, zskor),
                    p6_karar.kaynak_blok(kaynak[idx])])
    s = np.asarray(pk["kademe1"].predict_proba(Xd)[:, 1], float)
    if pk.get("kademe2") is not None:
        # IKINCI KADEME = KISA LISTE UZERINDE FP REDDEDICI.
        # Birinci gecisin YUKSEK GUVENLI secimleri TOHUM olur, periyodik yapi
        # olculeri cikar; ikinci model yalniz `kisa_esik`i gecen secenekleri
        # yeniden puanlar ve birinci kademe skorunu da OZNITELIK olarak alir.
        # Kisa liste DISI satirlar 0 kalir -- ikinci kademe birinci kademeyi
        # EZEMEZ, yalniz icinden secer. Tohumlar yalniz tahminden gelir; GT bu
        # yola HIC girmez.
        import kafes
        Pt, Dt = p6_karar.sec_ayrintili(
            P, idx, YD, s, tuple(pk["tohum_kural"]),
            nms_mm=float(pk["tohum_nms"]))[:2]
        kb = kafes.oznitelik(P[idx], YD, Pt, Dt)
        k = np.where(s >= float(pk.get("kisa_esik", 0.20)))[0]
        s2 = np.zeros(len(s))
        if len(k):
            X2 = np.hstack([Xd[k], kb[k], s[k][:, None]])
            s2[k] = pk["kademe2"].predict_proba(X2.astype(np.float32))[:, 1]
        s = s2
    P2, D2 = p6_karar.sec(P, idx, YD, s, tuple(pk["kural"]),
                          nms_mm=float(pk["nms"]))
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
