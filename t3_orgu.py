# -*- coding: utf-8 -*-
"""T3: KUTUP ORGUSU OZELLIKLERI -- tahminden turetilen, GT'siz orgu.

Y6 BULGUSU: ureticinin LISTELEDIGI CP'lerin %70.1'i kutup orgusunun bir dugumunde,
yanlis pozitiflerin yalniz %0.4'u. Devasa bir ayrim. AMA y6 adimi GT'DEN cikariyordu ve
calisma aninda GT YOK.

BU KOL: adimi, gate'ten ONCE var olan sinyalden cikarir -- TOPLULUK OYU (`votes`) ve
segmentasyon guveni (`conf`). Yuksek oylu adaylar CAPA olur, RANSAC ile en cok capayi
aciklayan adim vektoru secilir, sonra HER aday o orguye gore puanlanir.

NEDEN OLEN "UZAMSAL DUZEN" BLOGUNUN TEKRARI DEGIL: o blok YEREL yogunluk/simetri
sayiyordu (ayna-esi, satir/sutun tutarliligi, merkezilik). Bu, KURESEL bir orgu MODELI
uydurup her adayin o modele UYUMUNU olcuyor. Farkli nesne.

OZELLIKLER (6):
    org_artik    en yakin orgu dugumune uzaklik (mm)
    org_tam      tamsayi-adim hatasi |t - round(t)|
    org_destek   kazanan orguyu destekleyen capa sayisi
    org_guven    RANSAC ic-nokta orani
    org_es       +-1 adim otede baska bir aday VAR mi (karsi es)
    org_k        kac adim otede (buyuk k = orgunun ucunda)

ONCE UCUZ PROB: yalniz olcum kumesinde (194 parca) grup-capraz aday duzeyi. Sinyal
varsa tam korpusa yatirim yapilir (korpus npz'sinde NOKTA YOK, yeniden uretim gerekir).
"""
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AD = ["org_artik", "org_tam", "org_destek", "org_guven", "org_es", "org_k"]
TOL = 1.5          # mm, orgu dugumune uyum toleransi
MIN_CAPA = 3


def orgu_uydur(A, tol=TOL):
    """RANSAC: capa noktalarini en cok aciklayan ADIM vektorunu bul.

    Doner: (adim_vektoru, destek_sayisi, ic_nokta_orani) veya (None, 0, 0.0).
    """
    n = len(A)
    if n < MIN_CAPA:
        return None, 0, 0.0
    aday = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = A[j] - A[i]
            L = float(np.linalg.norm(v))
            if 1.5 < L < 60.0:
                aday.append(v)
    if not aday:
        return None, 0, 0.0
    aday = np.array(aday)
    # kisa vektorleri once dene (kutup adimi en kisa tekrarli farktir)
    sira = np.argsort(np.linalg.norm(aday, axis=1))
    en_iyi, en_s, en_o = None, 0, 0.0
    for idx in sira[:min(60, len(sira))]:
        s = aday[idx]; L2 = float(s @ s)
        if L2 < 1e-9:
            continue
        # kac capa, BIR BASKA capadan tamsayi adim otede?
        ic = 0
        for a in range(n):
            uy = False
            for b in range(n):
                if a == b:
                    continue
                v = A[a] - A[b]
                t = float(v @ s) / L2
                k = round(t)
                if k != 0 and np.linalg.norm(v - k * s) <= tol:
                    uy = True; break
            ic += int(uy)
        if ic > en_s:
            en_iyi, en_s, en_o = s, ic, ic / n
    return en_iyi, en_s, en_o


def orgu_ozellik(P, capa_idx, s, destek, guven):
    """Her aday icin 6 orgu sutunu."""
    F = np.zeros((len(P), len(AD)))
    F[:, 0] = 99.0; F[:, 1] = 0.5
    if s is None or len(capa_idx) == 0:
        return F
    L2 = float(s @ s)
    A = P[capa_idx]
    for i in range(len(P)):
        en_r, en_t, en_k = 99.0, 0.5, 0
        for a in A:
            v = P[i] - a
            t = float(v @ s) / L2
            k = int(round(t))
            if k == 0 and np.linalg.norm(v) < 1e-6:
                en_r, en_t, en_k = 0.0, 0.0, 0
                break
            r = float(np.linalg.norm(v - k * s))
            if r < en_r:
                en_r, en_t, en_k = r, abs(t - k), k
        F[i, 0] = min(en_r, 99.0); F[i, 1] = en_t
        F[i, 2] = float(destek); F[i, 3] = float(guven); F[i, 5] = float(abs(en_k))
        # KARSI ES: +-1 adim otede baska bir ADAY var mi
        es = 0
        for sg in (+1, -1):
            hedef = P[i] + sg * s
            if len(P) > 1 and np.min(np.linalg.norm(P - hedef, axis=1)) <= TOL:
                es += 1
        F[i, 4] = float(es)
    return F


def main():
    import gate_tezgah as T
    import olcum_kumesi
    import wire_gate
    from sina_kume import esle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from gece_kilit import bekci

    bekci("t3 baslangic")
    D = T.yukle()
    olcum_kumesi.rapor_bas(D["rap"])
    AD_TUM = D["ad"]
    i_votes = AD_TUM.index("votes"); i_conf = AD_TUM.index("conf")

    RX, RY, RG, RP, ORG = [], [], [], [], []
    kapsam = 0
    for r in D["DER"]:
        if r["X"] is None or r.get("XR") is None or not len(r["G"]):
            continue
        X58 = np.hstack([r["X"], r["XR"]])
        P = np.asarray(r["P"], float)
        # CAPA: en yuksek oy + guven ust yarisi (gate SKORU KULLANILMAZ -> dongusellik yok)
        # CAPA = TUM ADAYLAR. Ilk surumde capa "en yuksek oylu adaylar" idi ve OLCULDU:
        # capa sayisinin medyani 2, parcalarin %75'inde 3'ten az -> orgu yalniz %25 parcada
        # kurulabildi, yani ozellik cogu yerde HIC HESAPLANMADI. Tum adaylarla kapsam %98.
        # RANSAC zaten aykiri degere dayaniklidir; elemek gereksiz. Dongusellik YOK:
        # aday konumlari gate SKORUNDAN once vardir.
        capa = np.arange(len(P))
        s, destek, guven = orgu_uydur(P[capa]) if len(capa) >= MIN_CAPA else (None, 0, 0.0)
        kapsam += int(s is not None)
        FO = orgu_ozellik(P, capa, s, destek, guven)
        # GT etiketi (tespit toleransi)
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        diff = P[:, None, :] - G[None, :, :]
        al = (diff * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
        pe = np.where(np.abs(al) > 40, np.inf, pe)
        tt = max(3.0, 0.06 * float(r["diag"]))
        yy = np.zeros(len(P), int); up, ug = set(), set()
        for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(P))
                                 for b in range(len(G))):
            if not np.isfinite(d_) or d_ > tt or a_ in up or b_ in ug:
                continue
            up.add(a_); ug.add(b_); yy[a_] = 1
        for i in range(len(P)):
            RX.append(X58[i]); ORG.append(FO[i]); RY.append(yy[i])
            RG.append(r["geo"]); RP.append(r["pid"])
    RX = np.array(RX); ORG = np.array(ORG); RY = np.array(RY)
    RG = np.array(RG); RP = np.array(RP)
    print(f"\n{len(RY)} aday | pozitif {RY.mean():.1%} | orgu bulunan parca "
          f"{kapsam}/{len(D['DER'])} ({kapsam/len(D['DER']):.0%})")

    from t1_uretici_disi import auc_mw
    print(f"\n{'sutun':<12}{'AUC':>8}{'TP ort':>10}{'FP ort':>10}")
    for j, a in enumerate(AD):
        v = ORG[:, j]
        print(f"{a:<12}{auc_mw(v, RY.astype(bool)):>8.3f}"
              f"{v[RY == 1].mean():>10.3f}{v[RY == 0].mean():>10.3f}")

    def olc(M):
        o = np.zeros(len(RY))
        for tr, te in GroupKFold(n_splits=5).split(M, RY, RG):
            o[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                           random_state=0).fit(M[tr], RY[tr]).predict_proba(M[te])[:, 1]
        m = np.zeros(len(o), bool)
        for u in np.unique(RP):
            i = RP == u; vv = o[i]
            m[i] = (vv >= 0.5 * max(vv.max(), 1e-9)) & (vv >= 0.25)
        tp = int((RY.astype(bool) & m).sum()); fp = int((~RY.astype(bool) & m).sum())
        fn = int((RY.astype(bool) & ~m).sum())
        p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
        return 2 * p_ * r_ / max(p_ + r_, 1e-9)
    a58 = olc(RX); a64 = olc(np.hstack([RX, ORG]))
    print(f"\nADAY DUZEYI: 58 sutun {a58:.4f} | 58+orgu {a64:.4f} | fark {a64-a58:+.4f}")
    gecti = (a64 - a58) >= 0.01
    print(f"KARAR: {'SINYAL VAR -> tam korpusa yatirim' if gecti else 'SINYAL YOK -> T3 KAPANIR'}")
    with open("results/t3_orgu.pkl", "wb") as f:
        pickle.dump({"ORG": ORG, "RG": RG, "RP": RP, "RY": RY, "AD": AD}, f)
    with io.open("results/t3_orgu.json", "w", encoding="utf-8") as f:
        json.dump({"aday_58": float(a58), "aday_64": float(a64), "fark": float(a64 - a58),
                   "kapsam": kapsam / len(D["DER"]), "gecti": bool(gecti),
                   "auc": {a: float(auc_mw(ORG[:, j], RY.astype(bool)))
                           for j, a in enumerate(AD)}}, f, indent=1)
    print("makbuz -> results/t3_orgu.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
