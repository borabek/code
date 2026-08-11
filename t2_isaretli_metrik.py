# -*- coding: utf-8 -*-
"""T2: FIZIKSEL (ISARETLI) ROBOT METRIGI -- yon sozlesmesini olc ve ayir.

BULGU (dogrulandi): `sina_kume.esle` aciyi `abs(Pd . Gd)` ile oluyor -> 180 derece TERS bir
tahmin 0 derece sayiliyor. Ayrica robot_cp docstring'i yonu "telin girdigi yon" diye
tanimliyor ama OLCULDU: aday yonleri govdeden DISARI bakiyor, ureticinin InsertDirection'i
ICERI. Yani belge ile veri celisiyor ve metrik bunu goremiyor.

BU BETIK UC METRIGI AYIRIR:
    eksen-hazir      yanal<=2mm VE aci<=10 derece, ISARETSIZ  (bugune kadar raporlanan)
    takma-hazir(-)   ayni ama ISARETLI, takma_yonu = -disari_normal
    takma-hazir(+)   ayni ama ISARETLI, takma_yonu = +disari_normal

OKUMA:
  takma(-) ~ eksen-hazir  -> sozlesme "-disari" DOGRU; metrik zaten fiziksel, yalniz
                             ilan edilmemis. Duzeltme = alan adlandirmasi.
  takma(-) COKERSE        -> yonler aday BASINA tutarsiz; robot hedefi yeniden tabanlanmali.

Hicbir model degismez; bu bir OLCUM ayristirmasidir.
"""
import io
import json

import numpy as np

import gate_tezgah as T


def esle_isaretli(P, Pd, G, Gd, diag, lm, am, isaret):
    """sina_kume.esle ile AYNI greedy, ama aci ISARETLI olcülür.

    isaret = 0  -> abs (mevcut davranis)
    isaret = -1 -> takma yonu = -Pd  (disari normalin tersi)
    isaret = +1 -> takma yonu = +Pd
    """
    if not len(P) or not len(G):
        return 0, len(P), len(G)
    tol = lm if lm > 0 else max(3.0, 0.06 * float(diag))
    diff = P[:, None, :] - G[None, :, :]
    al = (diff * Gd[None, :, :]).sum(-1)
    pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
    D = Pd if isaret >= 0 else -Pd
    c = D @ Gd.T
    an = np.degrees(np.arccos(np.clip(np.abs(c) if isaret == 0 else c, -1, 1)))
    pe = np.where((np.abs(al) > 40) | (an > am), np.inf, pe)
    up, ug, tp = set(), set(), 0
    for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(P)) for b in range(len(G))):
        if not np.isfinite(d_) or d_ > tol or a_ in up or b_ in ug:
            continue
        up.add(a_); ug.add(b_); tp += 1
    return tp, len(P) - tp, len(G) - tp


def main():
    import olcum_kumesi
    import wire_gate
    from sklearn.ensemble import RandomForestClassifier

    D = T.yukle()
    olcum_kumesi.rapor_bas(D["rap"])
    cfg = D["cfg"]
    X, y, pid, keep = D["X"], D["y"], D["pid"], D["keep"]
    Z = np.zeros((len(X), X.shape[1] * 2))
    for u in np.unique(pid):
        i = np.where(pid == u)[0]
        Z[i] = wire_gate.parca_ici(X[i], D["donusum"])
    gate = {"clf": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                          random_state=0).fit(Z[keep], y[keep]),
            "n_feat": Z.shape[1], "donusum": D["donusum"]}

    KOL = {"eksen-hazir (isaretsiz)": 0, "takma-hazir (-disari)": -1, "takma-hazir (+disari)": +1}
    det = {k: [] for k in KOL}
    ISARET = []          # aday basina: -Pd mi +Pd mi GT ile ayni yone bakiyor
    for r in D["DER"]:
        P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
        if r["X"] is not None and r.get("XR") is not None:
            Xr = np.hstack([r["X"], r["XR"]])
            k = wire_gate.karar_maskesi(wire_gate.karar_skoru(gate, Xr))
            if k.any():
                P = r["P"][k].copy(); Pd = r["Pd"][k].copy()
                if cfg.get("robot_pose_head"):
                    c = [{"point": P[i], "direction": Pd[i]} for i in range(len(P))]
                    c = wire_gate.pose_duzelt(Xr[k], c)
                    if cfg.get("robot_aci_secici"):
                        c = wire_gate.aci_duzelt(Xr[k], c)
                    if cfg.get("robot_uye_secici") and r.get("UYE"):
                        c = wire_gate.uye_yonu_sec(Xr[k], c, r["UYE"])
                    P = np.array([x["point"] for x in c], float)
                    Pd = np.array([x["direction"] for x in c], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        for ad, s in KOL.items():
            det[ad].append((rj,) + esle_isaretli(P, Pd, G, Gd, r["diag"], 2.0, 10.0, s))
        # ISARET SAYIMI: eslesen ciftlerde -Pd mi +Pd mi GT'ye yakin
        if len(P) and len(G):
            diff = P[:, None, :] - G[None, :, :]
            al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) > 40, np.inf, pe)
            up, ug = set(), set()
            for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(P))
                                     for b in range(len(G))):
                if not np.isfinite(d_) or d_ > 2.0 or a_ in up or b_ in ug:
                    continue
                up.add(a_); ug.add(b_)
                ISARET.append(float(Pd[a_] @ Gd[b_]))

    print(f"\n{'kol':<26}{'robot F1':>10}")
    S = {}
    for ad in KOL:
        S[ad] = T.f1w(det[ad])
        print(f"{ad:<26}{S[ad]:>10.4f}")

    I = np.array(ISARET)
    print(f"\nISARET DAGILIMI ({len(I)} eslesen cift, Pd . Gd):")
    print(f"  NEGATIF (Pd disari, Gd iceri -> takma = -Pd): {float((I < 0).mean()):.1%}")
    print(f"  POZITIF (ayni yone bakiyor)                 : {float((I > 0).mean()):.1%}")
    print(f"  medyan {np.median(I):+.3f}")

    e = S["eksen-hazir (isaretsiz)"]
    m = S["takma-hazir (-disari)"]; p = S["takma-hazir (+disari)"]
    en_iyi = "-disari" if m >= p else "+disari"
    kayip = e - max(m, p)
    print(f"\nHUKUM: dogru sozlesme takma_yonu = {en_iyi}")
    print(f"  isaretli metrik {max(m,p):.4f} vs isaretsiz {e:.4f} -> kayip {kayip:+.4f}")
    if kayip <= 0.01:
        print("  -> Sozlesme TUTARLI. Metrik zaten fiziksel; eksik olan yalniz ILAN.")
    else:
        print("  -> Yonler aday BASINA TUTARSIZ. Robot hedefi isaretli metrikle YENIDEN tabanlanmali.")
    with io.open("results/t2_isaretli.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in S.items()} |
                  {"negatif_pay": float((I < 0).mean()), "medyan_dot": float(np.median(I)),
                   "dogru_sozlesme": en_iyi, "kayip": float(kayip)}, f, indent=1)
    print("makbuz -> results/t2_isaretli.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
