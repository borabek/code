# -*- coding: utf-8 -*-
"""T: IKI METRIGIN TAVAN MERDIVENI -- hedef 0.90 tespit / 0.75 robot NEREDE kilitli?

Tavani yukseltmek icin once NEREDE oldugunu ve NEYIN kilitledigini bilmek gerekir. Bu betik
her iki metrik icin ayni merdiveni kurar ve her basamakta KIM sinirliyor onu gosterir:

  0. SU AN                 : dagitilan urun (gate + goreli esik + cokus yonlendirme)
  1. + KAHIN GATE          : adaylar ayni, ama GT ile eslesenleri SECEBILSEYDIK (mukemmel karar)
  2. + KAHIN YON           : eslesenlerin acisi 0 olsaydi
  3. + KAHIN KONUM         : eslesenlerin yanal hatasi 0 olsaydi
  4. ADAY TAVANI           : bir GT'nin YAKININDA hic aday var mi? (turetmenin recall tavani)

Merdivenin mantigi: 1. basamak GATE'in, 4. basamak TURETMENIN (ag + cp_openings) tavanidir.
Ikisi arasindaki fark, gate'i mukemmellestirerek kazanilabilecek her seydir.

ONEMLI: kahin sayilari ULASILABILIR HEDEF DEGIL, UST SINIRDIR. Ayni veri uzerinde olculdugu
icin iyimserdir. Ama bir hedefin (0.90 / 0.75) hangi basamagin uzerinde oldugunu SOYLER --
tavanin altindaysa calisilabilir, ustundeyse once tavan yukseltilmelidir.
"""
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _esle_sayilari(P, Pd, G, Gd, diag, tol, am, pct):
    """sina_kume.esle ile AYNI mantik, ama esleme ciftlerini de dondurur."""
    hit = np.zeros(len(G), bool); used = set(); ciftler = []
    if len(P) and len(G):
        diff = P[:, None, :] - G[None, :, :]
        al = (diff * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
        an = np.degrees(np.arccos(np.clip(np.abs(Pd @ Gd.T), 0, 1)))
        tt = max(3.0, 0.06 * diag) if pct else tol
        pe2 = np.where((np.abs(al) > 40) | (an > am), np.inf, pe)
        for d_, a_, b_ in sorted((pe2[a, b], a, b)
                                 for a in range(len(P)) for b in range(len(G))):
            if d_ > tt or a_ in used or hit[b_]:
                continue
            hit[b_] = True; used.add(a_); ciftler.append((a_, b_))
    tp = int(hit.sum())
    return tp, len(P) - tp, len(G) - tp, ciftler


def main():
    import wire_gate
    from sina_kume import f1w
    from sklearn.ensemble import RandomForestClassifier

    with open("results/_u4_der.pkl", "rb") as f:
        DER = pickle.load(f)
    with open("results/_dev_val_kume.json", encoding="utf-8") as f:
        kume_of = json.load(f)
    d = np.load("results/gate_regrow_data_topo.npz", allow_pickle=True)
    with open("results/_strict_geometry_keys.json", encoding="utf-8") as f:
        gk = json.load(f)
    tr_pid = np.array([str(x) for x in d["pids"]])
    tr_grp = np.array([gk.get(p, "yok:" + p) for p in tr_pid])
    Xtr = np.asarray(d["X"], float); ytr = np.asarray(d["y"])
    keep = ~np.isin(tr_grp, list({gk.get(r["pid"], "yok:" + r["pid"]) for r in DER}))
    dag = wire_gate._load(wire_gate.MODEL_PATH)
    rf = lambda M: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                          random_state=0).fit(M[keep], ytr[keep])
    Ztr = np.zeros((len(Xtr), Xtr.shape[1] * 2))
    for u in np.unique(tr_pid):
        i = np.where(tr_pid == u)[0]
        Ztr[i] = wire_gate.parca_ici(Xtr[i], dag.get("donusum_z", "zskor"))
    m = {"clf": rf(Xtr), "clf_z": rf(Ztr), "n_feat": Xtr.shape[1],
         "donusum": dag.get("donusum"), "donusum_z": dag.get("donusum_z", "zskor")}
    mx = [float(m["clf"].predict_proba(Xtr[np.where((tr_pid == u) & keep)[0]])[:, 1].max())
          for u in np.unique(tr_pid[keep]) if ((tr_pid == u) & keep).any()]
    m["esik_cokus"] = float(np.quantile(mx, dag.get("yonlendirme_q", 0.10)))
    print(f"gate kuruldu (sizintisiz) | cokus esigi {m['esik_cokus']:.4f}\n", flush=True)

    BAS = ["0 SU AN", "1 +kahin GATE", "2 +kahin YON", "3 +kahin KONUM", "4 ADAY TAVANI"]
    det = {b: [] for b in BAS}; rob = {b: [] for b in BAS}
    for r in DER:
        rj = "cok" if r["n"] >= 8 else "dusuk"
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        Pt = r["P"] if r["X"] is not None else np.zeros((0, 3))
        Pdt = r["Pd"] if r["X"] is not None else np.zeros((0, 3))
        diag = float(r["diag"])

        # 0 SU AN: urunun karar yolu
        P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
        if r["X"] is not None:
            s = wire_gate.karar_skoru(m, r["X"])
            k = wire_gate.karar_maskesi(s)
            if k.any():
                P, Pd = Pt[k], Pdt[k]
        det["0 SU AN"].append((rj,) + _esle_sayilari(P, Pd, G, Gd, diag, 0.0, 180.0, True)[:3])
        rob["0 SU AN"].append((rj,) + _esle_sayilari(P, Pd, G, Gd, diag, 2.0, 10.0, False)[:3])

        # 1 KAHIN GATE: TUM adaylar icinde GT ile eslesenleri sec (mukemmel karar)
        tp, fp, fn, cift = _esle_sayilari(Pt, Pdt, G, Gd, diag, 0.0, 180.0, True)
        sec = np.zeros(len(Pt), bool)
        for a_, _ in cift:
            sec[a_] = True
        Pk, Pdk = Pt[sec], Pdt[sec]
        det["1 +kahin GATE"].append((rj,) + _esle_sayilari(Pk, Pdk, G, Gd, diag, 0.0, 180.0, True)[:3])
        rob["1 +kahin GATE"].append((rj,) + _esle_sayilari(Pk, Pdk, G, Gd, diag, 2.0, 10.0, False)[:3])

        # 2 +kahin YON: eslesen adaylarin yonu GT yonu yapilir
        Pd2 = Pdk.copy()
        for i2, (a_, b_) in enumerate(cift):
            Pd2[i2] = Gd[b_]
        det["2 +kahin YON"].append((rj,) + _esle_sayilari(Pk, Pd2, G, Gd, diag, 0.0, 180.0, True)[:3])
        rob["2 +kahin YON"].append((rj,) + _esle_sayilari(Pk, Pd2, G, Gd, diag, 2.0, 10.0, False)[:3])

        # 3 +kahin KONUM: eslesen adaylarin yanal sapmasi da sifirlanir (GT noktasina tasinir)
        P3 = Pk.copy()
        for i3, (a_, b_) in enumerate(cift):
            P3[i3] = G[b_]
        det["3 +kahin KONUM"].append((rj,) + _esle_sayilari(P3, Pd2, G, Gd, diag, 0.0, 180.0, True)[:3])
        rob["3 +kahin KONUM"].append((rj,) + _esle_sayilari(P3, Pd2, G, Gd, diag, 2.0, 10.0, False)[:3])

        # 4 ADAY TAVANI: her GT icin bir aday VARSA sayilir (turetmenin recall tavani)
        n_es = len(cift)
        det["4 ADAY TAVANI"].append((rj, n_es, 0, len(G) - n_es))
        rob["4 ADAY TAVANI"].append((rj, n_es, 0, len(G) - n_es))

    print(f"{'basamak':<18}{'TESPIT F1':>11}{'ROBOT F1':>11}{'tespit fark':>13}{'robot fark':>12}")
    onc_d = onc_r = None
    OUT = {}
    for b in BAS:
        fd, fr = f1w(det[b]), f1w(rob[b])
        print(f"{b:<18}{fd:>11.4f}{fr:>11.4f}"
              f"{('' if onc_d is None else f'{fd-onc_d:+.4f}'):>13}"
              f"{('' if onc_r is None else f'{fr-onc_r:+.4f}'):>12}")
        OUT[b] = {"tespit": float(fd), "robot": float(fr)}
        onc_d, onc_r = fd, fr

    su_d = OUT["0 SU AN"]["tespit"]; su_r = OUT["0 SU AN"]["robot"]
    kg_d = OUT["1 +kahin GATE"]["tespit"]; kg_r = OUT["1 +kahin GATE"]["robot"]
    at_d = OUT["4 ADAY TAVANI"]["tespit"]
    print(f"\nHEDEFLERE GORE:")
    print(f"  TESPIT 0.90 : aday tavani {at_d:.4f} -> "
          f"{'MUMKUN (tavan ustunde)' if at_d >= 0.90 else 'TAVANIN USTUNDE -- once TURETME recall''i artmali'}")
    print(f"                kahin GATE {kg_d:.4f} -> gate'i mukemmellestirmek "
          f"{'0.90 icin YETER' if kg_d >= 0.90 else 'YETMEZ'}")
    print(f"  ROBOT  0.75 : kahin yon+konum {OUT['3 +kahin KONUM']['robot']:.4f} -> "
          f"{'MUMKUN' if OUT['3 +kahin KONUM']['robot'] >= 0.75 else 'TAVANIN USTUNDE'}")
    print(f"\nNEREDE KILITLI:")
    print(f"  gate'i mukemmellestirme kazanci : tespit {kg_d-su_d:+.4f} | robot {kg_r-su_r:+.4f}")
    print(f"  yon+konumu mukemmellestirme     : robot {OUT['3 +kahin KONUM']['robot']-kg_r:+.4f}")
    print(f"  turetme recall tavani           : tespit {at_d:.4f} (bunun ustune CIKILAMAZ)")
    with open("results/t_tavan.json", "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=1)
    print("\nmakbuz -> results/t_tavan.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
