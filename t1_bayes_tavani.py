# -*- coding: utf-8 -*-
"""T1: BAYES TAVANINI YUKSELT -- "ayrilamaz" adaylari FIZIK ayiriyor mu?

TAVAN NEDIR: [[geometrik-bayes-tavani]] -- 73 sutunlu oznitelik uzayinda parca-disi
15-NN uyusmazligi %28.2, adaylarin **%32.5'i AYRILAMAZ** (neredeyse ozdes oznitelik
vektorune sahip bir TP ile bir FP var). Bayes hatasi ~%14.1 -> tavan F1 ~0.859.
Hedef 0.85 TAM TAVANDA; yani gate'i ne kadar iyilestirirsek iyilestirelim orasi son.

TAVAN MODELIN DEGIL OZNITELIK UZAYININ ozelligidir. Yukseltmenin TEK yolu, su an
ayrilamayan ciftleri ayiran YENI BILGI eklemektir. Bu betik dogru soruyu sorar:

    "ayrilamaz" denen ciftleri FIZIKSEL olcumler ayiriyor mu?

A4'te 194 parcanin 1186 CP'si icin fiziksel bayraklar zaten hesaplandi (govde ici /
onu kapali / agiz ic capi / ileri serbest mesafe) ve aletler A0'da dogrulandi
(is_inside %0.0 yanilma, mouth_width %1 sapma). Yani soru BEDAVAYA sinanabilir.

OLCUM:
  1. Mevcut oznitelik uzayinda (X) parca-disi k-NN uyusmazligi ve ayrilamaz pay
  2. X + FIZIKSEL olcumler ile AYNI hesap
  3. Fark = fizigin tavana katkisi

GO (F3-16'nin sarti): ayrilamaz pay >= %20 GORELI azalmali.
NOT: bu bir TAVAN olcumudur, uctan uca kazanc DEGIL. Tavan yukselirse F3 kollari
anlamli olur; yukselmezse o kollar tavana carpar ve bosa gider.
"""
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
import wire_gate

K = 15


def ayrilamaz_pay(X, y, grup, k=K):
    """Parca-DISI k-NN uyusmazligi ve 'ayrilamaz' pay.

    AYRILAMAZ = komsularinin en az yarisi TERS etiketli aday. Bu, hicbir modelin
    ayiramayacagi bolgedir (ayni yerde hem TP hem FP var).
    """
    from sklearn.neighbors import NearestNeighbors
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=min(k * 4 + 1, len(Xs)), n_jobs=2).fit(Xs)
    _, idx = nn.kneighbors(Xs)
    uyus = np.zeros(len(Xs)); ayr = np.zeros(len(Xs), bool)
    for i in range(len(Xs)):
        # PARCA-DISI: ayni parcadaki komsular ATILIR (ayni parca kolay eslesir, sisirir)
        komsu = [j for j in idx[i][1:] if grup[j] != grup[i]][:k]
        if len(komsu) < 3:
            continue
        ters = np.mean(y[komsu] != y[i])
        uyus[i] = ters
        ayr[i] = ters >= 0.5
    return float(uyus.mean()), float(ayr.mean())


def main():
    import protokol
    protokol.tez_dogrula()
    DER, gate, ek = T2.yukle()
    with open("results/_a4_bayraklar.pkl", "rb") as f:
        BAY = pickle.load(f)

    X0, FZ, Y, GR = [], [], [], []
    for r in DER:
        if r["X"] is None or r.get("XR") is None:
            continue
        M = np.hstack([r["X"], r["XR"]])
        if M.shape[1] * 2 != gate["n_feat"]:
            continue
        k = wire_gate.karar_maskesi(wire_gate.karar_skoru(gate, M))
        bl = BAY.get(r["pid"]) or []
        if not k.any() or len(bl) != int(k.sum()):
            continue
        P = np.asarray(r["P"], float)[k]
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        if len(G):
            d = P[:, None, :] - G[None, :, :]
            al = (d * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(d - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) > 40, np.inf, pe)
            iyi = (pe.min(1) <= max(3.0, 0.06 * float(r["diag"]))).astype(int)
        else:
            iyi = np.zeros(len(P), int)
        Mk = M[k]
        for i in range(len(P)):
            b = bl[i]
            il = float(b["ileri"]) if np.isfinite(b["ileri"]) else 60.0
            cp = float(b["ic_cap"]) if np.isfinite(b["ic_cap"]) else -1.0
            X0.append(Mk[i])
            FZ.append([1.0 if b["govde_ici"] else 0.0, min(il, 60.0), cp,
                       1.0 if il < 5.0 else 0.0, 1.0 if 0 < cp < 0.8 else 0.0])
            Y.append(iyi[i]); GR.append(r["geo"])
    X0 = np.array(X0, float); FZ = np.array(FZ, float)
    Y = np.array(Y); GR = np.array(GR)
    X1 = np.hstack([X0, FZ])
    print(f"aday {len(Y)} | iyi %{100*Y.mean():.1f} | sutun {X0.shape[1]} -> {X1.shape[1]}")

    print(f"\n{'oznitelik uzayi':<26}{'kNN uyusmazlik':>16}{'AYRILAMAZ pay':>16}{'tavan F1~':>11}")
    S = {}
    for ad, M in (("MEVCUT (gate ozn.)", X0), ("+ FIZIKSEL olcumler", X1),
                  ("YALNIZ fiziksel", FZ)):
        u, a = ayrilamaz_pay(M, Y, GR)
        tav = 1 - u / 2          # kaba: uyusmazligin yarisi duzeltilemez hata
        S[ad] = {"uyusmazlik": u, "ayrilamaz": a, "tavan": tav}
        print(f"{ad:<26}{100*u:>15.1f}%{100*a:>15.1f}%{tav:>11.4f}")

    a0 = S["MEVCUT (gate ozn.)"]["ayrilamaz"]
    a1 = S["+ FIZIKSEL olcumler"]["ayrilamaz"]
    gor = (a0 - a1) / max(a0, 1e-9)
    print(f"\nAYRILAMAZ PAY: %{100*a0:.1f} -> %{100*a1:.1f}  (GORELI azalma %{100*gor:.1f})")
    gecti = gor >= 0.20
    print(f"GO (F3-16 sarti: >=%20 goreli azalma) -> {'GECTI -- fizik TAVANI YUKSELTIYOR' if gecti else 'GECMEDI'}")
    if not gecti:
        print("  -> fiziksel olcumler tavani oynatmiyor; F3 kollari BASKA bilgi getirmeli")
    with io.open("results/t1_bayes_tavani.json", "w", encoding="utf-8") as f:
        json.dump({"k": K, "n": int(len(Y)), "sonuc": S,
                   "goreli_azalma": gor, "gecti": bool(gecti)}, f, indent=1)
    print("makbuz -> results/t1_bayes_tavani.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
