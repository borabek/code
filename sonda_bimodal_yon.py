# -*- coding: utf-8 -*-
"""YENI-7 -- IKI YONLU (BIMODAL) YON + DISARI ISARETI

TESHIS. NIT'te yon KAHINI 0.593, gerceklesen 0.150. Bugune kadarki en buyuk
tek bosluk. Iki olculmus basarisizlik ayni sebebe isaret ediyor:
  * K2.1 periyodiklik : tespit +0.0126 ama robot -0.0100  (yon KOPYALAMA)
  * dik_kipsel        : 0.064, dik_suzgec 0.166'nin ALTINDA (modal oylama)
Ikisi de "parcadaki yonu kopyala" fikrini denedi, ikisi de coktu.

HIPOTEZ. Bir klemens blogunda girisler TEK yonlu degildir: giris ve cikis
karsit yuzlerdedir. Isaretli aci metriginde 180 derece = TAM basarisizlik.
Modal oylama tek kumeyi secer, parcanin diger yarisi ters isaretlenir.
Yani kopyalama fikri yanlis degildi; ISARET ele alinmamisti.

COZUM. Yonu ISARETSIZ EKSEN olarak sec (+/- birlesir), isareti AYRICA
"govdeden disari" kosulundan koy. GT yon sozlesmesi denetimi disariligin
1.000 tutarli oldugunu olctu -- yani isaret ogrenilecek degil, TURETILECEK
bir buyukluktur.

ONCE DOGRULAMA: GT yonlerinin parca ici kip sayisi ISARETLI ve ISARETSIZ
olarak ayri sayilir. Isaretsizde belirgin sekilde azaliyorsa hipotez ayakta.

KOLLAR (hepsi GT KONUMUNDA, yani yon TEK BASINA yalitilir):
  bugunku       : en yuksek skorlu adayin yonu
  eksen_skor    : yerel isaretsiz modal eksen, isaret skordan
  eksen_disari  : yerel isaretsiz modal eksen, isaret DISARIDAN (merkez)
  eksen_dis_yer : ayni, ama yerel (15mm) merkeze gore disari
  parca_eksen   : PARCA capinda tek isaretsiz eksen + disari isareti
  kahin         : dogru yon adaylar arasinda VAR mi

KAPI: NIT'te herhangi bir kol `bugunku`yu >= 0.05 asacak.
D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_tam4")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402

KUME = os.environ.get("BY_KUME", "d6")
KAT_MIN = int(os.environ.get("BY_KAT_MIN", "40"))
ITER = int(os.environ.get("P6_ITER", "200"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "6"))
YAKIN_R = float(os.environ.get("BY_R", "2.0"))
EKSENEL = float(os.environ.get("BY_EKSENEL", "40.0"))
YEREL_R = float(os.environ.get("BY_YEREL", "15.0"))
MESH = {"d6": "results/_p1_olasilik",
        "tam": "results/_p1_olasilik_brepegit"}[KUME]
KOLLAR = ("bugunku", "eksen_skor", "eksen_disari", "eksen_dis_yer",
          "parca_eksen", "kahin")


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def temel(d):
    return np.hstack([p6_karar.donustur(d["X"]),
                      p6_karar.kaynak_blok(d["kaynak"][d["idx"]])]).astype(
                          np.float32)


def eksen_modal(Y, W=None):
    """ISARETSIZ modal eksen: +v ve -v AYNI sayilir.

    Oy = |cos| esigi uzerindeki komsu sayisi. Kazanani, kendisine hizali
    olanlarin isaret DUZELTILMIS ortalamasiyla inceltiriz.
    """
    if len(Y) == 1:
        return Y[0]
    c = np.abs(np.clip(Y @ Y.T, -1, 1))
    oy = (np.degrees(np.arccos(c)) <= K.ACI)
    a = oy.sum(1) if W is None else (oy * W[None, :]).sum(1)
    j = int(np.argmax(a))
    uy = Y[oy[j]]
    isaret = np.sign(uy @ Y[j])
    isaret[isaret == 0] = 1.0
    v = (uy * isaret[:, None]).mean(0)
    n = np.linalg.norm(v)
    return Y[j] if n < 1e-9 else v / n


def kip_sayisi(Y, isaretli):
    """acgozlu kumeleme ile kip sayisi (ACI toleransinda)."""
    kalan = list(range(len(Y)))
    n = 0
    while kalan:
        j = kalan[0]
        c = Y[kalan] @ Y[j]
        c = np.clip(c if isaretli else np.abs(c), -1, 1)
        yakin = np.degrees(np.arccos(c)) <= K.ACI
        kalan = [k for k, y in zip(kalan, yakin) if not y]
        n += 1
    return n


def main():
    t0 = time.time()
    veri = yukle(KUME, int(os.environ.get("P6_TR", "0")))
    for d in veri:
        d["y"] = np.asarray(d["y"], int)
        d["_M"] = temel(d)
    marka = collections.Counter(d["mfg"] for d in veri)
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"{len(veri)} parca | katlar {katlar}", flush=True)

    oof = [None] * len(veri)
    for b in katlar:
        ic = [i for i, d in enumerate(veri) if d["mfg"] != b]
        dis = [i for i, d in enumerate(veri) if d["mfg"] == b]
        n_s = sum(len(veri[i]["y"]) for i in ic)
        M = np.empty((n_s, veri[0]["_M"].shape[1]), np.float32)
        o = 0
        for i in ic:
            m_ = veri[i]["_M"]
            M[o:o + len(m_)] = m_
            o += len(m_)
        Y = np.concatenate([veri[i]["y"] for i in ic])
        rng = np.random.default_rng(0)
        poz, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]
        sec = np.concatenate([poz, rng.choice(
            neg, min(len(neg), NEG_KAT * max(len(poz), 1)), replace=False)])
        m = HistGradientBoostingClassifier(
            max_iter=ITER, learning_rate=0.06, max_leaf_nodes=63,
            l2_regularization=1.0, random_state=0).fit(M[sec], Y[sec])
        del M
        for i in dis:
            oof[i] = m.predict_proba(veri[i]["_M"])[:, 1]
        print(f"  OOF {b} ({time.time() - t0:.0f} s)", flush=True)

    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for d, s in zip(veri, oof):
        if s is None:
            continue
        G = np.asarray(d["G"], float)
        if len(G) < 2:
            continue
        Gn = _birim(np.asarray(d["Gd"], float))
        P = np.asarray(d["P"], float)
        idx = np.asarray(d["idx"], int)
        YD = _birim(np.asarray(d["YD"], float))
        s = np.asarray(s, float)

        # --- DOGRULAMA: GT yonlerinin kip sayisi
        a = ist[d["mfg"]]
        a["kip_imzali"].append(kip_sayisi(Gn, True))
        a["kip_eksen"].append(kip_sayisi(Gn, False))

        mf = f"{MESH}/{d['pid']}.npz"
        V = (np.asarray(np.load(mf)["V"], float)
             if os.path.exists(mf) else P)
        merkez = V.mean(0)

        # PARCA capinda tek isaretsiz eksen (ust skorlu adaylardan)
        ust = np.argsort(-s)[:max(30, 2 * len(G))]
        p_eksen = eksen_modal(YD[ust], s[ust])

        say = {k: 0 for k in KOLLAR}
        # ESLESME KUTUSU DUZELTMESI. Ilk kosuda adaylari GT'ye OKLID 2mm ile
        # esledim; oysa kabul kutusu CARPIMDIR: GT yonune gore yanal <= 2mm
        # ve eksenel <= 40mm. Bir aday eksende 30mm uzakta olup hala gecerli
        # eslesme olabilir. Oklid kullanmak `kahin`i 0.593'ten 0.0172'ye
        # dusuruyordu -- olculen sey mekanizma degil KUSURDU.
        Pk = P[idx]
        for j in range(len(G)):
            v = Pk - G[j][None, :]
            al = v @ Gn[j]
            yan = np.linalg.norm(v - al[:, None] * Gn[j][None, :], axis=1)
            m_ = (yan <= K.YANAL) & (np.abs(al) <= EKSENEL)
            if not m_.any():
                continue
            Yo, So = YD[m_], s[m_]
            aci = np.degrees(np.arccos(np.clip(Yo @ Gn[j], -1, 1)))
            if (aci <= K.ACI).any():
                say["kahin"] += 1
            if aci[int(np.argmax(So))] <= K.ACI:
                say["bugunku"] += 1

            e = eksen_modal(Yo, So)
            # KONUM SIZINTISI DUZELTMESI. Ilk kosuda `konum = G[j]`, yani
            # DISARI testi GT KONUMUNU kullaniyordu. Uruende elimizde yalniz
            # ADAY konumu var ve aday eksende 40mm'ye kadar kayabilir --
            # `e @ (konum - merkez)` isareti bu yuzden ters donebilir.
            # Dogrusu: o GT ile eslesen adaylarin EN YUKSEK SKORLUSUNUN
            # kendi konumu (uruende de bu bilinir).
            konum = Pk[m_][int(np.argmax(So))]
            # isaret: skor agirlikli izdusum
            pr = (So * (Yo @ e)).sum()
            v1 = e * (1.0 if pr >= 0 else -1.0)
            # isaret: kuresel merkezden disari
            r = konum - merkez
            v2 = e * (1.0 if (e @ r) >= 0 else -1.0)
            # isaret: YEREL merkezden disari
            yak = V[np.linalg.norm(V - konum, axis=1) <= YEREL_R]
            ry = konum - (yak.mean(0) if len(yak) >= 10 else merkez)
            v3 = e * (1.0 if (e @ ry) >= 0 else -1.0)
            # PARCA eksen + kuresel disari
            v4 = p_eksen * (1.0 if (p_eksen @ r) >= 0 else -1.0)
            for ad, v in (("eksen_skor", v1), ("eksen_disari", v2),
                          ("eksen_dis_yer", v3), ("parca_eksen", v4)):
                if np.degrees(np.arccos(
                        np.clip(float(v @ Gn[j]), -1, 1))) <= K.ACI:
                    say[ad] += 1
        a["gt"].append(len(G))
        for k_, v_ in say.items():
            a[k_].append(v_)
        n += 1
        if n % 80 == 0:
            print(f"  {n} parca ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{n} parca | GT KONUMUNDA yon dogrulugu (yon YALITILDI)")
    print(f"{'marka':<7}{'GT':>6}{'kip+-':>7}{'kipEks':>8}"
          + "".join(f"{k:>14}" for k in KOLLAR))
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = max(sum(a["gt"]), 1)
        r = {k: sum(a[k]) / g for k in KOLLAR}
        r["gt"] = g
        r["kip_imzali"] = float(np.median(a["kip_imzali"]))
        r["kip_eksen"] = float(np.median(a["kip_eksen"]))
        out[m_] = r
        print(f"{m_:<7}{g:>6}{r['kip_imzali']:>7.1f}{r['kip_eksen']:>8.1f}"
              + "".join(f"{r[k]:>14.4f}" for k in KOLLAR))
    print("\n=== BUGUNKUYE GORE (yogun marka NIT) ===")
    if "NIT" in out:
        h = out["NIT"]["bugunku"]
        for k_ in KOLLAR[1:-1]:
            f = out["NIT"][k_] - h
            print(f"  {k_:<14}{out['NIT'][k_]:.4f}   {f:+.4f}"
                  + ("  <- KAPI GECTI" if f >= 0.05 else ""))
        print(f"  {'kahin':<14}{out['NIT']['kahin']:.4f}   (ust sinir)")
    json.dump({"damga": makbuz_hash.damga(), "kume": KUME, "marka": out,
               "not": "Isaretsiz eksen + disari isareti. GT KONUMUNDA, yon "
                      "yalitildi. kip+- = GT yonlerinin isaretli kip sayisi, "
                      "kipEks = isaretsiz. D7'ye BAKILMADI."},
              open(f"results/bimodal_yon_{KUME}.json", "w"), indent=1)
    print(f"makbuz -> results/bimodal_yon_{KUME}.json")


if __name__ == "__main__":
    main()
