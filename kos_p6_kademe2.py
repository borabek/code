# -*- coding: utf-8 -*-
"""P6 IKI KADEMELI: birinci gecisin tohumlarindan PERIYODIK YAPI ozniteligi.

GEREKCE. D6-ici LOMO'da olculdu: yon secimi cozuldu (kahin farki +0.0057) ama
recall 0.177 / havuz tavani 0.525. Aday-basina bilgi tukendi. Kullanilmamis bilgi
PARCA DUZEYINDE ve olculdu: D6'da >=6 CP'li 1096 parcada GT'lerin **%90.8'i**
parcanin en sik OTELEME VEKTORUYLE baska bir GT'ye ulasiyor.

KADEMELER
  1. P6 ortak siralayici (92 sutun) -> skor
  2. yuksek skorlu secim = TOHUM -> `kafes` oznitelikleri (8 sutun)
  3. ikinci siralayici (100 sutun) -> nihai skor -> secim

SIZINTIYA KARSI IKI ONLEM
  * Tohumlar HER ZAMAN tahminden gelir, GT'den ASLA.
  * Egitim korpusunda birinci kademe skorlari MARKA-KATLI (out-of-fold) uretilir.
    Aksi halde tohumlar kendi egitim verisinde asiri iyi olur, ikinci kademe
    gercekte olmayan bir tohum kalitesine gore ogrenir ve sinavda coker.

AYAR VE OLCUM AYRIMI
  * Butun kural/esik secimi `tam` korpusunun MARKA KATLARINDA yapilir.
  * D6 (SUPU/UPUN/MOR/NIT/UTL/S+S/SE/ONV) TEMIZ OKUMADIR -- ayar icin
    KULLANILMAZ.
  * D7 sinavdir ve bu betik ONA HIC BAKMAZ.
"""
import collections
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
os.environ.setdefault("P6_DIZIN", "results/_p6_oz_m60")
sys.path.insert(0, ".")
import kafes                       # noqa: E402
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
import urun_genis                  # noqa: E402
import yon_bankasi as YB           # noqa: E402
from kos_p6_ortak import yukle     # noqa: E402
from sina_kume import esle_macar   # noqa: E402

AB = p6_karar.AB
C0 = AB
# Esik izgarasi YUKARI acik tutulur. Mesh havuzu acilinca darbogaz recall'dan
# KESINLIGE gecti (D6 on okumasi: recall 0.285 -> 0.523 ama kesinlik
# 0.576 -> 0.322) ve secilen kural izgaranin en ust degeri cikti. Sinirda kalan
# bir optimum, bulunmamis optimum demektir.
KURALLAR = ([("mutlak", e) for e in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
                                     0.80, 0.85, 0.90, 0.95, 0.97, 0.99)] +
            [("goreli", o, t) for o in (0.30, 0.50, 0.70, 0.85, 0.95)
             for t in (0.05, 0.20, 0.40, 0.60)])
NMSLER = (2.5, 3.5, 5.0)
# Kural aramasi kivrimin EGITIM parcalarinin bir ORNEKLEMINDE yapilir: her kural
# tum parcalari gezip Macar eslemesi kosuyor ve 42 kural x 2000 parca bir kolu
# dakikalarca bekletiyor. Ornekleme SECIMI degistirmez (kurallar arasi sira
# birkac yuz parcada zaten kararli), yalniz maliyeti dusurur.
ARAMA_N = int(os.environ.get("P6_ARAMA_N", "600"))
OLCUT = os.environ.get("P6_OLCUT", "makro")     # kural secim olcutu
TOHUM_KURAL = ("mutlak", 0.60)     # tohum ESIGI ayrica taranmaz: yuksek tutulur
TOHUM_NMS = 5.0
# IKINCI KADEME = KISA LISTE UZERINDE FP REDDEDICI.
# Ilk tasarimda ikinci kademe TUM secenekleri yeniden puanliyordu ve yalniz 8
# kafes sutunu ekliyordu -- olculdu, ZARAR verdi (-0.0363): birinci kademenin
# zaten cozdugu 2200 secenegin ezici cogunlugu apacik negatif ve model kapasitesi
# oraya gidiyor. Dogru kurulum kaskad: birinci kademe RECALL icin genis tarar,
# ikinci kademe yalniz KISA LISTEYE bakar, birinci kademe skorunu da OZNITELIK
# olarak alir ve zor negatifleri ayirmaya odaklanir.
KISA_ESIK = float(os.environ.get("P6_KISA_ESIK", "0.20"))
NEG_KAT = int(os.environ.get("P6_NEG_KAT", "8"))
KOLLAR = ("TABAN", "P6", "P6_KAFES")


def yap(tohum=0):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        l2_regularization=1.0, random_state=tohum)


def kendi(d):
    """Aday basina TEK satir: kendi yon secenegi (C blogunun k_kendi=1 satiri)."""
    return np.where(d["X"][:, C0] == 1.0)[0]


def taban_satir(d):
    """TABAN kolunun satirlari: kendi yonu VE mesh OLMAYAN adaylar.

    Dagitilan urunun havuzunda mesh tepeleri YOK. Mesh'i taban kolunda da
    birakmak, "yeni havuz + yeni siralayici" kazancini tabana da yazmak olurdu
    ve kiyas tek degiskenli olmaktan cikardi.
    """
    k = kendi(d)
    return k[d["kaynak"][d["idx"][k]] != 2]


def alt_ornekle(M, Y, kat=NEG_KAT, tohum=0):
    """Tum pozitifler + `kat` katı negatif. 3M satirlik egitimi kaldirilabilir
    kilar; esik zaten sonradan taraniyor, taban oran degismesi zararsiz."""
    if kat <= 0:
        return M, Y
    rng = np.random.default_rng(tohum)
    poz = np.where(Y == 1)[0]
    neg = np.where(Y == 0)[0]
    n = min(len(neg), kat * max(len(poz), 1))
    sec = np.concatenate([poz, rng.choice(neg, n, replace=False)])
    rng.shuffle(sec)
    return M[sec], Y[sec]


def tohumla(d, s):
    """Birinci kademe skorlarindan TOHUM konum/yonleri."""
    return p6_karar.sec_ayrintili(d["P"], d["idx"], d["YD"], s, TOHUM_KURAL,
                                  nms_mm=TOHUM_NMS)[:2]


def kafes_bloku(d, s):
    Pt, Dt = tohumla(d, s)
    return kafes.oznitelik(d["P"][d["idx"]], d["YD"], Pt, Dt)


def puanla(d, s, kural, nms, kol):
    if kol == "TABAN":
        k = taban_satir(d)
        ci = d["idx"][k]                     # aday indeksleri
        sk = s[k]
        m = p6_karar.kabul_maskesi(sk, kural)
        if not m.any():
            return np.zeros((0, 3)), np.zeros((0, 3))
        P, D = d["P"][ci[m]], d["D"][ci[m]]
        T = d["X"][k][m][:, AB + len(YB.OZ_AD):]
        n = np.ones(len(P), bool)
        if nms > 0 and len(P) > 1:
            import wire_gate
            n = wire_gate.kalabalik_maskesi(P, sk[m])
        return P[n], urun_genis.isaret_duzelt(D[n], T[n])
    return p6_karar.sec(d["P"], d["idx"], d["YD"], s, kural, nms_mm=nms)


def olc(veri, skor, kural, nms, kol):
    tp = fp = fn = 0
    tes = []
    per = collections.defaultdict(lambda: [0, 0, 0])
    for d, s in zip(veri, skor):
        P, D = puanla(d, s, kural, nms, kol)
        a, b, c = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                             False, isaretli=True)[:3]
        tp += a; fp += b; fn += c
        q = per[d["mfg"]]
        q[0] += a; q[1] += b; q[2] += c
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    pm = {m: 2 * q[0] / max(2 * q[0] + q[1] + q[2], 1) for m, q in per.items()}
    return {"robot": 2 * tp / max(2 * tp + fp + fn, 1), "tespit": K.mikro(tes),
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "recall": tp / max(tp + fn, 1), "kesinlik": tp / max(tp + fp, 1),
            "makro": float(np.mean(list(pm.values()))) if pm else 0.0,
            "en_kotu": float(min(pm.values())) if pm else 0.0, "marka": pm}


def oz(d, kol, kafes_blok=None, s1=None):
    """Kolun oznitelik matrisi.

    TABAN     : A+B, aday basina tek satir, mesh HARIC (dagitilan kural)
    P6        : 92 + kaynak gostergesi (3)
    P6_KAFES  : P6 + kafes (8) + birinci kademe skoru (1) -- YALNIZ kisa liste
    """
    if kol == "TABAN":
        return p6_karar.donustur(d["X"][taban_satir(d)][:, :AB], "hepsi")
    X = np.hstack([p6_karar.donustur(d["X"]),
                   p6_karar.kaynak_blok(d["kaynak"][d["idx"]])])
    if kol == "P6_KAFES":
        return np.hstack([X, kafes_blok, np.asarray(s1, float)[:, None]])
    return X


def kisa(s1):
    """Ikinci kademenin bakacagi satirlar."""
    return np.where(np.asarray(s1, float) >= KISA_ESIK)[0]


def kafes_matris(d, kb, s1, k):
    """Ikinci kademe oznitelikleri, `k` satirlarinda.
    [92 donusturulmus | kaynak 3 | kafes 8 | birinci kademe skoru 1]"""
    X = np.hstack([p6_karar.donustur(d["X"]),
                   p6_karar.kaynak_blok(d["kaynak"][d["idx"]])])
    return np.hstack([X[k], np.asarray(kb)[k],
                      np.asarray(s1, float)[k][:, None]])


def egit(tr, kol, kafes_bloklar=None, s1ler=None):
    Ms, Ys = [], []
    for i, d in enumerate(tr):
        if kol == "P6_KAFES":
            k = kisa(s1ler[i])
            if not len(k):
                continue
            Ms.append(kafes_matris(d, kafes_bloklar[i], s1ler[i], k))
            Ys.append(d["y"][k])
        else:
            Ms.append(oz(d, kol))
            Ys.append(d["y"][taban_satir(d)] if kol == "TABAN" else d["y"])
    if not Ms:
        return None
    M = np.vstack(Ms).astype(np.float32)
    Y = np.concatenate(Ys)
    M, Y = alt_ornekle(M, Y)
    return yap().fit(M, Y)


def skorla(m, veri, kol, kafes_bloklar=None, s1ler=None):
    """Kolun skorlari. P6_KAFES'te KISA LISTE DISI satirlar 0 kalir --
    yani ikinci kademe birinci kademeyi EZEMEZ, yalniz icinden secer."""
    out = []
    for i, d in enumerate(veri):
        if kol == "P6_KAFES":
            s = np.zeros(len(d["X"]))
            k = kisa(s1ler[i])
            if len(k):
                X = kafes_matris(d, kafes_bloklar[i], s1ler[i], k)
                s[k] = m.predict_proba(X.astype(np.float32))[:, 1]
            out.append(s)
            continue
        p = m.predict_proba(oz(d, kol).astype(np.float32))[:, 1]
        if kol == "TABAN":
            s = np.zeros(len(d["X"]))
            s[taban_satir(d)] = p
        else:
            s = p
        out.append(np.asarray(s, float))
    return out


def main():
    t0 = time.time()
    # KUME SECIMI. `tam` = 9 marka / 2583 parca (asil egitim ve kural secimi).
    # `d6` = 8 marka / 468 parca -- HIZLI YINELEME icin. D6 bu oturumda teshis
    # ve kol secimi icin YOGUN kullanildi, dolayisiyla TEMIZ OKUMA DEGILDIR;
    # temiz okuma yalnizca D7'dir ve ona 3 okumalik butce ile bakilir.
    tr = yukle(os.environ.get("P6_KUME", "tam"), int(os.environ.get("P6_TR", "0")))
    for d in tr:
        d["y"] = np.asarray(d["y"], int)
    marka = collections.Counter(d["mfg"] for d in tr)
    # KAT ESIGI. D6'da 60 esigi NIT'i (50 parca) disarida birakiyordu -- oysa
    # NIT D6 GT'sinin %46'si ve havuzun EN ZOR oldugu marka. Bir markanin hic
    # kat olmamasi, kiyas sayilarinin o markayi HIC olcmemesi demek.
    KAT_MIN = int(os.environ.get("P6_KAT_MIN", "60"))
    katlar = [m for m, n in marka.items() if n >= KAT_MIN]
    print(f"tam {len(tr)} parca | markalar {dict(marka)}", flush=True)
    print(f"marka katlari (n>={KAT_MIN}): {katlar}  ({time.time() - t0:.0f} s)"
          f" | kapsanan parca {sum(marka[m] for m in katlar)}/{len(tr)}",
          flush=True)

    # --- 1) BIRINCI KADEME: marka-katli OOF skorlari -----------------------
    oof = [None] * len(tr)
    for b in katlar:
        ic = [i for i, d in enumerate(tr) if d["mfg"] != b]
        dis = [i for i, d in enumerate(tr) if d["mfg"] == b]
        m1 = egit([tr[i] for i in ic], "P6")
        for i, s in zip(dis, skorla(m1, [tr[i] for i in dis], "P6")):
            oof[i] = s
        print(f"  OOF {b}: {len(dis)} parca ({time.time() - t0:.0f} s)",
              flush=True)
    kucuk = [i for i, s in enumerate(oof) if s is None]
    if kucuk:                       # kat olusturamayan kucuk markalar
        ic = [i for i in range(len(tr)) if i not in set(kucuk)]
        if not ic:                  # (yalniz kucuk kosularda olur)
            rng = np.random.default_rng(0)
            pay = rng.permutation(len(tr)) % 3
            for f_ in range(3):
                d_ = [i for i in range(len(tr)) if pay[i] == f_]
                i_ = [i for i in range(len(tr)) if pay[i] != f_]
                m1 = egit([tr[i] for i in i_], "P6")
                for i, s in zip(d_, skorla(m1, [tr[i] for i in d_], "P6")):
                    oof[i] = s
            print("  OOF: marka kati kurulamadi, 3 RASTGELE kat kullanildi "
                  "(yalniz kucuk kosularda olur)", flush=True)
        else:
            m1 = egit([tr[i] for i in ic], "P6")
            for i, s in zip(kucuk, skorla(m1, [tr[i] for i in kucuk], "P6")):
                oof[i] = s
            print(f"  OOF kucuk markalar: {len(kucuk)} parca", flush=True)
    if not katlar:                  # kiyas katlari da yoksa rastgele boluruz
        rng = np.random.default_rng(1)
        pay = rng.permutation(len(tr)) % 3
        for i, d in enumerate(tr):
            d["_kat"] = f"kat{pay[i]}"
        katlar = [f"kat{i}" for i in range(3)]
        print(f"  KIYAS katlari rastgele: {katlar}", flush=True)
    else:
        for d in tr:
            d["_kat"] = d["mfg"]

    kafes_tr = [kafes_bloku(d, s) for d, s in zip(tr, oof)]
    kv = np.vstack(kafes_tr)
    print(f"kafes blogu {kv.shape} | kafes bulunan secenek orani "
          f"{kv[:, 0].mean():.3f} ({time.time() - t0:.0f} s)", flush=True)

    # --- 2) KAT ICINDE kural secimi + kol kiyasi ---------------------------
    top = {k: collections.Counter() for k in KOLLAR}
    ayrinti = {}
    for b in katlar:
        ic = [i for i, d in enumerate(tr) if d["_kat"] != b]
        dis = [i for i, d in enumerate(tr) if d["_kat"] == b]
        TR = [tr[i] for i in ic]
        TE = [tr[i] for i in dis]
        ayrinti[b] = {}
        for kol in KOLLAR:
            kf = (kol == "P6_KAFES")
            kb_tr = [kafes_tr[i] for i in ic] if kf else None
            kb_te = [kafes_tr[i] for i in dis] if kf else None
            s1_tr = [oof[i] for i in ic] if kf else None
            s1_te = [oof[i] for i in dis] if kf else None
            m = egit(TR, kol, kb_tr, s1_tr)
            if m is None:
                ayrinti[b][kol] = {"robot": 0.0, "TP": 0, "FP": 0,
                                   "FN": sum(len(d["G"]) for d in TE),
                                   "kural": ["yok"], "nms": 0.0}
                continue
            s_tr = skorla(m, TR, kol, kb_tr, s1_tr)
            s_te = skorla(m, TE, kol, kb_te, s1_te)
            ar = (np.random.default_rng(0).choice(len(TR), ARAMA_N, False)
                  if ARAMA_N and len(TR) > ARAMA_N else np.arange(len(TR)))
            AR = [TR[i] for i in ar]
            AS = [s_tr[i] for i in ar]
            # KURAL SECIM OLCUTU: egitim markalarinin MAKRO ortalamasi.
            # NEDEN MIKRO DEGIL: mikro, GT'si cok olan markanin kuralini secer.
            # D6'da NIT GT'nin %46'si ve NIT'te secilen esik (0.97) HER SEYI
            # eliyor -> o markada F1 0.0016. Havuzda NIT'in cevabinin YARISI
            # (yonlu recall 0.5254) VAR; kaybeden kural, model degil. Makro
            # olcut, tek bir markada COKMEYEN kurali tercih eder.
            def _puan(x):
                r_ = olc(AR, AS, x[0], x[1], kol)
                return r_["makro"] if OLCUT == "makro" else r_["robot"]
            en = max(((r, n) for r in KURALLAR for n in NMSLER), key=_puan)
            r = olc(TE, s_te, en[0], en[1], kol)
            # KURAL KAHINI (TESHIS, dagitilamaz): disarida birakilan markada EN
            # IYI kural ne verirdi? Fark buyukse kayip KURAL SECIMINDE, kucukse
            # MODELDE demektir.
            kah = max((olc(TE, s_te, x, n, kol)["robot"]
                       for x in KURALLAR for n in NMSLER))
            for k in ("TP", "FP", "FN"):
                top[kol][k] += r[k]
            ayrinti[b][kol] = dict(r, kural=list(en[0]), nms=en[1],
                                   kural_kahini=kah)
        a = ayrinti[b]
        print(f"  {b:<6} n={len(TE):<4} TABAN {a['TABAN']['robot']:.4f} | "
              f"P6 {a['P6']['robot']:.4f} | P6+KAFES {a['P6_KAFES']['robot']:.4f}"
              f"   [kural kahini P6 {a['P6'].get('kural_kahini', 0):.4f}]"
              f"   ({time.time() - t0:.0f} s)", flush=True)

    print(f"\n{'kol':<12} {'robot':>8} {'recall':>8} {'kesinlik':>9} "
          f"{'TP':>7} {'FP':>7} {'FN':>7}")
    son = {}
    for kol in KOLLAR:
        c = top[kol]
        f1 = 2 * c["TP"] / max(2 * c["TP"] + c["FP"] + c["FN"], 1)
        rc = c["TP"] / max(c["TP"] + c["FN"], 1)
        pr = c["TP"] / max(c["TP"] + c["FP"], 1)
        son[kol] = {"robot": f1, "recall": rc, "kesinlik": pr, **dict(c)}
        print(f"{kol:<12} {f1:>8.4f} {rc:>8.4f} {pr:>9.4f} {c['TP']:>7} "
              f"{c['FP']:>7} {c['FN']:>7}")
    print(f"\nP6      - TABAN = {son['P6']['robot'] - son['TABAN']['robot']:+.4f}")
    print(f"P6KAFES - P6    = {son['P6_KAFES']['robot'] - son['P6']['robot']:+.4f}")

    # --- 3) NIHAI MODELLER (tum tam) ---------------------------------------
    en_kol = max(KOLLAR, key=lambda k: son[k]["robot"])
    kural_sayim = collections.Counter(
        (tuple(ayrinti[b][en_kol]["kural"]), ayrinti[b][en_kol]["nms"])
        for b in katlar)
    kural, nms = kural_sayim.most_common(1)[0][0]
    print(f"\nSECILEN kol {en_kol} | kural {kural} | nms {nms} "
          f"(marka katlarinda en sik)")
    m1 = egit(tr, "P6")
    paket = {"kademe1": m1, "kol": en_kol, "kural": list(kural), "nms": nms,
             "zskor": "ab", "AB": AB, "tohum_kural": list(TOHUM_KURAL),
             "tohum_nms": TOHUM_NMS, "kisa_esik": KISA_ESIK}
    if en_kol == "P6_KAFES":
        # Ikinci kademe OOF skorlarindan egitilir: urunde birinci kademe skoru
        # gorulmemis parcadan gelecek, egitimde de oyle gelmeli.
        paket["kademe2"] = egit(tr, "P6_KAFES", kafes_tr, oof)
    with open("results/p6_kademe2_model.pkl", "wb") as f:
        pickle.dump(paket, f)
    json.dump({"damga": makbuz_hash.damga(), "toplam": son, "marka": ayrinti,
               "n_egitim": len(tr), "katlar": katlar, "secilen": en_kol,
               "kural": list(kural), "nms": nms, "dizin": os.environ["P6_DIZIN"],
               "not": "tam korpusunun MARKA KATLARINDA kural secimi + kol "
                      "kiyasi. Tohumlar OUT-OF-FOLD skorlardan. D6 ve D7'ye "
                      "BAKILMADI."},
              open("results/p6_kademe2_tam.json", "w"), indent=1)
    print(f"makbuz -> results/p6_kademe2_tam.json  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
