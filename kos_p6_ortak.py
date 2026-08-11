# -*- coding: utf-8 -*-
"""P6 ORTAK SIRALAYICI: (konum x yon) seceneklerini TEK skorla puanla.

BUGUNKU URUN iki ayri karar veriyor:
  1. gate  -> bu aday CP mi?           (`urun_genis.sec`, HGB-derin, esik 0.05)
  2. yon   -> havuzun verdigi yon + isaret duzeltmesi + poz kafasi
Yon hicbir zaman SECILMIYOR; havuzdan ne geldiyse o.

BU BETIK ikisini birlestirir: her (konum, yon) secenegi tek modelle puanlanir,
secim skor sirasina gore acgozlu + NMS ile yapilir. Boylece
  * YON_YOK kovasi (D7'de 539 GT) dogrudan hedeflenir,
  * GATE_REDDI kovasi (580) da faydalanir: dogru yonu bulunan bir aday daha
    yuksek skor alir, esigi gecer.

OLCUM: D6. D7'ye BAKILMAZ (butce 3 okuma, hepsi kapida).

BLOK AYRIMI (`parca_ici` z-skoru):
  A+B (58+9) havuz/agiz olculeri MUTLAK buyukluklerdir -> parca-ici z-skor
             dagitilan gate'te en buyuk tek kazancti, KORUNUR.
  C+D (16+9) yon olculeri ZATEN goreli (aciya, destege, orana dayali) ->
             HAM birakilir. p5-v2'de goreli olculeri bir daha normalize etmek
             uctan uca 0.1649 -> 0.1465'e DUSURMUSTU.
  `P6_ZSKOR=hepsi` ile ikisi de z-skorlanir (ablasyon icin).
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
sys.path.insert(0, ".")
import d6_kayit                    # noqa: E402
import kanonik_d7 as K             # noqa: E402
import p6_karar                    # noqa: E402
import urun_genis                  # noqa: E402
import wire_gate                   # noqa: E402
import yon_bankasi as YB           # noqa: E402
from sina_kume import esle_macar   # noqa: E402

P6 = os.environ.get("P6_DIZIN", "results/_p6_oz")
# HAVUZ KAYNAGI SUZGECI: onbellek `kaynak` alanini tasir (0 seg / 1 B-rep /
# 2 mesh tepesi), boylece TEK cikarimdan farkli havuz kollari egitilebilir.
_KS = os.environ.get("P6_KAYNAK_EGIT", "")
KAYNAK_SUZ = tuple(int(c) for c in _KS) if _KS else None
AB = p6_karar.AB              # A(58) + B(9) -- z-skorlanan blok
ZSKOR = os.environ.get("P6_ZSKOR", "ab")
ESIKLER = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
_D6 = None


def kayitlar(pidler):
    """pid -> kayit (G, Gd, diag, mfg). Once kanonik, sonra D6 yedegi."""
    global _D6
    kay = K.yukle(pidler)
    eksik = [p for p in pidler if p not in kay]
    if eksik:
        if _D6 is None:
            _D6 = {str(p): r for p, r in d6_kayit.yukle().items()}
        kay.update({p: _D6[p] for p in eksik if p in _D6})
    return kay


def yukle(on, sinir=0):
    """Secenek tablolarini oku ve etiketle. Doner: liste[parca sozlugu]."""
    fs = sorted(f for f in os.listdir(P6)
                if f.startswith(on + "_") and f.endswith(".npz"))
    if sinir:
        fs = fs[:sinir]
    pidler = [f[len(on) + 1:-4] for f in fs]
    kay = kayitlar(pidler)
    out = []
    # SESSIZ KORPUS DARALMASI bu projede iki kez oldu (X genisligi 58-vs-22,
    # birlestir globu). Atlanan parca sayisi HER ZAMAN basilir.
    yok_kayit = yok_gt = 0
    for f, pid in zip(fs, pidler):
        r = kay.get(pid)
        if r is None:
            yok_kayit += 1
            continue
        if not len(r.get("G", [])):
            yok_gt += 1
            continue
        z = np.load(f"{P6}/{f}")
        X = np.asarray(z["X"], float)
        idx = np.asarray(z["idx"], int)
        YD = np.asarray(z["YD"], float)
        P = np.asarray(z["P"], float)
        Dham = np.asarray(z["D"], float)
        kayn = (np.asarray(z["kaynak"], int) if "kaynak" in z
                else np.zeros(len(P), int))
        if KAYNAK_SUZ is not None and "kaynak" in z:
            kay = kayn
            tut = np.isin(kay, KAYNAK_SUZ)
            # secenekler ADAY indeksine bagli; once secenekleri suz, sonra
            # aday indekslerini YENIDEN NUMARALA (aksi halde `idx` bos adaylara
            # isaret eder ve secim sessizce yanlis konumu doner).
            ysec = tut[idx]
            yeni = -np.ones(len(P), int)
            yeni[np.where(tut)[0]] = np.arange(int(tut.sum()))
            X, YD = X[ysec], YD[ysec]
            idx = yeni[idx[ysec]]
            P = P[tut]
            Dham = Dham[tut]
            kayn = kayn[tut]
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        y, _ = YB.etiketle(P[idx], YD, G, Gd)
        out.append({"pid": pid, "mfg": r["mfg"], "X": X, "idx": idx, "YD": YD,
                    "P": P, "D": Dham, "y": y, "kaynak": kayn,
                    "G": G, "Gd": Gd, "diag": float(r["diag"])})
    print(f"  {on}: {len(fs)} dosya -> {len(out)} parca "
          f"(kayit yok {yok_kayit}, GT yok {yok_gt})", flush=True)
    return out


def donustur(X):
    """URUNLE AYNI donusum -- `p6_karar` tek kaynaktir, burada kopyalanmaz."""
    return p6_karar.donustur(X, ZSKOR)


def olc(veri, skorlar, esik):
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    for d, s in zip(veri, skorlar):
        P, D = p6_karar.sec(d["P"], d["idx"], d["YD"], s, esik)
        tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL,
                                K.ACI, False, isaretli=True)[:3]
        a = rob[d["mfg"]]
        a[0] += tp; a[1] += fp; a[2] += fn
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    pm = {m: 2 * a[0] / max(2 * a[0] + a[1] + a[2], 1) for m, a in rob.items()}
    return {"robot": float(2 * sum(a[0] for a in rob.values()) /
                           max(sum(2 * a[0] + a[1] + a[2]
                                   for a in rob.values()), 1)),
            "tespit": K.mikro(tes),
            "makro": float(np.mean(list(pm.values()))),
            "en_kotu": float(min(pm.values())), "marka": pm,
            "TP": sum(a[0] for a in rob.values()),
            "FP": sum(a[1] for a in rob.values())}


def taban(veri):
    """DAGITILAN yol, ayni onbellekten yeniden kurulmus.

    `urun_genis.sec` ile ayni: X = A+B, parca-ici z-skor, HGB-derin, esik 0.05,
    kalabalik NMS, isaret duzeltme. Tek fark: burada onbellekten okunuyor.
    """
    model = urun_genis.model_yukle()
    if model is None:
        return None
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    for d in veri:
        kendi = np.where(d["X"][:, AB] == 1.0)[0]      # C blogunun ilk sutunu
        AB_ = d["X"][kendi, :AB]
        s = np.asarray(model.predict_proba(
            wire_gate.parca_ici(AB_, "zskor"))[:, 1], float)
        k = s >= urun_genis.ESIK
        P, D = (d["P"][k], d["D"][k]) if k.any() else (d["P"][:0], d["D"][:0])
        T = d["X"][kendi][k][:, AB + len(YB.OZ_AD):]
        if len(P) > 1:
            nm = wire_gate.kalabalik_maskesi(P, s[k])
            P, D, T = P[nm], D[nm], T[nm]
        D = urun_genis.isaret_duzelt(D, T) if len(D) else D
        tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL,
                                K.ACI, False, isaretli=True)[:3]
        a = rob[d["mfg"]]
        a[0] += tp; a[1] += fp; a[2] += fn
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    pm = {m: 2 * a[0] / max(2 * a[0] + a[1] + a[2], 1) for m, a in rob.items()}
    return {"robot": float(2 * sum(a[0] for a in rob.values()) /
                           max(sum(2 * a[0] + a[1] + a[2]
                                   for a in rob.values()), 1)),
            "tespit": K.mikro(tes),
            "makro": float(np.mean(list(pm.values()))),
            "en_kotu": float(min(pm.values())), "marka": pm,
            "TP": sum(a[0] for a in rob.values()),
            "FP": sum(a[1] for a in rob.values())}


def main():
    t0 = time.time()
    tr = yukle("tam", int(os.environ.get("P6_TR", "0")))
    dev = yukle("d6", int(os.environ.get("P6_DEV", "0")))
    print(f"egitim {len(tr)} parca | dev {len(dev)} parca "
          f"({time.time() - t0:.0f} s)", flush=True)
    if not tr or not dev:
        sys.exit("VERI YOK -- once `python kos_p6_oznitelik.py tam` kos.")

    M = np.vstack([donustur(d["X"]) for d in tr])
    Y = np.concatenate([d["y"] for d in tr])
    print(f"secenek {M.shape} | pozitif {Y.mean():.4f}", flush=True)

    m = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.06,
                                       max_leaf_nodes=63, l2_regularization=1.0,
                                       random_state=0).fit(M, Y)
    sk = [np.asarray(m.predict_proba(donustur(d["X"]))[:, 1], float)
          for d in dev]

    tb = taban(dev)
    print(f"\nTABAN (dagitilan yol, ayni parcalar): robot {tb['robot']:.4f} | "
          f"tespit {tb['tespit']:.4f} | makro {tb['makro']:.4f} | "
          f"TP {tb['TP']} FP {tb['FP']}", flush=True)
    print(f"\n{'esik':>6} {'robot':>8} {'tespit':>8} {'makro':>8} {'TP':>6} {'FP':>6}")
    sonuc = {}
    en = None
    for e in ESIKLER:
        r = olc(dev, sk, e)
        sonuc[f"{e:.2f}"] = r
        print(f"{e:>6.2f} {r['robot']:>8.4f} {r['tespit']:>8.4f} "
              f"{r['makro']:>8.4f} {r['TP']:>6} {r['FP']:>6}", flush=True)
        if en is None or r["robot"] > en[1]["robot"]:
            en = (e, r)
    e, r = en
    print(f"\nEN IYI esik {e:.2f} -> robot {r['robot']:.4f} "
          f"(taban {tb['robot']:.4f}, fark {r['robot'] - tb['robot']:+.4f})")
    with open("results/p6_ortak_model.pkl", "wb") as f:
        pickle.dump({"model": m, "esik": e, "zskor": ZSKOR, "AB": AB}, f)
    json.dump({"damga": makbuz_hash.damga(), "taban": tb, "esik_taramasi": sonuc,
               "en_iyi_esik": e, "zskor": ZSKOR, "n_dev": len(dev),
               "n_egitim": len(tr),
               "not": "P6 ortak (konum x yon) siralayici. D6 DEV -- D7'ye "
                      "BAKILMADI. Poz kafasi UYGULANMADI (ayri olculur)."},
              open("results/p6_ortak_d6.json", "w"), indent=1)
    print("makbuz -> results/p6_ortak_d6.json")


if __name__ == "__main__":
    main()
