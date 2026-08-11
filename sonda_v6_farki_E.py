# -*- coding: utf-8 -*-
"""v6 FARKI, ADIM E: esik D7'YE BAKMADAN secilir (durust kiyas).

ADIM C/D SONUCU: proje etiketiyle tez-saf kol 0.2009 (v6 seviyesi 0.1970 ->
fark KAPANDI), genisletilmis kol 0.2170 (dagitilan urun 0.2029). AMA esik
D7'nin KENDISINDE taranmisti ve iki kol FARKLI esikte tepe yapiyor:

  esik           tez-saf   genisletilmis
  mutlak 0.2      0.1749      **0.2170**
  mutlak 0.3      0.1957        0.1846
  mutlak 0.4    **0.2009**      0.1099
  goreli .5/.2    0.1851        0.2126
  goreli .5/.3    0.1957        0.1849

Yani sonuc esige BAGLI ve D7'de secmek sinav kumesine ayar yapmaktir
([[uclu-bolme-ve-sahte-kazanclar]]). Bu betik esigi **D6'da** secer (markalari
D7'den AYRIK) ve D7'ye oyle uygular. Egitim korpusu D6'yi ICERMEZ, boylece esik
secimi de ORNEKLEM-DISI olur.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import brep_havuz               # noqa: E402
import d6_kayit                 # noqa: E402
import kanonik_d7 as K          # noqa: E402
import wire_gate                # noqa: E402
from p1c_esik import maske      # noqa: E402
from sina_kume import esle_macar  # noqa: E402

OZ = "results/_brep_oz"
DONUSUM = "zskor"
EKSENEL = 40.0
KURALLAR = ([("mutlak", x) for x in (0.15, 0.20, 0.25, 0.30, 0.40)] +
            [("goreli", x) for x in ((0.5, 0.15), (0.5, 0.20), (0.5, 0.30))])

CY, AC = {}, {}
for _f in ("results/_brepegit_silindirler.pkl", "results/_d6_silindirler.pkl"):
    CY.update(pickle.load(open(_f, "rb")))
for _f in ("results/_brepegit_acikliklar.pkl", "results/_d6_acikliklar.pkl"):
    AC.update(pickle.load(open(_f, "rb")))


def proje_etiketi(P, G, Gd, diag):
    """`build_zengin_parite.py` ile BIREBIR: yanal + eksenel 40mm + bire-bir."""
    y = np.zeros(len(P), int)
    if not len(P) or not len(G):
        return y
    tol = max(3.0, 0.06 * float(diag))
    Gn = Gd / np.maximum(np.linalg.norm(Gd, axis=1, keepdims=True), 1e-12)
    diff = P[:, None, :] - G[None, :, :]
    al = (diff * Gn[None, :, :]).sum(-1)
    pe = np.linalg.norm(diff - al[..., None] * Gn[None, :, :], axis=-1)
    pe = np.where(np.abs(al) <= EKSENEL, pe, np.inf)
    up, ug = set(), set()
    for dd, a_, b_ in sorted((pe[a, b], a, b)
                             for a in range(len(P)) for b in range(len(G))):
        if dd > tol or a_ in up or b_ in ug:
            continue
        up.add(a_); ug.add(b_); y[a_] = 1
    return y


def parca_oku(on, pid, r, segtek):
    z = np.load(f"{OZ}/{on}_{pid}.npz")
    X = np.asarray(z["X"], float)
    nseg = len(np.asarray(r["P"], float))
    if nseg > len(X):
        return None
    kay = np.zeros(len(X), int)
    kay[nseg:] = 1
    if "P" in z:
        P = np.asarray(z["P"], float)
        D = np.asarray(z["D"], float)
    else:
        P, D, _ = brep_havuz.birlesik_havuz(
            np.asarray(r["P"], float), np.asarray(r["Pd"], float),
            CY.get(pid), AC.get(pid))
        if len(P) != len(X):
            return None
    m = (kay == 0) if segtek else np.ones(len(X), bool)
    if int(m.sum()) < 2:
        return None
    return X[m], P[m], D[m]


def kume_kur(segtek):
    """Doner: (egitim listesi, D6 esik-secim listesi). D6 EGITIME GIRMEZ."""
    R = {str(r["pid"]): r for r in pickle.load(open(K.KAYIT, "rb"))}
    R6 = d6_kayit.yukle(set(d6_kayit.sinav()["pidler"]))
    tr, dev = [], []
    for f in sorted(os.listdir(OZ)):
        if not f.endswith(".npz"):
            continue
        if f.startswith("tam_"):
            on, pid, hedef, r = "tam", f[4:-4], tr, None
        elif f.startswith("d6_"):
            on, pid, hedef, r = "d6", f[3:-4], dev, None
        else:
            continue
        r = R.get(pid) or R6.get(pid)
        if r is None or not len(r.get("G", [])):
            continue
        v = parca_oku(on, pid, r, segtek)
        if v is None:
            continue
        X, P, D = v
        hedef.append({"pid": pid, "mfg": r["mfg"], "X": X, "P": P, "D": D,
                      "G": np.asarray(r["G"], float),
                      "Gd": np.asarray(r["Gd"], float), "diag": float(r["diag"])})
    return tr, dev


def egit(tr):
    M, Y = [], []
    for d in tr:
        M.append(wire_gate.parca_ici(d["X"], DONUSUM))
        Y.append(proje_etiketi(d["P"], d["G"], d["Gd"], d["diag"]))
    M = np.vstack(M); Y = np.concatenate(Y)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(M, Y)
    return ({"clf": clf, "cols": None, "n_feat": M.shape[1], "donusum": DONUSUM},
            M.shape, float(Y.mean()))


def olc(model, veri, tip, e):
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    for d in veri:
        s = np.asarray(wire_gate.karar_skoru(model, d["X"]), float)
        k = (s >= e) if tip == "mutlak" else maske(s, e[0], e[1])
        P, D = (d["P"][k], d["D"][k]) if k.any() else (d["P"][:0], d["D"][:0])
        if len(P) > 1:
            nm = wire_gate.kalabalik_maskesi(P, s[k])
            P, D = P[nm], D[nm]
        tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                                False, isaretli=True)[:3]
        a = rob[d["mfg"]]
        a[0] += tp; a[1] += fp; a[2] += fn
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    pm = {m: 2 * a[0] / max(2 * a[0] + a[1] + a[2], 1) for m, a in rob.items()}
    mi = float(2 * sum(a[0] for a in rob.values()) /
               max(sum(2 * a[0] + a[1] + a[2] for a in rob.values()), 1))
    return {"robot": mi, "tespit": K.mikro(tes),
            "makro": float(np.mean(list(pm.values()))),
            "en_kotu": float(min(pm.values())), "marka": pm}


def sinav_kur(segtek):
    te = []
    kay = None
    for f in sorted(os.listdir(OZ)):
        if not (f.startswith("d7_") and f.endswith(".npz")):
            continue
        te.append(f[3:-4])
    kayit = K.yukle(te)
    out = []
    for pid in te:
        r = kayit[pid]
        z = np.load(f"{OZ}/d7_{pid}.npz")
        m = (z["kaynak"] == 0) if segtek else np.ones(len(z["kaynak"]), bool)
        if int(m.sum()) < 2:
            continue
        out.append({"pid": pid, "mfg": r["mfg"],
                    "X": np.asarray(z["X"], float)[m], "P": z["P"][m],
                    "D": z["D"][m], "G": np.asarray(r["G"], float),
                    "Gd": np.asarray(r["Gd"], float), "diag": float(r["diag"])})
    return out


def main():
    sonuc = {}
    modeller = {}
    for ad, segtek in (("TEZ-SAF", True), ("GENISLETILMIS", False)):
        tr, dev = kume_kur(segtek)
        m, sh, poz = egit(tr)
        modeller[ad] = m
        print(f"\n{ad}: egitim {len(tr)} parca {sh} pozitif {poz:.4f} | "
              f"esik secim kumesi (D6) {len(dev)} parca", flush=True)
        en = None
        for tip, e in KURALLAR:
            r = olc(m, dev, tip, e)
            print(f"   D6 {str((tip, e)):<20} robot {r['robot']:.4f}", flush=True)
            if en is None or r["robot"] > en[2]["robot"]:
                en = (tip, e, r)
        tip, e, dr = en
        te = sinav_kur(segtek)
        tr_ = olc(m, te, tip, e)
        sonuc[ad] = {"secilen_kural": f"{tip} {e}", "D6_robot": dr["robot"],
                     "D7": tr_}
        print(f"  SECILEN (D6'da) {tip} {e} -> D7 robot **{tr_['robot']:.4f}** | "
              f"tespit {tr_['tespit']:.4f} | makro {tr_['makro']:.4f} | "
              f"en kotu {tr_['en_kotu']:.4f}", flush=True)
    a = sonuc["TEZ-SAF"]["D7"]["robot"]
    b = sonuc["GENISLETILMIS"]["D7"]["robot"]
    print(f"\nORNEKLEM-DISI ESIKLE:")
    print(f"  TEZ-SAF        {a:.4f}")
    print(f"  GENISLETILMIS  {b:.4f}   (fark {b-a:+.4f})")
    print(f"  DAGITILAN URUN 0.2029")
    print("\nKARAR: " + ("GENISLETILMIS HAVUZ KAZANDI -- dagitilabilir"
                         if b > 0.2029 else
                         "urunu GECEMEDI -- dagitilmaz"))
    json.dump({"damga": makbuz_hash.damga(), "sonuc": sonuc, "urun": 0.2029,
               "not": "Esik D6'da secildi (D7'den marka olarak AYRIK), egitim "
                      "korpusu D6'yi ICERMEZ. D7 marka-disi, MIKRO."},
              open("results/v6_farki_E.json", "w"), indent=1)
    with open("results/v6_farki_E_modeller.pkl", "wb") as f:
        pickle.dump(modeller, f)
    print("modeller -> results/v6_farki_E_modeller.pkl")
    print("makbuz -> results/v6_farki_E.json")


if __name__ == "__main__":
    main()
