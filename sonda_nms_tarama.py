# -*- coding: utf-8 -*-
"""NMS yaricap taramasi + MARKA KARARLILIGI + tespit etkisi.

Ilk olcumde alti varyantin ALTISI da pozitifti ve yaricapla monoton artiyordu
(r=3.0'da +0.0093). Kucuk ama tutarli. Dagitim karari icin gereken uc sey:
(1) yaricap gercekten doyuyor mu, (2) kazanc MARKALARA yayilmis mi yoksa tek
markadan mi geliyor, (3) TESPIT F1'i bozuyor mu (aday siliyoruz).

D7 = DEV. Yaricap burada secilirse D7'ye ayarlanmis olur -> D6'da DOGRULANIR.
"""
import collections, json, os, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import d6_kayit, wire_gate, kanonik_d7 as K
from p1c_esik import maske
from sina_kume import esle_macar

RS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def nms(P, gs, r):
    if r <= 0:
        return np.ones(len(P), bool)
    sira = np.argsort(-gs); tut = np.ones(len(P), bool)
    for a, i in enumerate(sira):
        if not tut[i]:
            continue
        for j in sira[a + 1:]:
            if tut[j] and np.linalg.norm(P[i] - P[j]) < r:
                tut[j] = False
    return tut


def kos(kayit, x58f, ad):
    rob = {r: collections.defaultdict(lambda: [0, 0, 0]) for r in RS}
    tes = {r: [] for r in RS}
    for pid, rec in kayit.items():
        X = x58f(rec); G = np.asarray(rec.get("G", []), float)
        if X is None or not len(G):
            continue
        P = np.asarray(rec["P"], float); D = np.asarray(rec["Pd"], float)
        Gd = np.asarray(rec["Gd"], float); dg = rec["diag"]
        gs = np.asarray(wire_gate.karar_skoru(gate, X), float)
        m = maske(gs, 0.40, 0.30)
        Pm, Dm, gm = (P[m], D[m], gs[m]) if m.any() else (P[:0], D[:0], gs[:0])
        for r in RS:
            s = nms(Pm, gm, r) if len(Pm) else np.zeros(0, bool)
            tp, fp, fn = esle_macar(Pm[s], Dm[s], G, Gd, dg, K.YANAL, K.ACI,
                                    False, isaretli=True)[:3]
            a = rob[r][rec["mfg"]]; a[0] += tp; a[1] += fp; a[2] += fn
            t = esle_macar(Pm[s], Dm[s], G, Gd, dg,
                           max(3.0, 0.06 * dg), 180.0, True)[:3]
            tes[r].append((len(G),) + t)
    print(f"\n=== {ad} ===")
    print(f"{'r':>5} {'robot':>8} {'d':>8} {'tespit':>8} {'makro':>7} {'kotu':>7} {'+marka':>7}")
    t0 = None; cik = {}
    for r in RS:
        per = rob[r]
        mi = float(2 * sum(a[0] for a in per.values()) /
                   max(sum(2 * a[0] + a[1] + a[2] for a in per.values()), 1))
        pm = {m: 2*a[0]/max(2*a[0]+a[1]+a[2], 1) for m, a in per.items()}
        if t0 is None:
            t0 = mi; b0 = dict(pm)
        art = sum(1 for m in pm if pm[m] > b0[m] + 1e-9)
        cik[r] = {"robot": mi, "tespit": K.mikro(tes[r]),
                  "makro": float(np.mean(list(pm.values()))),
                  "en_kotu": float(min(pm.values())), "artan_marka": art,
                  "n_marka": len(pm)}
        c = cik[r]
        print(f"{r:>5.1f} {mi:>8.4f} {mi-t0:>+8.4f} {c['tespit']:>8.4f} "
              f"{c['makro']:>7.4f} {c['en_kotu']:>7.4f} {art:>4}/{len(pm)}")
    return cik


gate = K.gate_yukle()
d7 = kos(K.yukle(json.load(open("results/d7_sinav_kumesi.json"))["pidler"]),
         K.x58, "D7 (DEV, 835 parca, marka-disi)")
d6 = kos(d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])), d6_kayit.x58,
         "D6 (468 parca) -- BAGIMSIZ DOGRULAMA")
json.dump({"damga": makbuz_hash.damga(), "D7": d7, "D6": d6, "yaricaplar": RS,
           "not": "mesafe-NMS, gate skoruna gore. Yaricap D7'de secilirse D6 "
                  "dogrulamasi ZORUNLU. MIKRO."},
          open("results/nms_tarama.json", "w"), indent=1)
