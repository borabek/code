# -*- coding: utf-8 -*-
"""DAGITILAN gate v6'yi GENISLETILMIS havuza vermek: calisiyor mu?

Tek degiskenli kiyas havuzun robot'a +0.0386 kattigini gosterdi, ama benim elde
kurdugum gate ayni havuzda 0.1159 -- dagitilan v6 tez-saf havuzda 0.2029 veriyor.
Yani recete farki (~0.09) havuz kazancindan buyuk.

En ucuz yol: v6'yi OLDUGU GIBI genisletilmis havuza uygulamak. RISK: v6 B-rep
adaylarini HIC gormedi ([[gate-refit-minv4]] dersi tam bunu yasaklar). O yuzden
bu bir DAGITIM KARARI DEGIL, bir OLCUM: v6'nin ozellik uzayi B-rep adaylarina
genelleniyor mu?

Uc kol: yalniz seg (referans) / seg + B-rep / B-rep'e skor CEZASI ile.
"""
import collections, json, os, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import wire_gate, kanonik_d7 as K
from p1c_esik import maske
from sina_kume import esle_macar

OZ = "results/_brep_oz"
te = []
for f in sorted(os.listdir(OZ)):
    if f.startswith("d7_") and f.endswith(".npz"):
        z = np.load(f"{OZ}/{f}")
        te.append({"pid": f[3:-4], "X": z["X"], "P": z["P"], "D": z["D"],
                   "kaynak": z["kaynak"]})
kay = K.yukle([d["pid"] for d in te])
for d in te:
    r = kay[d["pid"]]
    d.update({"mfg": r["mfg"], "G": np.asarray(r["G"], float),
              "Gd": np.asarray(r["Gd"], float), "diag": r["diag"]})
g6 = K.gate_yukle()
print(f"D7 {len(te)} parca | gate v6 n_feat={g6.get('n_feat')} "
      f"donusum={g6.get('donusum')!r}", flush=True)


def kos(segtek, ceza):
    en = None
    for tip, e in ([("mutlak", x) for x in (0.30, 0.40, 0.50)] +
                   [("goreli", x) for x in ((0.5, 0.20), (0.5, 0.30), (0.4, 0.25))]):
        rob = collections.defaultdict(lambda: [0, 0, 0]); tes = []
        for d in te:
            m0 = (d["kaynak"] == 0) if segtek else np.ones(len(d["kaynak"]), bool)
            X = np.asarray(d["X"][m0], float)
            if len(X) < 2:
                continue
            s = np.asarray(wire_gate.karar_skoru(g6, X), float)
            if ceza and not segtek:
                s = s - ceza * (d["kaynak"][m0] == 1)
            k = (s >= e) if tip == "mutlak" else maske(s, e[0], e[1])
            P, D = d["P"][m0], d["D"][m0]
            P, D = (P[k], D[k]) if k.any() else (P[:0], D[:0])
            if len(P) > 1:
                nm = wire_gate.kalabalik_maskesi(P, s[k]); P, D = P[nm], D[nm]
            tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL,
                                    K.ACI, False, isaretli=True)[:3]
            a = rob[d["mfg"]]; a[0] += tp; a[1] += fp; a[2] += fn
            tes.append((len(d["G"]),) + esle_macar(P, D, d["G"], d["Gd"], d["diag"],
                       max(3.0, 0.06*d["diag"]), 180.0, True)[:3])
        pm = {m: 2*a[0]/max(2*a[0]+a[1]+a[2], 1) for m, a in rob.items()}
        mi = float(2*sum(a[0] for a in rob.values()) /
                   max(sum(2*a[0]+a[1]+a[2] for a in rob.values()), 1))
        r = {"kural": f"{tip} {e}", "robot": mi, "tespit": K.mikro(tes),
             "makro": float(np.mean(list(pm.values()))),
             "en_kotu": float(min(pm.values())), "marka": pm}
        if en is None or r["robot"] > en["robot"]:
            en = r
    return en


out = {}
for ad, segtek, ceza in (("v6 | TEZ-SAF (referans)", True, 0.0),
                         ("v6 | GENISLETILMIS", False, 0.0),
                         ("v6 | GENISLETILMIS -0.05 ceza", False, 0.05),
                         ("v6 | GENISLETILMIS -0.10 ceza", False, 0.10),
                         ("v6 | GENISLETILMIS -0.20 ceza", False, 0.20)):
    out[ad] = kos(segtek, ceza)
    r = out[ad]
    print(f"{ad:<32} robot {r['robot']:.4f} | tespit {r['tespit']:.4f} | makro "
          f"{r['makro']:.4f} | en kotu {r['en_kotu']:.4f} | {r['kural']}", flush=True)
t = out["v6 | TEZ-SAF (referans)"]["robot"]
print()
for ad in out:
    if ad != "v6 | TEZ-SAF (referans)":
        print(f"  {ad:<32} {out[ad]['robot']-t:+.4f}")
json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
           "not": "DAGITILAN gate v6, degistirilmeden, genisletilmis havuza. "
                  "OLCUM -- dagitim karari DEGIL (v6 B-rep adaylarini gormedi). "
                  "D7 marka-disi, MIKRO."},
          open("results/v6_genisletilmis.json", "w"), indent=1)
