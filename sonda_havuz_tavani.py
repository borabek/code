# -*- coding: utf-8 -*-
"""HAVUZ TAVANI: MUKEMMEL secici ile seg-tek vs genisletilmis havuz.

Tam olcekli gate refit saatler surer. Once sunu bilmek gerekir: o is bitse bile
tavan nerede? Kahin = her parcada, ROBOT olcutune uyan adaylari secen mukemmel
secici (bire-bir Macar; secmemek de bir secenek, yani fazla aday CEZA YAZMAZ --
`gate-once-poz-sonra-tavani-kirpiyor` kaydindaki 2 numarali hata tekrarlanmaz).

Bu bir TAVAN: gercek bir gate bunun ALTINDA kalir.
"""
import collections, json, os, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K
from sina_kume import esle_macar

OZ = "results/_brep_oz"
te = []
for f in sorted(os.listdir(OZ)):
    if f.startswith("d7_") and f.endswith(".npz"):
        z = np.load(f"{OZ}/{f}")
        te.append({"pid": f[3:-4], "P": z["P"], "D": z["D"], "kaynak": z["kaynak"]})
kay = K.yukle([d["pid"] for d in te])
for d in te:
    r = kay[d["pid"]]
    d.update({"mfg": r["mfg"], "G": np.asarray(r["G"], float),
              "Gd": np.asarray(r["Gd"], float), "diag": r["diag"]})
print(f"D7 {len(te)} parca", flush=True)


def kahin(P, D, G, Gd, diag, yanal, aci, tespit):
    """MUKEMMEL secim: olcute uyan adaylari sec, digerlerini ATA (ceza yok)."""
    if not len(P):
        return 0, 0, len(G)
    if tespit:
        tp, fp, fn = esle_macar(P, D, G, Gd, diag, max(3.0, 0.06*diag), 180.0,
                                True)[:3]
    else:
        tp, fp, fn = esle_macar(P, D, G, Gd, diag, yanal, aci, False,
                                isaretli=True)[:3]
    return tp, 0, fn                      # secilmeyen adaylar cikti DEGIL -> FP yok


out = {}
for ad, segtek in (("SEG-TEK havuz", True), ("GENISLETILMIS havuz", False)):
    for olcut, tespit in (("robot", False), ("tespit", True)):
        per = collections.defaultdict(lambda: [0, 0, 0])
        for d in te:
            m = d["kaynak"] == 0 if segtek else np.ones(len(d["kaynak"]), bool)
            tp, fp, fn = kahin(d["P"][m], d["D"][m], d["G"], d["Gd"], d["diag"],
                               K.YANAL, K.ACI, tespit)
            a = per[d["mfg"]]; a[0] += tp; a[1] += fp; a[2] += fn
        mi = float(2*sum(a[0] for a in per.values()) /
                   max(sum(2*a[0]+a[1]+a[2] for a in per.values()), 1))
        pm = {m: 2*a[0]/max(2*a[0]+a[1]+a[2], 1) for m, a in per.items()}
        out[f"{ad} | {olcut}"] = {"tavan": mi,
                                  "makro": float(np.mean(list(pm.values()))),
                                  "en_kotu": float(min(pm.values())), "marka": pm}
        print(f"{ad:<22} {olcut:<7} TAVAN {mi:.4f} | makro "
              f"{np.mean(list(pm.values())):.4f} | en kotu {min(pm.values()):.4f}",
              flush=True)
r0 = out["SEG-TEK havuz | robot"]["tavan"]; r1 = out["GENISLETILMIS havuz | robot"]["tavan"]
print(f"\nROBOT TAVANI: seg-tek {r0:.4f} -> genisletilmis {r1:.4f} ({r1-r0:+.4f})")
print(f"Mevcut urun 0.2029, yani genisletilmis havuzun tavaninin %{100*0.2029/max(r1,1e-9):.1f}'i")
print("0.50 HEDEFI icin gereken: tavan >= 0.50 " +
      ("SAGLANDI" if r1 >= 0.50 else f"SAGLANMADI (tavan {r1:.4f})"))
a = out["SEG-TEK havuz | robot"]["marka"]; b = out["GENISLETILMIS havuz | robot"]["marka"]
print(f"\n{'marka':<8} {'seg':>8} {'genis':>8} {'fark':>8}")
for m in sorted(a, key=lambda k: a[k]):
    print(f"  {m:<7} {a[m]:>7.4f} {b[m]:>8.4f} {b[m]-a[m]:>+8.4f}")
json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
           "not": "MUKEMMEL secici tavani. Secilmeyen aday cikti DEGIL (FP yok). "
                  "D7 marka-disi, MIKRO."},
          open("results/havuz_tavani.json", "w"), indent=1)
