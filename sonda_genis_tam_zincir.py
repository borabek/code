# -*- coding: utf-8 -*-
"""DAGITIM KAPISI: genisletilmis havuz + duzeltilmis etiketli gate, TAM ZINCIRDE.

`results/v6_farki_E.json`: esik D6'da secilerek genisletilmis havuz D7'de 0.2079,
tez-saf kontrol 0.1981 (+0.0098). AMA o olcum urunun TAM zincirini kosmuyordu --
POZ KAFASI (`urun_zinciri.tam_poz`) yoktu. Dagitim karari tam zincirde verilir
([[olcum-yolu-ve-secim-kusurlari]]: olcum yolu urunun KENDI yolunu kosmali).

Bu betik iki kolu da TAM zincirle olcer:
  A) TEZ-SAF havuz   + duzeltilmis-etiket gate  (kontrol)
  B) GENISLETILMIS   + duzeltilmis-etiket gate  (deney)
Referans: dagitilan urun (gate v6 + NMS + poz kafasi) = robot 0.2029 / tespit 0.4523.

Esik D6'da secilmis kurallardir, D7'de YENIDEN TARANMAZ:
  tez-saf -> goreli (0.5, 0.30) | genisletilmis -> mutlak 0.15
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import kanonik_d7 as K          # noqa: E402
import urun_zinciri             # noqa: E402
import wire_gate                # noqa: E402
from p1c_esik import maske      # noqa: E402
from sina_kume import esle_macar  # noqa: E402

OZ = "results/_brep_oz"
OB = "results/_p1_olasilik_d7"
KURAL = {"TEZ-SAF": ("goreli", (0.5, 0.30)), "GENISLETILMIS": ("mutlak", 0.15)}


def main():
    S = K.step_haritasi()
    modeller = pickle.load(open("results/v6_farki_E_modeller.pkl", "rb")) \
        if os.path.exists("results/v6_farki_E_modeller.pkl") else None
    if modeller is None:
        raise SystemExit("Modeller kaydedilmemis -- sonda_v6_farki_E.py once "
                         "modelleri diske yazmali (asagida duzeltildi)")
    pidler = [f[3:-4] for f in sorted(os.listdir(OZ))
              if f.startswith("d7_") and f.endswith(".npz")]
    kay = K.yukle(pidler)
    out = {}
    for ad in ("TEZ-SAF", "GENISLETILMIS"):
        model = modeller[ad]
        tip, e = KURAL[ad]
        rob = collections.defaultdict(lambda: [0, 0, 0])
        tes = []
        pozsuz = 0
        for pid in pidler:
            r = kay[pid]
            G = np.asarray(r.get("G", []), float)
            if not len(G):
                continue
            z = np.load(f"{OZ}/d7_{pid}.npz")
            m = (z["kaynak"] == 0) if ad == "TEZ-SAF" \
                else np.ones(len(z["kaynak"]), bool)
            X = np.asarray(z["X"], float)[m]
            if len(X) < 2:
                continue
            P, D = z["P"][m], z["D"][m]
            s = np.asarray(wire_gate.karar_skoru(model, X), float)
            k = (s >= e) if tip == "mutlak" else maske(s, e[0], e[1])
            P, D = (P[k], D[k]) if k.any() else (P[:0], D[:0])
            if len(P) > 1:
                nm = wire_gate.kalabalik_maskesi(P, s[k])
                P, D = P[nm], D[nm]
            # TAM ZINCIR: poz kafasi
            f = f"{OB}/{pid}.npz"
            if len(P) and os.path.exists(f):
                zz = np.load(f)
                P, D = urun_zinciri.tam_poz(
                    np.ascontiguousarray(zz["V"], np.float64),
                    np.ascontiguousarray(zz["F"], np.int64),
                    np.asarray(zz["pbs"], float).mean(0), P, D,
                    step_path=S.get(pid))
            elif len(P):
                pozsuz += 1
            Gd = np.asarray(r["Gd"], float)
            dg = float(r["diag"])
            tp, fp, fn = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False,
                                    isaretli=True)[:3]
            a = rob[r["mfg"]]
            a[0] += tp; a[1] += fp; a[2] += fn
            tes.append((len(G),) + esle_macar(P, D, G, Gd, dg,
                       max(3.0, 0.06 * dg), 180.0, True)[:3])
        pm = {m: 2 * a[0] / max(2 * a[0] + a[1] + a[2], 1) for m, a in rob.items()}
        mi = float(2 * sum(a[0] for a in rob.values()) /
                   max(sum(2 * a[0] + a[1] + a[2] for a in rob.values()), 1))
        out[ad] = {"robot": mi, "tespit": K.mikro(tes),
                   "makro": float(np.mean(list(pm.values()))),
                   "en_kotu": float(min(pm.values())), "marka": pm,
                   "kural": f"{tip} {e}", "poz_kafasiz": pozsuz}
        c = out[ad]
        print(f"{ad:<16} robot {mi:.4f} | tespit {c['tespit']:.4f} | makro "
              f"{c['makro']:.4f} | en kotu {c['en_kotu']:.4f} | {c['kural']}",
              flush=True)
    a, b = out["TEZ-SAF"]["robot"], out["GENISLETILMIS"]["robot"]
    print(f"\nTAM ZINCIRDE: tez-saf {a:.4f} | genisletilmis {b:.4f} "
          f"({b-a:+.4f}) | dagitilan urun 0.2029")
    print("KARAR: " + ("DAGITILABILIR" if b > 0.2029 else "urunu gecemedi"))
    json.dump({"damga": makbuz_hash.damga(), "sonuc": out, "urun": 0.2029,
               "not": "TAM URUN ZINCIRI (poz kafasi dahil). Esikler D6'da "
                      "secildi, D7'de yeniden taranmadi. MIKRO."},
              open("results/genis_tam_zincir.json", "w"), indent=1)
    print("makbuz -> results/genis_tam_zincir.json")


if __name__ == "__main__":
    main()
