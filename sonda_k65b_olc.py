"""K6.5-b OLCUM: few-shot adapte edilmis aglari UCTAN UCA olcer.

Egitimden AYRI betik: egitim bir kez kosar (saatler), olcum tekrar tekrar
kosulabilir (dakikalar). `kos_k65b_fewshot_seg.py` makbuzunu okur.

PROTOKOL:
  * k=0 TABANI: degistirilmemis urun agi (g10) ayni parcalarda
  * adaptasyon parcalari OLCUME GIRMEZ (makbuzdaki `adapt` listesi dislanir)
  * TAM zincir: gate -> urun_zinciri.tam_poz -> Macar eslestirme
  * tespit VE robot AYRI raporlanir; k=0'a gore fark verilir
  * cekilisler arasi ortalama +- std (tek cekilis gurultulu)

HATA YUTULMAZ: bir kosum olculemezse betik PATLAR (bkz. g10 kill kapisi dersi).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")

import d6_kayit                      # noqa: E402
import robot_cp                      # noqa: E402
import wire_gate                     # noqa: E402
import urun_zinciri                  # noqa: E402
from p1c_esik import maske           # noqa: E402
from sina_kume import esle_macar, f1w  # noqa: E402
from korpus_kimlik import step_kimlik as SK  # noqa: E402

ROBOT_YANAL, ROBOT_ACI = 2.0, 10.0
TABAN_OB = "results/_p1_olasilik_g10"     # k=0: degistirilmemis urun agi


def olc(pidler, ob_dir, kayit, gate, S):
    """Bir onbellek dizini icin (tespit_F1, robot_F1). Hata YUTULMAZ."""
    T, R = [], []
    for pid in pidler:
        f = f"{ob_dir}/{pid}.npz"
        if not os.path.exists(f):
            raise RuntimeError(f"onbellekte YOK: {f} -- olcum eksik kalirdi")
        r = kayit[pid]
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        if not len(G):
            continue
        d = np.load(f)
        V = np.ascontiguousarray(d["V"], np.float64)
        F = np.ascontiguousarray(d["F"], np.int64)
        pbs = [np.asarray(q, float) for q in d["pbs"]]
        cps, _op, _cok, _per = robot_cp.adaylari_uret(V, F, pbs, S.get(pid))
        if not cps:
            T.append((len(G), 0.0, 0.0, 0.0))
            R.append((len(G), 0.0, 0.0, 0.0))
            continue
        # ANAHTAR "direction" -- "dir" DEGIL. Ilk surumumde `c.get("dir",[0,0,1])`
        # yazmistim: TUM yonler [0,0,1] oluyordu ve isaretli aci metrigi her yerde
        # dusuyordu -> robot TAM 0.0000. Varsayilan vermek yerine PATLIYORUZ.
        P = np.asarray([c["point"] for c in cps], float)
        D = np.asarray([c["direction"] for c in cps], float)

        # GATE OZNITELIKLERI YENI ADAYLAR ICIN HESAPLANIR.
        # Ilk surumumde `d6_kayit.x58(r)` kullaniyordum: o oznitelikler KAYITTAKI
        # (eski agdan gelen) adaylar icin uretilmisti. Uzunluklar tutmadigi icin
        # (45 vs 23) gate SESSIZCE ATLANIYORDU -- yani "uctan uca" dedigim olcum
        # gate'siz kosuyordu. Simdi urunun kendi fonksiyonu kullaniliyor.
        avg = np.asarray(d["pbs"], float).mean(0)
        Xp = wire_gate.feats_for(V, F, avg, cps, robot_cp.CE, robot_cp.CT,
                                 step_path=S.get(pid))
        Xp = np.asarray(Xp, float)
        if len(Xp) != len(P):
            raise RuntimeError(f"{pid}: gate oznitelik {len(Xp)} != aday {len(P)}")
        k = maske(np.asarray(wire_gate.karar_skoru(gate, Xp), float), 0.40, 0.30)
        if k.any():
            P, D = P[k], D[k]
        P, D = urun_zinciri.tam_poz(V, F, np.asarray(d["pbs"], float).mean(0),
                                    P, D, step_path=S.get(pid))
        T.append((len(G),) + esle_macar(P, D, G, Gd, r["diag"], 0.0, 180.0, True)[:3])
        R.append((len(G),) + esle_macar(P, D, G, Gd, r["diag"], ROBOT_YANAL,
                                        ROBOT_ACI, False, isaretli=True)[:3])
    return f1w(T), f1w(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--makbuz", default="results/k65b_fewshot_seg.json")
    ap.add_argument("--cikti", default="results/k65b_fewshot_olcum.json")
    a = ap.parse_args()

    with open(a.makbuz) as f:
        mk = json.load(f)
    marka = mk["marka"]
    sv = d6_kayit.sinav()
    kayit = d6_kayit.yukle(set(sv["pidler"]))
    gate = pickle.load(open("results/wire_gate_v5.pkl", "rb"))
    import glob
    S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
    hepsi = sorted(p for p, r in kayit.items() if r["mfg"] == marka)
    print(f"=== {marka}: {len(hepsi)} parca ===\n")

    sonuc = {}
    for k, kosumlar in sorted(mk["kosumlar"].items(), key=lambda t: int(t[0])):
        tl, rl = [], []
        for ko in kosumlar:
            adapt = set(map(str, ko["adapt"]))
            olcp = [p for p in hepsi if p not in adapt]
            # k=0 TABANI AYNI PARCALARDA: adaptasyon parcalari burada da disarida,
            # yoksa iki kol FARKLI kumede olculur ve fark anlamsizlasir.
            t0, r0 = olc(olcp, TABAN_OB, kayit, gate, S)
            t1, r1 = olc(olcp, ko["onbellek"], kayit, gate, S)
            tl.append((t0, t1))
            rl.append((r0, r1))
            print(f"  k={k} cekilis {ko['cekilis']}: tespit {t0:.4f} -> {t1:.4f} "
                  f"({t1-t0:+.4f}) | robot {r0:.4f} -> {r1:.4f} ({r1-r0:+.4f})",
                  flush=True)
        t0m = float(np.mean([x[0] for x in tl])); t1m = float(np.mean([x[1] for x in tl]))
        r0m = float(np.mean([x[0] for x in rl])); r1m = float(np.mean([x[1] for x in rl]))
        sonuc[k] = {"tespit_k0": t0m, "tespit_k": t1m, "tespit_fark": t1m - t0m,
                    "robot_k0": r0m, "robot_k": r1m, "robot_fark": r1m - r0m,
                    "robot_std": float(np.std([x[1] for x in rl])),
                    "n_cekilis": len(kosumlar)}
        print(f"  --> k={k} ORTALAMA: tespit {t1m-t0m:+.4f} | robot {r1m-r0m:+.4f} "
              f"(std {sonuc[k]['robot_std']:.4f})\n", flush=True)

    with open(a.cikti, "w") as f:
        json.dump({"marka": marka, "sonuc": sonuc, "taban": TABAN_OB,
                   "not": "k=0 tabani AYNI parcalarda olculdu; adaptasyon parcalari "
                          "her iki kolda da DISARIDA. Son-epoch ckpt kullanildi "
                          "(genel val'e gore secim adaptasyonu cezalandirir)."},
                  f, indent=1)
    print(f"makbuz -> {a.cikti}")


if __name__ == "__main__":
    main()
