# -*- coding: utf-8 -*-
"""SONDA: yon bankasinin DAGITILABILIR tavani. D6'da olculur, D7'ye DOKUNULMAZ.

Soru: dagitilan havuzun (B-rep birlesik havuz -- `urun_genis.havuz` ile AYNI)
uzerine `yon_bankasi.secenekler` konursa, MUKEMMEL secici ne kadar alir?

Karsilastirma tek degiskenli:
  Y0  : havuzun kendi yonu (bugun dagitilan durum)
  BANK: yon_bankasi (kendi + komsu + silindir + ana), aday basina <= MAX_SEC

Ayrica DAGITILABILIRLIK olculur: aday basina ortalama secenek sayisi ve parca
basina toplam secenek. Tavan betikleri (`sonda_tavan_085.py`) parca basina
binlerce konum kullaniyordu; bu bank onu YAPMAZ.

D7 BUTCESI: bu betik D7'ye BAKMAZ. Ayar ve karar D6'da verilir.
"""
import json
import os
import pickle
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import brep_havuz          # noqa: E402
import d6_kayit            # noqa: E402
import yon_bankasi as YB   # noqa: E402

YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
OB = "results/_p1_olasilik"
N = int(os.environ.get("SONDA_N", "0"))


def recall_tavan(P, D, G, Gd):
    """Kac GT icin KABUL EDILEBILIR bir (konum,yon) cifti var?"""
    if not len(P) or not len(G):
        return 0
    Dn = YB.birim(D)
    Gn = YB.birim(Gd)
    diff = np.asarray(P, float)[:, None, :] - np.asarray(G, float)[None, :, :]
    al = (diff * Gn[None, :, :]).sum(-1)
    yan = np.linalg.norm(diff - al[..., None] * Gn[None, :, :], axis=-1)
    an = np.degrees(np.arccos(np.clip(Dn @ Gn.T, -1.0, 1.0)))
    return int(((yan <= YANAL) & (np.abs(al) <= EKSENEL) & (an <= ACI))
               .any(0).sum())


def main():
    cy = pickle.load(open("results/_d6_silindirler.pkl", "rb"))
    ac = pickle.load(open("results/_d6_acikliklar.pkl", "rb"))
    pidler = [f[:-4] for f in sorted(os.listdir(OB)) if f.endswith(".npz")]
    kay = d6_kayit.yukle(pidler)
    secili = [p for p in pidler if p in kay and len(kay[p].get("G", []))]
    if N:
        secili = secili[:N]
    print(f"D6 {len(secili)} parca", flush=True)

    n_gt = 0
    tut = {"Y0": 0, "BANK": 0}
    n_aday = n_sec = n_parca = 0
    for i, pid in enumerate(secili, 1):
        r = kay[pid]
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        P0 = np.asarray(r["P"], float).reshape(-1, 3)
        D0 = YB.birim(r["Pd"]) if len(P0) else np.zeros((0, 3))
        cyl = cy.get(str(pid))
        Pb, Db, _k = brep_havuz.birlesik_havuz(P0, D0, cyl, ac.get(str(pid)))
        if not len(Pb):
            n_gt += len(G)
            continue
        V = np.asarray(np.load(f"{OB}/{pid}.npz")["V"], float)
        idx, YD, _OZ = YB.secenekler(Pb, Db, cyl, V)
        n_gt += len(G)
        tut["Y0"] += recall_tavan(Pb, Db, G, Gd)
        tut["BANK"] += recall_tavan(Pb[idx], YD, G, Gd)
        n_aday += len(Pb)
        n_sec += len(idx)
        n_parca += 1
        if i % 100 == 0:
            print(f"  {i}/{len(secili)}", flush=True)

    out = {"n_gt": n_gt, "n_parca": n_parca,
           "aday_basina_secenek": n_sec / max(n_aday, 1),
           "parca_basina_secenek": n_sec / max(n_parca, 1),
           "parca_basina_aday": n_aday / max(n_parca, 1), "kol": {}}
    print(f"\n{'kol':<8} {'recall':>8} {'F1 tavani':>10}")
    for ad in ("Y0", "BANK"):
        rc = tut[ad] / max(n_gt, 1)
        f1 = 2 * rc / (1 + rc)
        out["kol"][ad] = {"recall": rc, "f1_tavani": f1, "tutan": tut[ad]}
        print(f"{ad:<8} {rc:>8.4f} {f1:>10.4f}")
    d = out["kol"]["BANK"]["f1_tavani"] - out["kol"]["Y0"]["f1_tavani"]
    print(f"\nBANK - Y0 tavan farki: {d:+.4f}")
    print(f"aday basina secenek: {out['aday_basina_secenek']:.1f} | "
          f"parca basina aday {out['parca_basina_aday']:.0f} -> "
          f"secenek {out['parca_basina_secenek']:.0f}")
    json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
               "not": "Yon bankasi TAVANI, dagitilan havuz uzerinde. MUKEMMEL "
                      "secici. D6 -- D7'ye BAKILMADI."},
              open("results/yon_bankasi_tavan_d6.json", "w"), indent=1)
    print("makbuz -> results/yon_bankasi_tavan_d6.json")


if __name__ == "__main__":
    main()
