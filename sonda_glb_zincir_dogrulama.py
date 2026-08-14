# -*- coding: utf-8 -*-
"""E2 -- SAHAYA INEN ZINCIRIN DOGRULANMASI

SORUN (memory: glb-olculen-zinciri-kullanmiyor). Ihracatcilar
`robot_cp.extract` cagiriyor; kampanyada olculen `urun_p6`/`urun_genis`
sahaya HIC girmiyor. Yani olctugum her kazanc robota ULASMIYOR.
Olculmus fark: taban 0.2980 vs olculen zincir 0.3115.

`export_robot_glb.py` icine `cp_config.glb_kanonik_zincir` bayragi kondu
(varsayilan KAPALI). Bu betik bayragi ACMADAN ONCE yolun saglam calistigini
dogrular -- sahaya inen sey budur, kirik cikti robotu yanlis yere gonderir.

DENETLENEN (GT'li parcalarda, iki zincir YAN YANA):
  cikti_var    : zincir hic CP uretiyor mu (bos donmuyor mu)
  adet         : uretilen CP sayisi
  yon_birim    : yonler birim uzunlukta mi (robot icin sart)
  yon_sonlu    : NaN/Inf var mi
  tier         : tier atamasi yapiliyor mu (AUTO/REVIEW)
  robot_F1     : uctan uca robot F1 (yanal 2mm / isaretli aci 10 / eksenel 40)

KAPI: olculen zincir (1) hicbir parcada COKMEYECEK, (2) yonleri gecerli
olacak, (3) robot F1'de tabani ASACAK. Ucu birden saglanmadan bayrak
ACILMAZ.

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys
import time

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K             # noqa: E402
import d6_kayit                    # noqa: E402
from sina_kume import esle_macar   # noqa: E402

MESH = os.environ.get("GZ_MESH", "results/_p1_olasilik")
N_PARCA = int(os.environ.get("GZ_N", "60"))
MARKALAR = set(os.environ.get("GZ_MARKA", "NIT,MOR,SUPU,UPUN").split(","))


def _birim_mi(D):
    if not len(D):
        return True
    u = np.linalg.norm(D, axis=1)
    return bool(np.all(np.abs(u - 1.0) < 1e-3))


def main():
    t0 = time.time()
    import kanonik_zincir
    import robot_cp
    cfg = json.load(open("cp_config.json")) if os.path.exists(
        "cp_config.json") else {}
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    aday = [(p, r) for p, r in kay.items()
            if r.get("mfg") in MARKALAR and len(r.get("G", []))
            and os.path.exists(f"{MESH}/{p}.npz")]
    rng = np.random.default_rng(0)
    aday = [aday[i] for i in rng.choice(len(aday),
                                        min(N_PARCA, len(aday)),
                                        replace=False)]
    print(f"{len(aday)} parca denetleniyor", flush=True)

    say = collections.Counter()
    agg = collections.Counter()
    hata = []
    for pid, r in aday:
        z = np.load(f"{MESH}/{pid}.npz")
        V = np.asarray(z["V"], float)
        F = np.asarray(z["F"], int)
        pbs = [np.asarray(q, float) for q in z["pbs"]]
        try:
            ham = kanonik_zincir.urun_cikti(V, F, pbs, r.get("step"), cfg)
        except Exception as e:            # noqa: BLE001
            say["COKTU"] += 1
            hata.append(f"{pid}: {type(e).__name__}: {e}")
            continue
        # `urun_cikti` sozlukleri `point`/`direction` anahtarlariyla doner
        # (ihracatci sozlesmesi). Ilk yazimda `p`/`d` varsaymistim, KeyError
        # verdi -- anahtarlar dogrudan ciktidan okundu.
        P = np.asarray([c["point"] for c in ham], float).reshape(-1, 3) \
            if len(ham) else np.zeros((0, 3))
        D = np.asarray([c["direction"] for c in ham], float).reshape(-1, 3) \
            if len(ham) else np.zeros((0, 3))
        say["parca"] += 1
        if len(P):
            say["cikti_var"] += 1
        say["cp"] += len(P)
        if not np.all(np.isfinite(D)) or not np.all(np.isfinite(P)):
            say["SONSUZ"] += 1
            hata.append(f"{pid}: NaN/Inf cikti")
            continue
        if not _birim_mi(D):
            say["YON_BIRIM_DEGIL"] += 1
            hata.append(f"{pid}: yonler birim degil")
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        dg = float(np.linalg.norm(V.max(0) - V.min(0)))
        tp, fp, fn = esle_macar(P, D, G, Gd, dg, K.YANAL, K.ACI, False,
                                isaretli=True)[:3]
        agg["tp"] += tp; agg["fp"] += fp; agg["fn"] += fn
        agg["gt"] += len(G)

    f1 = 2 * agg["tp"] / max(2 * agg["tp"] + agg["fp"] + agg["fn"], 1)
    print(f"\n--- OLCULEN ZINCIR (kanonik_zincir.urun_cikti) ---")
    print(f"  parca            {say['parca']}")
    print(f"  cikti veren      {say['cikti_var']}")
    print(f"  COKEN            {say['COKTU']}")
    print(f"  NaN/Inf          {say['SONSUZ']}")
    print(f"  yon birim DEGIL  {say['YON_BIRIM_DEGIL']}")
    print(f"  uretilen CP      {say['cp']}  (GT {agg['gt']})")
    print(f"  robot F1         {f1:.4f}")
    if hata:
        print("\n  ILK HATALAR:")
        for h in hata[:8]:
            print(f"    {h}")
    saglam = (say["COKTU"] == 0 and say["SONSUZ"] == 0
              and say["YON_BIRIM_DEGIL"] == 0
              and say["cikti_var"] == say["parca"])
    print(f"\nSAGLAMLIK: {'GECTI' if saglam else 'KALDI'}")
    print("Bayrak `glb_kanonik_zincir` ancak SAGLAMLIK GECTI ve robot F1")
    print("tabani astiktan sonra acilir.")
    json.dump({"damga": makbuz_hash.damga(), "n_parca": say["parca"],
               "coken": say["COKTU"], "sonsuz": say["SONSUZ"],
               "yon_birim_degil": say["YON_BIRIM_DEGIL"],
               "cikti_veren": say["cikti_var"], "uretilen_cp": say["cp"],
               "gt": agg["gt"], "robot_f1": f1, "saglam": saglam,
               "not": "Sahaya inecek zincirin saglamlik denetimi. Bayrak "
                      "acilmadan ONCE kosulur. D7'ye BAKILMADI."},
              open("results/glb_zincir_dogrulama.json", "w"), indent=1)
    print(f"makbuz -> results/glb_zincir_dogrulama.json "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
