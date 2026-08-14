# -*- coding: utf-8 -*-
"""SEG-6 — ONBELLEK vs TAZE OLASILIK: zincirin basi BAYAT mi?

BULGU. `results/_p1_olasilik/*.npz` dosyalari **6 Agustos** tarihli ve
`pbs` sekli **(2, N, 5)** -- yani YALNIZ IKI model. Canli yapilandirma
(`cp_config.robot_vote2_checkpoints`) ise **DORT** kontrol noktasi
kullaniyor. Tum P6 oznitelik zinciri bu onbellekten besleniyor.

Sabahki otopsi "NIT'te segmentasyon ayrimi 1.19x" dedi ve o sayi bu
ONBELLEKTEN geldi. Ayni olcu, CANLI dort modelle taze hesaplandiginda
16 parcalik bir on denemede **6.34x** cikti. Fark mekanizmadan degil
BAYAT ONBELLEKTEN geliyor olabilir.

Onemi: eger taze olasilik belirgin sekilde daha ayirt ediciyse, YENIDEN
EGITIM OLMADAN, yalnizca onbellegi tazeleyerek zincirin basi duzelir.

BU BETIK ayni parcalarda IKISINI birden olcer:
  onbellek : npz icindeki `pbs` (bugun kullanilan)
  taze     : canli dort kontrol noktasiyla yeniden cikarim
Olcut ayni: GT'deki olasilik / rastgele yuzeydeki olasilik, ve AUC.

D7'ye BAKILMAZ (olcum d6 parcalarinda).
"""
import collections
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import connector3d       # noqa: E402
import d6_kayit          # noqa: E402

CE = int(connector3d.CABLE_ENTRY)
CT = int(connector3d.CONTACT)
MESH = os.environ.get("OT_MESH", "results/_p1_olasilik")
MARKALAR = set(os.environ.get("OT_MARKA", "NIT,MOR,SUPU,UPUN").split(","))
N_PARCA = int(os.environ.get("OT_N", "60"))


def auc(poz, neg):
    if not len(poz) or not len(neg):
        return float("nan")
    h = np.concatenate([poz, neg])
    r = np.argsort(np.argsort(h)) + 1.0
    return float((r[:len(poz)].sum() - len(poz) * (len(poz) + 1) / 2.0)
                 / (len(poz) * len(neg)))


def main():
    import torch
    import diffusionnet as D
    from infer_step_cp import load_any
    cfg = json.load(open("cp_config.json"))
    cks = cfg["robot_vote2_checkpoints"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modeller = [load_any(c, dev=dev)[:2] for c in cks]
    print(f"canli kontrol noktasi: {len(cks)} | cihaz {dev}", flush=True)

    kay = d6_kayit.yukle()
    aday = [(str(p), r) for p, r in kay.items()
            if r.get("mfg") in MARKALAR and len(r.get("G", []))
            and os.path.exists(f"{MESH}/{p}.npz")]
    rng = np.random.default_rng(0)
    if len(aday) > N_PARCA:
        aday = [aday[i] for i in rng.choice(len(aday), N_PARCA,
                                            replace=False)]
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n_ob = None
    for i, (pid, r) in enumerate(aday, 1):
        z = np.load(f"{MESH}/{pid}.npz")
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        ob = np.asarray(z["pbs"], float)
        n_ob = ob.shape[0]
        p_ob = ob.mean(0)[:, CE] + ob.mean(0)[:, CT]
        tz = []
        for m, meta in modeller:
            _, pb = D.predict(m, meta, V, F, device=dev, return_probs=True)
            tz.append(np.asarray(pb, float))
        tz = np.mean(tz, axis=0)
        p_tz = tz[:, CE] + tz[:, CT]

        G = np.asarray(r["G"], float)
        yak = np.argmin(np.linalg.norm(G[:, None, :] - V[None, :, :],
                                       axis=-1), axis=1)
        uzak = np.where(np.min(np.linalg.norm(
            V[:, None, :] - G[None, :, :], axis=-1), axis=1) >= 4.0)[0]
        if len(uzak) < 50:
            continue
        rr = np.random.default_rng(0)
        ri = rr.choice(uzak, min(400, len(uzak)), replace=False)
        a = ist[r["mfg"]]
        a["gt"].append(len(G))
        for ad, p in (("onbellek", p_ob), ("taze", p_tz)):
            a[f"{ad}_gt"].append(float(np.median(p[yak])))
            a[f"{ad}_zemin"].append(float(np.median(p[ri])))
            a[f"{ad}_auc"].append(auc(p[yak], p[ri]))
        if i % 15 == 0:
            print(f"  {i}/{len(aday)}", flush=True)

    print(f"\nonbellekteki model sayisi: {n_ob} | canli: {len(cks)}")
    print(f"{'marka':<7}{'GT':>6}{'ONB oran':>11}{'ONB auc':>10}"
          f"{'TAZE oran':>12}{'TAZE auc':>10}{'auc farki':>11}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        r = {}
        for ad in ("onbellek", "taze"):
            g = float(np.median(a[f"{ad}_gt"]))
            z = float(np.median(a[f"{ad}_zemin"]))
            r[f"{ad}_oran"] = g / max(z, 1e-9)
            r[f"{ad}_auc"] = float(np.nanmean(a[f"{ad}_auc"]))
        r["gt"] = int(sum(a["gt"]))
        r["auc_farki"] = r["taze_auc"] - r["onbellek_auc"]
        out[m_] = r
        print(f"{m_:<7}{r['gt']:>6}{r['onbellek_oran']:>10.2f}x"
              f"{r['onbellek_auc']:>10.4f}{r['taze_oran']:>11.2f}x"
              f"{r['taze_auc']:>10.4f}{r['auc_farki']:>+11.4f}")
    print("\nOKUMA: TAZE auc belirgin yuksekse zincirin basi BAYAT ONBELLEK")
    print("       yuzunden zayif demektir -- yeniden egitim GEREKMEDEN,")
    print("       yalnizca onbellegi tazeleyerek duzelir.")
    json.dump({"onbellek_model": n_ob, "canli_model": len(cks),
               "marka": out,
               "not": "Onbellekteki pbs vs canli modellerle taze cikarim, "
                      "AYNI parcalarda. D7'ye BAKILMADI."},
              open("results/onbellek_vs_taze.json", "w"), indent=1)
    print("makbuz -> results/onbellek_vs_taze.json")


if __name__ == "__main__":
    main()
