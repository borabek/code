# -*- coding: utf-8 -*-
"""SEG-3 — YENI KONTROL NOKTASI NIT'te AYIRT EDIYOR MU?

Bu, `sonda_otopsi_segmentasyon.py`'nin AYNI olcumudur; tek fark olasiliklarin
onbellekten degil VERILEN KONTROL NOKTASINDAN hesaplanmasidir. Boylece A
(kontrol) ve B (+GT korpusu) kollari AYNI olcutle kiyaslanir.

BUGUN OLCULEN TABAN (canli, 27 Temmuz kontrol noktalari):

| marka | GT'de olasilik | rastgele yuzey | oran |
|---|---|---|---|
| SUPU | 0.4815 | 0.0058 | 83x |
| UPUN | 0.4448 | 0.0219 | 20x |
| MOR  | 0.2075 | 0.1289 | 1.6x |
| NIT  | 0.5244 | 0.4394 | **1.19x** |

KAPI (ONCE ILAN EDILDI): B kolu NIT'te oran >= 2.0 yapacak. Yapmazsa
segmentasyon kolu OLU ve zincirin basi duzelmiyor demektir.

OLCUM d6 PARCALARINDA yapilir -- onlar egitime GIRMEDI (sizinti bekcisi
835 d6/d7 parcasini disarida tuttu), yani bu temiz bir okumadir.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import connector3d       # noqa: E402
import d6_kayit          # noqa: E402


def auc(poz, neg):
    """Mann-Whitney AUC. ORAN (ortanca tabanli) ile AUC (sira tabanli)
    AYRI seyler ve ters yonde hareket edebiliyorlar -- onbellek/taze
    kiyasinda tam bu oldu (oran yukseldi, AUC dustu). Siralama icin
    belirleyici olan AUC'dir, cunku secici SIRALIYOR."""
    import numpy as _np
    if not len(poz) or not len(neg):
        return float("nan")
    h = _np.concatenate([poz, neg])
    r = _np.argsort(_np.argsort(h)) + 1.0
    return float((r[:len(poz)].sum() - len(poz) * (len(poz) + 1) / 2.0)
                 / (len(poz) * len(neg)))

CE = int(connector3d.CABLE_ENTRY)
CT = int(connector3d.CONTACT)
MESH = os.environ.get("SY_MESH", "results/_p1_olasilik")
MARKALAR = set(os.environ.get("SY_MARKA", "NIT,MOR,SUPU,UPUN").split(","))
N_PARCA = int(os.environ.get("SY_N", "120"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--etiket", default="")
    a = ap.parse_args()

    import torch
    import diffusionnet as D
    from infer_step_cp import load_any
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modeller = [load_any(c, dev=dev)[:2] for c in a.ckpt]
    print(f"{len(modeller)} kontrol noktasi | cihaz {dev}", flush=True)

    kay = d6_kayit.yukle()
    aday = [(str(p), r) for p, r in kay.items()
            if r.get("mfg") in MARKALAR and len(r.get("G", []))
            and os.path.exists(f"{MESH}/{p}.npz")]
    rng = np.random.default_rng(0)
    if len(aday) > N_PARCA:
        aday = [aday[i] for i in rng.choice(len(aday), N_PARCA,
                                            replace=False)]
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, (pid, r) in enumerate(aday, 1):
        z = np.load(f"{MESH}/{pid}.npz")
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        pbs = []
        for m, meta in modeller:
            _, pb = D.predict(m, meta, V, F, device=dev, return_probs=True)
            pbs.append(np.asarray(pb, float))
        pb = np.mean(pbs, axis=0)
        p = pb[:, CE] + pb[:, CT]
        G = np.asarray(r["G"], float)
        yak = np.argmin(np.linalg.norm(G[:, None, :] - V[None, :, :],
                                       axis=-1), axis=1)
        rr = np.random.default_rng(0)
        aa = ist[r["mfg"]]
        aa["gt"].append(len(G))
        aa["gt_p"].append(float(np.median(p[yak])))
        _uzak = np.where(np.min(np.linalg.norm(
            V[:, None, :] - G[None, :, :], axis=-1), axis=1) >= 4.0)[0]
        _ri = (rr.choice(_uzak, min(400, len(_uzak)), replace=False)
               if len(_uzak) >= 50
               else rr.choice(len(V), min(400, len(V)), replace=False))
        aa["zemin"].append(float(np.median(p[_ri])))
        aa["auc"].append(auc(p[yak], p[_ri]))
        if i % 20 == 0:
            print(f"  {i}/{len(aday)}", flush=True)

    print(f"\n{'marka':<7}{'GT':>7}{'GT olasilik':>13}{'rastgele':>11}"
          f"{'oran':>9}{'AUC':>9}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        aa = ist[m_]
        g = float(np.median(aa["gt_p"]))
        z = float(np.median(aa["zemin"]))
        o = g / max(z, 1e-6)
        u = float(np.nanmean(aa["auc"])) if aa["auc"] else float("nan")
        out[m_] = {"gt": int(sum(aa["gt"])), "gt_olasilik": g,
                   "zemin": z, "oran": o, "auc": u}
        print(f"{m_:<7}{sum(aa['gt']):>7}{g:>13.4f}{z:>11.4f}{o:>9.2f}x"
              f"{u:>9.4f}")
    if "NIT" in out:
        o = out["NIT"]["oran"]
        print(f"\nKAPI: NIT orani >= 2.0 (bugunku taban 1.19x)")
        print(f"  olculen {o:.2f}x -> "
              f"{'GECTI' if o >= 2.0 else 'KALDI'}")
    ad = a.etiket or os.path.basename(a.ckpt[0]).replace(".pt", "")
    json.dump({"ckpt": a.ckpt, "marka": out,
               "not": "Yeni kontrol noktasiyla segmentasyon ayrimi. Olcum "
                      "d6 parcalarinda; d6 egitime GIRMEDI. D7'ye BAKILMADI."},
              open(f"results/seg_ckpt_{ad}.json", "w"), indent=1)
    print(f"makbuz -> results/seg_ckpt_{ad}.json")


if __name__ == "__main__":
    main()
