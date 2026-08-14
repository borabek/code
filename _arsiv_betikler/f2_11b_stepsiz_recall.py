# -*- coding: utf-8 -*-
"""F2-11b: STEP'SIZ 2759 PARCANIN JSON AGI KULLANILABILIR MI? (STEP GEREKMEZ)

F2-11 SONUCU: JSON agi genel olarak parite SAGLAMADI (aday recall 0.8579 -> 0.5374).
AMA uretici kirilimi kapiyi kapatmadi:

    PXC   STEP recall 0.8450 -> JSON 0.2999   (-0.5451)  COKUS
    WEI   STEP recall 0.8764 -> JSON 0.8858   (+0.0094)  KAYIP YOK

Ve bu COZUNURLUK DEGIL: iki ureticinin JSON agi neredeyse ayni yogunlukta (medyan
1934 vs 1926 tepe). Yani sorun URETICIYE OZGU TESSELASYON BICIMI -- WEI'nin agi
acikliklari koruyor, PXC'ninki kapatiyor.

BU BETIGIN FIKRI: aday recall'i olcmek icin STEP GEREKMEZ -- yalniz adaylar ve GT yeter.
Dolayisiyla STEP'i HIC OLMAYAN ureticiler (A-B 822, CWT 785, ABB 324, SIE 311, KLM 101)
icin de recall DOGRUDAN olculebilir. Sonuc WEI'ye benziyorsa o veri KULLANILABILIR;
PXC'ye benziyorsa kullanilamaz.

TEZ CIZGISI: ayni ag, ayni uniform ~6000 remesh, ayni aday ureticisi. Degisen tek sey
mesh kaynagi. STEP olmadigi icin B-rep ozellikleri uretilmez (gate'e girmez) ama RECALL
gate'ten ONCEKI asamadir -- bu olcum ondan etkilenmez.

UC SISME KAYNAGI VE NASIL KAPATILDIGI (bu olcum kolayca yaniltir):

  1. ADAY SAYISI SISMESI. Recall, aday uretirsen BEDAVAYA yukselir. Bu yuzden her
     satirda ADAY/GT ORANI da raporlanir. (Referans: STEP kolu 2.43 aday/GT, JSON kolu
     1.67 -- JSON daha AZ aday uretip daha DUSUK recall veriyor, yani tutarli.)

  2. HIZALAMA AVANTAJI. JSON yolunda GT zaten AYNI CERCEVEDE, `align_frames` YOK.
     WEI'nin +0.0094'u TAMAMEN bu olabilir -- gercek ustunluk degil, EKSIK HATA KAYNAGI.
     Bu yuzden JSON recall'i STEP'ten YUKSEK cikan hicbir sonuc "iyilesme" sayilmaz.

  3. EN TEHLIKELISI -- URETICI TANIDIKLIGI. Ag agirlikli olarak WEI ve PXC uzerinde
     egitildi (1926 uygun parcanin 1011'i WEI, 896'si PXC). SIE/A-B/CWT'de dusuk recall
     "JSON agi kotu" mu demek, "ag o ureticiyi tanimiyor" mu? YALNIZ JSON ILE AYRISTIRILAMAZ.

     COZUM: olcum IKIYE ayrilir.
       ESLESTIRILMIS  (hem STEP hem JSON olan: PXC, WEI, DIN, WAGO) -> fark YALNIZ mesh
                      kaynagini izole eder, tanidiklik sadelesir. ASIL KANIT BUDUR.
       ESLESTIRILMEMIS (A-B, CWT, ABB, SIE, KLM...) -> mutlak recall; TANIDIKLIKLA
                      KARISIK, tek basina kanit DEGIL, yalniz gosterge.

KILL (yalniz ESLESTIRILMIS kolda gecerli): STEP->JSON recall kaybi <= 0.05 ise o
ureticinin JSON agi kullanilabilir. (WEI kayip +0.0094 = kayipsiz; PXC -0.5451 = cokus.)
"""
import argparse
import collections
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f2_11_parite import json_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12, help="uretici basina parca")
    a = ap.parse_args()

    import protokol
    protokol.tez_dogrula()
    import torch
    import diffusionnet as D_
    import robot_cp as RC
    import thesis_remesh
    from infer_step_cp import load_any

    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    man = json.load(io.open("results/manifest_korpus.json", encoding="utf-8"))
    # STEP'i OLMAYAN, CP'si ve mesh'i olan parcalar
    aday = [m for m in man if not m["step"] and m["cp"] > 0 and m["g3d_tepe"] > 0]
    grup = collections.defaultdict(list)
    for m in aday:
        grup[m["uretici"]].append(m)
    hedef = []
    rng = np.random.RandomState(0)
    for u, v in sorted(grup.items(), key=lambda x: -len(x[1])):
        if len(v) < 5:
            continue
        idx = rng.permutation(len(v))[:a.per]
        hedef += [v[i] for i in idx]
    print(f"STEP'siz uretici sayisi {len(grup)} | orneklenen {len(hedef)} parca")
    print("  " + ", ".join(f"{u}:{min(len(v),a.per)}" for u, v in
                           sorted(grup.items(), key=lambda x: -len(x[1])) if len(v) >= 5))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cks = cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]
    models = [load_any(c, dev=dev)[:2] for c in cks]

    SON = collections.defaultdict(lambda: [0, 0, 0, []])
    t0 = time.time(); hata = 0
    for k, m in enumerate(hedef, 1):
        if k % 15 == 0:
            print(f"  {k}/{len(hedef)}  {time.time()-t0:.0f}s  hata={hata}", flush=True)
        jf = os.path.join("_ds1/DataSet", m["dosya"])
        try:
            Vj, Fj, j = json_mesh(jf)
            if Vj is None or len(Fj) < 4:
                hata += 1; continue
            V, F = thesis_remesh.remesh_uniform(Vj, Fj, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            pbs = []
            for model, meta in models:
                _, pb = D_.predict(model, meta, V, F, device=dev,
                                   op_cache_dir=f"results/step_infer/ops_json_k{int(meta.get('k_eig',64))}",
                                   return_probs=True)
                pbs.append(np.asarray(pb, float))
            cps, probs, _, _ = RC.adaylari_uret(V, F, pbs, None, cfg=cfg)
            P = np.array([c["point"] for c in cps], float) if cps else np.zeros((0, 3))
            g = j.get("ConnectionPoints") or []
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in g], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in g], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            diag = float(np.linalg.norm(V.max(0) - V.min(0)))
            hit = 0
            if len(P) and len(G):
                d = P[:, None, :] - G[None, :, :]
                al = (d * Gd[None, :, :]).sum(-1)
                pe = np.linalg.norm(d - al[..., None] * Gd[None, :, :], axis=-1)
                pe = np.where(np.abs(al) > 40, np.inf, pe)
                hit = int((pe.min(0) <= max(3.0, 0.06 * diag)).sum())
            s = SON[m["uretici"]]
            s[0] += hit; s[1] += len(G); s[2] += len(P); s[3].append(len(Vj))
        except Exception as e:
            hata += 1
            if hata <= 3:
                print(f"    {m['parca']}: {type(e).__name__}: {str(e)[:70]}")

    print(f"\n{'='*84}\nESLESTIRILMEMIS KOL -- STEP'SIZ URETICILERDE JSON-AGI ADAY RECALL\n"
          f"{'='*84}")
    print("DIKKAT: bu satirlar TANIDIKLIKLA KARISIK. Ag agirlikli olarak WEI+PXC uzerinde")
    print("egitildi; gorulmemis bir ureticide dusuk recall 'JSON agi kotu' DEGIL 'ag bu")
    print("ureticiyi tanimiyor' da olabilir. TEK BASINA KANIT DEGIL -- gosterge.\n")
    print(f"{'uretici':<10}{'parca':>6}{'GT':>6}{'aday':>7}{'aday/GT':>9}{'RECALL':>9}"
          f"{'JSON tepe':>11}   gosterge")
    R = {}
    for u in sorted(SON, key=lambda x: -SON[x][1]):
        h, n, na, jt = SON[u]
        rc = h / max(n, 1)
        oran = na / max(n, 1)
        # ADAY SISMESI KONTROLU: aday/GT orani referansin (STEP 2.43) COK USTUNDEYSE
        # yuksek recall bedavaya gelmis olabilir; isaretle.
        sis = " [ADAY SISMESI?]" if oran > 3.5 else ""
        hk = ("WEI-benzeri" if rc >= 0.70 else
              "SINIRDA" if rc >= 0.55 else "PXC-benzeri (cokus)")
        R[u] = {"recall": rc, "gt": n, "aday": na, "aday_per_gt": oran, "parca": len(jt),
                "json_tepe_medyan": float(np.median(jt)) if jt else 0}
        print(f"{u:<10}{len(jt):>6}{n:>6}{na:>7}{oran:>9.2f}{rc:>9.4f}"
              f"{np.median(jt) if jt else 0:>11.0f}   {hk}{sis}")
    print(f"\n  REFERANS (ESLESTIRILMIS, tanidiklik sadelesmis):")
    print(f"    WEI  STEP 0.8764 -> JSON 0.8858  (+0.0094)  KAYIPSIZ")
    print(f"    PXC  STEP 0.8450 -> JSON 0.2999  (-0.5451)  COKUS")
    print(f"    aday/GT referansi: STEP kolu 2.43 | JSON kolu 1.67")
    ok = [u for u, v in R.items() if v["recall"] >= 0.70 and v["aday_per_gt"] <= 3.5]
    print(f"\n  WEI-benzeri davranan (gosterge): {ok if ok else 'YOK'}")
    print("  KESIN HUKUM ICIN o ureticiden STEP'li birkac parca gerekir (eslestirilmis olcum).")
    with io.open("results/f2_11b_stepsiz_recall.json", "w", encoding="utf-8") as f:
        json.dump({"referans_eslestirilmis": {"WEI": {"step": 0.8764, "json": 0.8858},
                                              "PXC": {"step": 0.8450, "json": 0.2999}},
                   "eslestirilmemis_gosterge": R, "wei_benzeri": ok,
                   "uyari": "tanidiklikla karisik; tek basina kanit degil"},
                  f, indent=1, ensure_ascii=False)
    print("makbuz -> results/f2_11b_stepsiz_recall.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
