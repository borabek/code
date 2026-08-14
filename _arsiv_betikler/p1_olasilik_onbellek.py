# -*- coding: utf-8 -*-
"""P1 ONKOSUL: parca basina AG OLASILIKLARINI bir kez hesapla, diske yaz.

NEDEN: P1 (aday birlesmesini engelle) vote-pooling / cluster / dedupe yaricaplarini
taramak istiyor. Bu parametreler `cp_openings.connection_points` icinde, yani
adaylari YENIDEN uretmek gerekiyor -- ama turetme kayitlari yalnizca SONUC adaylari
tasiyor, olasiliklari degil.

Olasiliklari BIR KEZ hesaplayip onbellege alirsak (parca basina ~15s), sonraki her
parametre kombinasyonu SANIYELER surer. Ayni onbellek P4 (poz secenekleri) ve P5
(gate+poz birlikte) icin de girdi olur.

TEZ DEGISMEZ: ayni ~6000 uniform izotropik remesh, ayni 5 sinif, ayni ckpt'ler.
Yalnizca ara sonuc saklaniyor.

Disk: parca basina ~6000 tepe x 5 sinif x 2 model x float16 = ~120 KB.
468 parca -> ~56 MB. Mesh de saklanir (V, F float32/int32) -> ~300 KB/parca.
"""
import argparse
import glob
import io
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ONBELLEK = "results/_p1_olasilik"
DISK_ESIK_GB = 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", default="results/d6_sinav_kumesi.json")
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/seg_g5/g5_s0.pt", "results/seg_g5/g5_s1.pt"])
    ap.add_argument("--vardiya", type=int, default=0)
    ap.add_argument("--toplam", type=int, default=1)
    ap.add_argument("--ek", default="")
    a = ap.parse_args()
    import protokol
    protokol.tez_dogrula()
    import torch
    import diffusionnet as D_
    import thesis_remesh
    from infer_step_cp import load_any, step_to_mesh
    from korpus_kimlik import step_kimlik as SK

    kok = ONBELLEK + a.ek
    os.makedirs(kok, exist_ok=True)
    with io.open(a.kume, encoding="utf-8") as f:
        sv = json.load(f)
    S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
    var = {os.path.splitext(x)[0] for x in os.listdir(kok) if x.endswith(".npz")}
    hedef = [p for p in sv["pidler"] if p in S and p not in var]
    if a.toplam > 1:
        hedef = [p for i, p in enumerate(hedef) if i % a.toplam == a.vardiya]
    print(f"kume {a.kume} muhur {sv['sha16']} | onbellekte {len(var)} | "
          f"HESAPLANACAK {len(hedef)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in a.ckpt]
    print(f"  ckpt: {[c.split('/')[-1] for c in a.ckpt]} | cihaz {dev}", flush=True)

    t0 = time.time(); hata = 0
    for k, pid in enumerate(hedef, 1):
        if k % 20 == 0:
            hiz = (time.time() - t0) / k
            print(f"  {k}/{len(hedef)}  {hiz:.1f}s/parca  hata={hata}  "
                  f"kalan ~{hiz*(len(hedef)-k)/60:.0f} dk", flush=True)
            if shutil.disk_usage(".").free / 1e9 < DISK_ESIK_GB:
                print("  ** DISK ESIGI -- TEMIZ DURULUYOR **", flush=True); break
        try:
            Vr, Fr = step_to_mesh(S[pid])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            pbs = []
            for model, meta in models:
                _, pb = D_.predict(model, meta, V, F, device=dev,
                                   op_cache_dir=None, return_probs=True)
                pbs.append(np.asarray(pb, np.float16))
            np.savez_compressed(os.path.join(kok, pid + ".npz"),
                                V=V.astype(np.float32), F=F.astype(np.int32),
                                pbs=np.stack(pbs))
        except Exception as e:
            hata += 1
            if hata <= 5:
                print(f"    {pid}: {type(e).__name__}: {str(e)[:70]}", flush=True)
    print(f"BITTI: {len(os.listdir(kok))} parca onbellekte, hata {hata}")


if __name__ == "__main__":
    main()
