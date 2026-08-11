# -*- coding: utf-8 -*-
"""DUMAN TESTI: URUN GERCEKTEN CALISIYOR MU? (olcumler degil, URUN)

NEDEN YAZILDI (2026-08-04): `diffusion_net` .venv'den dusmustu ve URUNUN DAGITILAN
DORT CHECKPOINT'I DE YUKLENEMIYORDU -- yani robot yeni bir WSCAD parcasini isleyemezdi.
Bunu HICBIR SEY yakalamadi:
  * butun olcum betikleri `_der_tam.pkl` / `_r4_sozluk.pkl` ONBELLEKLERINDEN okuyor,
    model yuklemiyorlar -> hepsi yesil kosmaya devam etti;
  * `run_selftests.py` eksik `diffusion_net`'i acikca **SKIP** sayiyor, FAIL degil;
  * `run_on_file.py` eski `connector3d` hattini kosuyor, guncel robot yolunu degil.
Sonuc: urun bozuktu ve gostergelerin hepsi yesildi. Bu ancak demo gunu anlasilirdi.

BU TEST TEK SEYI YAPAR: gercek bir STEP dosyasini bastan sona URUNUN yolundan gecirir
    STEP -> mesh -> tez remesh (6000) -> 4 checkpoint cikarimi -> aday uretimi
    -> wire-gate karari -> poz/aci/uye/yon duzeltmeleri
ve sonunda CP CIKMASINI SART KOSAR. Cikmazsa EXIT 1.

Kullanim:
    python duman_testi.py            # korpustan ilk uygun parca
    python duman_testi.py <step>     # belirli bir dosya
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import io
    import json
    import torch
    import robot_cp as RC
    import thesis_remesh
    import wire_gate
    import diffusionnet as D_
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from infer_step_cp import load_any, step_to_mesh

    t0 = time.time()
    with io.open("cp_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cks = cfg["current_product"].get("checkpoints") or cfg["robot_vote2_checkpoints"]

    if len(sys.argv) > 1:
        stp = sys.argv[1]
    else:
        from big_arbiter import eligible
        E = eligible()
        assert E, "korpusta uygun parca yok"
        stp = E[0][3]
    print(f"parca: {os.path.basename(stp)}")

    # --- 1) CHECKPOINT'LER YUKLENIYOR MU (asil kirilma noktasi)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for c in cks:
        models.append(load_any(c, dev=dev)[:2])
    print(f"[1/5] {len(models)} checkpoint yuklendi ({dev})")

    # --- 2) STEP -> mesh -> tez remesh
    Vr, Fr = step_to_mesh(stp)
    V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    assert len(V) > 1000, f"remesh cok az tepe uretti: {len(V)}"
    print(f"[2/5] mesh {len(Vr)} -> remesh {len(V)} tepe / {len(F)} yuz")

    # --- 3) cikarim
    pbs = []
    for model, meta in models:
        _, pb = D_.predict(model, meta, V, F, device=dev,
                           op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig',64))}",
                           return_probs=True)
        pbs.append(np.asarray(pb, float))
    assert len(pbs) == len(cks)
    print(f"[3/5] cikarim tamam, olasilik boyutu {pbs[0].shape}")

    # --- 4) aday uretimi
    cps, probs, is_hi, uyeler = RC.adaylari_uret(V, F, pbs, stp, cfg=cfg)
    assert cps, "HIC ADAY URETILMEDI"
    print(f"[4/5] {len(cps)} aday (cok-CP rejimi: {is_hi})")

    # --- 5) gate karari + duzeltmeler = URUNUN CIKTISI
    X = wire_gate.feats_for(V, F, probs, cps, CE, CT, step_path=stp)
    gate = wire_gate._load()
    assert gate is not None, "wire_gate.pkl yuklenemedi"
    sk = wire_gate.karar_skoru(gate, X)
    k = wire_gate.karar_maskesi(sk)
    assert k.any(), f"GATE HEPSINI REDDETTI (skorlar {sk.min():.3f}-{sk.max():.3f})"
    c = [{"point": np.asarray(cps[i]["point"], float),
          "direction": np.asarray(cps[i]["direction"], float)}
         for i in np.where(k)[0]]
    Xk = X[k]
    c = wire_gate.pose_duzelt(Xk, c)
    if cfg.get("robot_aci_secici"):
        c = wire_gate.aci_duzelt(Xk, c)
    if cfg.get("robot_uye_secici") and uyeler:
        c = wire_gate.uye_yonu_sec(Xk, c, uyeler)
    if cfg.get("robot_yon_secici"):
        c = wire_gate.yon_sozluk_sec(Xk, c, V, step_path=stp, uyeler=uyeler)
    P = np.array([x["point"] for x in c], float)
    D = np.array([x["direction"] for x in c], float)
    assert len(P) and np.isfinite(P).all(), "CP konumlari gecersiz"
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0, atol=1e-3), "yonler birim degil"
    print(f"[5/5] URUN CIKTISI: {len(P)} CP")
    for i in range(min(3, len(P))):
        print(f"      CP{i}  ({P[i,0]:8.2f},{P[i,1]:8.2f},{P[i,2]:8.2f})  "
              f"yon ({D[i,0]:+.3f},{D[i,1]:+.3f},{D[i,2]:+.3f})")
    print(f"\nDUMAN TESTI GECTI  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    import traceback
    try:
        rc = main()
    except Exception:
        # SystemExit / KeyboardInterrupt YAKALANMAZ: `except BaseException` yazarsam
        # basarili kosudaki sys.exit(0) da buraya duser ve test GECERKEN "URUN CALISMIYOR"
        # basar -- yani gostergeyi tam da bu betigin onlemek icin var oldugu sekilde bozar.
        traceback.print_exc()
        print("\nDUMAN TESTI KALDI -- URUN CALISMIYOR")
        rc = 1
    sys.exit(rc)
