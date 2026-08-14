# -*- coding: utf-8 -*-
"""OFFSET + SEED BASI — tepe basina CP merkezine kayma vektoru ogren.

GEREKCE (21.60, 21.75, 21.76). Baglayici kisit YANAL KONUM (+0.358) ve
konum bugun "segmentlenen bolgenin agirlik merkezi"nden turetiliyor --
literaturde bunun bilinen kusuru degen ayni nesneleri ayiramamak.
Onerilen: her TEPE kendi CP merkezine bir kayma vektoru tahmin etsin,
tepeler kaydirilip kumelensin.

TAVAN OLCULDU (`sonda_offset_tavani.py`, egitim YOK): kumeleme bandi
4 mm iken **1.0 mm** offset hatasinda tespit F1 **0.9933**. Yani gereken
hassasiyet ~1 mm. Kiyas: pose head'in bugun ulastigi artik 0.67 mm.

BU BETIGIN CEVAPLADIGI SORU (mutlak, esik sorusu):
    **Ag, tepe basina offseti <=1.0 mm ile tahmin edebiliyor mu?**
Bu bir "kucuk fark" sorusu DEGIL; bu yuzden tohum gurultusu (0.046, F1
farklari icin) burada BAGLAYICI DEGIL ve tek kosu hukum verebilir.

CIKTI: 4 kanal -- [ox, oy, oz, seed_logit].
KAYIP: seed tepelerinde offset icin L1 + tum tepelerde seed icin BCE.
HEDEF YENI ETIKET GEREKTIRMEZ: (en yakin GT CP - tepe konumu).

SIZINTI KAPISI: VAL ve LOCKED parcalari VE VAL'in GEOMETRI GRUPLARI
egitimden cikarilir (5793 parca kalir).
"""
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED_MM = float(os.environ.get("OB_SEED_MM", "6.0"))   # oy veren bant
N_EGITIM = int(os.environ.get("OB_N", "600"))          # parca sayisi
EPOK = int(os.environ.get("OB_EPOK", "8"))
KEIG = int(os.environ.get("OB_KEIG", "96"))
LR = float(os.environ.get("OB_LR", "1e-3"))
CIKTI = os.environ.get("OB_OUT", "results/offset_bas.pt")
OPS_DIR = os.environ.get("OB_OPS", "results/offset_ops")


def veri_havuzu():
    import d6_kayit
    kay = d6_kayit.yukle()
    s3 = json.load(io.open("results/split3.json", encoding="utf-8"))
    val = {str(x) for x in s3["val"]["parts"]}
    lock = {str(x) for x in s3["locked"]["parts"]}
    gk = json.load(io.open("results/_strict_geometry_keys.json",
                           encoding="utf-8"))
    vg = {gk[p] for p in val if p in gk}
    uygun = [str(p) for p in kay
             if str(p) not in val and str(p) not in lock
             and gk.get(str(p)) not in vg]
    assert uygun, "egitim havuzu BOS -- sizinti kapisi yanlis kurulmus"
    print(f"sizinti kapisi: {len(kay)} kayit -> {len(uygun)} uygun "
          f"(VAL/LOCKED ve VAL geometri gruplari HARIC)", flush=True)
    return kay, uygun


def hedefler(V, G):
    """her tepe icin (offset, seed). offset = en yakin CP - tepe."""
    d = np.linalg.norm(V[:, None, :] - G[None, :, :], axis=-1)
    j = d.argmin(1)
    m = d.min(1)
    off = G[j] - V
    seed = (m <= SEED_MM).astype(np.float32)
    return off.astype(np.float32), seed


def main():
    import torch
    import diffusionnet as D
    import thesis_remesh
    import kanonik_d7 as K
    from export_robot_glb import step_to_mesh

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kay, uygun = veri_havuzu()
    STEP = K.step_haritasi()
    rng = np.random.default_rng(0)
    rng.shuffle(uygun)
    uygun = [p for p in uygun if p in STEP][:N_EGITIM]
    print(f"egitim parcasi: {len(uygun)} | cihaz {dev} | k_eig {KEIG} | "
          f"epok {EPOK}", flush=True)

    cfg = {"input_features": "xyz", "c_width": 128,
           "n_diffusion_blocks": 4, "n_eig": KEIG, "dropout": 0.0,
           "loss": "ce"}                       # ce -> HAM cikti (logit)
    model, meta = D.build_diffusionnet(cfg, n_classes=4)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = torch.nn.BCEWithLogitsLoss()

    # VERIYI BIR KEZ HAZIRLA (operator hesabi pahali, epoklar arasi tekrar
    # edilmesin). Bellek icin mesh basina yalniz gerekli sey tutulur.
    hazir = []
    t0 = time.time()
    for i, pid in enumerate(uygun):
        try:
            Vr, Fr = step_to_mesh(STEP[pid])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64)
            F = np.ascontiguousarray(F, np.int64)
            G = np.asarray(kay[pid]["G"], float).reshape(-1, 3)
            if not len(G):
                continue
            off, seed = hedefler(V, G)
            if seed.sum() < 3:
                continue
            # OPERATORLER RAM'DE TUTULMAZ: parca basina evecs/gradX ~10 MB;
            # 800 parca birkac GB eder. Diske onbelleklenir, epok basinda
            # oradan okunur (ilk epoktan sonra maliyet ~sifir).
            D.precompute_operators(V, F, KEIG, OPS_DIR)
            hazir.append((V, F, off, seed))
        except Exception as e:                          # noqa: BLE001
            print(f"  {pid}: {type(e).__name__} -- atlandi", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  hazirlik {i+1}/{len(uygun)} "
                  f"({time.time()-t0:.0f} s, kullanilan {len(hazir)})",
                  flush=True)
    print(f"\nhazir parca: {len(hazir)} ({time.time()-t0:.0f} s)\n",
          flush=True)
    if not hazir:
        print("HAZIR PARCA YOK -- egitim yapilamaz")
        return 1

    for ep in range(EPOK):
        model.train()
        top_l, top_mm, n = 0.0, [], 0
        sira = rng.permutation(len(hazir))
        for k in sira:
            V, F, off, seed = hazir[k]
            ops = D.precompute_operators(V, F, KEIG, OPS_DIR)
            o = {kk: (v.to(dev) if hasattr(v, "to") else v)
                 for kk, v in ops.items()}
            out = D._forward(model, o, D._model_input(o, meta))
            t_off = torch.as_tensor(off, device=dev)
            t_sd = torch.as_tensor(seed, device=dev)
            m = t_sd > 0.5
            kayip = bce(out[:, 3], t_sd)
            if m.any():
                kayip = kayip + (out[m][:, :3] - t_off[m]).abs().mean()
            opt.zero_grad()
            kayip.backward()
            opt.step()
            with torch.no_grad():
                if m.any():
                    hata = (out[m][:, :3] - t_off[m]).norm(dim=-1)
                    top_mm.append(float(hata.median()))
            top_l += float(kayip)
            n += 1
        ort = float(np.median(top_mm)) if top_mm else float("nan")
        print(f"epok {ep+1}/{EPOK}  kayip {top_l/max(n,1):.4f}  "
              f"ORTANCA OFFSET HATASI {ort:.3f} mm"
              f"   {'<-- 1.0 mm ESIGININ ALTINDA' if ort < 1.0 else ''}",
              flush=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg,
                    "meta": meta, "epok": ep + 1, "ortanca_mm": ort},
                   CIKTI)
    print(f"\n-> {CIKTI}")
    print("SORU: ag tepe basina offseti <=1.0 mm ile tahmin edebiliyor mu?")
    print(f"CEVAP (egitim kumesi ortancasi): {ort:.3f} mm")
    print("NOT: bu EGITIM kumesi hatasidir; genelleme icin VAL'de "
          "olculmelidir (sonraki adim).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
