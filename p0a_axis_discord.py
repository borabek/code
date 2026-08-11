# -*- coding: utf-8 -*-
"""P0-a: EKSEN HIPOTEZI TESTI (CPU, insan-etiketi YOK, mevcut havuz).
Hipotez: wire-CP'ler ile tool/vida acikliklari FARKLI EKSEN ailelerinde yasar (tel ONDEN, vida/pusher
USTTEN girer) -- gate'in 13 lokal feature'inda bu bilgi YOK.
Test: her parcada adaylari EKSENE gore kumele (parca-ici, frame-invariant) -> kumelerin TP/FP saflini
NULL modele (etiketleri parca-ici permute) karsi olc.
MONEY METRIK: 'sifir-TP kumesinde yasayan FP kutlesi' = yapisal olarak oldurulebilir FP orani.
GO: saflik >> null (z>5) VE saf-FP kumelerinde FP kutlesi >=%40."""
import json, sys
import numpy as np

POOL = sys.argv[1] if len(sys.argv) > 1 else "results/wei_aggr_pool.json"
ANG = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0     # ayni eksen sayilma esigi (derece)
SIGNED = "--unsigned" not in sys.argv                        # yon isareti onemli mi (on vs arka giris)
rng = np.random.RandomState(0)


def cluster_axes(D, ang_deg, signed=True):
    """greedy eksen kumeleme: cosine benzerligi ang_deg icinde olanlar ayni kume."""
    thr = np.cos(np.deg2rad(ang_deg))
    n = len(D); lab = -np.ones(n, int); k = 0
    for i in range(n):
        if lab[i] >= 0: continue
        c = D[i]; lab[i] = k
        for j in range(i + 1, n):
            if lab[j] >= 0: continue
            d = float(D[i] @ D[j])
            if (d if signed else abs(d)) >= thr: lab[j] = k
        k += 1
    return lab, k


def purity(lab, y):
    """agirlikli saflik: her kumede cogunluk-sinifin payi."""
    tot = 0
    for c in np.unique(lab):
        m = lab == c; nt = int(y[m].sum()); nf = int(m.sum() - nt)
        tot += max(nt, nf)
    return tot / max(len(y), 1)


d = json.load(open(POOL))
obs_p = []; null_p = []; z_all = []
fp_in_pure = 0; fp_tot = 0; tp_tot = 0
clust_stats = []
for pid, v in d.items():
    D = np.array(v["dir"], float); y = np.array(v["y"], int)
    if len(y) < 4 or y.sum() == 0: continue
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    lab, k = cluster_axes(D, ANG, SIGNED)
    op = purity(lab, y)
    # NULL: etiketleri parca-ici permute (ayni kume boyutlari, ayni TP sayisi)
    nulls = []
    for _ in range(200):
        yp = rng.permutation(y); nulls.append(purity(lab, yp))
    nm, ns = float(np.mean(nulls)), float(np.std(nulls) + 1e-9)
    obs_p.append(op); null_p.append(nm); z_all.append((op - nm) / ns)
    # money metrik: sifir-TP kumelerindeki FP
    for c in np.unique(lab):
        m = lab == c; nt = int(y[m].sum()); nf = int(m.sum() - nt)
        if nt == 0: fp_in_pure += nf
        fp_tot += nf; tp_tot += nt
    clust_stats.append((k, len(y)))

obs_p = np.array(obs_p); null_p = np.array(null_p); z_all = np.array(z_all)
kk = np.array([c[0] for c in clust_stats]); nn = np.array([c[1] for c in clust_stats])
print(f"POOL {POOL} | {len(obs_p)} parca | aday/parca ort {nn.mean():.1f} | eksen-kumesi/parca ort {kk.mean():.1f}"
      f" | esik {ANG:.0f}deg {'signed' if SIGNED else 'unsigned'}")
print(f"\n=== EKSEN KUMELERI TP/FP AYIRIYOR MU (null-model testi) ===")
print(f"  gozlenen saflik : {obs_p.mean():.3f}")
print(f"  NULL saflik     : {null_p.mean():.3f}  (etiketler parca-ici permute)")
print(f"  fark            : {obs_p.mean()-null_p.mean():+.3f}   ortalama z = {z_all.mean():+.2f}")
print(f"  parcalarin %{100*np.mean(z_all>2):.0f}'inde z>2 (eksen bilgi tasiyor)")
print(f"\n=== MONEY METRIK: yapisal olarak oldurulebilir FP ===")
print(f"  toplam FP {fp_tot} | toplam TP {tp_tot}")
print(f"  SIFIR-TP eksen kumesinde yasayan FP: {fp_in_pure}/{fp_tot} = {fp_in_pure/max(fp_tot,1):.3f}")
print(f"  -> bu FP'ler hicbir gercek wire-CP ile ayni ekseni PAYLASMIYOR = eksen-kuralIyla oldurulebilir")
go1 = z_all.mean() > 5; go2 = fp_in_pure / max(fp_tot, 1) >= 0.40
print(f"\nKARAR: saflik-sinyali {'GECTI' if go1 else 'ZAYIF'} (z {z_all.mean():+.2f}, esik >5) | "
      f"saf-FP kutlesi {'GECTI' if go2 else 'DUSUK'} ({fp_in_pure/max(fp_tot,1):.2f}, esik >=0.40)")
print("  IKISI DE GECTI -> P0-b/c'ye devam (kume-havuzlu gate)." if (go1 and go2)
      else "  -> eksen tek basina zayif; P0-b'de yuz/konum ile birlestir ya da durustce kapat.")
