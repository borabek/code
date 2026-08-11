# -*- coding: utf-8 -*-
"""Y2: FP'LERI INSAN DENETIMINE HAZIRLA -- derinlik haritasi + baglam gorunumu.

SORU: urunun "yanlis" saydigi bu nokta GERCEKTEN aciklik mi, yoksa duz yuzey/vida basi mi?

YONTEM -- DERINLIK HARITASI (kesin ayirt edici):
    FP yonu boyunca (disaridan iceri) bir isin izgarasi at, her isinin gövdeye ilk carpma
    mesafesini olc.
        GERCEK ACIKLIK  -> merkez bolgede isinlar KANALA girer: derinlik ziplar (veya hic
                           carpmaz) ve etrafinda sig bir halka olur.
        DUZ YUZEY       -> tum derinlikler ayni; desen yok.
        VIDA BASI/CUKUR -> hafif, YUMUSAK bir cukur; kanal gibi keskin sicrama yok.
    Bu, "bana delik gibi geldi" degil OLCULEN bir sey: merkez ile halka arasindaki derinlik
    farki (mm) her kare icin YAZILIR.

trimesh.ray KULLANILMAZ: rtree kurulu degil ve trimesh'in isin/contains cagrilari HER
seferinde patliyor ([[trimesh-rtree-silent-failure]] -- `except: continue` bunu sessizce
yutmustu). Kendi Moller-Trumbore kesisimimizi yaziyoruz; yerel kirpma sayesinde ucuz.

CIKTI: results/fp_denetim/<pid>_<i>.png  + results/fp_denetim_olcum.json (derinlik sayilari)
"""
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

DIZIN = "results/fp_denetim"
YARICAP = 9.0      # mm, izgara yaricapi
N = 56             # izgara cozunurlugu
GERI = 10.0        # mm, isin baslangici gövdenin disinda


def kesisim(orig, d, V, F):
    """Moller-Trumbore, TEK yon / COK ucgen. Doner: en kucuk pozitif t (yoksa inf)."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    m = np.abs(a) > 1e-9
    if not m.any():
        return np.inf
    f = np.zeros_like(a); f[m] = 1.0 / a[m]
    s = orig - V[F[:, 0]]
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    ok = m & (u >= -1e-6) & (u <= 1 + 1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-6)
    return float(t[ok].min()) if ok.any() else np.inf


def derinlik_haritasi(V, F, p, d):
    """FP yonu boyunca derinlik haritasi + merkez/halka farki."""
    d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-9)
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(d @ a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u) + 1e-9
    v = np.cross(d, u)
    # YEREL KIRPMA: yalniz yakin ucgenler (hem hizli hem komsu gövdeyi disarida birakir)
    c = V[F].mean(1)
    yakin = np.linalg.norm(c - p, axis=1) <= (YARICAP * 2.5 + GERI + 15.0)
    Fk = F[yakin]
    if not len(Fk):
        return None, None, None
    g = np.linspace(-YARICAP, YARICAP, N)
    D = np.full((N, N), np.inf)
    for i, yy in enumerate(g):
        for j, xx in enumerate(g):
            o = p + d * GERI + u * xx + v * yy
            D[i, j] = kesisim(o, -d, V, Fk)
    son = np.where(np.isfinite(D), D - GERI, np.nan)      # yuzeye gore derinlik
    yy, xx = np.meshgrid(g, g, indexing="ij")
    rr = np.sqrt(xx ** 2 + yy ** 2)
    merkez = np.nanmedian(son[rr <= 2.0]) if np.isfinite(son[rr <= 2.0]).any() else np.nan
    halka = np.nanmedian(son[(rr >= 5.0) & (rr <= 8.0)]) if np.isfinite(
        son[(rr >= 5.0) & (rr <= 8.0)]).any() else np.nan
    delik_orani = float(np.mean(~np.isfinite(D[rr <= 2.0])))
    return son, (merkez, halka, delik_orani), (u, v)


def main():
    import thesis_remesh
    from big_arbiter import eligible
    from infer_step_cp import step_to_mesh

    os.makedirs(DIZIN, exist_ok=True)
    with io.open("results/fp_denetim.json", encoding="utf-8") as f:
        FD = json.load(f)
    FP = FD["hepsi"]; sec = FD["ornek_idx"]
    stp_of = {p: s for m, p, jf, s in eligible()}
    print(f"{len(sec)} FP render edilecek ({len({FP[i]['pid'] for i in sec})} parca)")

    # parca basina grupla -> her parcayi BIR KEZ remesh et
    byp = {}
    for i in sec:
        byp.setdefault(FP[i]["pid"], []).append(i)

    OLCUM = []
    t0 = time.time()
    for k, (pid, idxs) in enumerate(sorted(byp.items()), 1):
        try:
            Vr, Fr = step_to_mesh(stp_of[pid])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, float); F = np.ascontiguousarray(F, np.int64)
        except Exception as e:
            print(f"  {pid}: mesh HATA {type(e).__name__}"); continue
        for i in idxs:
            f_ = FP[i]
            p = np.array(f_["nokta"], float); d = np.array(f_["yon"], float)
            son, olc, _ = derinlik_haritasi(V, F, p, d)
            if son is None:
                continue
            merkez, halka, dor = olc
            fig = plt.figure(figsize=(9.0, 4.2))
            ax = fig.add_subplot(121)
            im = ax.imshow(son, origin="lower", extent=[-YARICAP, YARICAP, -YARICAP, YARICAP],
                           cmap="viridis")
            ax.plot(0, 0, "r+", ms=14, mew=2)
            ax.set_title(f"derinlik (mm)  merkez {merkez:.2f} / halka {halka:.2f}"
                         f"\nfark {merkez-halka:+.2f} mm | delik {dor:.0%}", fontsize=9)
            ax.set_xlabel("mm"); fig.colorbar(im, ax=ax, fraction=0.046)
            # baglam: parcanin siluetii + FP
            ax2 = fig.add_subplot(122)
            ax2.scatter(V[:, 0], V[:, 1], s=0.4, c="0.75", linewidths=0)
            ax2.scatter([p[0]], [p[1]], s=70, c="red", marker="x")
            ax2.set_aspect("equal"); ax2.set_title(
                f"{pid} ({f_['mfg']}, {f_['rejim']}-CP)\nen yakin GT "
                f"{('%.1f mm' % f_['gt_uzaklik']) if f_['gt_uzaklik'] else 'GT YOK'}", fontsize=9)
            ax2.set_xticks([]); ax2.set_yticks([])
            fig.tight_layout()
            yol = f"{DIZIN}/{pid}_{f_['aday_i']}.png"
            fig.savefig(yol, dpi=88, bbox_inches="tight"); plt.close(fig)
            OLCUM.append({"idx": i, "pid": pid, "mfg": f_["mfg"], "rejim": f_["rejim"],
                          "png": yol, "merkez": None if np.isnan(merkez) else float(merkez),
                          "halka": None if np.isnan(halka) else float(halka),
                          "delik_orani": dor, "gt_uzaklik": f_["gt_uzaklik"]})
        if k % 10 == 0:
            print(f"  {k}/{len(byp)} parca  {time.time()-t0:.0f}s", flush=True)
    with io.open("results/fp_denetim_olcum.json", "w", encoding="utf-8") as f:
        json.dump(OLCUM, f, indent=1)
    print(f"\n{len(OLCUM)} kare -> {DIZIN}/")
    d = [o for o in OLCUM if o["merkez"] is not None and o["halka"] is not None]
    if d:
        fark = np.array([o["merkez"] - o["halka"] for o in d])
        print(f"derinlik farki: medyan {np.median(fark):.2f} mm | "
              f">1mm {np.mean(fark > 1):.0%} | >3mm {np.mean(fark > 3):.0%}")
    print("makbuz -> results/fp_denetim_olcum.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
