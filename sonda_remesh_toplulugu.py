# -*- coding: utf-8 -*-
"""Y16 -- REMESH-VARYANT TOPLULUGU. Uc remesh hedefinin ciktilarini birlestirir.

GEREKCE. Zincir her parcayi 6000 tepeye remesh eder. Remesh KENDISI bir
gurultu kaynagidir: farkli hedef cozunurluk farkli ucgenleme, farkli
ozdeger tabani, farkli tahmin uretir. Ayni model uc farkli hedefte
kosulup ciktilar birlestirilirse bu gurultunun bir kismi stabilize
olabilir. **Yeni egitim YOK** -- bu yuzden tohum gurultusune (0.046) tabi
DEGIL; kol tek kosuda hukum giyebilir.

IKI VARYANT olculur:
  * BIRLESIM (oy >= 1): recall'u acar, kesinligi dusurur.
  * OYLAMA  (oy >= 2): kesinligi acar, recall'u dusurur.

Kiyas tabani, ayni sondanin 6000 hedefli kosusudur (yani mevcut davranis).
Karar ESLI PARCA BOOTSTRAP ile verilir -- kiyas ayni parcalarda.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sina_kume import esle_macar                       # noqa: E402

KUME_MM = float(os.environ.get("RT_KUME", "5.0"))
YOL = os.environ.get("RT_YOL", "saha")


def _birim(v):
    v = np.asarray(v, float).reshape(-1, 3)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _yukle(fp):
    """dokum -> {pid: kayit}, yalniz istenen yol."""
    out = {}
    for r in json.load(open(fp)):
        if r.get("yol") != YOL:
            continue
        out[str(r["pid"])] = r
    return out


def birlestir(listeler, min_oy):
    """Uc varyantin (P, D) listelerini KUME_MM'de birlestirir.

    Ilk varyant cipa alinir; her kume, kendisine KUME_MM icinde en yakin
    adaylari toplar. Kume oyu = o kumeye katki veren VARYANT sayisi
    (ayni varyanttan iki aday oyu 1 sayar -- yoksa tek varyant tek basina
    coklu oy uretirdi).
    """
    P = np.concatenate([l[0] for l in listeler]) if listeler else np.zeros((0, 3))
    D = np.concatenate([l[1] for l in listeler]) if listeler else np.zeros((0, 3))
    kaynak = np.concatenate([np.full(len(l[0]), i) for i, l in
                             enumerate(listeler)]) if listeler else np.zeros(0, int)
    if not len(P):
        return np.zeros((0, 3)), np.zeros((0, 3))
    kullanildi = np.zeros(len(P), bool)
    oP, oD = [], []
    # cipa sirasi: once ilk varyant (6000 = mevcut davranis), sonra digerleri
    sira = np.argsort(kaynak, kind="stable")
    for i in sira:
        if kullanildi[i]:
            continue
        d = np.linalg.norm(P - P[i][None, :], axis=1)
        uye = np.where((d <= KUME_MM) & (~kullanildi))[0]
        kullanildi[uye] = True
        oy = len(set(kaynak[uye].tolist()))
        if oy < min_oy:
            continue
        # konum: uyelerin ortalamasi. yon: cipanin yonuyle ayni yarikureye
        # hizalanmis uye yonlerinin ortalamasi (isaret kacmasini onler).
        p = P[uye].mean(0)
        d0 = D[i]
        Du = D[uye] * np.sign(np.maximum(D[uye] @ d0, -1.0))[:, None]
        Du = np.where(np.abs(D[uye] @ d0)[:, None] > 0, Du, D[uye])
        v = Du.mean(0)
        n = float(np.linalg.norm(v))
        oP.append(p)
        oD.append(v / n if n > 1e-9 else d0)
    return np.asarray(oP).reshape(-1, 3), np.asarray(oD).reshape(-1, 3)


def olc(kayitlar, secici):
    """secici(pid, r) -> (P, D). Doner: (tespit, rob_isaretsiz, rob_isaretli,
    parca_basina_liste)."""
    tot = {k: [0, 0, 0] for k in ("tespit", "rob", "rbi")}
    parca = []
    for pid, r in sorted(kayitlar.items()):
        G = np.asarray(r["G"], float).reshape(-1, 3)
        Gd = _birim(r["Gd"])
        diag = float(r["diag"])
        P, D = secici(pid, r)
        satir = {"pid": pid}
        for ad, kw in (("tespit", dict(tol=2.0, am=180.0, isaretli=False)),
                       ("rob", dict(tol=2.0, am=10.0, isaretli=False)),
                       ("rbi", dict(tol=2.0, am=10.0, isaretli=True))):
            tp, fp, fn, _ = esle_macar(P, D, G, Gd, diag, kw["tol"],
                                       kw["am"], False,
                                       isaretli=kw["isaretli"])
            tot[ad][0] += tp
            tot[ad][1] += fp
            tot[ad][2] += fn
            satir[ad] = (tp, fp, fn)
        parca.append(satir)
    def f1(t):
        tp, fp, fn = t
        return 2 * tp / max(2 * tp + fp + fn, 1)
    return {k: f1(v) for k, v in tot.items()}, parca


def esli_bootstrap(pa, pb, ad, n=4000, tohum=0):
    """parca duzeyinde esli bootstrap; FARKIN dagilimi."""
    rng = np.random.default_rng(tohum)
    A = np.asarray([r[ad] for r in pa], float)
    B = np.asarray([r[ad] for r in pb], float)
    m = len(A)
    fk = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        a, b = A[i].sum(0), B[i].sum(0)
        f = lambda t: 2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)
        fk.append(f(b) - f(a))
    fk = np.asarray(fk)
    return float(fk.mean()), float(np.percentile(fk, 2.5)), \
        float(np.percentile(fk, 97.5)), float((fk > 0).mean())


def main():
    hedefler = [6000, 5000, 7200]          # 6000 ONCE: cipa = mevcut davranis
    dosya = {t: f"results/_dokum_remesh{t}.json" for t in hedefler}
    dosya[6000] = os.environ.get("RT_TABAN", "results/_dokum_taban.json")
    K = {}
    for t in hedefler:
        if not os.path.exists(dosya[t]):
            print(f"EKSIK: {dosya[t]} -- kol tamamlanmadi")
            return 1
        K[t] = _yukle(dosya[t])
    ortak = sorted(set.intersection(*[set(K[t]) for t in hedefler]))
    print(f"yol={YOL} | ortak parca: {len(ortak)} "
          f"(tekil: {[len(K[t]) for t in hedefler]})")
    if not ortak:
        print("ORTAK PARCA YOK")
        return 1
    K = {t: {p: K[t][p] for p in ortak} for t in hedefler}

    def tek(t):
        return lambda pid, r: (np.asarray(K[t][pid]["P"], float).reshape(-1, 3),
                               _birim(K[t][pid]["D"])
                               if len(K[t][pid]["P"]) else np.zeros((0, 3)))

    def top(min_oy):
        def f(pid, r):
            L = []
            for t in hedefler:
                p = np.asarray(K[t][pid]["P"], float).reshape(-1, 3)
                d = _birim(K[t][pid]["D"]) if len(p) else np.zeros((0, 3))
                L.append((p, d))
            return birlestir(L, min_oy)
        return f

    sonuc = {}
    taban_parca = None
    print(f"\n{'varyant':28s} {'tespit':>8s} {'rob':>8s} {'rob-ISR':>8s}")
    for ad, sec in [("TABAN (6000)", tek(6000)),
                    ("tek 5000", tek(5000)),
                    ("tek 7200", tek(7200)),
                    (f"TOPLULUK birlesim(>=1)", top(1)),
                    (f"TOPLULUK oylama(>=2)", top(2)),
                    (f"TOPLULUK oybirligi(>=3)", top(3))]:
        m, parca = olc(K[6000], sec)
        sonuc[ad] = {"metrik": m, "parca": parca}
        if taban_parca is None:
            taban_parca = parca
        print(f"{ad:28s} {m['tespit']:8.4f} {m['rob']:8.4f} {m['rbi']:8.4f}")

    print("\n--- ESLI PARCA BOOTSTRAP (TABAN'a gore fark) ---")
    print(f"{'varyant':28s} {'metrik':>8s} {'fark':>9s} "
          f"{'%95 GA':>22s} {'poz%':>6s}")
    for ad in sonuc:
        if ad.startswith("TABAN"):
            continue
        for mad in ("tespit", "rob", "rbi"):
            f, lo, hi, pz = esli_bootstrap(taban_parca, sonuc[ad]["parca"], mad)
            yildiz = " *" if (lo > 0 or hi < 0) else ""
            print(f"{ad:28s} {mad:>8s} {f:+9.4f} "
                  f"[{lo:+.4f},{hi:+.4f}]{yildiz:>3s} {100*pz:5.1f}")

    with open("results/remesh_toplulugu.json", "w") as fh:
        json.dump({ad: v["metrik"] for ad, v in sonuc.items()}, fh, indent=1)
    print("\n-> results/remesh_toplulugu.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
