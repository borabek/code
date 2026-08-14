# -*- coding: utf-8 -*-
"""CP SAYISINI GEOMETRIDEN TAHMIN ET -> parca-basina uyarlanan gate esigi.

NEDEN BU, NEDEN SIMDI
  Olculmus gercek: uretici CP sayisi BILINIYORSA (metadata modu, top-N) CP-F1 0.750 -> 0.775.
  Bu +0.025 su an katalog verisi gerektirdigi icin GORULMEMIS parcada kullanilamiyor.
  Ama rejim router'i geometriden ">=8 CP mi?" sorusunu AUC 0.9945 ile cevapliyor -- yani
  geometri CP sayisi hakkinda cok sey biliyor. Sayinin KENDISINI tahmin edebilirsek o kazanci
  metadata olmadan aliriz.

  Ayrica: esik ayari GLOBAL ve REJIM bazinda tukendi (results/pitstop2_gate_nested.json,
  ice-ice secim mevcut degerleri dogruladi). PARCA bazinda hic denenmedi.

IKI KULLANIM SEKLI, IKISI DE OLCULUR
  A) sert top-N : tahmin edilen N kadar en yuksek skorlu adayi tut
  B) yumusak     : tahmin N'e gore esigi parca basina kaydir (cok aday -> esik yukselt)
  (B) daha guvenli, cunku tahmin hatasi recall'u tamamen kesmiyor.

KILL (olcumden ONCE yazildi)
  * korpus-agirlikli CP-F1 katkisi < +0.02  -> OLU
  * VEYA uretici-disi sondada 0.01'den fazla kayip -> REDDEDILIR

VERI: gate_regrow.py'nin urettigi npz (parca basina ngt + aday basina 13 ozellik). Ayri bir
cikarim KOSULMAZ -- ayni GPU isinden yeniden yararlanilir.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NPZ = "results/gate_regrow_data.npz"
OUT = "results/count_prior.json"


def part_features(X, votes, groups):
    """Aday-basina ozellikleri PARCA-basina ozet vektorune cevir (sayi tahmini icin girdi).

    Sayiyi belirleyen sey tek bir adayin sekli degil, adaylarin TOPLU dagilimi: kac tane var,
    ne kadar sik duruyorlar, ne kadar buyukler. Bu yuzden ozet istatistikler kullanilir.
    """
    gids = np.unique(groups)
    F, ids = [], []
    for g in gids:
        m = groups == g
        Xg = X[m]
        if len(Xg) == 0:
            continue
        q = lambda col: [float(np.mean(col)), float(np.median(col)), float(np.std(col)),
                         float(np.percentile(col, 25)), float(np.percentile(col, 75))]
        feat = [float(len(Xg))]                       # aday sayisi -- en guclu tekil sinyal
        feat += q(Xg[:, 0])                           # size
        feat += q(Xg[:, 1])                           # depth
        feat += q(Xg[:, 2])                           # nn_dist  (adaylar arasi mesafe = adim)
        feat += q(Xg[:, 3])                           # n_close  (yogunluk)
        feat += q(Xg[:, 5])                           # nverts
        feat += [float(np.mean(votes[m])), float((votes[m] >= 2).mean())]
        F.append(feat); ids.append(int(g))
    return np.array(F, float), np.array(ids)


def main():
    if not os.path.exists(NPZ):
        print(f"{NPZ} yok -- once gate_regrow.py cikarimini bitir"); return 1
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold

    d = np.load(NPZ, allow_pickle=True)
    X, y, votes, groups = d["X"], d["y"], d["votes"], d["groups"]
    ngt_of = dict(zip(d["grp_ids"].tolist(), d["ngt"].tolist()))
    fams = d["fams"] if "fams" in d else None

    Pf, pids = part_features(X, votes, groups)
    target = np.array([ngt_of.get(int(p), 0) for p in pids], float)
    keep = target > 0
    Pf, pids, target = Pf[keep], pids[keep], target[keep]
    # aile bazli gruplama: ayni ailenin varyantlari egitim ve teste BOLUNMEZ
    if fams is not None:
        fam_of = {}
        for g, f in zip(groups, fams):
            fam_of.setdefault(int(g), str(f))
        gkey = np.array([fam_of.get(int(p), str(p)) for p in pids])
    else:
        gkey = pids.astype(str)

    print(f"{len(pids)} parca | gercek CP sayisi: medyan {np.median(target):.0f} "
          f"araligi {target.min():.0f}-{target.max():.0f}")

    oof = np.zeros(len(target))
    for tr, te in GroupKFold(n_splits=5).split(Pf, target, gkey):
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                                   random_state=0).fit(Pf[tr], target[tr])
        oof[te] = rf.predict(Pf[te])
    pred = np.clip(np.rint(oof), 1, None)
    err = pred - target
    mae = float(np.abs(err).mean())

    # TABAN CIZGISI: "aday sayisini oldugu gibi kullan" -- tahminci bunu GECMELI
    base = np.clip(np.rint(Pf[:, 0]), 1, None)
    base_mae = float(np.abs(base - target).mean())

    print(f"\n{'yontem':<28}{'MAE':>8}{'+-1 icinde':>12}{'tam isabet':>12}")
    print(f"{'aday sayisi (taban)':<28}{base_mae:>8.2f}"
          f"{100*np.mean(np.abs(base-target)<=1):>11.0f}%{100*np.mean(base==target):>11.0f}%")
    print(f"{'geometri tahmincisi':<28}{mae:>8.2f}"
          f"{100*np.mean(np.abs(err)<=1):>11.0f}%{100*np.mean(err==0):>11.0f}%")

    rec = {"n_parts": int(len(pids)), "mae": mae, "baseline_mae": base_mae,
           "within_1": float(np.mean(np.abs(err) <= 1)),
           "exact": float(np.mean(err == 0)),
           "beats_baseline": bool(mae < base_mae)}

    # KARAR: tahmin yeterince iyi mi? top-N modu ancak +-1 dogrulukta anlamli.
    ok = rec["within_1"] >= 0.60 and mae < base_mae
    rec["usable_for_topn"] = bool(ok)
    print(f"\nKAPI (+-1 icinde >= %60 VE tabani gecsin): {'GECTI' if ok else 'KALDI'}")
    if not ok:
        print("  -> sert top-N riskli; yumusak esik kaydirmasi denenmeli (B yolu)")

    json.dump(rec, open(OUT, "w"), indent=1)
    print(f"makbuz -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
