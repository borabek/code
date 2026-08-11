# -*- coding: utf-8 -*-
"""KAYIP DENETIMI: puan tam olarak HANGI ASAMADA kayboluyor?

NEDEN BU, NEDEN FIKIR DEGIL
  Bu depoda olculen 12 kaldiracin hepsi "modele yeni bilgi/ozellik/veri ekle" ailesindendi ve
  HEPSI oldu. Yasayan 6 degisikligin hicbiri yeni bilgi eklemedi; hepsi BOZUK ya da YANLIS
  DAGITILMIS bir seyi duzeltti (min_v10 aritmetik hatasi, router girdi uyusmazligi, rejime tek
  esik, yanlis eksen yuvarlamasi). Yani bu boru hattinda puani KAYIP AVI artiriyor, fikir avi
  degil.

  Bu betik fikir uretmez. Elde olan 1903 parcalik veride puanin nerede kaybedildigini
  ASAMA ASAMA ve REJIM AYRIMLI olcer, boylece bir sonraki is tahminle degil olcumle secilir.

ASAMALAR
  1. ADAY OLUSUMU : GT acikligin yaninda HIC aday uretildi mi? (uretilmediyse gate'in sansi yok)
  2. GATE         : aday vardi ama gate onu attiysa, kayip buradadir
  3. PRECISION    : gate'ten gecen ama GT'ye karsilik gelmeyen adaylar

Cikti, her rejim icin: tavan (mukemmel gate), bugunku, ve ikisi arasindaki farkin nereye gittigi.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NPZ = "results/gate_regrow_data.npz"
W = {"dusuk": 0.895, "cok": 0.105}
THR = 0.35


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    d = np.load(NPZ, allow_pickle=True)
    X, y, groups, fams = d["X"], d["y"], d["groups"], d["fams"]
    ngt_of = dict(zip(d["grp_ids"].tolist(), d["ngt"].tolist()))
    fam_of = {}
    for g, f in zip(groups, fams):
        fam_of.setdefault(int(g), str(f))
    gkey = np.array([fam_of[int(g)] for g in groups])

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, gkey):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                     random_state=0).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    # rejim basina asama muhasebesi
    acc = {k: dict(n_gt=0, cand_tp=0, kept_tp=0, kept_fp=0, n_parts=0) for k in W}
    for g in np.unique(groups):
        m = groups == g
        n_gt = int(ngt_of.get(int(g), 0))
        if n_gt <= 0:
            continue
        k = "cok" if n_gt >= 8 else "dusuk"
        a = acc[k]
        a["n_parts"] += 1
        a["n_gt"] += n_gt
        a["cand_tp"] += int((y[m] == 1).sum())          # GT'nin kac tanesi aday olarak URETILDI
        sel = oof[m] >= THR
        a["kept_tp"] += int((y[m][sel] == 1).sum())     # kac tanesi gate'ten GECTI
        a["kept_fp"] += int((y[m][sel] == 0).sum())

    print(f"{'rejim':<8}{'parca':>7}{'GT':>7}{'aday-R':>9}{'gate-R':>9}"
          f"{'P':>8}{'F1':>8}{'tavan-F1':>10}")
    rows = {}
    for k, a in acc.items():
        cand_r = a["cand_tp"] / max(a["n_gt"], 1)
        gate_r = a["kept_tp"] / max(a["n_gt"], 1)
        p = a["kept_tp"] / max(a["kept_tp"] + a["kept_fp"], 1)
        f1 = 2 * p * gate_r / max(p + gate_r, 1e-9)
        ceil = 2 * 1.0 * cand_r / max(1.0 + cand_r, 1e-9)     # mukemmel gate: P=1
        rows[k] = dict(cand_r=cand_r, gate_r=gate_r, prec=p, f1=f1, ceil=ceil, **a)
        print(f"{k:<8}{a['n_parts']:>7}{a['n_gt']:>7}{cand_r:>9.3f}{gate_r:>9.3f}"
              f"{p:>8.3f}{f1:>8.4f}{ceil:>10.4f}")

    wf1 = sum(W[k] * rows[k]["f1"] for k in W)
    wceil = sum(W[k] * rows[k]["ceil"] for k in W)
    print(f"\n{'agirlikli':<8}{'':>7}{'':>7}{'':>9}{'':>9}{'':>8}{wf1:>8.4f}{wceil:>10.4f}")

    print("\n--- KAYIP MUHASEBESI (korpus agirlikli, F1 puani cinsinden) ---")
    gap_total = wceil - wf1
    # aday olusumunda kaybedilen: tavani 1.0'dan asagi ceken kisim
    perfect = 1.0
    loss_formation = perfect - wceil
    print(f"  1) ADAY OLUSUMU : {loss_formation:.4f}  (GT acikligin yaninda hic aday yok)")
    print(f"  2) GATE         : {gap_total:.4f}  (aday vardi, gate atti VEYA yanlis aday tuttu)")
    for k in W:
        r = rows[k]
        miss_f = (1 - r["cand_r"]) * r["n_gt"]
        miss_g = (r["cand_r"] - r["gate_r"]) * r["n_gt"]
        print(f"     {k:<6} GT {r['n_gt']:>5} | aday YOK {miss_f:>6.0f}  "
              f"| aday VARDI gate ATTI {miss_g:>6.0f}  | yanlis tutulan {r['kept_fp']:>6}")

    print("\n--- EN BUYUK TEK KAYIP ---")
    cands = []
    for k in W:
        r = rows[k]
        cands.append((W[k] * (1 - r["cand_r"]), f"{k}: aday olusumu (GT'nin %{100*(1-r['cand_r']):.0f}'i icin aday yok)"))
        cands.append((W[k] * (r["cand_r"] - r["gate_r"]), f"{k}: gate gercek adaylari atiyor"))
        fp_rate = r["kept_fp"] / max(r["kept_tp"] + r["kept_fp"], 1)
        cands.append((W[k] * fp_rate, f"{k}: gate yanlis adaylari tutuyor (precision)"))
    for w, name in sorted(cands, reverse=True)[:4]:
        print(f"  {w:.4f}  {name}")

    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in rows.items()},
              open("results/loss_audit.json", "w"), indent=1)
    print("\nmakbuz -> results/loss_audit.json")


if __name__ == "__main__":
    main()
