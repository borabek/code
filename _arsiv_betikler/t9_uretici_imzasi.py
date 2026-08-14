# -*- coding: utf-8 -*-
"""T9: hangi SUTUNLAR uretici imzasi tasiyor? (siralama tedavisinin teshisi)

T2 OLCTU: uretici-disi cokusun ~%58'i SIRALAMA hatasi (AUC 0.895 -> 0.792 / 0.694), yani
kalibrasyon duzeltilse bile kalan kayip. Sebep: bazi ozellikler URETICIYE OZGU degerler
tasiyor ve model onlari ezberliyor.

IKI OLCUM, her sutun icin:
  (1) URETICI AYIRT EDICILIGI: sutun tek basina "bu aday hangi ureticiden" sorusunu ne kadar
      cevapliyor (AUC). Yuksekse o sutun bir URETICI IMZASIDIR.
  (2) TRANSFER KAYBI: sutunun TP/FP ayirt ediciligi (AUC) uretici icinde ve uretici-disi
      arasinda ne kadar dusuyor.

Iyi bir ozellik: (1) DUSUK, (2) DUSUK. Kotu: (1) yuksek -- model onu kestirme olarak kullanir
ve yeni ureticide yaniltir.

TEDAVI HIPOTEZI: olcek-bagimli sutunlar (mm cinsinden mutlak buyukluk, kose sayisi) uretici
imzasi tasir; parca capina bolununce imza silinir ama bilgi kalir. FIZIKSEL sutunlar (brep_r =
gercek delik yaricapi) mutlak kalmali -- tel capi ureticiden bagimsizdir ve T1 zaten fiziksel
sutunlarin transferi +0.047 iyilestirdigini olctu.
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from t1_uretici_disi import auc_mw


def main():
    import wire_gate
    d = np.load("results/gate_regrow_data_fiz.npz", allow_pickle=True)
    X = d["X"][:, :18]; y = d["y"].astype(bool)
    mfg = np.array([str(x) for x in d["mfg"]])
    isim = list(wire_gate.FEAT_NAMES_13) + list(wire_gate.FEAT_NAMES_FIZ)
    u0 = mfg == sorted(set(mfg))[0]

    print(f"{'sutun':<16}{'URETICI imzasi':>15}{'TP/FP ic':>10}{'TP/FP disi':>12}{'transfer':>10}")
    out = {}
    for i, n in enumerate(isim):
        imza = abs(auc_mw(X[:, i], u0) - 0.5) + 0.5        # ureticiyi ne kadar ele veriyor
        ic = abs(auc_mw(X[u0, i], y[u0]) - 0.5) + 0.5      # kendi ureticisinde ayirt edicilik
        dis = abs(auc_mw(X[~u0, i], y[~u0]) - 0.5) + 0.5   # oteki ureticide
        out[n] = {"imza": imza, "ic": ic, "dis": dis, "transfer": dis - ic}
        print(f"{n:<16}{imza:>15.3f}{ic:>10.3f}{dis:>12.3f}{dis-ic:>10.3f}")

    sirali = sorted(out.items(), key=lambda kv: -kv[1]["imza"])
    print("\nEN COK URETICI IMZASI TASIYAN 6 SUTUN:")
    for n, v in sirali[:6]:
        print(f"  {n:<16} imza {v['imza']:.3f}")
    print("\nEN AZ TASIYANLAR (guvenli cekirdek):")
    for n, v in sirali[-6:]:
        print(f"  {n:<16} imza {v['imza']:.3f}")

    # Ezberci sutunlari ATMAK transferi duzeltir mi? (L2 dersi: tanidik veride kaybettirmisti)
    print(f"\n{'kume':<34}{'uretici-disi F1 (en kotu)':>26}")
    THR = float(json.load(open("cp_config.json", encoding="utf-8"))["robot_wire_gate_threshold"])
    def kotu(cols):
        v = []
        for u in sorted(set(mfg)):
            te = mfg == u
            s = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                       random_state=0).fit(X[~te][:, cols], y[~te]).predict_proba(
                                           X[te][:, cols])[:, 1]
            m = s >= THR
            tp = int((y[te] & m).sum()); fp = int((~y[te] & m).sum()); fn = int((y[te] & ~m).sum())
            p_ = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
            v.append(2 * p_ * r_ / max(p_ + r_, 1e-9))
        return min(v), v
    tum = list(range(18))
    b, v = kotu(tum); print(f"{'18 sutun (mevcut)':<34}{b:>26.4f}   {[round(x,4) for x in v]}")
    for k in (2, 3, 4):
        at = {isim.index(n) for n, _ in sirali[:k]}
        cols = [i for i in tum if i not in at]
        b2, v2 = kotu(cols)
        print(f"{'en imzali ' + str(k) + ' atildi (' + str(18-k) + ')':<34}{b2:>26.4f}"
              f"   {[round(x,4) for x in v2]}")
        out[f"at{k}"] = {"en_kotu": b2, "hepsi": v2}
    json.dump(out, open("results/t9_uretici_imzasi.json", "w"), indent=1)
    print("\nmakbuz -> results/t9_uretici_imzasi.json")


if __name__ == "__main__":
    main()
