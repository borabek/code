# -*- coding: utf-8 -*-
"""Gate v6: YENI aday dagiliminda (g7 agi + 1/2/2 birlesme yaricaplari) egit.

[[gate-refit-minv4]]: gate, gorecegi aday dagiliminin AYNISIYLA egitilmeli. P1
yaricaplari degistirdi (aday x1.46) ve g7 farkli bir ag -- eski gate bu havuzu hic
gormedi. Refit yapilmazsa P1'in kazanci TERSINE doner.

DOGRULAMA: URETICI-DISI (manufacturer-out) capraz gecerleme, EGITIM ureticileri
icinde. D7'ye DOKUNULMAZ -- o TEK ATIS icin saklidir.
"""
import collections, io, json, pickle, sys, os
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_gate_v5 import egit

V6 = "results/zengin_parite_v6.npz"
MODEL = "results/wire_gate_v6.pkl"


def main():
    import protokol
    protokol.tez_dogrula()
    from sklearn.metrics import roc_auc_score
    d = np.load(V6, allow_pickle=True)
    X = np.hstack([d["X22"], d["XR"]]).astype(float)
    y = np.asarray(d["y"]); pid = np.asarray(d["pids"], str); mfg = np.asarray(d["mfg"], str)
    print(f"korpus: {len(y)} aday / {len(set(pid))} parca / {len(set(mfg))} uretici | "
          f"pozitif %{100*y.mean():.1f}")
    import wire_gate
    # URETICI-DISI CV: her buyuk uretici sirayla DISARIDA
    buyuk = [m for m, n in collections.Counter(mfg).most_common() if n >= 1000]
    print(f"\nURETICI-DISI CV (n>=1000): {buyuk}")
    print(f"{'disarida':<8}{'egitim':>9}{'test':>8}{'AUC':>8}")
    aucs = {}
    for m in buyuk:
        tr = mfg != m; te = ~tr
        mod = egit(X[tr], y[tr], pid[tr])
        s = wire_gate.karar_skoru(mod, X[te])
        a = float(roc_auc_score(y[te], s)) if len(set(y[te])) > 1 else float("nan")
        aucs[m] = a
        print(f"{m:<8}{int(tr.sum()):>9}{int(te.sum()):>8}{a:>8.4f}")
    print(f"  ORTALAMA uretici-disi AUC: {np.nanmean(list(aucs.values())):.4f}")
    tam = egit(X, y, pid)
    with open(MODEL, "wb") as f:
        pickle.dump(tam, f)
    print(f"\nTAM MODEL -> {MODEL} (n_feat={tam['n_feat']})")
    with io.open("results/p6_gate_v6.json", "w", encoding="utf-8") as f:
        json.dump({"korpus": V6, "aday": int(len(y)), "parca": len(set(pid)),
                   "uretici": sorted(set(mfg)), "uretici_disi_auc": aucs,
                   "ort_auc": float(np.nanmean(list(aucs.values())))}, f, indent=1)


if __name__ == "__main__":
    main()
