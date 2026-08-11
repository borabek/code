# -*- coding: utf-8 -*-
"""F2-12: VERI-ONLY KONTROL -- korpus buyumesi ve uretici cesitliligi F1'i oynatiyor mu?

ASIL KOL. Olculmus kazanci olan tek mekanizma veri
([[ogrenme-egrisi-fiyat-etiketi]] +0.033/ln(grup), [[cesitlilik-ve-fn-profili]] +0.0443).

IKI GATE, TEK FARK VERI:
    TABAN : results/zengin_parite_v3_taban.npz   (yalniz ESKI parcalar)
    v3    : results/zengin_parite_v3.npz         (eski + YENI parcalar)
Ikisi de AYNI protokol filtresinden gecti, AYNI siniflandirici, AYNI hiperparametre,
AYNI tohum. Bu sart: [[gate-refit-minv4]] dersi "iki gate AYNI dagilimda egitilmeli".

URUNUN KARAR YOLU KULLANILIR: skor `wire_gate.karar_skoru`, maske `karar_maskesi`.
Olcum betiklerinin gate'i ELDE yeniden kurmasi 2026-08-01'de yakalanmis bir hataydi --
urunun `apply()` yolu arada yonlendirme yapiyor ve iki yol ayrisirsa olculen sey urunun
YAPTIGI sey olmaz. Bu yuzden model sozlugu dagitilanla AYNI alanlarla kurulur.

TEZ DEGISMEZ: ag egitilmez, remesh degismez, `v_o` degismez, aday ureticisi degismez.
"""
import collections
import io
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wire_gate

TABAN = "results/zengin_parite_v3_taban.npz"
V3 = "results/zengin_parite_v3.npz"
MAKBUZ = "results/f2_12_veri_kolu.json"


def egit(npz, tohum=0):
    """Dagitilan gate ile AYNI recete: RF 400 / leaf 3, 58 ham -> parca-ici z-skor -> 116."""
    from sklearn.ensemble import RandomForestClassifier
    d = np.load(npz, allow_pickle=True)
    X = np.hstack([d["X22"], d["XR"]]).astype(float)
    pid = np.asarray(d["pids"], str)
    # PARCA-ICI z-skor PARCA BAZINDA uygulanir (calisma aninda da oyle)
    Z = np.zeros((len(X), X.shape[1] * 2), float)
    for p in np.unique(pid):
        m = pid == p
        Z[m] = wire_gate.parca_ici(X[m], "zskor")
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=tohum)
    clf.fit(Z, d["y"])
    model = {"clf": clf, "n_feat": Z.shape[1], "donusum": "zskor", "cols": None,
             "feat_names": None}
    return model, {"aday": int(len(d["y"])), "parca": int(len(np.unique(pid))),
                   "uretici": int(len(set(map(str, d["mfg"])))),
                   "pozitif": float(np.mean(d["y"]))}


def olc(model, DER):
    """Tespit F1 -- urunun karar yolu + metrigin esleme kurali."""
    tp = fp = fn = 0
    per = collections.defaultdict(lambda: [0, 0, 0])
    for r in DER:
        if r["X"] is None or r.get("XR") is None:
            continue
        M = np.hstack([r["X"], r["XR"]]).astype(float)
        if M.shape[1] * 2 != model["n_feat"]:
            continue
        k = wire_gate.karar_maskesi(wire_gate.karar_skoru(model, M))
        P = np.asarray(r["P"], float)[k]
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        tol = max(3.0, 0.06 * float(r["diag"]))
        e = 0
        if len(P) and len(G):
            d = P[:, None, :] - G[None, :, :]
            al = (d * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(d - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
            up, ug = set(), set()
            for dd, a_, b_ in sorted((pe[a, b], a, b)
                                     for a in range(len(P)) for b in range(len(G))):
                if dd > tol or a_ in up or b_ in ug:
                    continue
                up.add(a_); ug.add(b_); e += 1
        tp += e; fp += len(P) - e; fn += len(G) - e
        s = per[r["mfg"]]; s[0] += e; s[1] += len(P) - e; s[2] += len(G) - e
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    pm = {m: 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1) for m, s in per.items()}
    return {"F1": f1, "TP": tp, "FP": fp, "FN": fn, "uretici": pm}


def main():
    import protokol
    protokol.tez_dogrula()
    import olcum_kumesi as OK
    DER, _ = OK.kume("results/_der_tam.pkl")
    print(f"OLCUM: {len(DER)} parca\n")

    S = {}
    for ad, npz in (("TABAN (yalniz eski)", TABAN), ("v3 (eski + YENI)", V3)):
        model, bilgi = egit(npz)
        r = olc(model, DER)
        S[ad] = {**r, "korpus": bilgi}
        print(f"{ad:<22} korpus {bilgi['parca']:>5} parca / {bilgi['uretici']:>2} uretici "
              f"| TESPIT F1 {r['F1']:.4f}  (TP {r['TP']} FP {r['FP']} FN {r['FN']})")

    a, b = S["TABAN (yalniz eski)"], S["v3 (eski + YENI)"]
    d = b["F1"] - a["F1"]
    print(f"\nFARK (v3 - taban): {d:+.4f}")
    ort = set(a["uretici"]) & set(b["uretici"])
    # URETICI KAPISI ICIN EN AZ 5 PARCA SARTI (2026-08-04'te ogrenildi):
    # ilk kosuda WAGO 0.80 -> 0.50 dustu ve kapiyi DUSURDU. Olcum kumesinde WAGO'dan
    # **TEK parca** var; tek parcalik bir kova gurultu uretir, gerileme degil. Kapiyi
    # tetikleyebilmek icin ureticinin en az 5 parcasi olmali; digerleri RAPORLANIR ama
    # karar vermez.
    say = collections.Counter(r["mfg"] for r in DER)
    print(f"\n{'uretici':<8}{'parca':>7}{'taban':>9}{'v3':>9}{'fark':>9}  {'kapi':>6}")
    kotu = []
    for m in sorted(ort):
        f = b["uretici"][m] - a["uretici"][m]
        n = say.get(m, 0)
        kapi = n >= 5
        print(f"{m:<8}{n:>7}{a['uretici'][m]:>9.4f}{b['uretici'][m]:>9.4f}{f:>+9.4f}"
              f"  {'EVET' if kapi else 'yok(n<5)':>6}")
        if f < -0.005 and kapi:
            kotu.append(m)
    print(f"\nGO: havuzlanmis >= +0.015  -> {'GECTI' if d >= 0.015 else 'gecmedi'}")
    print(f"    hicbir uretici < -0.005 -> {'GECTI' if not kotu else 'GECMEDI ' + str(kotu)}")
    print("\nNOT: bu ARA olcumdur -- turetme %62'de durdu, korpus henuz tam degil.")
    print("     Ayrica olcum kumesi PXC+WEI agirlikli; gorulmemis uretici sorusu D5-4 ile yanitlanir.")
    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"taban": a, "v3": b, "fark": d, "kotulesen_uretici": kotu,
                   "ara_olcum": True}, f, indent=1, ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
