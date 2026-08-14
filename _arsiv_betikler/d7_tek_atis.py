# -*- coding: utf-8 -*-
"""D7 FINAL SINAVI -- TEK ATIS. Bu betik BIR KEZ calisir ve sonucu MUHURLENIR.

KUME: `results/d7_sinav_kumesi.json` (838 parca / 12 uretici / 3201 CP, muhur
4bdb5c2ed81fe89f). Bu 12 uretici:
  * gate v6 egitim korpusunda YOK (uretici duzeyi dislama, dogrulandi)
  * g7 seg agi'nin oto-etiket korpusunda YOK (ETIKET_DISLA ile dislandi)
Yani YENI boru hattinin hicbir bileseni bu ureticilerden tek parca gormedi.

DEGERLENDIRILEN YIGIN (hepsi D6 uzerinde ya da D7'ye dokunmadan secildi):
  seg      : g7_s0 (16 uretici / 2367 etiket)
  yaricap  : cluster 1 / dedupe 2 / oy havuzu 2   (P1, D6'da secildi)
  gate     : wire_gate_v6 (yeni aday dagiliminda refit; uretici-disi AUC 0.8379)
  esik     : goreli 0.40 / taban 0.30             (D6'nin DEV yarisinda secildi)
  secici   : p3c eksen secici                     (D6'nin DEV yarisinda secildi)

BILINEN SINIRLILIK -- ACIKCA YAZILIYOR: eksen secici g5 aday dagiliminda egitildi,
g7'de degil. Ozellikleri geometrik (aci/mesafe/yaricap/paralel sayisi) oldugu icin
tasinmasi beklenir; yine de bu MUHAFAZAKAR bir secimdir -- etkisi varsa sayiyi
DUSURUR, sismez.

BUNDAN SONRA TUNING YAPILIRSA D7 DE DEV OLUR ve final icin D8 gerekir.
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

import d6_kayit

KUME = "results/d7_sinav_kumesi.json"
DESEN = "results/_der_yeni_G7BIRLESIK.pkl"
MAKBUZ = "results/D7_FINAL.json"
ORAN, TABAN = 0.40, 0.30
ROBOT_YANAL, ROBOT_ACI = 2.0, 10.0


def pr(rows):
    tp = sum(r[1] for r in rows); fp = sum(r[2] for r in rows); fn = sum(r[3] for r in rows)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / max(p + r, 1e-9))


def main():
    import protokol
    protokol.tez_dogrula()
    import p3c_eksen_secici as P3C
    from p0_tavan_yeniden import kod_muhru
    from sina_kume import esle_macar, f1w

    sv = json.load(io.open(KUME, encoding="utf-8"))
    print(f"D7 FINAL: {sv['n_parca']} parca | {len(sv['uretici'])} uretici | "
          f"GT {sv['gt_toplam']} CP | muhur {sv['sha16']}")
    kayit = d6_kayit.yukle(set(sv["pidler"]), desen=DESEN)
    print(f"turetme kaydi: {len(kayit)}/{sv['n_parca']}  (g7 + yaricap 1/2/2)\n")

    with open("results/wire_gate_v6.pkl", "rb") as f:
        gate = pickle.load(f)
    with open("results/_d7_silindirler.pkl", "rb") as f:
        cyl = pickle.load(f)
    with open("results/p3c_eksen_secici.pkl", "rb") as f:
        sec = pickle.load(f)["clf"]

    T, R = [], []
    Tm = collections.defaultdict(list); Rm = collections.defaultdict(list)
    Tr = collections.defaultdict(list); Rr = collections.defaultdict(list)
    pid_sira = []
    for pid, r in kayit.items():
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        P = np.zeros((0, 3)); D = np.zeros((0, 3))
        pak = P3C.parca_adaylari(r, gate, cyl, oran=ORAN, taban=TABAN)
        if pak is not None:
            P, D, Sk, komsu = pak
            cy = cyl.get(pid) or []
            P2 = P.copy(); D2 = D.copy()
            for i in range(len(P)):
                opt = P3C.secenekler(cy, P[i], D[i], r["diag"], float(Sk[i]), komsu, len(P))
                if len(opt) > 1:
                    s = sec.predict_proba(np.asarray([o[2] for o in opt], float))[:, 1]
                    j = int(np.argmax(s))
                    if j != 0:
                        P2[i] = opt[j][0]; D2[i] = opt[j][1]
            P, D = P2, D2
        t = (rj,) + esle_macar(P, D, G, Gd, r["diag"], 0.0, 180.0, True)[:3]
        q = (rj,) + esle_macar(P, D, G, Gd, r["diag"], ROBOT_YANAL, ROBOT_ACI,
                               False, isaretli=True)[:3]
        T.append(t); R.append(q)
        Tm[r["mfg"]].append(t); Rm[r["mfg"]].append(q)
        Tr[rj].append(t); Rr[rj].append(q)
        pid_sira.append(pid)

    tf, rf = f1w(T), f1w(R)
    tp_, tr_, _ = pr(T); rp_, rr_, _ = pr(R)
    print(f"{'':<22}{'F1':>9}{'kesinlik':>10}{'recall':>9}")
    print(f"{'TESPIT':<22}{tf:>9.4f}{tp_:>10.4f}{tr_:>9.4f}")
    print(f"{'ROBOT':<22}{rf:>9.4f}{rp_:>10.4f}{rr_:>9.4f}")
    print(f"{'donusum':<22}{rf/max(tf,1e-9):>8.1%}")
    print(f"\n{'rejim':<10}{'n':>6}{'TESPIT':>9}{'ROBOT':>9}")
    for rj in ("dusuk", "cok"):
        if Tr[rj]:
            print(f"{rj:<10}{len(Tr[rj]):>6}{f1w(Tr[rj]):>9.4f}{f1w(Rr[rj]):>9.4f}")
    print(f"\n{'uretici':<8}{'n':>5}{'TESPIT':>9}{'ROBOT':>9}")
    for m in sorted(Tm, key=lambda x: -len(Tm[x])):
        print(f"{m:<8}{len(Tm[m]):>5}{f1w(Tm[m]):>9.4f}{f1w(Rm[m]):>9.4f}")

    # URETICI BOOTSTRAP (%95 alt sinir) -- uretici duzeyinde yeniden ornekle
    rng = np.random.RandomState(0)
    urs = sorted(Rm)
    boot = []
    for _ in range(2000):
        sec_u = rng.choice(urs, len(urs), replace=True)
        satir = [s for u in sec_u for s in Rm[u]]
        boot.append(f1w(satir))
    alt = float(np.percentile(boot, 2.5)); ust = float(np.percentile(boot, 97.5))
    print(f"\nURETICI BOOTSTRAP robot F1: {rf:.4f}  %95 arali [{alt:.4f}, {ust:.4f}]")

    kabul = {
        "robot >= 0.75": rf >= 0.75,
        "bootstrap alt >= 0.70": alt >= 0.70,
        "kesinlik >= 0.70": rp_ >= 0.70,
        "recall >= 0.70": rr_ >= 0.70,
        "dusuk-CP >= 0.80": (f1w(Rr["dusuk"]) >= 0.80) if Rr["dusuk"] else False,
        "cok-CP >= 0.50": (f1w(Rr["cok"]) >= 0.50) if Rr["cok"] else False,
        ">=20 parcali uretici < 0.50 YOK":
            all(f1w(Rm[m]) >= 0.50 for m in Rm if len(Rm[m]) >= 20),
    }
    print(f"\nFINAL KABUL OLCUTLERI:")
    for k, v in kabul.items():
        print(f"  [{'GECTI' if v else 'KALDI'}] {k}")
    print(f"\nSONUC: {'KABUL' if all(kabul.values()) else 'KABUL EDILMEDI'}")

    with io.open(MAKBUZ, "w", encoding="utf-8") as f:
        json.dump({"kume": KUME, "muhur": sv["sha16"], "n": len(kayit),
                   "tespit": {"f1": tf, "kesinlik": tp_, "recall": tr_},
                   "robot": {"f1": rf, "kesinlik": rp_, "recall": rr_},
                   "donusum": rf / max(tf, 1e-9),
                   "rejim": {rj: {"tespit": f1w(Tr[rj]), "robot": f1w(Rr[rj])}
                             for rj in Tr},
                   "uretici": {m: {"n": len(Tm[m]), "tespit": f1w(Tm[m]),
                                   "robot": f1w(Rm[m])} for m in Tm},
                   "bootstrap": {"alt": alt, "ust": ust},
                   "kabul": kabul, "hash": kod_muhru(),
                   "yigin": {"seg": "g7_s0", "yaricap": "1/2/2", "gate": "wire_gate_v6",
                             "esik": f"{ORAN}/{TABAN}", "secici": "p3c"},
                   "sinirlilik": ("eksen secici g5 aday dagiliminda egitildi; "
                                  "muhafazakar secim -- etkisi varsa sayiyi DUSURUR")},
                  f, indent=1, ensure_ascii=False)
    print(f"makbuz -> {MAKBUZ}   [D7 HARCANDI]")


if __name__ == "__main__":
    main()
