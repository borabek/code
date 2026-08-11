# -*- coding: utf-8 -*-
"""R9c: ISARETLI OZNITELIK KARARI -- esigi kaydirmadan, kanit guclendirerek.

R9b: taban 0.5818 -> ESKI +0.0235 / YENI +0.0321, FARK **+0.0086**. Onceden yazdigim
kill esigi +0.01'di ve TUTMADI. Esigi kaydirmiyorum. Bunun yerine kararin dayanagini
iki dogru istatistikle guclendiriyorum:

  1. ESLESTIRILMIS GA: (YENI - ESKI) farkinin grup bootstrap araligi. Iki kol ayni
     parcalarda, ayni katlarda kosuyor; farkin kendisinin GA'si, iki ayri GA'ya bakmaktan
     DOGRU istatistiktir. Sifiri disliyorsa etki GERCEKTIR (buyuklugu ayri konu).
  2. VERI HACMI EGRISI: ayni deney satirlarin %25/%50/%100'uyle tekrarlanir. Dagitilan
     secici 154k satirla egitildi, buradaki test 27k. Eger FARK veriyle BUYUYORSA,
     154k'da esigi asacagi cikarimi mesrudur ve R10 (45 dk turetme) hak edilir. Fark
     veriyle KUCULUYORSA ya da sabitse, R10 kosulmaz -- bu, esigi kaydirmadan verilmis
     bir karardir.

Hicbir sey dagitilmaz.
"""
import io, json, os, pickle, sys
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
import olcum_kumesi
from big_arbiter import eligible
from sina_kume import esle, f1w
from r9b_taban_duzeltme import satirlari_kur, geo_kur, R4

MARJ = 0.02      # R9b'de iki kol da burada en iyiydi


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    DER, gate, ek = T2.yukle()
    stp = {p: s for m, p, jf, s in eligible()}
    PARCA = pickle.load(open(R4, "rb"))
    GEO = geo_kur(PARCA, stp)
    EX, YX, LY, LG, LK = satirlari_kur(DER, PARCA, GEO)
    print(f"satir {len(LY)} | dogru giris {LY.mean():.1%}", flush=True)

    ug = np.unique(LG); rng = np.random.RandomState(0); rng.shuffle(ug)
    kat = {g: j % 5 for j, g in enumerate(ug)}
    kk_ = np.array([kat[g] for g in LG])

    def oof(M, pay):
        """pay: egitim katlarindan kullanilacak GRUP orani (veri hacmi egrisi icin)."""
        o = np.zeros(len(LY))
        rs = np.random.RandomState(7)
        for f_ in range(5):
            tr = kk_ != f_
            if pay < 1.0:
                gtr = np.unique(LG[tr])
                sec = set(rs.choice(gtr, max(2, int(round(len(gtr) * pay))), replace=False))
                tr = tr & np.array([g in sec for g in LG])
            c = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1,
                                       random_state=0).fit(M[tr], LY[tr])
            o[~(kk_ != f_)] = c.predict_proba(M[kk_ == f_])[:, 1]
        return o

    def satir_uret(o):
        SC = {}
        for j, (pid, i, gi) in enumerate(LK):
            SC.setdefault((pid, i), {})[gi] = o[j]
        rob, gg = [], []
        for r in DER:
            d_ = PARCA.get(r["pid"])
            G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
            rj = "cok" if r["n"] >= 8 else "dusuk"
            if d_ is None:
                P = np.zeros((0, 3)); Pn = np.zeros((0, 3))
            else:
                P = d_["P"]; Pn = d_["Pd"].copy()
                if o is not None:
                    for i in range(len(P)):
                        s = SC.get((r["pid"], i))
                        if not s:
                            continue
                        g = max(s, key=s.get)
                        if g != 0 and s[g] - s.get(0, 0.0) >= MARJ:
                            Pn[i] = d_["SY"][i][g][1]
            rob.append((rj,) + esle(P, Pn, G, Gd, r["diag"], 2.0, 10.0, False, isaretli=True))
            gg.append(r["geo"])
        return rob, gg

    fn = lambda rows: f1w([q for _, q in rows]) - f1w([p for p, _ in rows])
    print(f"\n{'pay':>6}{'satir':>8}{'ESKI':>9}{'YENI':>9}{'FARK':>9}"
          f"{'ESLESTIRILMIS GA (YENI-ESKI)':>32}{'AUC E':>8}{'AUC Y':>8}")
    SON = {}
    for pay in (0.25, 0.50, 1.00):
        oE = oof(EX, pay); oY = oof(YX, pay)
        rE, gg = satir_uret(oE); rY, _ = satir_uret(oY)
        fE, fY = f1w(rE), f1w(rY)
        _, lo, hi = olcum_kumesi.grup_bootstrap(list(zip(rE, rY)), gg, fn, n=3000)
        aE = float(roc_auc_score(LY, oE)); aY = float(roc_auc_score(LY, oY))
        gerc = "GERCEK" if (lo > 0 or hi < 0) else "gurultu"
        print(f"{pay:>6.2f}{int(len(LY)*pay):>8}{fE:>9.4f}{fY:>9.4f}{fY-fE:>+9.4f}"
              f"        [{lo:+.4f},{hi:+.4f}] {gerc:<8}{aE:>8.4f}{aY:>8.4f}")
        SON[str(pay)] = {"eski": fE, "yeni": fY, "fark": fY - fE, "ga": [lo, hi],
                         "auc_eski": aE, "auc_yeni": aY, "gercek": bool(lo > 0 or hi < 0)}
    f25 = SON["0.25"]["fark"]; f50 = SON["0.5"]["fark"]; f100 = SON["1.0"]["fark"]
    buyuyor = f100 > f50 > f25
    tam = SON["1.0"]
    print(f"\nFARK egrisi: %25 {f25:+.4f} -> %50 {f50:+.4f} -> %100 {f100:+.4f} "
          f"-> {'BUYUYOR' if buyuyor else 'BUYUMUYOR'}")
    karar = bool(tam["gercek"] and buyuyor)
    print(f"\nKARAR: fark GERCEK mi={tam['gercek']} & veriyle BUYUYOR mu={buyuyor} "
          f"-> R10 {'HAK EDILDI' if karar else 'KOSULMAZ'}")
    if not karar and tam["gercek"]:
        print("  (fark gercek ama veriyle buyumuyor: 154k satirda esigi asacagi "
              "cikarimi DAYANAKSIZ -- esigi kaydirmiyorum)")
    with io.open("results/r9c_karar.json", "w", encoding="utf-8") as f:
        json.dump({"marj": MARJ, "egri": SON, "buyuyor": bool(buyuyor),
                   "karar_r10": karar}, f, indent=1)
    print("makbuz -> results/r9c_karar.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
