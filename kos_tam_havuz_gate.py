# -*- coding: utf-8 -*-
"""TAM HAVUZ gate: egit, esigi D6'da sec, D7'de TAM ZINCIRLE olc.

Havuz = segmentasyon `v_o` + B-rep agizlari + mesh tepeleri (p_pos>=0.50,
2mm seyreltme; yon = yerel normal). Olculen TAVAN 0.8347 / ~318 aday/parca
(`results/tavan_080_eksensiz.json`).

Kiyas noktalari (hepsi D7 marka-disi, MIKRO, tam zincir):
  dagitilan urun (v6 + NMS)                 robot 0.2029 | tespit 0.4523
  B-rep havuzu + duzeltilmis etiket (gece)  robot 0.2249 | tespit 0.4518

KURALLAR:
* Etiket PROJENIN tanimi (yanal + eksenel 40 + acgozlu bire-bir) -- oznitelik
  uretiminde zaten oyle yazildi.
* Esik **D6'da** secilir (markalari D7'den ayrik), D7'de YENIDEN TARANMAZ.
* Egitim korpusu D6'yi ICERMEZ.
* Kollar TEK DEGISKENLI: yalniz havuz degisir.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import d6_kayit                 # noqa: E402
import kanonik_d7 as K          # noqa: E402
import urun_zinciri             # noqa: E402
import wire_gate                # noqa: E402
from p1c_esik import maske      # noqa: E402
from sina_kume import esle_macar  # noqa: E402

OZ = "results/_tam_oz"
TAN = "results/_tan_hizali"
TANLI = os.environ.get("TANLI", "0") not in ("0", "")
OB = "results/_p1_olasilik_d7"
DONUSUM = "zskor"
KURALLAR = ([("mutlak", x) for x in (0.05, 0.10, 0.15, 0.20, 0.30)] +
            [("goreli", x) for x in ((0.5, 0.05), (0.5, 0.10), (0.5, 0.20))])
# kaynak: 0=segmentasyon 1=B-rep 2=mesh tepesi
KOLLAR = json.loads(os.environ.get("TG_KOLLAR", '{"TEZ-SAF": [0], "B-REP": [0,1], "TAM HAVUZ": [0,1,2]}'))


def oku(on, kaynaklar):
    v = []
    for f in sorted(os.listdir(OZ)):
        if not (f.startswith(on + "_") and f.endswith(".npz")):
            continue
        z = np.load(f"{OZ}/{f}")
        kay = np.asarray(z["kaynak"], int)
        m = np.isin(kay, kaynaklar)
        if int(m.sum()) < 2:
            continue
        X = np.asarray(z["X"], float)
        if TANLI:
            tf = f"{TAN}/{f}"
            if not os.path.exists(tf):
                continue
            T = np.asarray(np.load(tf)["T"], float)
            if len(T) != len(X):
                raise SystemExit(f"{f}: tanimlayici {len(T)} vs oznitelik "
                                 f"{len(X)} -- HIZALAMA BOZUK, duruyorum")
            X = np.hstack([X, T])
        v.append({"pid": f[len(on) + 1:-4], "X": X[m],
                  "y": np.asarray(z["y"], int)[m],
                  "P": np.asarray(z["P"], float)[m],
                  "D": np.asarray(z["D"], float)[m]})
    return v


_D6 = None


def kimlikle(v):
    """pid -> kayit. D6 parcalari kanonik `_der_yeni_G7BIRLESIK.pkl` icinde YOK
    (olculdu: 0/468), bu yuzden `d6_kayit`'a DUSULUR. Bu dusme olmadan esik
    secim kumesi SESSIZCE bos kaliyordu."""
    global _D6
    kay = K.yukle([d["pid"] for d in v])
    eksik = [d["pid"] for d in v if d["pid"] not in kay]
    if eksik:
        if _D6 is None:
            _D6 = {str(p): r for p, r in
                   d6_kayit.yukle(set(d6_kayit.sinav()["pidler"])).items()}
        kay.update({p: _D6[p] for p in eksik if p in _D6})
    out = []
    for d in v:
        r = kay.get(d["pid"])
        if r is None:
            continue
        d.update({"mfg": r["mfg"], "G": np.asarray(r["G"], float),
                  "Gd": np.asarray(r["Gd"], float), "diag": float(r["diag"])})
        out.append(d)
    return out


def egit(tr):
    M = np.vstack([wire_gate.parca_ici(d["X"], DONUSUM) for d in tr])
    Y = np.concatenate([d["y"] for d in tr])
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(M, Y)
    return ({"clf": clf, "cols": None, "n_feat": M.shape[1],
             "donusum": DONUSUM}, M.shape, float(Y.mean()))


def olc(model, veri, tip, e, tam_zincir=False, S=None):
    rob = collections.defaultdict(lambda: [0, 0, 0])
    tes = []
    for d in veri:
        s = np.asarray(wire_gate.karar_skoru(model, d["X"]), float)
        k = (s >= e) if tip == "mutlak" else maske(s, e[0], e[1])
        P, D = (d["P"][k], d["D"][k]) if k.any() else (d["P"][:0], d["D"][:0])
        if len(P) > 1:
            nm = wire_gate.kalabalik_maskesi(P, s[k])
            P, D = P[nm], D[nm]
        if tam_zincir and len(P):
            f = f"{OB}/{d['pid']}.npz"
            if os.path.exists(f):
                z = np.load(f)
                P, D = urun_zinciri.tam_poz(
                    np.ascontiguousarray(z["V"], np.float64),
                    np.ascontiguousarray(z["F"], np.int64),
                    np.asarray(z["pbs"], float).mean(0), P, D,
                    step_path=(S or {}).get(d["pid"]))
        tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL,
                                K.ACI, False, isaretli=True)[:3]
        a = rob[d["mfg"]]
        a[0] += tp; a[1] += fp; a[2] += fn
        tes.append((len(d["G"]),) + esle_macar(
            P, D, d["G"], d["Gd"], d["diag"], max(3.0, 0.06 * d["diag"]),
            180.0, True)[:3])
    pm = {m: 2 * a[0] / max(2 * a[0] + a[1] + a[2], 1) for m, a in rob.items()}
    mi = float(2 * sum(a[0] for a in rob.values()) /
               max(sum(2 * a[0] + a[1] + a[2] for a in rob.values()), 1))
    return {"robot": mi, "tespit": K.mikro(tes),
            "makro": float(np.mean(list(pm.values()))),
            "en_kotu": float(min(pm.values())), "marka": pm}


def main():
    S = K.step_haritasi()
    sonuc = {}
    modeller = {}
    for ad, kaynaklar in KOLLAR.items():
        tr = kimlikle(oku("tam", kaynaklar))
        dev = kimlikle(oku("d6", kaynaklar))
        te = kimlikle(oku("d7", kaynaklar))
        if not tr or not dev or not te:
            raise SystemExit(f"{ad}: veri eksik (tr {len(tr)} dev {len(dev)} "
                             f"te {len(te)}) -- sessiz atlamak yerine DURUYORUM")
        m, sh, poz = egit(tr)
        modeller[ad] = m
        ap = float(np.mean([len(d["X"]) for d in te]))
        print(f"\n{ad}: egitim {len(tr)} parca {sh} pozitif {poz:.4f} | "
              f"D7 aday/parca {ap:.1f}", flush=True)
        en = None
        for tip, e in KURALLAR:
            r = olc(m, dev, tip, e)
            print(f"   D6 {str((tip, e)):<20} robot {r['robot']:.4f}", flush=True)
            if en is None or r["robot"] > en[2]["robot"]:
                en = (tip, e, r)
        tip, e, _ = en
        r7 = olc(m, te, tip, e, tam_zincir=True, S=S)
        sonuc[ad] = {"kural": f"{tip} {e}", "aday_per_parca": ap, "D7": r7}
        print(f"  SECILEN {tip} {e} -> D7 (TAM ZINCIR) robot **{r7['robot']:.4f}** "
              f"| tespit {r7['tespit']:.4f} | makro {r7['makro']:.4f} | "
              f"en kotu {r7['en_kotu']:.4f}", flush=True)
    print("\n" + "=" * 64)
    print(f"{'kol':<14}{'aday/parca':>12}{'robot':>9}{'tespit':>9}{'makro':>9}")
    for ad, c in sonuc.items():
        print(f"{ad:<14}{c['aday_per_parca']:>12.1f}{c['D7']['robot']:>9.4f}"
              f"{c['D7']['tespit']:>9.4f}{c['D7']['makro']:>9.4f}")
    print(f"{'URUN (v6+NMS)':<14}{13.4:>12.1f}{0.2029:>9.4f}{0.4523:>9.4f}"
          f"{0.2146:>9.4f}")
    with open(os.environ.get("TG_MODEL", "results/tam_havuz_gate_modeller.pkl"), "wb") as f:
        pickle.dump(modeller, f)
    json.dump({"damga": makbuz_hash.damga(), "sonuc": sonuc,
               "urun": {"robot": 0.2029, "tespit": 0.4523, "makro": 0.2146},
               "not": "Esik D6'da secildi, D7'de yeniden taranmadi. Egitim D6'yi "
                      "icermez. TAM URUN ZINCIRI (poz kafasi). MIKRO."},
              open(os.environ.get("TG_CIKTI", "results/tam_havuz_gate.json"), "w"), indent=1)
    print("makbuz -> results/tam_havuz_gate.json")


if __name__ == "__main__":
    main()
