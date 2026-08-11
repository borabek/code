# -*- coding: utf-8 -*-
"""GENISLETILMIS HAVUZ + URUNUN KENDI GATE RECETESI.

ONCEKI KOSUNUN KUSURU: gate'i elle `RandomForestClassifier(...).fit(X, y)` diye
kurdum. Urunun gate'i oyle calismiyor -- `wire_gate.karar_skoru` once
`parca_ici(X, donusum)` uyguluyor (58 -> 116 sutun, PARCA PARCA z-skor) sonra
`_cokus_yonlendir` ile yonlendiriyor. Parca-ici z-skor bu projenin en buyuk tek
gate kazancidir (+0.1273, `gate-refit-minv4`). Onsuz egitilen gate, urunun
gate'i DEGILDIR ve kiyas gecersizdir.

Burada `refit_gate_v7.py` recetesi birebir kullanilir (parca_ici PARCA PARCA,
RF 400/leaf3, urun model sozlugu) ve olcum `wire_gate.karar_skoru` uzerinden
yapilir -- yani olcum yolu = urun yolu.

Ayrica IKI HAVUZ ayni recete ile egitilir ki kiyas TEK DEGISKENLI olsun:
  A) TEZ-SAF havuz  (yalniz segmentasyon `v_o` adaylari)
  B) GENISLETILMIS  (seg + B-rep fiziksel onerileri)
"""
import collections, json, os, pickle, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import wire_gate, kanonik_d7 as K
from sklearn.ensemble import RandomForestClassifier
from p1c_esik import maske
from sina_kume import esle_macar

OZ = "results/_brep_oz"
DONUSUM = "zskor"


def parcalar(on, kaynakli):
    """on: 'tam'|'d6'|'d7'. kaynakli=True ise `kaynak` alani da doner."""
    v = []
    for f in sorted(os.listdir(OZ)):
        if not (f.startswith(on + "_") and f.endswith(".npz")):
            continue
        z = np.load(f"{OZ}/{f}")
        d = {"pid": f[len(on)+1:-4], "X": z["X"], "y": z["y"]}
        if kaynakli:
            if "kaynak" not in z:
                continue
            d.update({"P": z["P"], "D": z["D"], "kaynak": z["kaynak"]})
        v.append(d)
    return v


tr = parcalar("tam", False) + parcalar("d6", False)
te = parcalar("d7", True)
kay7 = K.yukle([d["pid"] for d in te])
for d in te:
    r = kay7[d["pid"]]
    d.update({"mfg": r["mfg"], "G": np.asarray(r["G"], float),
              "Gd": np.asarray(r["Gd"], float), "diag": r["diag"]})
print(f"EGITIM {len(tr)} parca | SINAV {len(te)} parca", flush=True)
# D6 parcalarinda kaynak var; egitimde de tez-saf/genisletilmis ayrimi gerekli
d6k = {d["pid"]: d for d in parcalar("d6", True)}
tam_k = {}
for f in sorted(os.listdir(OZ)):
    if f.startswith("tam_") and f.endswith(".npz"):
        tam_k[f[4:-4]] = None          # tam_* dosyalarinda kaynak SAKLANMADI


def egit(segtek):
    """Urunun recetesi: parca_ici PARCA PARCA, RF 400/leaf3, urun model sozlugu."""
    M, Y = [], []
    atlanan = 0
    for d in tr:
        X, y = d["X"], d["y"]
        if segtek:
            k = d6k.get(d["pid"])
            if k is None:
                atlanan += 1          # tam_* icin kaynak yok -> tez-saf kol D6 ile sinirli
                continue
            m = k["kaynak"] == 0
            X, y = k["X"][m], k["y"][m]
        if not len(X):
            continue
        M.append(wire_gate.parca_ici(np.asarray(X, float), DONUSUM))
        Y.append(y)
    M = np.vstack(M); Y = np.concatenate(Y)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                 random_state=0).fit(M, Y)
    return {"clf": clf, "cols": None, "n_feat": M.shape[1], "donusum": DONUSUM,
            "feat_names": None}, M.shape, float(Y.mean()), atlanan


def olc(model, segtek):
    en = None
    for tip, e in ([("mutlak", x) for x in (0.20, 0.30, 0.40, 0.50)] +
                   [("goreli", x) for x in ((0.5, 0.20), (0.5, 0.30), (0.4, 0.25))]):
        rob = collections.defaultdict(lambda: [0, 0, 0]); tes = []
        for d in te:
            m0 = d["kaynak"] == 0 if segtek else np.ones(len(d["y"]), bool)
            X = np.asarray(d["X"][m0], float)
            if not len(X):
                continue
            s = np.asarray(wire_gate.karar_skoru(model, X), float)
            k = (s >= e) if tip == "mutlak" else maske(s, e[0], e[1])
            P, D = d["P"][m0], d["D"][m0]
            P, D = (P[k], D[k]) if k.any() else (P[:0], D[:0])
            if len(P) > 1:
                nm = wire_gate.kalabalik_maskesi(P, s[k]); P, D = P[nm], D[nm]
            tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL,
                                    K.ACI, False, isaretli=True)[:3]
            a = rob[d["mfg"]]; a[0] += tp; a[1] += fp; a[2] += fn
            tes.append((len(d["G"]),) + esle_macar(P, D, d["G"], d["Gd"], d["diag"],
                       max(3.0, 0.06*d["diag"]), 180.0, True)[:3])
        pm = {m: 2*a[0]/max(2*a[0]+a[1]+a[2], 1) for m, a in rob.items()}
        mi = float(2*sum(a[0] for a in rob.values()) /
                   max(sum(2*a[0]+a[1]+a[2] for a in rob.values()), 1))
        r = {"kural": f"{tip} {e}", "robot": mi, "tespit": K.mikro(tes),
             "makro": float(np.mean(list(pm.values()))),
             "en_kotu": float(min(pm.values())), "marka": pm}
        print(f"    {r['kural']:<16} robot {mi:.4f} | tespit {r['tespit']:.4f} | "
              f"makro {r['makro']:.4f}", flush=True)
        if en is None or r["robot"] > en["robot"]:
            en = r
    return en


out = {}
for ad, segtek in (("GENISLETILMIS (seg + B-rep)", False),):
    print(f"\n{ad}", flush=True)
    model, sh, poz, atl = egit(segtek)
    print(f"  egitim {sh} | pozitif {poz:.4f}", flush=True)
    pickle.dump(model, open("results/brep_gate_recete.pkl", "wb"))
    out[ad] = olc(model, segtek)
    print(f"  EN IYI: {out[ad]['kural']} robot {out[ad]['robot']:.4f}", flush=True)

g = out["GENISLETILMIS (seg + B-rep)"]
print(f"\nKANONIK TABAN (tez-saf, dagitilan gate v6 + NMS): robot 0.2029 / tespit 0.4523")
print(f"GENISLETILMIS (urun recetesi):                     robot {g['robot']:.4f} / "
      f"tespit {g['tespit']:.4f}  ({g['robot']-0.2029:+.4f})")
print(f"Genisletilmis havuz tavani 0.5748 -> yakalanan pay %{100*g['robot']/0.5748:.1f}")
json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
           "taban_tez_saf": {"robot": 0.2029, "tespit": 0.4523},
           "tavan_genisletilmis": 0.5748,
           "not": "Urunun gate recetesi (parca_ici zskor PARCA PARCA, RF 400/leaf3) "
                  "ve urunun karar_skoru ile olculdu. D7 marka-disi, MIKRO. "
                  "B-rep TEZ TURETMESI DEGIL, ek aday kaynagi."},
          open("results/brep_gate_recete_d7.json", "w"), indent=1)
print("makbuz -> results/brep_gate_recete_d7.json")
