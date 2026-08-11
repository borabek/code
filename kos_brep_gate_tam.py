# -*- coding: utf-8 -*-
"""TAM OLCEKLI B-rep gate: 2583 parcada egit, D7'de (marka-disi) olc.

GEREKCE ZINCIRI (hepsi olculdu, makbuzlari var):
 1. D7 seg-tek havuz recall 0.6654 -> robot TAVANI 0.3412 => `.50` eski havuzda
    IMKANSIZDI (`results/havuz_recall_d7.json`, `results/havuz_tavani.json`).
 2. B-rep onerileri eklenince recall 0.8465, robot tavani **0.5748**.
 3. Esit egitim buyuklugunde havuzun uctan uca etkisi **+0.0330 robot**;
    ilk kosudaki dusus HAVUZDAN DEGIL gate'in 468 parcalik kucuk egitim
    kumesinden geliyordu (`results/brep_kontrol.json`).
 => Bu betik o karistiriciyi kaldirir: gate 2583 parcayla egitilir.

BEKLENTI DURUSTCE: mevcut secici seg-tek tavanin %57.7'sini yakaliyor; ayni oran
genisletilmis havuzda ~0.33 eder. 0.50 icin secici veriminin de artmasi gerekir.

SIZINTI: egitim kumesi `results/brep_egitim_kumesi.json` (muhur ffb7950c349ad534),
D7 kesisimi SIFIR olacak sekilde kuruldu ve burada YENIDEN dogrulanir.
"""
import collections, json, os, pickle, sys, time
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import connector3d, wire_gate, brep_havuz, makbuz_hash
import kanonik_d7 as K
from sina_kume import esle_macar

CE, CT = int(connector3d.CABLE_ENTRY), int(connector3d.CONTACT)
OZ = "results/_brep_oz"; os.makedirs(OZ, exist_ok=True)
OB = "results/_p1_olasilik_brepegit"
S = K.step_haritasi()

egit_pid = [str(p) for p in json.load(open("results/brep_egitim_kumesi.json"))["pidler"]]
d7set = set(map(str, json.load(open("results/d7_sinav_kumesi.json"))["pidler"]))
kesisim = set(egit_pid) & d7set
assert not kesisim, f"SIZINTI: egitim kumesinde {len(kesisim)} D7 parcasi"
print(f"egitim {len(egit_pid)} parca | D7 kesisimi 0 DOGRULANDI", flush=True)

R = {str(r["pid"]): r for r in pickle.load(open(K.KAYIT, "rb"))}
cy = pickle.load(open("results/_brepegit_silindirler.pkl", "rb"))
ac = pickle.load(open("results/_brepegit_acikliklar.pkl", "rb"))
print(f"silindir onbellegi {len(cy)} | aciklik onbellegi {len(ac)}", flush=True)

tr = []; t0 = time.time(); yok = 0
for i, pid in enumerate(egit_pid):
    r = R.get(pid)
    if r is None or not len(r.get("G", [])):
        continue
    yol = f"{OZ}/tam_{pid}.npz"
    if os.path.exists(yol):
        try:
            z = np.load(yol); tr.append({"X": z["X"], "y": z["y"]}); continue
        except Exception:
            os.remove(yol)
    f = f"{OB}/{pid}.npz"
    if not os.path.exists(f):
        yok += 1
        continue
    z = np.load(f)
    V = np.ascontiguousarray(z["V"], np.float64)
    F = np.ascontiguousarray(z["F"], np.int64)
    pb = np.asarray(z["pbs"], float).mean(0)
    P, D, kay = brep_havuz.birlesik_havuz(r["P"], r["Pd"], cy.get(pid), ac.get(pid))
    if not len(P):
        continue
    X = np.asarray(wire_gate.feats_for(
        V, F, pb, [{"point": P[j], "direction": D[j]} for j in range(len(P))],
        CE, CT, step_path=S.get(pid)), float)
    G = np.asarray(r["G"], float)
    tol = max(3.0, 0.06 * r["diag"])
    y = (np.linalg.norm(P[:, None] - G[None], axis=-1).min(1) <= tol).astype(np.int8)
    np.savez_compressed(yol, X=X, y=y)
    tr.append({"X": X, "y": y})
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(egit_pid)} ({time.time()-t0:.0f}s, onbellek yok {yok})",
              flush=True)
# D6 de EGITIME katilir: D7'den marka olarak AYRIK ve ozniteligi zaten onbellekte.
# Ayrica D6 YUKSEK-CP agirlikli; korpus tek basina dusuk-CP agirlikli oldugu icin
# (pozitif orani 0.1238 -> 0.0475) skor kalibrasyonu kayiyordu.
d6ek = 0
for f in sorted(os.listdir(OZ)):
    if f.startswith("d6_") and f.endswith(".npz"):
        z = np.load(f"{OZ}/{f}")
        tr.append({"X": z["X"], "y": z["y"]}); d6ek += 1
print(f"EGITIM {len(tr)} parca hazir (korpus {len(tr)-d6ek}, D6 {d6ek}, "
      f"olasiligi olmayan {yok})", flush=True)

Xtr = np.vstack([d["X"] for d in tr]); ytr = np.concatenate([d["y"] for d in tr])
print(f"X {Xtr.shape} | pozitif {ytr.mean():.4f}", flush=True)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                             class_weight="balanced_subsample", random_state=0)
clf.fit(Xtr, ytr)
pickle.dump({"clf": clf, "n_feat": Xtr.shape[1], "n_parca": len(tr)},
            open(os.environ.get("BREP_MODEL", "results/brep_gate_tam.pkl"), "wb"))
print("gate egitildi -> results/brep_gate_tam.pkl", flush=True)

te = []
for f in sorted(os.listdir(OZ)):
    if f.startswith("d7_") and f.endswith(".npz"):
        z = np.load(f"{OZ}/{f}")
        te.append({"pid": f[3:-4], "X": z["X"], "P": z["P"], "D": z["D"],
                   "kaynak": z["kaynak"]})
kay7 = K.yukle([d["pid"] for d in te])
for d in te:
    r = kay7[d["pid"]]
    d.update({"mfg": r["mfg"], "G": np.asarray(r["G"], float),
              "Gd": np.asarray(r["Gd"], float), "diag": r["diag"]})
print(f"SINAV {len(te)} parca", flush=True)

out = {}
# Esikler ASAGI uzatildi: ara kosuda en iyi esik taranan araligin EN ALTINDAYDI
# (0.20) ve monoton artiyordu -- yani optimum altta kalmisti.
# GORELI kol: urunun `p1c_esik.maske` kurali (parca-ici goreli + mutlak taban);
# olculdu ki dagilim kaymasini emiyor (gate-uretici-disi-cokusu).
from p1c_esik import maske as _goreli
KOLLAR = [("mutlak", e) for e in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)] +          [("goreli", (o, t)) for o, t in ((0.5, 0.05), (0.5, 0.10), (0.4, 0.15),
                                          (0.3, 0.20))]
for _tip, esik in KOLLAR:
    rob = collections.defaultdict(lambda: [0, 0, 0]); tes = []
    for d in te:
        s = clf.predict_proba(d["X"])[:, 1]
        k = (s >= esik) if _tip == "mutlak" else _goreli(s, esik[0], esik[1])
        P, D = (d["P"][k], d["D"][k]) if k.any() else (d["P"][:0], d["D"][:0])
        if len(P) > 1:
            nm = wire_gate.kalabalik_maskesi(P, s[k]); P, D = P[nm], D[nm]
        tp, fp, fn = esle_macar(P, D, d["G"], d["Gd"], d["diag"], K.YANAL, K.ACI,
                                False, isaretli=True)[:3]
        a = rob[d["mfg"]]; a[0] += tp; a[1] += fp; a[2] += fn
        tes.append((len(d["G"]),) + esle_macar(P, D, d["G"], d["Gd"], d["diag"],
                   max(3.0, 0.06*d["diag"]), 180.0, True)[:3])
    pm = {m: 2*a[0]/max(2*a[0]+a[1]+a[2], 1) for m, a in rob.items()}
    mi = float(2*sum(a[0] for a in rob.values()) /
               max(sum(2*a[0]+a[1]+a[2] for a in rob.values()), 1))
    out[f"{_tip} {esik}"] = {"robot": mi, "tespit": K.mikro(tes),
                      "makro": float(np.mean(list(pm.values()))),
                      "en_kotu": float(min(pm.values())), "marka": pm}
    _a = f"{_tip} {esik}"
    print(f"{_a:<16} robot {mi:.4f} | tespit {out[_a]['tespit']:.4f} | "
          f"makro {out[_a]['makro']:.4f} | en kotu {out[_a]['en_kotu']:.4f}", flush=True)
en = max(out, key=lambda e: out[e]["robot"])
print(f"\nEN IYI esik {en}: robot {out[en]['robot']:.4f} "
      f"(kanonik taban 0.2029, fark {out[en]['robot']-0.2029:+.4f})")
print(f"Genisletilmis havuz tavani 0.5748 -> yakalanan pay "
      f"%{100*out[en]['robot']/0.5748:.1f}")
json.dump({"damga": makbuz_hash.damga(), "sonuc": out, "en_iyi_esik": en,
           "n_egitim_parca": len(tr), "taban_kanonik_robot": 0.2029,
           "tavan_genisletilmis": 0.5748,
           "not": "TAM OLCEKLI gate, genisletilmis havuz (seg + B-rep). D7 "
                  "marka-disi, MIKRO. B-rep TEZ TURETMESI DEGIL, ek aday kaynagi."},
          open(os.environ.get("BREP_CIKTI", "results/brep_gate_tam_d7.json"), "w"), indent=1)
print("makbuz -> results/brep_gate_tam_d7.json")
