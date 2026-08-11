# -*- coding: utf-8 -*-
"""B2: YON_YOK kovasi (660 GT, %21.4) ISARET hatasi mi, EKSEN hatasi mi?

Kova aritmetigi: yalniz bu kova kurtarilirsa F1 0.2773 -> 0.474. Ama nasil
kurtarilacagi, hatanin TURUNE bagli:
  * ISARET (180 derece ters): tek bir ISARET siniflandiricisi yeter -- kolay
  * EKSEN (gercekten baska yon): tam yon secicisi gerekir -- zor, ve o kol
    duzeltilmis etiketle bile -0.0059 verdi
Ayrica adayin YEREL MESH NORMALI o yonu tasiyor mu diye bakilir; tavan olcumu
normalin 'yon yok'u %23.0 -> %1.9 indirdigini soyluyordu (B1 kolunun dayanagi).
"""
import collections, json, os, pickle, sys
import numpy as np
import makbuz_hash
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"]="1"; os.environ["WG_TOPO"]="1"; os.environ["WG_ZENGIN"]="1"
sys.path.insert(0, ".")
import brep_havuz, kanonik_d7 as K
OZ = "results/_tam_oz"; OB = "results/_p1_olasilik_d7"
YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
say = collections.Counter()
pids = [f[3:-4] for f in sorted(os.listdir(OZ))
        if f.startswith("d7_") and f.endswith(".npz")]
kay = K.yukle(pids)
for pid in pids:
    r = kay.get(pid)
    if r is None or not len(r.get("G", [])):
        continue
    z = np.load(f"{OZ}/d7_{pid}.npz")
    m = np.isin(np.asarray(z["kaynak"], int), (0, 1))
    P, D = np.asarray(z["P"], float)[m], np.asarray(z["D"], float)[m]
    if len(P) < 2:
        continue
    G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
    f = f"{OB}/{pid}.npz"
    V = NV = None
    if os.path.exists(f):
        zz = np.load(f)
        V = np.asarray(zz["V"], float)
        NV = brep_havuz.tepe_normalleri(V, np.asarray(zz["F"], np.int64))
    Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    for j in range(len(G)):
        n = np.linalg.norm(Gd[j])
        if n < 1e-9:
            continue
        u = Gd[j] / n
        w = P - G[j]
        e = w @ u
        yan = np.linalg.norm(w - e[:, None] * u[None], axis=1)
        kon = (yan <= YANAL) & (np.abs(e) <= EKSENEL)
        if not kon.any():
            continue                       # ADAY_YOK kovasi, burada degil
        aci = np.degrees(np.arccos(np.clip(Dn[kon] @ u, -1.0, 1.0)))
        if (aci <= ACI).any():
            say["YON TAMAM"] += 1
            continue
        say["YON_YOK toplam"] += 1
        # ISARET mi? yonu ters cevirince duzeliyor mu
        if (180.0 - aci <= ACI).any():
            say["  ISARET (180 ters)"] += 1
        elif (aci <= 30.0).any():
            say["  YAKIN (10-30 derece)"] += 1
        else:
            say["  EKSEN (>30 derece)"] += 1
        # yerel mesh normali o yonu tasiyor mu (B1'in dayanagi)
        if V is not None:
            for i in np.where(kon)[0]:
                d = np.linalg.norm(V - P[i], axis=1)
                yk = d <= 2.0
                if not yk.any():
                    continue
                nn = NV[yk]
                a2 = np.degrees(np.arccos(np.clip(
                    np.concatenate([nn, -nn]) @ u, -1.0, 1.0)))
                if (a2 <= ACI).any():
                    say["  MESH NORMALI KURTARIR"] += 1
                    break
t = say["YON_YOK toplam"]
print(f"YON TAMAM {say['YON TAMAM']} | YON_YOK {t}\n")
for k in ("  ISARET (180 ters)", "  YAKIN (10-30 derece)", "  EKSEN (>30 derece)",
          "  MESH NORMALI KURTARIR"):
    print(f"{k:<28} {say[k]:>5}  %{100*say[k]/max(t,1):.1f}")
json.dump({"damga": makbuz_hash.damga(), "sayim": dict(say),
           "not": "YON_YOK kovasinin ic kirilimi. D7 marka-disi, B-rep havuzu."},
          open("results/b2_yon_isaret.json", "w"), indent=1)
print("\nmakbuz -> results/b2_yon_isaret.json")
