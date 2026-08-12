# -*- coding: utf-8 -*-
"""YON, MESH GEOMETRISINDEN OKUNABILIR MI? (NIT'in darbogazina dogrudan saldiri)

MANTIK ZINCIRI:
 * III.2 isaret problemini YEREL DIS NORMALLE %100 cozdu -- ama yalniz B-rep
   agzi OLAN yerlerde.
 * NIT'te B-rep agzi CP'lerin uzerinde YOK (konum 0.011). Dort bagimsiz olcum
   ayni yeri gosteriyor.
 * AMA MESH VAR: hafizadaki olcume gore GT'lerin %98.1'i 2mm'de bir tepeye
   komsu; tavan-24 havuzunda NIT konum recall'u 0.8429.
 * VE FIZIK: bir deligin agzinda DIS YUZEYIN NORMALI zaten delik EKSENIDIR.

Yani yon, B-rep'e hic ihtiyac duymadan MESH'ten okunabilir olmali. Eger
tutuyorsa, yayilim kolunu 0.527'den 0.077'ye dusuren YON SECIMI darbogazi
GEOMETRIYLE kapanir.

OLCULEN (her GT icin, konum kutusunda bir mesh tepesi varken):
  n_tepe    : tepenin KENDI normali 10 derece icinde mi (ISARETLI)
  n_komsu   : 2mm komsulugun ORTALAMA normali
  n_kipsel  : komsulukta EN COK oy alan normal (delik duvari gurultusune karsi)
  n_duzlem  : komsulugun PCA duzlemi normali (en kucuk ozvektor)

D7'ye BAKILMAZ.
"""
import collections
import json
import os
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")
import kanonik_d7 as K   # noqa: E402
import d6_kayit          # noqa: E402

MESH_DIZ = os.environ.get("MY_MESH", "results/_p1_olasilik")
MARKALAR = set(os.environ.get("MY_MARKA", "NIT,MOR,SUPU,UPUN").split(","))
YANAL, EKSENEL = 2.0, 40.0
KOMSU_R = float(os.environ.get("MY_R", "2.0"))


def _birim(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    import trimesh
    kay = K.yukle()
    kay.update({str(p): r for p, r in d6_kayit.yukle().items()
                if str(p) not in kay})
    ist = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for pid, r in kay.items():
        if r.get("mfg") not in MARKALAR:
            continue
        G = np.asarray(r.get("G", []), float)
        if not len(G):
            continue
        mf = f"{MESH_DIZ}/{pid}.npz"
        if not os.path.exists(mf):
            continue
        z = np.load(mf)
        V = np.ascontiguousarray(z["V"], np.float64)
        F = np.ascontiguousarray(z["F"], np.int64)
        mesh = trimesh.Trimesh(V, F, process=False)
        VN = _birim(np.asarray(mesh.vertex_normals, float))
        Gn = _birim(np.asarray(r["Gd"], float))

        # konum kutusu: tepe, GT'nin kutusunda mi
        v = V[:, None, :] - G[None, :, :]
        al = (v * Gn[None, :, :]).sum(-1)
        yan = np.linalg.norm(v - al[..., None] * Gn[None, :, :], axis=-1)
        konum = (yan <= YANAL) & (np.abs(al) <= EKSENEL)     # (tepe, gt)

        a = ist[r["mfg"]]
        a["gt"].append(len(G))
        a["konum"].append(int(konum.any(0).sum()))

        ok_t = ok_k = ok_m = ok_d = 0
        for j in range(len(G)):
            t_ler = np.where(konum[:, j])[0]
            if not len(t_ler):
                continue
            g = Gn[j]
            # 1) tepenin kendi normali
            aci = np.degrees(np.arccos(np.clip(VN[t_ler] @ g, -1, 1)))
            if (aci <= K.ACI).any():
                ok_t += 1
            # komsuluk: kutudaki tepelerin cevresi
            merkez = V[t_ler].mean(0)
            komsu = np.where(np.linalg.norm(V - merkez, axis=1) <= KOMSU_R)[0]
            if len(komsu) < 3:
                continue
            NK = VN[komsu]
            # 2) ortalama normal
            ort = _birim(NK.mean(0))
            if np.degrees(np.arccos(np.clip(float(ort @ g), -1, 1))) <= K.ACI:
                ok_k += 1
            # 3) KIPSEL normal: en cok komsusu olan yon (delik duvari
            #    normalleri her yone bakar; dis yuz normalleri hizalidir)
            cos = np.clip(NK @ NK.T, -1, 1)
            oy = (np.degrees(np.arccos(cos)) <= K.ACI).sum(1)
            kip = NK[int(np.argmax(oy))]
            if np.degrees(np.arccos(np.clip(float(kip @ g), -1, 1))) <= K.ACI:
                ok_m += 1
            # 4) PCA duzlem normali (isaret: kipselle ayni tarafa)
            Q = V[komsu] - V[komsu].mean(0)
            try:
                _, _, Vt = np.linalg.svd(Q, full_matrices=False)
                nd = Vt[-1]
                if float(nd @ kip) < 0:
                    nd = -nd
                if np.degrees(np.arccos(np.clip(float(nd @ g), -1, 1))) <= K.ACI:
                    ok_d += 1
            except Exception:
                pass
        a["n_tepe"].append(ok_t)
        a["n_komsu"].append(ok_k)
        a["n_kipsel"].append(ok_m)
        a["n_duzlem"].append(ok_d)
        n += 1
        if n % 40 == 0:
            print(f"  {n} parca", flush=True)

    print(f"\n{'marka':<7}{'GT':>7}{'KONUM':>8}{'tepe N':>9}{'komsu N':>9}"
          f"{'KIPSEL N':>10}{'duzlem N':>10}")
    out = {}
    for m_ in sorted(ist, key=lambda x: -sum(ist[x]["gt"])):
        a = ist[m_]
        g = max(sum(a["gt"]), 1)
        r = {k: sum(a[k]) / g for k in ("konum", "n_tepe", "n_komsu",
                                        "n_kipsel", "n_duzlem")}
        r["gt"] = g
        out[m_] = r
        print(f"{m_:<7}{g:>7}{r['konum']:>8.3f}{r['n_tepe']:>9.3f}"
              f"{r['n_komsu']:>9.3f}{r['n_kipsel']:>10.3f}"
              f"{r['n_duzlem']:>10.3f}")
    json.dump({"komsu_r": KOMSU_R, "marka": out,
               "not": "Yon, MESH yuzey normalinden (ISARETLI aci). Dort "
                      "turetme: tepe normali / komsu ortalamasi / KIPSEL "
                      "normal / PCA duzlem normali. D7'ye BAKILMADI."},
              open("results/mesh_yon.json", "w"), indent=1)
    print("\nmakbuz -> results/mesh_yon.json")
    print("OKUMA: NIT'te yuksek cikarsa yon darbogazi GEOMETRIYLE kapanir")
    print("       ve B-rep'e bagimlilik ortadan kalkar.")


if __name__ == "__main__":
    main()
