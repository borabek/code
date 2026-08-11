# -*- coding: utf-8 -*-
"""CWT CEPHESI: D7 GT'sinin %37'si orada ve F1 0.0466. Neden?

CWT 226 parca / 1146 GT / 5.1 CP-parca. Taban 0.0370, P6 0.0466 -- ikisi de
neredeyse sifir. NIT'te (D6) ayni desen vardi ve orada kok neden TEMSILDI:
havuzda cevabin yarisi olmasina ragmen model siralama uretemiyordu.

Bu betik CWT icin AYNI uc soruyu sorar ve cevabi D7'ye BAKMADAN karar vermek
icin degil, NEREYE YATIRIM YAPILACAGINI bilmek icin kullanir:

  1. HAVUZ: yalniz konum / konum+yon recall'u ne?
  2. SIRALAMA: esikten bagimsiz recall@k, rastgeleye gore kac kat?
  3. YOGUNLUK: CWT parcalari NIT gibi mi (cok aday / az pozitif)?

D7 SINAV KUMESIDIR. Burada YALNIZCA TESHIS yapilir; hicbir esik/kural/kol
secimi bu ciktilara bakilarak yapilmaz. Sonuc raporda "teshis" olarak gecer.
"""
import collections
import json
import os
import sys

import numpy as np

import makbuz_hash

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, ".")
import brep_havuz              # noqa: E402
import connector3d             # noqa: E402
import havuz_seyrelt           # noqa: E402
import kanonik_d7 as K         # noqa: E402
import yon_bankasi as YB       # noqa: E402

YANAL, ACI, EKSENEL = 2.0, 10.0, 40.0
OB = "results/_p1_olasilik_d7"
OZ = "results/_tam_oz"
MARKALAR = os.environ.get("CEPHE_MARKA", "CWT,WIE").split(",")


def rec(P, D, G, Gd, yon=True):
    if not len(P) or not len(G):
        return 0
    Gn = YB.birim(Gd)
    df = np.asarray(P, float)[:, None, :] - np.asarray(G, float)[None, :, :]
    al = (df * Gn[None, :, :]).sum(-1)
    yan = np.linalg.norm(df - al[..., None] * Gn[None, :, :], axis=-1)
    ok = (yan <= YANAL) & (np.abs(al) <= EKSENEL)
    if yon:
        an = np.degrees(np.arccos(np.clip(YB.birim(D) @ Gn.T, -1.0, 1.0)))
        ok = ok & (an <= ACI)
    return int(ok.any(0).sum())


def main():
    kay = K.yukle()
    agg = collections.defaultdict(collections.Counter)
    for f in sorted(os.listdir(OZ)):
        if not f.startswith("d7_"):
            continue
        pid = f[3:-4]
        r = kay.get(pid)
        if r is None or not len(r.get("G", [])):
            continue
        if r["mfg"] not in MARKALAR:
            continue
        mf = f"{OB}/{pid}.npz"
        if not os.path.exists(mf):
            continue
        z = np.load(f"{OZ}/{f}")
        kk = np.asarray(z["kaynak"], int)
        P = np.asarray(z["P"], float)
        D = np.asarray(z["D"], float)
        zz = np.load(mf)
        V = np.ascontiguousarray(zz["V"], np.float64)
        Fc = np.ascontiguousarray(zz["F"], np.int64)
        pb = np.asarray(zz["pbs"], float).mean(0)
        pp = havuz_seyrelt.ppos(pb, connector3d.CABLE_ENTRY,
                                connector3d.CONTACT)
        G = np.asarray(r["G"], float)
        Gd = np.asarray(r["Gd"], float)
        a = agg[r["mfg"]]
        a["parca"] += 1
        a["gt"] += len(G)
        m01 = np.isin(kk, (0, 1))
        a["n01"] += int(m01.sum())
        a["nmesh"] += int((kk == 2).sum())
        a["konum01"] += rec(P[m01], D[m01], G, Gd, yon=False)
        a["yon01"] += rec(P[m01], D[m01], G, Gd, yon=True)
        a["konum012"] += rec(P, D, G, Gd, yon=False)
        # TUM mesh tepeleri (seyreltmesiz) -- konum bilgisi meshte var mi?
        Pm, Dm = brep_havuz.mesh_adaylari(V, Fc, pp, brep_havuz.MESH_ESIK,
                                          brep_havuz.MESH_DEDUPE_MM)
        a["nmesh_ham"] += len(Pm)
        Pt = np.vstack([P[m01], Pm]) if len(Pm) else P[m01]
        a["konum_tam"] += rec(Pt, None, G, Gd, yon=False)
        # yon bankasi ile
        idx, YD, _ = YB.secenekler(P, D, None, V)
        a["yon_banka"] += rec(P[idx], YD, G, Gd, yon=True)
        a["nsec"] += len(idx)

    print(f"{'marka':<6}{'parca':>6}{'GT':>7}{'CP/p':>6}{'n01/p':>7}"
          f"{'mesh/p':>8}{'mesh_ham':>9}")
    out = {}
    for m, a in agg.items():
        p = max(a["parca"], 1)
        g = max(a["gt"], 1)
        o = {"parca": a["parca"], "gt": a["gt"], "cp_parca": a["gt"] / p,
             "n01_parca": a["n01"] / p, "mesh_parca": a["nmesh"] / p,
             "mesh_ham_parca": a["nmesh_ham"] / p,
             "konum_recall_01": a["konum01"] / g,
             "yon_recall_01": a["yon01"] / g,
             "konum_recall_012": a["konum012"] / g,
             "konum_recall_TAM_MESH": a["konum_tam"] / g,
             "yon_recall_BANKA": a["yon_banka"] / g,
             "secenek_parca": a["nsec"] / p}
        out[m] = o
        print(f"{m:<6}{a['parca']:>6}{a['gt']:>7}{o['cp_parca']:>6.1f}"
              f"{o['n01_parca']:>7.0f}{o['mesh_parca']:>8.0f}"
              f"{o['mesh_ham_parca']:>9.0f}")
    print(f"\n{'marka':<6}{'konum01':>9}{'yon01':>8}{'konum012':>10}"
          f"{'konumTAM':>10}{'yonBANKA':>10}")
    for m, o in out.items():
        print(f"{m:<6}{o['konum_recall_01']:>9.4f}{o['yon_recall_01']:>8.4f}"
              f"{o['konum_recall_012']:>10.4f}"
              f"{o['konum_recall_TAM_MESH']:>10.4f}"
              f"{o['yon_recall_BANKA']:>10.4f}")
    json.dump({"damga": makbuz_hash.damga(), "sonuc": out,
               "not": "CWT/WIE cephe TESHISI. D7 SINAV kumesidir; buradan "
                      "hicbir esik/kural/kol secimi YAPILMAZ, yalnizca nereye "
                      "yatirim yapilacagi belirlenir."},
              open("results/cwt_cephesi.json", "w"), indent=1)
    print("\nmakbuz -> results/cwt_cephesi.json")


if __name__ == "__main__":
    main()
