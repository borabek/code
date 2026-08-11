# -*- coding: utf-8 -*-
"""A1b: EKSENEL TOLERANSIN DOGRU KURULUSU -- GT'yi KENDI AGZINA tasi, SONRA sik tolerans.

A1'IN ILK KURULUSU YANLISTI (kendi olcumumle yakalandi). "Eksenel tolerans 40 -> 15mm"
dedigimde tespit 0.7584 -> 0.5214 dustu ve 25 -> 15 arasinda 160 TP birden gitti. Bu
UCURUM, kesilen seyin HATA degil TANIM FARKININ KUYRUGU oldugunu gosteriyor:

    tez CP'yi ACIKLIGIN AGZINDA (v_o) tanimlar
    uretici CP'yi KONTAKTA (seat) tanimlar
    aradaki mesafe olculdu: medyan 7.5-10.6mm, ama KUYRUGU 20mm+ olan parcalar var

Seat'e gore 15mm dayatmak, dogru yerlestirilmis bir CP'yi "yanlis" ilan eder. Metrigi
duzeltmek yerine URUNU cezalandirmis olurduk.

DOGRUSU: uretici seat'ini `cp_geometry.seat_to_mouth` ile KENDI AGZINA tasi (bu, tezin
`v_o` insasinin ta kendisi -- ayni tanim), sonra eksenel toleransi SIK tut. Boylece
agiz-agiza karsilastirma olur ve kalan eksenel fark GERCEK hatadir.

Bu betik GT agizlarini uretir (mesh gerekir), onbellege alir ve UC olcumu yan yana koyar:
    (a) SEAT'e gore, eksen 40   = bugunku manset
    (b) SEAT'e gore, eksen 15   = A1'in HATALI kurulusu
    (c) AGIZ'a gore, eksen 15/10/6 = A1'in DOGRU kurulusu
Ayrica her birinde F1_kesin (belirsiz eslesmeler kredilendirilmez, A2).
"""
import io
import json
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"; os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tezgah2 as T2
from sina_kume import esle_detay, f1w
from a1a3_metrik_cerrahi import urun_ciktisi

ONB = "results/_gt_agiz.pkl"


def gt_agizlari(DER):
    """Her parca icin uretici GT'sinin KENDI AGZI (seat_to_mouth) + offset."""
    if os.path.exists(ONB):
        d = pickle.load(open(ONB, "rb"))
        print(f"GT agizlari onbellekten: {len(d)} parca", flush=True)
        return d
    import trimesh
    import thesis_remesh
    from big_arbiter import eligible
    from cp_geometry import seat_to_mouth
    from infer_step_cp import step_to_mesh
    stp = {p: s for m, p, jf, s in eligible()}
    out = {}
    t0 = time.time()
    for k, r in enumerate(DER, 1):
        if k % 25 == 0:
            print(f"  agiz {k}/{len(DER)}  {time.time()-t0:.0f}s", flush=True)
            pickle.dump(out, open(ONB, "wb"))
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        if not len(G) or r["pid"] not in stp:
            out[r["pid"]] = (G.copy(), np.zeros(len(G)))
            continue
        try:
            Vr, Fr = step_to_mesh(stp[r["pid"]])
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            mesh = trimesh.Trimesh(vertices=np.ascontiguousarray(V, float),
                                   faces=np.ascontiguousarray(F, np.int64), process=False)
        except Exception:
            out[r["pid"]] = (G.copy(), np.zeros(len(G)))
            continue
        M = np.zeros_like(G); off = np.zeros(len(G))
        for i in range(len(G)):
            try:
                m_, o_ = seat_to_mouth(mesh, G[i], Gd[i])
                M[i] = np.asarray(m_, float); off[i] = float(o_)
            except Exception:
                M[i] = G[i]
        out[r["pid"]] = (M, off)
    pickle.dump(out, open(ONB, "wb"))
    print(f"-> {ONB}", flush=True)
    return out


def main():
    import protokol
    protokol.tez_dogrula()
    DER, gate, ek = T2.yukle()
    AGIZ = gt_agizlari(DER)
    CIKTI = {r["pid"]: urun_ciktisi(r, gate) for r in DER}

    off = np.concatenate([np.abs(AGIZ[r["pid"]][1]) for r in DER if len(AGIZ[r["pid"]][1])])
    print(f"\nseat->agiz mesafesi (n={len(off)}): medyan {np.median(off):.2f}mm | "
          f"%75 {np.percentile(off,75):.2f} | %90 {np.percentile(off,90):.2f} | "
          f"max {off.max():.2f} | sifir olan %{100*(off<0.01).mean():.0f}")

    def olc(hedef, eksen_tol, kesin=False, robot=False):
        rows, bel, top = [], 0, 0
        for r in DER:
            P, Pd = CIKTI[r["pid"]]
            Gd = np.asarray(r["Gd"], float)
            G = np.asarray(r["G"], float) if hedef == "seat" else AGIZ[r["pid"]][0]
            rj = "cok" if r["n"] >= 8 else "dusuk"
            if robot:
                tp, fp, fn, b = esle_detay(P, Pd, G, Gd, r["diag"], 2.0, 10.0, False,
                                           isaretli=True, eksen_tol=eksen_tol)
            else:
                tp, fp, fn, b = esle_detay(P, Pd, G, Gd, r["diag"], 0.0, 180.0, True,
                                           eksen_tol=eksen_tol)
            nb = b["belirsiz"]; bel += nb; top += tp
            if kesin and nb:
                tp -= nb; fp += nb; fn += nb
            rows.append((rj, tp, fp, fn))
        return f1w(rows), bel, top

    print(f"\n{'olcum':<34}{'tespit F1':>11}{'F1_kesin':>11}{'TP':>6}{'belirsiz':>10}")
    S = {}
    for ad, hedef, et in (("(a) SEAT'e gore, eksen 40  [BUGUN]", "seat", 40.0),
                          ("(b) SEAT'e gore, eksen 15  [HATALI]", "seat", 15.0),
                          ("(c) AGIZ'a gore, eksen 15", "agiz", 15.0),
                          ("(d) AGIZ'a gore, eksen 10", "agiz", 10.0),
                          ("(e) AGIZ'a gore, eksen  6", "agiz", 6.0)):
        f1, bel, top = olc(hedef, et)
        fk, _, _ = olc(hedef, et, kesin=True)
        S[ad] = {"f1": f1, "f1_kesin": fk, "tp": top, "belirsiz": bel}
        print(f"{ad:<34}{f1:>11.4f}{fk:>11.4f}{top:>6}{bel:>10}")

    print(f"\n{'ROBOT (FIZIKSEL)':<34}{'robot F1':>11}{'F1_kesin':>11}{'TP':>6}{'belirsiz':>10}")
    for ad, hedef, et in (("(a) SEAT'e gore, eksen 40  [BUGUN]", "seat", 40.0),
                          ("(c) AGIZ'a gore, eksen 15", "agiz", 15.0),
                          ("(d) AGIZ'a gore, eksen 10", "agiz", 10.0)):
        f1, bel, top = olc(hedef, et, robot=True)
        fk, _, _ = olc(hedef, et, kesin=True, robot=True)
        S["ROBOT " + ad] = {"f1": f1, "f1_kesin": fk, "tp": top, "belirsiz": bel}
        print(f"{ad:<34}{f1:>11.4f}{fk:>11.4f}{top:>6}{bel:>10}")

    with io.open("results/a1b_agiz_hizalamali.json", "w", encoding="utf-8") as f:
        json.dump({"seat_agiz_medyan": float(np.median(off)),
                   "seat_agiz_p90": float(np.percentile(off, 90)),
                   "olcumler": S}, f, indent=1, ensure_ascii=False)
    print("\nmakbuz -> results/a1b_agiz_hizalamali.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
