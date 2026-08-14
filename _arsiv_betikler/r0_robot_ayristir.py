# -*- coding: utf-8 -*-
"""R0: robot-hazir kaybinin AYRISTIRMASI -- yanal mi, aci mi, ikisi mi?

robot-hazir (0.4500) ile tespit (0.7439) arasindaki fark, AYNI tahminlerin daha DAR bir olcute
sokulmasindan geliyor:

    tespit      : yanal <= max(3mm, %6 x kosegen), aci SERBEST (180)
    robot-hazir : yanal <= 2.0mm  VE  aci <= 10 deg

Yani kayip YALNIZ iki kaynaktan olabilir. Hangisi oldugunu bilmeden plan yapmak, bu gece dort kez
yaptigim hatanin aynisi olur.

NOT (olcuyu okurken cikti): `esle` aciyi `abs(Pd . Gd)` ile hesapliyor -> 180 derece TERS bir
tahmin 0 derece sayiliyor. Yani bu metrik takma YONUNU denetlemiyor, yalnizca EKSEN DOGRUSUNU.
Bu bir kusur olabilir ama METRIGIN TANIMI; burada degistirilmez, raporlanir.

CIKTI: tespitte eslesen her GT icin yanal mesafe ve aci dagilimi; kaybin yuzde kaci hangi
sarttan; rejim ve DEV/VAL kirilimi. Plan bunun UZERINE kurulacak.
"""
import collections
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def eslesme_ayrinti(P, Pd, G, Gd, diag):
    """`esle`nin TESPIT modunun ayni esleme sirasi, ama her eslesmenin (yanal, aci) degeriyle."""
    if not len(P) or not len(G):
        return []
    diff = P[:, None, :] - G[None, :, :]
    al = (diff * Gd[None, :, :]).sum(-1)
    pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
    an = np.degrees(np.arccos(np.clip(np.abs(Pd @ Gd.T), 0, 1)))
    pe_t = np.where(np.abs(al) > 40, np.inf, pe)          # tespit modu: aci serbest
    tt = max(3.0, 0.06 * diag)
    out, used, hit = [], set(), set()
    for d_, a_, b_ in sorted((pe_t[a, b], a, b)
                             for a in range(len(P)) for b in range(len(G))):
        if d_ > tt or a_ in used or b_ in hit:
            continue
        used.add(a_); hit.add(b_)
        out.append({"yanal": float(pe[a_, b_]), "aci": float(an[a_, b_]),
                    "eksenel": float(abs(al[a_, b_])), "tol": float(tt)})
    return out


def main():
    import wire_gate
    from big_arbiter import eligible
    from sklearn.ensemble import RandomForestClassifier

    mfg_of = {p: m for m, p, jf, s in eligible()}
    with open("results/_u4_der.pkl", "rb") as f:
        DER = pickle.load(f)
    with open("results/_dev_val_kume.json", encoding="utf-8") as f:
        kume_of = json.load(f)
    d = np.load("results/gate_regrow_data_topo.npz", allow_pickle=True)
    with open("results/_strict_geometry_keys.json", encoding="utf-8") as f:
        gk = json.load(f)
    tr_pid = np.array([str(x) for x in d["pids"]])
    tr_grp = np.array([gk.get(p, "yok:" + p) for p in tr_pid])
    Xtr = np.asarray(d["X"], float); ytr = np.asarray(d["y"])
    keep = ~np.isin(tr_grp, list({gk.get(r["pid"], "yok:" + r["pid"]) for r in DER}))

    dag = wire_gate._load(wire_gate.MODEL_PATH)
    rf = lambda M: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                          random_state=0).fit(M[keep], ytr[keep])
    Ztr = np.zeros((len(Xtr), Xtr.shape[1] * 2))
    for u in np.unique(tr_pid):
        i = np.where(tr_pid == u)[0]
        Ztr[i] = wire_gate.parca_ici(Xtr[i], dag.get("donusum_z", "zskor"))
    m = {"clf": rf(Xtr), "clf_z": rf(Ztr), "n_feat": Xtr.shape[1],
         "donusum": dag.get("donusum"), "donusum_z": dag.get("donusum_z", "zskor"),
         "topo_r": dag.get("topo_r")}
    mx = [float(m["clf"].predict_proba(Xtr[np.where((tr_pid == u) & keep)[0]])[:, 1].max())
          for u in np.unique(tr_pid[keep]) if ((tr_pid == u) & keep).any()]
    m["esik_cokus"] = float(np.quantile(mx, dag.get("yonlendirme_q", 0.10)))
    print(f"gate kuruldu (sizintisiz) | cokus esigi {m['esik_cokus']:.4f}", flush=True)

    KAYIT = []
    for r in DER:
        P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
        if r["X"] is not None:
            s = wire_gate.karar_skoru(m, r["X"])
            k = wire_gate.karar_maskesi(s)
            if k.any():
                P = r["P"][k]; Pd = r["Pd"][k]
        for e in eslesme_ayrinti(P, Pd, r["G"], r["Gd"], r["diag"]):
            e.update(pid=r["pid"], rejim="cok" if r["n"] >= 8 else "dusuk",
                     kume=kume_of.get(r["pid"]), mfg=mfg_of.get(r["pid"], "?"),
                     diag=float(r["diag"]))
            KAYIT.append(e)

    ya = np.array([e["yanal"] for e in KAYIT])
    ac = np.array([e["aci"] for e in KAYIT])
    to = np.array([e["tol"] for e in KAYIT])
    print(f"\nTESPITTE ESLESEN {len(KAYIT)} GT noktasi")
    print(f"  tespit toleransi   : medyan {np.median(to):.2f}mm  (2.0mm'nin "
          f"{np.median(to)/2.0:.1f} kati)")
    print(f"  YANAL mesafe       : medyan {np.median(ya):.2f}mm | ort {ya.mean():.2f} | "
          f"%75 {np.percentile(ya,75):.2f} | %90 {np.percentile(ya,90):.2f} | maks {ya.max():.2f}")
    print(f"  ACI                : medyan {np.median(ac):.2f} deg | ort {ac.mean():.2f} | "
          f"%75 {np.percentile(ac,75):.2f} | %90 {np.percentile(ac,90):.2f} | maks {ac.max():.2f}")

    y_ok = ya <= 2.0; a_ok = ac <= 10.0
    n = len(KAYIT)
    print(f"\nKAYBIN AYRISTIRMASI (tespitte eslesen {n} nokta):")
    print(f"  ikisi de OK (robot-hazir sayilir): {int((y_ok & a_ok).sum()):>5}  "
          f"({(y_ok & a_ok).mean():.1%})")
    print(f"  YALNIZ yanal >2mm               : {int((~y_ok & a_ok).sum()):>5}  "
          f"({(~y_ok & a_ok).mean():.1%})")
    print(f"  YALNIZ aci >10deg               : {int((y_ok & ~a_ok).sum()):>5}  "
          f"({(y_ok & ~a_ok).mean():.1%})")
    print(f"  IKISI birden                    : {int((~y_ok & ~a_ok).sum()):>5}  "
          f"({(~y_ok & ~a_ok).mean():.1%})")
    kayip = int((~(y_ok & a_ok)).sum())
    if kayip:
        print(f"\n  KAYBEDILEN {kayip} noktanin dagilimi:")
        print(f"    yanal sucu     : {int((~y_ok).sum())/kayip:.1%}")
        print(f"    aci sucu       : {int((~a_ok).sum())/kayip:.1%}")

    print("\n  'yanal 2mm'ye ne kadar yakin?' (kaybedilenler icinde):")
    kb = ya[~y_ok]
    if len(kb):
        for u in (2.5, 3.0, 4.0, 6.0, 10.0):
            print(f"    <= {u:>4.1f}mm : {(kb <= u).mean():>6.1%}  "
                  f"({int((kb <= u).sum())}/{len(kb)})")

    print("\n  'aci 10 dereceye ne kadar yakin?' (kaybedilenler icinde):")
    ab = ac[~a_ok]
    if len(ab):
        for u in (15, 20, 30, 45, 90):
            print(f"    <= {u:>3} deg : {(ab <= u).mean():>6.1%}  ({int((ab <= u).sum())}/{len(ab)})")

    print(f"\n{'kirilim':<16}{'n':>6}{'yanal med':>11}{'aci med':>9}{'yanal OK':>10}{'aci OK':>9}"
          f"{'ikisi OK':>10}")
    for alan, adlar in (("rejim", ["dusuk", "cok"]), ("kume", ["dev", "val"]),
                        ("mfg", ["PXC", "WEI"])):
        for ad in adlar:
            i = np.array([e[alan] == ad for e in KAYIT])
            if not i.any():
                continue
            print(f"{alan + '=' + ad:<16}{int(i.sum()):>6}{np.median(ya[i]):>11.2f}"
                  f"{np.median(ac[i]):>9.2f}{y_ok[i].mean():>10.1%}{a_ok[i].mean():>9.1%}"
                  f"{(y_ok & a_ok)[i].mean():>10.1%}")

    # TAVAN: butun yanal hatalar sifirlansaydi / butun acilar sifirlansaydi ne olurdu?
    print(f"\nTAVAN KESTIRIMI (tespitte eslesen noktalar uzerinde):")
    print(f"  su an robot-hazir sayilan            : {(y_ok & a_ok).mean():.1%}")
    print(f"  YANAL tamamen cozulseydi (aci ayni)  : {a_ok.mean():.1%}")
    print(f"  ACI tamamen cozulseydi (yanal ayni)  : {y_ok.mean():.1%}")
    with open("results/r0_robot_ayristir.json", "w", encoding="utf-8") as f:
        json.dump({"n": n, "yanal_medyan": float(np.median(ya)), "aci_medyan": float(np.median(ac)),
                   "ikisi_ok": float((y_ok & a_ok).mean()), "yanal_ok": float(y_ok.mean()),
                   "aci_ok": float(a_ok.mean()),
                   "yalniz_yanal": float((~y_ok & a_ok).mean()),
                   "yalniz_aci": float((y_ok & ~a_ok).mean()),
                   "ikisi_kotu": float((~y_ok & ~a_ok).mean()),
                   "tespit_tol_medyan": float(np.median(to))}, f, indent=1)
    with open("results/r0_kayit.pkl", "wb") as f:
        pickle.dump(KAYIT, f)
    print("makbuz -> results/r0_robot_ayristir.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
