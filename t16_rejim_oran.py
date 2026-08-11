# -*- coding: utf-8 -*-
"""T16: GORELI ORAN rejime gore ayarlanmali mi? (dusuk-CP vs cok-CP)

DAGITILAN kural: aday, parcasindaki en yuksek skorun 0.5 katini gecmeli VE 0.25 tabanini asmali.
Tek oran, tum parcalar. Ama iki rejim cok farkli:
  dusuk-CP  (<8 CP, korpusun %89.5'i) -- az sayida gercek aciklik, KESINLIK onemli
  cok-CP    (>=8 CP)                  -- cok sayida gercek aciklik; 0.5 orani cok SIKI olabilir,
                                         cunku en guclu adayin yarisini gecemeyen GERCEK CP'ler
                                         elenir. (Ayni sebeple esik zaten 0.40/0.35 idi.)
Rejimi `robot_cp._highcp_router` GEOMETRIDEN belirliyor (uretici metadata'si YOK, aile-disi AUC 0.994).

TASARIM: adaylar ve skorlar BIR KEZ; sonra oran cift-ligi bedava taranir. Hem TANIDIK hem
GORULMEMIS URETICI raporlanir -- bu gecenin ekseni transfer.

KILL (onceden yazili): bir (dusuk_oran, cok_oran) cifti, tanidik tespiti DUSURMEDEN
(>= -0.005) gorulmemis uretici EN KOTU durumunu artirmazsa ALINMAZ.
"""
import os, sys, json, pickle, collections
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"; os.environ["WG_TOPO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import cp_openings, robot_cp, wire_gate
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from sklearn.ensemble import RandomForestClassifier
    from big_arbiter import eligible
    from sina_kume import esle, f1w, pr

    cfg = json.load(open("cp_config.json", encoding="utf-8"))
    pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"]); CL = float(pp["cluster_mm"])
    TABAN = float(cfg["gate_goreli_taban"])
    mfg_of = {p: m for m, p, jf, s in eligible()}

    cache = []
    for k in ("dev", "val"):
        cf = f"results/_probs_{k}.pkl"
        if not os.path.exists(cf) and k == "dev":
            cf = "results/_h_probs.pkl"
        for r in pickle.load(open(cf, "rb")):
            r.setdefault("pid", os.path.basename(r["stp"]).split("_")[1])
            cache.append(r)

    DER = []
    for i, r in enumerate(cache, 1):
        if i % 50 == 0:
            print(f"  turetme {i}/{len(cache)}", flush=True)
        V = np.ascontiguousarray(r["V"], np.float64)
        F = np.ascontiguousarray(r["F"], np.int64)
        plist = [np.asarray(pb, np.float64) for pb in r["pbs"]]
        mk = lambda pr_, **kw: cp_openings.connection_points(
            V, F, pr_.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
            probs=pr_, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
            step_path=r["stp"], **kw)
        merge = lambda L: robot_cp._vote2(L, min_votes=1)
        base = merge([mk(pb) for pb in plist])
        is_hi = robot_cp._highcp_router(r["stp"], V, len(base))
        cps = merge([mk(pb, conn_promote=0.25) for pb in plist]) if is_hi else base
        Xc = (wire_gate.feats_for(V, F, sum(plist) / len(plist), cps, CE, CT,
                                  step_path=r["stp"]) if cps else None)
        DER.append(dict(pid=r["pid"], mfg=mfg_of.get(r["pid"], "?"), X=Xc, is_hi=is_hi,
                        P=np.array([c["point"] for c in cps], float) if cps else np.zeros((0, 3)),
                        Pd=np.array([c["direction"] for c in cps], float) if cps else np.zeros((0, 3)),
                        G=r["G"], Gd=r["Gd"], n=r["n"], diag=r["diag"]))
    hi = sum(1 for r in DER if r["is_hi"])
    print(f"turetme bitti | yonlendirici: {hi} cok-CP / {len(DER)-hi} dusuk-CP", flush=True)

    d = np.load("results/gate_regrow_data_topo.npz", allow_pickle=True)
    gk = json.load(open("results/_strict_geometry_keys.json"))
    tr_pid = np.array([str(x) for x in d["pids"]])
    tr_grp = np.array([gk.get(p, "yok:" + p) for p in tr_pid])
    tr_mfg = np.array([str(x) for x in d["mfg"]])
    kod = {k: collections.Counter(mfg_of.get(p, "?") for p in tr_pid[tr_mfg == k]).most_common(1)[0][0]
           for k in np.unique(tr_mfg)}
    tg = {gk.get(r["pid"], "yok:" + r["pid"]) for r in DER}

    def skorla(keep, alt):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                     random_state=0).fit(d["X"][keep][:, :22], d["y"][keep])
        return [(r, (clf.predict_proba(r["X"][:, :22])[:, 1] if r["X"] is not None
                     else np.zeros(0))) for r in alt]

    def puanla(skorlu, o_dusuk, o_cok):
        det = []
        for r, s in skorlu:
            P = np.zeros((0, 3)); Pd = np.zeros((0, 3))
            if len(s):
                o_ = o_cok if r["is_hi"] else o_dusuk
                m = (s >= o_ * max(float(s.max()), 1e-9)) & (s >= TABAN)
                if m.any():
                    P = r["P"][m]; Pd = r["Pd"][m]
            det.append(("cok" if r["n"] >= 8 else "dusuk",) +
                       esle(P, Pd, r["G"], r["Gd"], r["diag"], 0.0, 180.0, True))
        return det

    S = {"tanidik": skorla(~np.isin(tr_grp, list(tg)), DER)}
    for k, mad in kod.items():
        alt = [x for x in DER if x["mfg"] == mad]
        if len(alt) >= 10:
            S[mad] = skorla((tr_mfg != k) & ~np.isin(tr_grp, list(tg)), alt)

    CIFT = [(0.5, 0.5), (0.5, 0.4), (0.5, 0.35), (0.5, 0.3), (0.55, 0.4), (0.45, 0.4), (0.6, 0.4)]
    ad = sorted(S)
    print(f"\n{'dusuk/cok oran':<16}" + "".join(f"{a:>11}" for a in ad) + f"{'EN KOTU URT':>13}")
    out = {}
    for od, oc in CIFT:
        v = {a: f1w(puanla(S[a], od, oc)) for a in ad}
        ek = min(v[a] for a in ad if a != "tanidik")
        print(f"{f'{od} / {oc}':<16}" + "".join(f"{v[a]:>11.4f}" for a in ad) + f"{ek:>13.4f}",
              flush=True)
        out[f"{od}/{oc}"] = {"degerler": v, "en_kotu_uretici": ek}
    t = out["0.5/0.5"]
    print("\nKARAR (kill: tanidik >= -0.005 VE en kotu uretici artmali):")
    for k, v in out.items():
        if k == "0.5/0.5":
            continue
        dt = v["degerler"]["tanidik"] - t["degerler"]["tanidik"]
        de = v["en_kotu_uretici"] - t["en_kotu_uretici"]
        print(f"  {k:<12} tanidik {dt:+.4f} | en kotu {de:+.4f} -> "
              f"{'GECTI' if (dt >= -0.005 and de > 0) else '-'}")
    json.dump(out, open("results/t16_rejim_oran.json", "w"), indent=1)
    print("makbuz -> results/t16_rejim_oran.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
