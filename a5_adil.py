# -*- coding: utf-8 -*-
"""P4 YIGIN TESTI: uzlasma yukseltici (P1) + RENK ozelligi, ESLESEN gate ile.

P1 tek basina +0.0056 (baraj 0.01 -> tek basina duser) ama MEKANIZMA dogrulandi.
Renk en iyi varyanti +0.0078 (dar kural, %36 kapsama). Iki mekanizma BAGIMSIZ:
biri oy cozunurlugu, digeri malzeme sinyali. Emsal: stacked-levers +0.053.
YIGIN BARAJI: toplam >= +0.02.

Kollar (hepsi KENDI dagilimiyla egitilmis gate ile, test aileleri disarida):
  A = 4 uye, 13 ozellik            (urun)
  B = 5 uye, 13 ozellik            (P1 tek basina)
  C = 5 uye, 13 + RENK(3+bayrak)   (P1 + P2 YIGIN)
  D = 4 uye, 13 + RENK             (P2 tek basina, referans)
"""

ILK SINYAL: -0.0184 ama tek-oy gercek payi %54 -> %47 (mekanizma CALISTI). Dusus, gate'in
4-uyeli oy dagilimiyla egitilmis olmasindan -- bayatlik yasasi. Adil test: her kol KENDI
dagilimiyla egitilmis gate'le skorlanir, iki gate de ayni 400 egitim parcasindan, test
aileleri her ikisinden de cikarilmis.

KARAR KURALI (onceden): B - A >= +0.01 -> A5 final yapilandirmaya girer; degilse duser.
"""
import os, sys, json, copy, pickle
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = {"dusuk": 0.895, "cok": 0.105}
N_TRAIN = 400


def main():
    import torch, thesis_remesh, cp_openings, robot_cp, wire_gate, diffusionnet as D
    from cad_eval import align_frames
    from infer_step_cp import step_to_mesh, load_any
    from big_arbiter import eligible
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from json_dataset import family_key
    from sklearn.ensemble import RandomForestClassifier

    cfg = json.load(open("cp_config.json")); pp = cfg["prediction_postproc"]
    MINV = int(pp["min_vertices"]); VC = float(pp["vertex_confidence_mask"]); CL = float(pp["cluster_mm"])
    TL = float(cfg["robot_wire_gate_threshold"]); TH = float(cfg["robot_wire_gate_threshold_highcp"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LOCK = set(json.load(open("results/split_lock.json"))["locked_parts"])
    models = [load_any(c, dev=dev)[:2] for c in cfg["robot_vote2_checkpoints"]]

    parts = []
    for m, p, jf, s in eligible():
        if p in LOCK: continue
        try: n = len(json.load(open(jf, encoding="utf-8-sig"))["ConnectionPoints"])
        except Exception: continue
        if n > 0: parts.append((m, p, jf, s, n))

    # ayni 48-parcalik test kumesi (seed 13 -- a1_a2 ile birebir)
    rng = np.random.RandomState(13)
    lo = [x for x in parts if x[4] < 8]; hi = [x for x in parts if x[4] >= 8]
    sel = ([lo[i] for i in rng.choice(len(lo), 26, replace=False)] +
           [hi[i] for i in rng.choice(len(hi), 22, replace=False)])
    test_fams = {family_key(p[1]) for p in sel}
    test_ids = {p[1] for p in sel}

    # 400 egitim parcasi: test AILELERI tamamen disarida
    pool = [p for p in parts if family_key(p[1]) not in test_fams]
    rng2 = np.random.RandomState(77)
    tr_parts = [pool[i] for i in rng2.choice(len(pool), min(N_TRAIN, len(pool)), replace=False)]
    print(f"test {len(sel)} parca ({len(test_fams)} aile disarida) | egitim {len(tr_parts)} parca",
          flush=True)

    def load_part(mfg, pid, jf, stp, n):
        Vr, Fr = step_to_mesh(stp)
        V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
        V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
        pbs = []
        for model, meta in models:
            _, pb = D.predict(model, meta, V, F, device=dev,
                              op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig',64))}",
                              return_probs=True)
            pbs.append(np.asarray(pb, float))
        j = json.load(open(jf, encoding="utf-8-sig"))
        Vj = np.array([[q[c] for c in "XYZ"] for q in j["Graphic3d"]["Points"]], float)
        G = np.array([[c["Point"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
        Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
        Gd /= np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9
        R, t, _ = align_frames(Vr, Vj)
        return dict(V=V, F=F, pbs=pbs, stp=stp, n=n, G=(G - t) @ R, Gd=Gd @ R,
                    tol=max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0)))))

    def derive(r, plist, promote):
        return [cp_openings.connection_points(
            r["V"], r["F"], pb.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
            probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
            conn_promote=promote) for pb in plist]

    def runtime_cands(r, add_avg):
        plist = r["pbs"] + [sum(r["pbs"]) / len(r["pbs"])] if add_avg else r["pbs"]
        base = robot_cp._vote2(derive(r, plist, 0.0), min_votes=1)
        is_hi = robot_cp._highcp_router(r["stp"], r["V"], len(base))
        cps = robot_cp._vote2(derive(r, plist, 0.25), min_votes=1) if is_hi else base
        return cps, is_hi

    def label(r, cps):
        P = np.array([c["point"] for c in cps], float)
        lab = np.zeros(len(P), int)
        if len(P) and len(r["G"]):
            diff = P[:, None, :] - r["G"][None, :, :]
            al = (diff * r["Gd"][None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * r["Gd"][None, :, :], axis=-1)
            pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
            used = set(); hit = np.zeros(len(r["G"]), bool)
            for dd, a, b in sorted((pe[a, b], a, b) for a in range(len(P)) for b in range(len(r["G"]))):
                if dd > r["tol"] or a in used or hit[b]: continue
                used.add(a); hit[b] = True; lab[a] = 1
        return lab

    # ---- egitim verisi: iki kol paralel toplanir (ayni parcalar, ayni cikarim) ----
    XA, yA, XB, yB = [], [], [], []
    for k, prt in enumerate(tr_parts, 1):
        try:
            r = load_part(*prt)
            probs = sum(r["pbs"]) / len(r["pbs"])
            for add_avg, Xs, ys in ((False, XA, yA), (True, XB, yB)):
                cps, _ = runtime_cands(r, add_avg)
                if not cps: continue
                Xs.append(wire_gate.feats_for(r["V"], r["F"], probs, cps, CE, CT))
                ys.append(label(r, cps))
        except Exception:
            continue
        if k % 50 == 0:
            print(f"  egitim {k}/{len(tr_parts)}", flush=True)
    XA = np.vstack(XA); yA = np.concatenate(yA)
    XB = np.vstack(XB); yB = np.concatenate(yB)
    print(f"egitim adaylari: A(4uye) {len(yA)} | B(5uye) {len(yB)}", flush=True)
    gA = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                random_state=0).fit(XA, yA)
    gB = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                random_state=0).fit(XB, yB)
    pickle.dump({"clf": gB, "feat_names": wire_gate.FEAT_NAMES},
                open("results/_a5_gate5.pkl", "wb"))

    # ---- test ----
    tests = []
    for prt in sel:
        try:
            tests.append(load_part(*prt))
        except Exception:
            continue
    print(f"test onbellek {len(tests)} parca", flush=True)

    def score(clf, add_avg, use_col=False):
        agg = {"dusuk": [0, 0, 0], "cok": [0, 0, 0]}
        for r in tests:
            cps, is_hi = runtime_cands(r, add_avg)
            probs = sum(r["pbs"]) / len(r["pbs"])
            if cps:
                X = wire_gate.feats_for(r["V"], r["F"], probs, cps, CE, CT)
                sc = clf.predict_proba(X)[:, 1]
                thr = TH if is_hi else TL
                keep = [c for c, s_ in zip(cps, sc) if s_ >= thr]
            else:
                keep = []
            Q = np.array([c["point"] for c in keep], float) if keep else np.zeros((0, 3))
            G, Gd = r["G"], r["Gd"]; hit = np.zeros(len(G), bool); used = set()
            if len(Q) and len(G):
                diff = Q[:, None, :] - G[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
                pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
                pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
                for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(Q)) for b in range(len(G))):
                    if d_ > r["tol"] or a_ in used or hit[b_]: continue
                    hit[b_] = True; used.add(a_)
            tp = int(hit.sum()); k = "cok" if r["n"] >= 8 else "dusuk"
            agg[k][0] += tp; agg[k][1] += len(Q) - tp; agg[k][2] += len(G) - tp
        out = {}
        for k, (T, Fp, Fn) in agg.items():
            p = T / max(T + Fp, 1); rc = T / max(T + Fn, 1)
            out[k] = 2 * p * rc / max(p + rc, 1e-9)
        out["w"] = sum(W[k] * out[k] for k in W)
        return out

    a = score(gA, False)
    b = score(gB, True)
    print(f"\n{'kol':<28}{'dusuk':>9}{'cok':>9}{'agirlikli':>11}")
    print(f"{'A: 4 uye + eslesen gate':<28}{a['dusuk']:>9.4f}{a['cok']:>9.4f}{a['w']:>11.4f}")
    print(f"{'B: 5 uye + eslesen gate':<28}{b['dusuk']:>9.4f}{b['cok']:>9.4f}{b['w']:>11.4f}")
    d = b["w"] - a["w"]
    print(f"\nA5 ADIL FARK: {d:+.4f}")
    print(f"KARAR (>= +0.01): {'A5 FINALE GIRER' if d >= 0.01 else 'A5 DUSER'}")
    json.dump({"A": a, "B": b, "delta": d}, open("results/a5_adil.json", "w"), indent=1)
    print("makbuz -> results/a5_adil.json")


if __name__ == "__main__":
    main()
