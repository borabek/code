# -*- coding: utf-8 -*-
"""P4 YIGIN TESTI: uzlasma yukseltici (P1) + RENK ozelligi (P2), ESLESEN gate ile.

NEDEN YIGIN: P1 tek basina +0.0056, P2'nin en iyi varyanti +0.0078 -- ikisi de tek basina
+0.01/+0.02 barajini gecemedi. Ama mekanizmalari BAGIMSIZ (biri oy cozunurlugu, digeri
malzeme sinyali) ve bu projede alt-esik kaldirac YIGINI daha once +0.053 olculdu
([[stacked-levers-2026-07-29]]). Bar yigin seviyesine yazilir, esik dusurulmez.

KILL (olcumden ONCE): yigin (C) - urun (A) >= +0.02 -> finale girer, degilse duser.

KOLLAR -- hepsi KENDI dagilimiyla egitilmis gate ile, test aileleri egitimden CIKARILMIS:
  A = 4 uye, 13 ozellik            (urun, referans)
  B = 5 uye, 13 ozellik            (P1 tek basina -- a5_adil'i dogrular)
  C = 5 uye, 13 + RENK             (YIGIN)
  D = 4 uye, 13 + RENK             (P2 tek basina)

RENK: dar kural (en iyi olculen varyant), NaN = renk yok -> -1 + eksiklik bayragi.
"""
import os, sys, json, pickle
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ.setdefault("CP_METAL_WIDE", "0")        # dar kural: olculen en iyi varyant
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = {"dusuk": 0.895, "cok": 0.105}
N_TRAIN = 400


def main():
    import torch, thesis_remesh, cp_openings, robot_cp, wire_gate, diffusionnet as D
    import step_face_color_link as L
    from cad_eval import align_frames
    from infer_step_cp import step_to_mesh, load_any
    from big_arbiter import eligible
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from json_dataset import family_key
    from k7_dip_metal import dip_features
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
    rng = np.random.RandomState(13)
    lo = [x for x in parts if x[4] < 8]; hi = [x for x in parts if x[4] >= 8]
    sel = ([lo[i] for i in rng.choice(len(lo), 26, replace=False)] +
           [hi[i] for i in rng.choice(len(hi), 22, replace=False)])
    test_fams = {family_key(p[1]) for p in sel}
    pool = [p for p in parts if family_key(p[1]) not in test_fams]
    rng2 = np.random.RandomState(77)
    tr_parts = [pool[i] for i in rng2.choice(len(pool), min(N_TRAIN, len(pool)), replace=False)]
    print(f"test {len(sel)} | egitim {len(tr_parts)} (test aileleri disarida)", flush=True)

    def metal_of(stp, Vr):
        """(tum_noktalar_bayrakli, metal_noktalar) -- mesh cercevesinde. Yoksa (None, None)."""
        try:
            faces = L.face_vertices_from_text(stp)
            if not faces or not any(x["is_metal"] for x in faces):
                return None, None
            allp = np.vstack([x["pts"] for x in faces])
            flag = np.concatenate([np.full(len(x["pts"]), 1.0 if x["is_metal"] else 0.0)
                                   for x in faces])
            allv = L.all_vertex_points(stp)
            R, t, res = align_frames(Vr, allv if len(allv) else allp)
            if res > 1.0:                       # SIKI esik (gevsetmek olculdu: zarar veriyor)
                return None, None
            allm = (allp - t) @ R
            return np.column_stack([allm, flag]), allm[flag > 0.5]
        except Exception:
            return None, None

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
        mp_all, mpts = metal_of(stp, Vr)
        return dict(V=V, F=F, pbs=pbs, stp=stp, n=n, G=(G - t) @ R, Gd=Gd @ R,
                    mp_all=mp_all, mpts=mpts,
                    tol=max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0)))))

    def cands(r, add_avg):
        plist = r["pbs"] + [sum(r["pbs"]) / len(r["pbs"])] if add_avg else r["pbs"]
        der = lambda pr: [cp_openings.connection_points(
            r["V"], r["F"], pb.argmax(-1), min_v=MINV, classes=(CE, CT), dedupe_mm=10.0,
            probs=pb, vertex_conf=VC, ct_depth_min_mm=1.0, cluster_mm=CL,
            conn_promote=pr) for pb in plist]
        base = robot_cp._vote2(der(0.0), min_votes=1)
        is_hi = robot_cp._highcp_router(r["stp"], r["V"], len(base))
        return (robot_cp._vote2(der(0.25), min_votes=1) if is_hi else base), is_hi

    def feats(r, cps, use_col):
        probs = sum(r["pbs"]) / len(r["pbs"])
        X = wire_gate.feats_for(r["V"], r["F"], probs, cps, CE, CT)
        if not use_col:
            return X
        C = np.full((len(cps), 3), np.nan)
        if r["mp_all"] is not None:
            for i, c in enumerate(cps):
                C[i] = dip_features(r["mp_all"], r["mpts"], c["point"],
                                    c.get("direction", (0, 0, 1)))
        miss = np.isnan(C).any(1).astype(float)[:, None]
        return np.hstack([X, np.where(np.isnan(C), -1.0, C), miss])

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

    ARMS = {"A_4uye_13": (False, False), "B_5uye_13": (True, False),
            "C_5uye_RENK": (True, True), "D_4uye_RENK": (False, True)}
    TX = {k: [] for k in ARMS}; TY = {k: [] for k in ARMS}
    n_col = 0
    for k, prt in enumerate(tr_parts, 1):
        try:
            r = load_part(*prt)
            n_col += int(r["mp_all"] is not None)
            for name, (avg, col) in ARMS.items():
                cps, _ = cands(r, avg)
                if not cps: continue
                TX[name].append(feats(r, cps, col)); TY[name].append(label(r, cps))
        except Exception:
            continue
        if k % 100 == 0:
            print(f"  egitim {k}/{len(tr_parts)} ({n_col} renkli)", flush=True)
    gates = {}
    for name in ARMS:
        X = np.vstack(TX[name]); y = np.concatenate(TY[name])
        gates[name] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                             random_state=0).fit(X, y)
        print(f"  {name}: {len(y)} aday, {X.shape[1]} ozellik", flush=True)

    tests = []
    for prt in sel:
        try: tests.append(load_part(*prt))
        except Exception: continue
    tc = sum(1 for r in tests if r["mp_all"] is not None)
    print(f"test onbellek {len(tests)} parca ({tc} renkli)\n", flush=True)

    def score(name):
        avg, col = ARMS[name]; clf = gates[name]
        agg = {"dusuk": [0, 0, 0], "cok": [0, 0, 0]}
        for r in tests:
            cps, is_hi = cands(r, avg)
            keep = []
            if cps:
                sc = clf.predict_proba(feats(r, cps, col))[:, 1]
                thr = TH if is_hi else TL
                keep = [c for c, s_ in zip(cps, sc) if s_ >= thr]
            Q = np.array([c["point"] for c in keep], float) if keep else np.zeros((0, 3))
            G, Gd = r["G"], r["Gd"]; hit = np.zeros(len(G), bool); used = set()
            if len(Q) and len(G):
                diff = Q[:, None, :] - G[None, :, :]; al = (diff * Gd[None, :, :]).sum(-1)
                pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
                pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
                for d_, a_, b_ in sorted((pe[a, b], a, b) for a in range(len(Q)) for b in range(len(G))):
                    if d_ > r["tol"] or a_ in used or hit[b_]: continue
                    hit[b_] = True; used.add(a_)
            tp = int(hit.sum()); kk = "cok" if r["n"] >= 8 else "dusuk"
            agg[kk][0] += tp; agg[kk][1] += len(Q) - tp; agg[kk][2] += len(G) - tp
        out = {}
        for kk, (T, Fp, Fn) in agg.items():
            p = T / max(T + Fp, 1); rc = T / max(T + Fn, 1)
            out[kk] = 2 * p * rc / max(p + rc, 1e-9)
        out["w"] = sum(W[kk] * out[kk] for kk in W)
        return out

    res = {name: score(name) for name in ARMS}
    print(f"{'kol':<16}{'dusuk':>9}{'cok':>9}{'agirlikli':>11}{'fark':>10}")
    base = res["A_4uye_13"]["w"]
    for name in ("A_4uye_13", "B_5uye_13", "D_4uye_RENK", "C_5uye_RENK"):
        r = res[name]
        print(f"{name:<16}{r['dusuk']:>9.4f}{r['cok']:>9.4f}{r['w']:>11.4f}{r['w']-base:>+10.4f}")
    dC = res["C_5uye_RENK"]["w"] - base
    print(f"\nYIGIN (C-A): {dC:+.4f}")
    print(f"KAPI (>= +0.02): {'GECTI -- finale girer' if dC >= 0.02 else 'OLU'}")
    json.dump({k: v for k, v in res.items()} | {"stack_delta": dC},
              open("results/p4_yigin.json", "w"), indent=1)
    print("makbuz -> results/p4_yigin.json")


if __name__ == "__main__":
    main()
