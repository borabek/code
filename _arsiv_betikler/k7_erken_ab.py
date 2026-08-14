# -*- coding: utf-8 -*-
"""K7.4 ERKEN SINYAL: renk ozelligi gate'e eklenince CP-F1 degisiyor mu? (64 parca)

Tam korpus cikarimi ~70dk surerken bu kucuk olcekli A/B erken karar verdirir.
Aile-disi OOF, ayni adaylar, tek fark ozellik seti. KUCUK ORNEKLEM -- yon gostergesi,
kesin sayi degil; tam olcum gate_regrow_data_k7 ile yapilacak.
"""
import os, sys, glob, json, pickle
import numpy as np
os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CACHE = "results/pitstop2_cache"
W = {"dusuk": 0.895, "cok": 0.105}


def main():
    import wire_gate, step_face_color_link as L
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from cad_eval import align_frames
    from big_arbiter import eligible
    from infer_step_cp import step_to_mesh
    from json_dataset import family_key
    from k7_dip_metal import dip_features

    stp_of = {p: s for _, p, _, s in eligible()}
    X13, XC, Y, GRP, FAM, REG, NGT = [], [], [], [], [], [], {}
    n_col = 0
    for f in sorted(glob.glob(f"{CACHE}/*.npz")):
        pid = os.path.basename(f)[:-4]
        pk = f"{CACHE}/{pid}.cps.pkl"
        if pid not in stp_of or not os.path.exists(pk):
            continue
        d = np.load(f, allow_pickle=True)
        cps = pickle.load(open(pk, "rb"))
        if not cps:
            continue
        V, F, probs = d["V"], d["F"], d["probs"]
        G, Gd, tol = d["G"], d["Gd"], float(d["tol"])
        n = int(d["n"])
        feats = wire_gate.feats_for(V, F, probs, cps, CE, CT)
        P = np.array([c["point"] for c in cps], float)
        D = np.array([c.get("direction", (0, 0, 1)) for c in cps], float)
        # renk ozellikleri (bulunamazsa NaN -- 0 YAZILMAZ)
        col = np.full((len(cps), 3), np.nan)
        try:
            faces = L.face_vertices_from_text(stp_of[pid])
            met = [x for x in faces if x["is_metal"]]
            if faces and met:
                Vr, _ = step_to_mesh(stp_of[pid])
                allp = np.vstack([x["pts"] for x in faces])
                flag = np.concatenate([np.full(len(x["pts"]), 1.0 if x["is_metal"] else 0.0)
                                       for x in faces])
                R, t, res = align_frames(Vr, allp)
                if res <= 1.0:
                    allm = (allp - t) @ R
                    mp_all = np.column_stack([allm, flag])
                    mpts = allm[flag > 0.5]
                    for i in range(len(cps)):
                        col[i] = dip_features(mp_all, mpts, P[i], D[i])
                    n_col += 1
        except Exception:
            pass
        lab = np.zeros(len(P), int)
        if len(G):
            diff = P[:, None, :] - G[None, :, :]
            al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) <= 40.0, pe, np.inf)
            used = set(); hit = np.zeros(len(G), bool)
            for dd, a, b in sorted((pe[a, b], a, b) for a in range(len(P)) for b in range(len(G))):
                if dd > tol or a in used or hit[b]:
                    continue
                used.add(a); hit[b] = True; lab[a] = 1
        X13.append(feats); XC.append(col); Y.append(lab)
        GRP += [pid] * len(cps); FAM += [family_key(pid)] * len(cps)
        REG += [("cok" if n >= 8 else "dusuk")] * len(cps)
        NGT[pid] = (len(G), "cok" if n >= 8 else "dusuk")
    X13 = np.vstack(X13); XC = np.vstack(XC); y = np.concatenate(Y)
    fam = np.array(FAM); reg = np.array(REG)
    print(f"{len(NGT)} parca ({n_col} tanesinde renk) | {len(y)} aday | pozitif %{100*y.mean():.0f}\n")
    tot = {k: sum(v[0] for v in NGT.values() if v[1] == k) for k in ("dusuk", "cok")}

    def oof(Xm):
        s = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(Xm, y, fam):
            s[te] = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                           random_state=0).fit(Xm[tr], y[tr]).predict_proba(Xm[te])[:, 1]
        return s

    def best_f1(s):
        out = {}
        for k in ("dusuk", "cok"):
            bb = 0
            for thr in np.arange(0.20, 0.71, 0.05):
                m = reg == k; sel = (s >= thr) & m
                tp = int((y[sel] == 1).sum()); fp = int(sel.sum()) - tp
                p = tp / max(tp + fp, 1); r = tp / max(tot[k], 1)
                bb = max(bb, 2 * p * r / max(p + r, 1e-9))
            out[k] = bb
        out["w"] = sum(W[k] * out[k] for k in W)
        return out

    # NaN -> RandomForest tolere etmez; eksiklik BAYRAGI ile birlikte -1 kodlanir (bilgi kaybi yok)
    miss = np.isnan(XC).any(axis=1).astype(float)[:, None]
    XCf = np.where(np.isnan(XC), -1.0, XC)
    a = best_f1(oof(X13))
    b = best_f1(oof(np.hstack([X13, XCf, miss])))
    print(f"{'ozellik seti':<28}{'dusuk':>9}{'cok':>9}{'agirlikli':>11}")
    print(f"{'13 mevcut':<28}{a['dusuk']:>9.4f}{a['cok']:>9.4f}{a['w']:>11.4f}")
    print(f"{'13 + RENK (3+bayrak)':<28}{b['dusuk']:>9.4f}{b['cok']:>9.4f}{b['w']:>11.4f}")
    print(f"\nRENK katkisi: {b['w']-a['w']:+.4f}")
    print(f"KAPI (>= +0.02): {'GECTI' if b['w']-a['w'] >= 0.02 else 'OLU'}   [KUCUK ORNEKLEM]")
    json.dump({"base": a, "color": b, "delta": b["w"] - a["w"], "n_parts": len(NGT),
               "n_with_color": n_col}, open("results/k7_erken_ab.json", "w"), indent=1)
    print("makbuz -> results/k7_erken_ab.json")


if __name__ == "__main__":
    main()
