# -*- coding: utf-8 -*-
"""TEL-B'yi GERCEK korpusa kos: her aday icin 10 kanal ozelligi + TP/FP etiketi.

TEL-A'nin gerekcesi: hayatta kalan 263 FP'de mevcut 41 ozelligin EN IYISI AUC 0.36/0.61 -- kor.
TEL-B hipotezi: tel girisi KONTAKTA biter, alet agzi kol/yay yuvasinda -- fark ICERIDE.

ONCELIK NOTU: korpusun %89.5'i DUSUK-CP ve orada tavan 0.894 (potansiyel +0.054 GENEL);
cok-CP dali sadece +0.016. Bu yuzden ONCE dusuk-CP altkumesi.

Cikti: results/tel_b_feats.npz  (kanal ozellikleri + etiket + parca kimligi)
Not: sentetik dogrulama tel_b_kanal_profili.py icinde, 3/3 gecti (tel-girisi/alet-agzi/tunel).
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "results/tel_b_feats.npz"
HIGH_CP = 8
SAVE_EVERY = 25


def main():
    import torch, trimesh
    import thesis_remesh, robot_cp
    from cad_eval import align_frames
    from infer_step_cp import step_to_mesh, load_any
    from big_arbiter import eligible
    from connector_constants import CABLE_ENTRY as CE, CONTACT as CT
    from tel_b_kanal_profili import feats_for, FEAT_NAMES
    import wire_gate

    lock = json.load(open("results/split_lock.json"))
    LOCK = set(lock["locked_parts"])
    cfg = json.load(open("cp_config.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_any(c, dev=dev)[:2] for c in cfg["current_product"]["checkpoints"]]

    # DUSUK-CP altkumesi, kilitli olmayan
    parts = []
    for mfg, pid, jf, stp in eligible():
        if pid in LOCK:
            continue
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            n = len(j["ConnectionPoints"])
        except Exception:
            continue
        if 0 < n < HIGH_CP:
            parts.append((mfg, pid, jf, stp, n))
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = np.random.RandomState(0)
    if len(parts) > lim:
        parts = [parts[i] for i in sorted(rng.choice(len(parts), lim, replace=False))]
    print(f"{len(parts)} dusuk-CP parca islenecek", flush=True)

    X, XB, Y, PID, MFG = [], [], [], [], []
    done = set()
    if os.path.exists(OUT):                      # DEVAM-EDEBILIRLIK (uzun is, kesilebilir)
        z = np.load(OUT, allow_pickle=True)
        X = [z["X"]]; XB = [z["X13"]]; Y = [z["y"]]; PID = list(z["pid"]); MFG = list(z["mfg"])
        done = {str(q) for q in z["pid"]}
        print(f"  [devam] {len(done)} parca zaten islenmis", flush=True)

    t0 = time.time()
    for k, (mfg, pid, jf, stp, ngt) in enumerate(parts, 1):
        if pid in done:
            continue
        try:
            with open("results/_telb_current.txt", "w") as fh:
                fh.write(f"{mfg}.{pid}\t{k}/{len(parts)}\t{time.strftime('%H:%M:%S')}")
            cps = robot_cp.extract(models, stp, dev,
                                   conf_auto=float(cfg.get("robot_conf_auto", 0.5)))
            if not cps:
                continue
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            import diffusionnet as D
            acc = None
            for model, meta in models:
                _, pb = D.predict(model, meta, V, F, device=dev,
                                  op_cache_dir=f"results/step_infer/ops_k{int(meta.get('k_eig',64))}",
                                  return_probs=True)
                pb = np.asarray(pb, float); acc = pb if acc is None else acc + pb
            probs = acc / len(models)

            xb = feats_for(V, F, probs, cps, CE, CT)          # (n_aday, 10) kanal ozellikleri
            x13 = wire_gate.feats_for(V, F, probs, cps, CE, CT)   # mevcut gate seti (AYNI adaylar/sira)

            # ETIKET: uretici GT'sine eksene-duyarli eslesme (big_arbiter ile ayni konvansiyon)
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[q[c] for c in "XYZ"] for q in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
            Gd = np.array([[c["InsertDirection"][q] for q in "XYZ"] for c in j["ConnectionPoints"]], float)
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            R, t, _ = align_frames(Vr, Vj)
            Gm = (G - t) @ R; Gdm = Gd @ R
            P = np.array([c["point"] for c in cps], float)
            tol = max(3.0, 0.06 * float(np.linalg.norm(V.max(0) - V.min(0))))
            diff = P[:, None, :] - Gm[None, :, :]
            al = (diff * Gdm[None, :, :]).sum(-1)
            perp = np.linalg.norm(diff - al[..., None] * Gdm[None, :, :], axis=-1)
            perp = np.where(np.abs(al) <= 40.0, perp, np.inf)
            used_g, lab = set(), np.zeros(len(P), int)
            for d_, a_, b_ in sorted((perp[a_, b_], a_, b_)
                                     for a_ in range(len(P)) for b_ in range(len(Gm))):
                if d_ > tol or lab[a_] or b_ in used_g:
                    continue
                lab[a_] = 1; used_g.add(b_)

            X.append(xb); XB.append(x13); Y.append(lab)
            PID += [pid] * len(lab); MFG += [mfg] * len(lab)
            done.add(pid)
        except Exception as e:
            print(f"  {pid}: HATA {type(e).__name__} {str(e)[:60]}", flush=True)
            continue
        if k % SAVE_EVERY == 0:
            np.savez(OUT, X=np.vstack(X), X13=np.vstack(XB), y=np.concatenate(Y),
                     pid=np.array(PID), mfg=np.array(MFG), names=np.array(FEAT_NAMES))
            print(f"  {k}/{len(parts)}  {len(done)} ok  {time.time()-t0:.0f}s (kaydedildi)", flush=True)

    np.savez(OUT, X=np.vstack(X), X13=np.vstack(XB), y=np.concatenate(Y),
             pid=np.array(PID), mfg=np.array(MFG), names=np.array(FEAT_NAMES))
    yy = np.concatenate(Y)
    print(f"-> {OUT}  {len(yy)} aday ({int(yy.sum())} TP / {int((yy==0).sum())} FP), "
          f"{len(done)} parca  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
