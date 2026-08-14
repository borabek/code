# -*- coding: utf-8 -*-
"""COK-OLCEKLI inference plato-kirici (ROBOT_STRATEJI yedek plani #3).

9k EGITIMI reddedildi (domain gap) ama cok-olcekli INFERENCE farkli: TEK urun modeli iki
cozunurlukte (6k + 9k) kosulur, CP'ler BIRLESTIRILIR. Hipotez: minik acikliklar 9k'da,
buyukler 6k'da yakalanir -> union'dan farkli olarak AYNI modelin olcek-tamamlayiciligi,
precision'i cökertmeden recall ekleyebilir. thesis_remesh STEP frame'i korur -> olcekler ayni
koordinatta, dogrudan birlestirilir. Ayni parca setinde 6k-tek / 9k-tek / union'i skorlar.

Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe multiscale_arbiter.py \
    --ckpt results/seg_extra/recall_hard_s2.pt --only-mfg WEI --only-parts $(cat _hw_r3.txt) \
    --axis-aware --scales 6000 9000 --tag ms_wei
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, cp_openings, thesis_remesh
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import greedy, eligible, CE, CT, OP
from union_arbiter import union_cps


def derive(V, F, probs, min_v, vc, cluster_mm):
    lab = probs.argmax(-1)
    return cp_openings.connection_points(V, F, lab, min_v=min_v, classes=(CE, CT), dedupe_mm=10.0,
                                         probs=probs, vertex_conf=vc, ct_depth_min_mm=1.0, cluster_mm=cluster_mm)


def score_cps(parts, a, pick):
    """pick(per_scale_cps) -> chosen cp list for the part."""
    T = Fp = Fn = 0
    for pid, Vr, Vj, G, Gd, per_scale in parts:
        cps = pick(per_scale)
        R, t, _ = align_frames(Vr, Vj)
        P = (np.array([np.asarray(c["point"]) for c in cps], float) @ R.T + t) if cps else np.zeros((0, 3))
        tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
        tp, fp, fn = greedy(P, G, tol, Gd, a.axis_tol) if a.axis_aware else greedy(P, G, tol)
        T += tp; Fp += fp; Fn += fn
    pr = T / max(T + Fp, 1); rc = T / max(T + Fn, 1); f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    return f1, pr, rc, T, Fp, Fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/seg_extra/recall_hard_s2.pt")
    ap.add_argument("--only-mfg", default="")
    ap.add_argument("--only-parts", nargs="+", default=[])
    ap.add_argument("--scales", nargs="+", type=int, default=[6000, 9000])
    ap.add_argument("--axis-aware", action="store_true")
    ap.add_argument("--axis-tol", type=float, default=40.0)
    ap.add_argument("--cluster-mm", type=float, default=5.0)
    ap.add_argument("--min-v", type=int, default=30)
    ap.add_argument("--vertex-conf", type=float, default=0.5)
    ap.add_argument("--tag", default="ms")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, meta, _ = load_any(a.ckpt, dev=a.device)
    os.environ["BA_ALLOW_SEEN"] = "1"
    raw = eligible()
    if a.only_mfg: raw = [p for p in raw if p[0] == a.only_mfg]
    if a.only_parts:
        keep = set(a.only_parts); raw = [p for p in raw if p[1] in keep]
    print(f"{len(raw)} parca | {os.path.basename(a.ckpt)} | olcekler {a.scales}", flush=True)

    parts = []; t0 = time.time()
    for k, (mfg, pid, jf, stp) in enumerate(raw, 1):
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            if not len(G): continue
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vr, Fr = step_to_mesh(stp)
            per_scale = []
            for tgt in a.scales:
                V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=tgt)
                V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
                _, pb = D.predict(model, meta, V, F, device=a.device, op_cache_dir=OP, return_probs=True)
                per_scale.append(derive(V, F, np.asarray(pb, float), a.min_v, a.vertex_conf, a.cluster_mm))
            parts.append((pid, Vr, Vj, G, Gd, per_scale))
        except Exception:
            continue
        if k % 25 == 0:
            print(f"  {k}/{len(raw)}  {time.time()-t0:.0f}s", flush=True)

    print(f"\n=== SONUC ({len(parts)} parca) — axis_aware={a.axis_aware} ===")
    for i, tgt in enumerate(a.scales):
        f1, pr, rc, T, Fp, Fn = score_cps(parts, a, lambda ps, i=i: ps[i])
        print(f"  TEK-OLCEK {tgt:5d}          F1={f1:.3f}  P={pr:.3f}  R={rc:.3f}  (TP{T} FP{Fp} FN{Fn})")
    f1, pr, rc, T, Fp, Fn = score_cps(parts, a, lambda ps: union_cps(ps, a.cluster_mm, 1))
    print(f"  UNION (herhangi olcek)      F1={f1:.3f}  P={pr:.3f}  R={rc:.3f}  (TP{T} FP{Fp} FN{Fn})")
    f1, pr, rc, T, Fp, Fn = score_cps(parts, a, lambda ps: union_cps(ps, a.cluster_mm, len(a.scales)))
    print(f"  KESISIM (her olcek anlasir) F1={f1:.3f}  P={pr:.3f}  R={rc:.3f}  (TP{T} FP{Fp} FN{Fn})")
    json.dump({"tag": a.tag, "ckpt": a.ckpt, "scales": a.scales}, open(f"results/ms_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
