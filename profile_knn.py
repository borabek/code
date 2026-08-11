"""Profiler: where does a knngraph epoch actually spend its time?

Loads the real corpus, reports vertex-size distribution, then times:
  - kNN graph build (CPU, scipy cKDTree)
  - CPU→GPU transfer (nbr tensor)
  - Feature tensor construction (xyz / +normals / +curvature)
  - Forward pass (GPU)
  - Backward + optimiser step (GPU)

Also benchmarks graph-BATCHING: compares serial (one part at a time) vs
batched (multiple small parts concatenated into one GPU call) throughput.

Usage:
  python profile_knn.py corpus/                         (cpu, cap=7000)
  python profile_knn.py corpus/ cuda                    (gpu)
  python profile_knn.py corpus/ cuda 7000 amp           (gpu + AMP)
  python profile_knn.py corpus/ cuda 7000 amp --batch   (also benchmark batching)
  python profile_knn.py corpus/ cuda 7000 amp --full    (all stages + batching)
"""
import argparse
import sys
import time

import numpy as np
import torch

import json_dataset as jd
import cp_regressor as cpr
import cp_targets as ct


def _parse_args():
    ap = argparse.ArgumentParser(description="knngraph epoch profiler")
    ap.add_argument("source", nargs="?",
                    default=r"C:\Users\DE00024082\Desktop\JSON")
    ap.add_argument("device", nargs="?", default="cpu")
    ap.add_argument("cap", nargs="?", type=int, default=7000)
    ap.add_argument("amp_flag", nargs="?", default="",
                    help="pass 'amp' to enable AMP")
    ap.add_argument("--batch", action="store_true",
                    help="benchmark graph-batching vs serial throughput")
    ap.add_argument("--full", action="store_true",
                    help="enable all sub-timings + batching benchmark")
    ap.add_argument("--normals", action="store_true",
                    help="include normal features in profiling")
    ap.add_argument("--curvature", action="store_true",
                    help="include curvature features (implies --normals)")
    ap.add_argument("--k", type=int, default=16,
                    help="kNN graph degree (default 16)")
    ap.add_argument("--n-sample", type=int, default=12, dest="n_sample",
                    help="number of parts to sample for timing")
    ap.add_argument("--backbone", default="knngraph",
                    choices=["knngraph", "hierpoint"],
                    help="which backbone to profile (default knngraph)")
    return ap.parse_args()


def _sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _t():
    return time.perf_counter()


def main():
    args = _parse_args()
    DEV = args.device
    CAP = args.cap
    K = args.k
    use_amp = args.amp_flag.lower() in ("amp", "1", "true") and DEV == "cuda"
    use_normals = args.normals or args.curvature or args.full
    use_curvature = args.curvature or args.full
    do_batch = args.batch or args.full

    t0 = _t()
    parts = [p for p in jd.iter_parts(args.source) if p.n_cps > 0]
    print(f"load: {len(parts)} parts with CPs in {_t()-t0:.1f}s")

    nv = np.array([p.n_vertices for p in parts])
    print(f"verts: min={nv.min()} median={int(np.median(nv))} "
          f"mean={int(nv.mean())} max={nv.max()}  "
          f">cap({CAP}): {(nv>CAP).sum()} parts  >60k: {(nv>60000).sum()}")
    print(f"total verts: {nv.sum():,}  (capped: {np.minimum(nv,CAP).sum():,})")

    # global_feat=False: matches what training actually uses (a True here used to
    # profile a DIFFERENT model than the one being trained)
    if args.backbone == "hierpoint":
        cfg = {"k": K, "normals": use_normals, "curvature": use_curvature,
               "concavity": False, "edge_dist": False}
        import hierpoint as hp
    else:
        cfg = {"k": K, "c_width": 128, "n_layers": 4, "global_feat": False,
               "normals": use_normals, "curvature": use_curvature}
        hp = None
    model, meta = cpr.build_regressor(args.backbone, cfg)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    from torch.amp.grad_scaler import GradScaler
    scaler = GradScaler("cuda", enabled=use_amp)
    print(f"\nbackbone={args.backbone}  device={DEV}  cap={CAP}  k={K}  "
          f"amp={'ON' if use_amp else 'off'}  "
          f"normals={use_normals}  curvature={use_curvature}")

    # sample: spread of sizes + biggest
    order = np.argsort(nv)
    idx_s = list(order[:: max(1, len(order) // args.n_sample)]) + [int(order[-1])]
    idx_s = idx_s[:args.n_sample + 1]

    print(f"\n{'n_verts':>8} {'used':>6} | {'graph':>8} {'cpu->gpu':>8} "
          f"{'feat':>8} {'fwd':>8} {'bwd':>8} | {'total':>9}")
    print("-" * 82)

    t_graph_all = t_transfer_all = t_feat_all = t_fwd_all = t_bwd_all = 0.0
    n_parts_timed = 0

    for i in idx_s:
        p = parts[int(i)]
        s = cpr.prepare_sample(p, dedup=True)
        v = s["verts_norm"]; tgt = s["target"]; msk = s["mask"]
        n = len(v)

        if n > CAP:
            heat = tgt[:, ct.HEATMAP]; peaks = np.where(heat >= 1 - 1e-4)[0]
            rest = np.setdiff1d(np.arange(n), peaks, assume_unique=True)
            extra = np.random.RandomState(0).choice(
                rest, min(CAP - len(peaks), len(rest)), replace=False)
            sel = np.sort(np.concatenate([peaks, extra]))
            v, tgt, msk = v[sel], tgt[sel], msk[sel]
        used = len(v)

        # --- kNN graph build (+ pooling hierarchy for hierpoint) ---
        tb = _t()
        nbr = cpr._knn_graph(v, K)
        hier = hp.build_hierarchy(v) if hp is not None else None
        g_ms = (_t() - tb) * 1e3

        # --- CPU → GPU transfer ---
        _sync(DEV)
        tc_ = _t()
        if hp is not None:
            nb_gpu = hp.collate([hier], [nbr], device=DEV)
        else:
            nb_gpu = nbr.to(DEV)
        _sync(DEV)
        tr_ms = (_t() - tc_) * 1e3

        # --- feature tensor (incl. normal/curvature compute on CPU) ---
        _sync(DEV)
        tf0 = _t()
        x = cpr._knn_feature_tensor(v, nbr, use_normals, DEV,
                                    use_curvature, False, False)
        _sync(DEV)
        ft_ms = (_t() - tf0) * 1e3

        t_in = torch.tensor(tgt, dtype=torch.float32, device=DEV)
        m_in = torch.tensor(msk, dtype=torch.bool, device=DEV)

        # --- forward ---
        _sync(DEV)
        t_f = _t()
        with torch.autocast(device_type="cuda" if DEV == "cuda" else "cpu",
                            dtype=torch.float16,
                            enabled=use_amp and DEV == "cuda"):
            out = model(x, nb_gpu)
            loss, _ = cpr.cp_loss(out, t_in, m_in, heat_loss="centernet")
        _sync(DEV)
        f_ms = (_t() - t_f) * 1e3

        # --- backward + step ---
        _sync(DEV)
        t_b = _t()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        _sync(DEV)
        b_ms = (_t() - t_b) * 1e3

        total_ms = g_ms + tr_ms + ft_ms + f_ms + b_ms
        mem = ""
        if DEV == "cuda" and torch.cuda.is_available():
            # peak reserved is the WDDM-cliff tripwire: silently spilling past
            # ~3.7 GB on a 4 GB card is a 10x slowdown, not an OOM error
            mem = f"  peak {torch.cuda.max_memory_reserved() / (1 << 20):>5.0f}MB"
            torch.cuda.reset_peak_memory_stats()
        print(f"{n:>8} {used:>6} | {g_ms:>8.1f} {tr_ms:>8.2f} "
              f"{ft_ms:>8.1f} {f_ms:>8.1f} {b_ms:>8.1f} | {total_ms:>9.1f}{mem}")

        t_graph_all += g_ms; t_transfer_all += tr_ms; t_feat_all += ft_ms
        t_fwd_all += f_ms; t_bwd_all += b_ms
        n_parts_timed += 1

    ns = max(1, n_parts_timed)
    print(f"\n{'per-part avg (ms)':30s}: "
          f"graph={t_graph_all/ns:.0f}  "
          f"cpu→gpu={t_transfer_all/ns:.1f}  "
          f"feat={t_feat_all/ns:.0f}  "
          f"fwd={t_fwd_all/ns:.0f}  "
          f"bwd={t_bwd_all/ns:.0f}  "
          f"total={(t_graph_all+t_transfer_all+t_feat_all+t_fwd_all+t_bwd_all)/ns:.0f}")

    total_part_ms = (t_graph_all + t_transfer_all + t_feat_all +
                     t_fwd_all + t_bwd_all) / ns
    est1 = total_part_ms * len(parts) / 1e3
    step_only = (t_fwd_all + t_bwd_all) / ns
    estN = step_only * len(parts) / 1e3
    print(f"\nEST epoch 1 (build+train): {est1:.0f}s = {est1/60:.1f} min  "
          f"| later epochs (cached): {estN:.0f}s = {estN/60:.1f} min")
    print(f"EST epochs in 60 min: ~{int(max(0, (3600 - est1) / max(estN, 1)) + 1)}")

    # -----------------------------------------------------------------------
    # Batching benchmark: compare serial vs batched throughput on small parts
    # -----------------------------------------------------------------------
    if do_batch and hp is not None:
        print("\n(batching benchmark is knngraph-specific -- for hierpoint the "
              "trainer's collate covers it; skipped)")
        return
    if do_batch:
        print("\n" + "=" * 60)
        print("GRAPH-BATCHING benchmark (small parts, GPU)")
        print("=" * 60)

        small = [p for p in parts if p.n_vertices <= CAP // 4]
        if len(small) < 4:
            print(f"Only {len(small)} parts <= {CAP//4} verts — skip batching bench")
            return

        rng = np.random.default_rng(42)
        sample_b = [small[i] for i in rng.choice(len(small),
                                                   min(16, len(small)),
                                                   replace=False)]

        # Prepare all
        prep = []
        for p in sample_b:
            s = cpr.prepare_sample(p, dedup=True)
            v = s["verts_norm"]; tgt = s["target"]; msk = s["mask"]
            nbr = cpr._knn_graph(v, K)
            prep.append((v, tgt, msk, nbr))

        _sync(DEV)

        # SERIAL
        t_ser = _t()
        for v, tgt, msk, nbr in prep:
            x = cpr._knn_feature_tensor(v, nbr, use_normals, DEV,
                                        use_curvature, False, False)
            nb_gpu = nbr.to(DEV)
            t_in = torch.tensor(tgt, dtype=torch.float32, device=DEV)
            m_in = torch.tensor(msk, dtype=torch.bool, device=DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if DEV == "cuda" else "cpu",
                                dtype=torch.float16,
                                enabled=use_amp and DEV == "cuda"):
                out = model(x, nb_gpu)
                loss, _ = cpr.cp_loss(out, t_in, m_in, heat_loss="centernet")
            loss.backward()
            opt.step()
        _sync(DEV)
        ser_ms = (_t() - t_ser) * 1e3
        print(f"Serial  ({len(prep)} parts): {ser_ms:.0f} ms  "
              f"({ser_ms/len(prep):.1f} ms/part)")

        # BATCHED: concatenate graphs
        t_bat = _t()
        # build concatenated graph
        offset = 0
        cat_V_list, cat_tgt_list, cat_msk_list, cat_nbr_list = [], [], [], []
        n_list = []
        for v, tgt, msk, nbr in prep:
            cat_V_list.append(v)
            cat_tgt_list.append(tgt)
            cat_msk_list.append(msk)
            cat_nbr_list.append(nbr + offset)
            n_list.append(len(v))
            offset += len(v)
        cat_V   = np.concatenate(cat_V_list)
        cat_tgt = np.concatenate(cat_tgt_list)
        cat_msk = np.concatenate(cat_msk_list)
        cat_nbr = torch.cat(cat_nbr_list)

        x_bat = cpr._knn_feature_tensor(cat_V, cat_nbr.numpy(), use_normals, DEV,
                                        use_curvature, False, False)
        nb_bat = cat_nbr.to(DEV)
        t_bat_g = torch.tensor(cat_tgt, dtype=torch.float32, device=DEV)
        m_bat_g = torch.tensor(cat_msk, dtype=torch.bool, device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if DEV == "cuda" else "cpu",
                            dtype=torch.float16,
                            enabled=use_amp and DEV == "cuda"):
            out_bat = model(x_bat, nb_bat)
            loss_bat, _ = cpr.cp_loss(out_bat, t_bat_g, m_bat_g, heat_loss="centernet")
        loss_bat.backward()
        opt.step()
        _sync(DEV)
        bat_ms = (_t() - t_bat) * 1e3
        speedup = ser_ms / max(bat_ms, 0.1)
        print(f"Batched ({len(prep)} parts): {bat_ms:.0f} ms  "
              f"({bat_ms/len(prep):.1f} ms/part)  speedup={speedup:.1f}x")
        total_verts = sum(n_list)
        print(f"  ({len(prep)} parts, {total_verts} total verts "
              f"in one {total_verts}-vert forward)")


if __name__ == "__main__":
    main()
