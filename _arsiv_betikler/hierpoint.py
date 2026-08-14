# hierpoint.py -- hierarchical point U-Net backbone ("hierpoint") for the CP
# detector. Replaces the flat 4x EdgeConv knngraph stack with a PointNet++-style
# encoder/decoder sized for a 4 GB (T1200/WDDM) card:
#
#   stem Linear(c_in->w0)
#   L0:  LeanEdgeConv(w0)                       @ N       (uses the existing kNN graph)
#   SA1: group-max downsample -> N/r1, w0->w1;  L1: LeanEdgeConv(w1)
#   SA2: -> N/r2, w1->w2;                       L2: LeanEdgeConv(w2) x2
#   SA3: -> N/r3, w2->w3;                       L3: LeanEdgeConv(w3) + per-graph
#                                                   global max-pool context
#   FP(3->2) -> FP(2->1) -> FP(1->0): 3-NN inverse-distance upsample + skip concat
#   refine: LeanEdgeConv(w0) @ N (keeps the decoded heatmap peak sharp)
#   head: Dropout -> Linear(w0 -> 7)
#
# WHY: (1) receptive field -- the flat knngraph sees ~4 hops of 16-NN at full
# density (~1-2 mm on a 70 mm part), and ~55% of its false negatives are "MODEL
# BLIND" (no heat near the GT): a vertex inside a recessed wire entry never sees
# the terminal pitch or the block face. Here the coarsest level covers the whole
# patch and feature propagation hands that context back to every vertex.
# (2) memory -- the stock EdgeConv materialises ~6 (N,k,C..2C) tensors per layer
# (~1.6 GB at N=7000), which is what forced the 7000-vert cap and the chunked
# forward that halved throughput. LeanEdgeConv keeps ONE (N,k,C) edge tensor and
# the SA/FP levels shrink N by 4x each, so a 14000-vert patch fits in ~0.4 GB.
#
# All pooling structure (FPS + grouping kNN + within-level kNN + 3-NN upsample
# weights) is a pure function of the geometry: it is precomputed on CPU
# (numpy/scipy, GIL-releasing cKDTree) at prep time exactly like the cached
# d["nbr"] graphs, and merged per accumulation chunk by collate() with index
# offsets -- disconnected graphs never mix (the global pool is segment-wise), so
# chunked forwards + summed backwards stay gradient-identical to a fused forward
# (the same invariant train_knngraph_regressor documents for knngraph).
# torch-cluster could accelerate the FPS but is deliberately NOT a dependency:
# numpy FPS at <=14k points costs ~0.1 s/graph, once, cached.

import hashlib
import os

import numpy as np

N_CHANNELS = 7   # 1 heat logit + 3 offset + 3 direction (matches cp_targets)

HP_DEFAULTS = {
    "widths": (64, 128, 256, 256),   # L0..L3 feature widths
    "ratios": (4, 16, 64),           # level sizes: N/4, N/16, N/64 (floor 8 pts)
    "k_levels": (16, 16, 8),         # within-level kNN at L1..L3
    "group_k": 16,                   # SA grouping kNN (into the finer level)
    "up_k": 3,                       # feature-propagation interpolation neighbours
    "layers": (1, 1, 2, 1),          # LeanEdgeConv blocks per level L0..L3
    "k": 16,                         # L0 kNN (shared with d["nbr"] / feature calc)
    "dropout": 0.1,
    # geometric input features default ON (c_in=9): normals alone took precision
    # 36%->77% on v14; v19 regressed to raw xyz and plateaued.
    "normals": True, "curvature": True, "concavity": True, "edge_dist": True,
}

_MIN_LEVEL_PTS = 8


def _disk_has_room(path, min_free_gb=3.0):
    """True if `path`'s drive has at least min_free_gb free -- the hierarchy
    cache is an optimisation and must never be the thing that fills the disk out
    from under the training run it exists to speed up (see the disk-full/OOM
    cascades in RESULTS). Mirrors cp_regressor._disk_has_room; kept local to
    avoid a circular import (cp_regressor imports this module)."""
    import shutil
    try:
        return shutil.disk_usage(
            path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        ).free >= min_free_gb * (1 << 30)
    except OSError:
        return False


def _require_torch():
    import torch
    return torch


# --------------------------------------------------------------------------- #
# hierarchy construction (CPU, numpy/scipy, deterministic, cacheable)
# --------------------------------------------------------------------------- #

def _fps(V, m, start=0):
    """Farthest-point sampling: m indices into V, deterministic (start=0).
    O(m*N) with vectorised numpy distance updates -- ~0.1 s at N=14000, m=N/4."""
    n = len(V)
    m = int(min(m, n))
    sel = np.empty(m, dtype=np.int64)
    sel[0] = start
    d = ((V - V[start]) ** 2).sum(axis=1)
    for i in range(1, m):
        j = int(np.argmax(d))
        sel[i] = j
        np.minimum(d, ((V - V[j]) ** 2).sum(axis=1), out=d)
    return sel


def _knn_idx(src_pts, query_pts, k, drop_self=False):
    """(len(query), k) int32 neighbour indices into src_pts via cKDTree.
    drop_self assumes query_pts IS src_pts (col 0 = the point itself). Tiny sets
    (fewer than k neighbours available) are padded by repeating the last column."""
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]
    tree = cKDTree(src_pts)
    kq = int(min(k + (1 if drop_self else 0), len(src_pts)))
    _, idx = tree.query(query_pts, k=kq, workers=-1)
    idx = np.atleast_2d(idx)
    if idx.ndim == 1:
        idx = idx[:, None]
    if drop_self and idx.shape[1] > 1:
        idx = idx[:, 1:]
    if idx.shape[1] < k:
        idx = np.concatenate(
            [idx, np.repeat(idx[:, -1:], k - idx.shape[1], axis=1)], axis=1)
    return np.ascontiguousarray(idx, dtype=np.int32)


def _up_weights(coarse_pts, fine_pts, k=3):
    """3-NN inverse-distance interpolation: (idx int32 (Nf,k), w float32 (Nf,k)),
    rows of w sum to 1. Standard PointNet++ feature propagation."""
    from scipy.spatial import cKDTree  # type: ignore[attr-defined]
    tree = cKDTree(coarse_pts)
    kq = int(min(k, len(coarse_pts)))
    dist, idx = tree.query(fine_pts, k=kq, workers=-1)
    dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
    if dist.ndim == 1:
        dist = dist[:, None]; idx = idx[:, None]
    if idx.shape[1] < k:
        pad = k - idx.shape[1]
        idx = np.concatenate([idx, np.repeat(idx[:, -1:], pad, axis=1)], axis=1)
        dist = np.concatenate([dist, np.repeat(dist[:, -1:], pad, axis=1)], axis=1)
    w = 1.0 / np.maximum(dist, 1e-8)
    w = w / w.sum(axis=1, keepdims=True)
    return (np.ascontiguousarray(idx, dtype=np.int32),
            np.ascontiguousarray(w, dtype=np.float32))


def build_hierarchy(verts, ratios=(4, 16, 64), k_levels=(16, 16, 8),
                    group_k=16, up_k=3, cache_dir=None):
    """Pooling structure for ONE graph/patch. Returns a dict of CPU torch
    tensors (indices int32 -- collate() widens to long after the device copy):

      n     : [N, N1, N2, N3] level sizes
      pos   : [pos1, pos2, pos3] float32 (Nl, 3) level coordinates
      gnbr  : [g1, g2, g3] (Nl, group_k) grouping kNN into the PREVIOUS level
              (a centre is its own nearest previous-level neighbour -> included)
      lnbr  : [l1, l2, l3] (Nl, k_l) within-level kNN (self dropped, EdgeConv)
      up    : [(u1, w1), (u2, w2), (u3, w3)] -- ul: (N_{l-1}, up_k) indices into
              level l + inverse-distance weights, rows sum to 1

    Level l points are the FPS subset of level l-1 (deterministic, start=0), so
    the whole structure is a pure function of `verts` + the parameters -- safe to
    disk-cache (SHA256 of the float32 vertex bytes + params, like _knn_graph_disk).
    """
    torch = _require_torch()
    V0 = np.ascontiguousarray(np.asarray(verts, dtype=np.float64))
    key_path = None
    if cache_dir:
        key = hashlib.sha256(
            np.asarray(verts, dtype=np.float32).tobytes()).hexdigest()[:24]
        tag = "r%s_k%s_g%d_u%d" % ("-".join(map(str, ratios)),
                                   "-".join(map(str, k_levels)), group_k, up_k)
        key_path = os.path.join(cache_dir, f"hier_{tag}_{key}.npz")
        if os.path.exists(key_path):
            try:
                z = np.load(key_path)
                return _hier_from_arrays(z, torch)
            except Exception:                    # noqa: BLE001 -- corrupt entry
                pass

    N = len(V0)
    pos_prev = V0
    n_sizes = [N]
    pos_l, gnbr_l, lnbr_l, up_l = [], [], [], []
    for li, r in enumerate(ratios):
        n_l = int(min(len(pos_prev), max(_MIN_LEVEL_PTS, int(np.ceil(N / r)))))
        sel = _fps(pos_prev, n_l)
        pts = pos_prev[sel]
        gnbr_l.append(_knn_idx(pos_prev, pts, group_k))
        lnbr_l.append(_knn_idx(pts, pts, int(k_levels[li]), drop_self=True))
        u_idx, u_w = _up_weights(pts, pos_prev, k=up_k)
        up_l.append((u_idx, u_w))
        pos_l.append(np.ascontiguousarray(pts, dtype=np.float32))
        pos_prev = pts
        n_sizes.append(n_l)

    if key_path and _disk_has_room(cache_dir):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = key_path + ".tmp.npz"
            arrs = {"n": np.asarray(n_sizes, dtype=np.int64)}
            for i in range(len(ratios)):
                arrs[f"pos{i}"] = pos_l[i]
                arrs[f"gnbr{i}"] = gnbr_l[i]
                arrs[f"lnbr{i}"] = lnbr_l[i]
                arrs[f"up{i}"] = up_l[i][0]
                arrs[f"upw{i}"] = up_l[i][1]
            np.savez(tmp, **arrs)
            os.replace(tmp, key_path)
        except Exception:                        # noqa: BLE001 -- cache is optional
            pass

    return {"n": n_sizes,
            "pos": [torch.from_numpy(p) for p in pos_l],
            "gnbr": [torch.from_numpy(g) for g in gnbr_l],
            "lnbr": [torch.from_numpy(l) for l in lnbr_l],
            "up": [(torch.from_numpy(u), torch.from_numpy(w)) for u, w in up_l]}


def _hier_from_arrays(z, torch):
    n_sizes = [int(v) for v in z["n"]]
    L = len(n_sizes) - 1
    return {"n": n_sizes,
            "pos": [torch.from_numpy(z[f"pos{i}"]) for i in range(L)],
            "gnbr": [torch.from_numpy(z[f"gnbr{i}"]) for i in range(L)],
            "lnbr": [torch.from_numpy(z[f"lnbr{i}"]) for i in range(L)],
            "up": [(torch.from_numpy(z[f"up{i}"]),
                    torch.from_numpy(z[f"upw{i}"])) for i in range(L)]}


def collate(hiers, nbr0s, device="cpu"):
    """Merge per-graph hierarchies into ONE disconnected batch structure.

    Index arrays are offset per level (a level-l index of graph g points into the
    concatenated level-l tensor), widened to int64 AFTER the (cheaper) int32 host
    copy. seg = per-point graph id at the COARSEST level, for the segment-wise
    global max-pool -- the only op that could mix graphs, made per-graph so the
    chunked-forward == fused-forward gradient identity holds for hierpoint too.
    """
    torch = _require_torch()
    L = len(hiers[0]["pos"])
    offs = [0] * (L + 1)                      # running per-level point offsets
    pos = [[] for _ in range(L)]
    gnbr = [[] for _ in range(L)]
    lnbr = [[] for _ in range(L)]
    up_i = [[] for _ in range(L)]
    up_w = [[] for _ in range(L)]
    nbr0, seg = [], []
    for gi, (h, nb0) in enumerate(zip(hiers, nbr0s)):
        nbr0.append(nb0 + offs[0])
        for li in range(L):
            pos[li].append(h["pos"][li])
            gnbr[li].append(h["gnbr"][li].to(torch.int64) + offs[li])
            lnbr[li].append(h["lnbr"][li].to(torch.int64) + offs[li + 1])
            ui, uw = h["up"][li]
            up_i[li].append(ui.to(torch.int64) + offs[li + 1])
            up_w[li].append(uw)
        seg.append(torch.full((h["n"][L],), gi, dtype=torch.int64))
        for li in range(L + 1):
            offs[li] += h["n"][li]
    out = {
        "nbr0": torch.cat(nbr0, dim=0).to(device),
        "pos": [torch.cat(p, dim=0).to(device) for p in pos],
        "gnbr": [torch.cat(g, dim=0).to(device) for g in gnbr],
        "lnbr": [torch.cat(l, dim=0).to(device) for l in lnbr],
        "up": [(torch.cat(ui, dim=0).to(device), torch.cat(uw, dim=0).to(device))
               for ui, uw in zip(up_i, up_w)],
        "seg": torch.cat(seg, dim=0).to(device),
        "n_graphs": len(hiers),
    }
    return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def build_hierpoint_regressor(config=None):
    """Build the hierarchical point U-Net regressor (C_out=7).
    Returns (model, meta). forward(x, hier) with hier from collate()."""
    torch = _require_torch()
    import torch.nn as nn

    cfg = {**HP_DEFAULTS, **(config or {})}
    widths = tuple(int(w) for w in cfg["widths"])
    ratios = tuple(int(r) for r in cfg["ratios"])
    k_levels = tuple(int(k) for k in cfg["k_levels"])
    layers = tuple(int(x) for x in cfg["layers"])
    group_k = int(cfg["group_k"]); up_k = int(cfg["up_k"])
    k0 = int(cfg["k"]); dropout = float(cfg.get("dropout", 0.0))
    use_normals = bool(cfg.get("normals", True))
    use_curvature = bool(cfg.get("curvature", True)) and use_normals
    use_concavity = bool(cfg.get("concavity", True)) and use_normals
    use_edge_dist = bool(cfg.get("edge_dist", True)) and use_normals
    c_in = 3
    if use_normals:    c_in += 3
    if use_curvature:  c_in += 1
    if use_concavity:  c_in += 1
    if use_edge_dist:  c_in += 1
    L = len(ratios)
    assert len(widths) == L + 1 and len(k_levels) == L and len(layers) == L + 1

    class LeanEdgeConv(nn.Module):
        """EdgeConv with the first edge Linear decomposed:
        Linear([h_i, h_j-h_i]) == W_a h_i + W_b (h_j - h_i) + bias, so only ONE
        (N,k,C) edge tensor is materialised (relu input); aggregation is max,
        then a per-VERTEX LN->Linear->LN. The stock EdgeConv retains ~6
        (N,k,C..2C) tensors for backward -- the memory cliff that forced the
        7000-vert cap; this keeps ~1-2."""
        def __init__(self, c):
            super().__init__()
            self.lin_a = nn.Linear(c, c)
            self.lin_b = nn.Linear(c, c, bias=False)
            self.post = nn.Sequential(nn.LayerNorm(c), nn.ReLU(),
                                      nn.Linear(c, c), nn.LayerNorm(c))

        def forward(self, h, nbr):
            a = self.lin_a(h)                                # (N,C)
            b = self.lin_b(h)                                # (N,C)
            e = b[nbr] - b.unsqueeze(1) + a.unsqueeze(1)     # (N,k,C) -- the one
            e = torch.relu(e).amax(dim=1)                    # big edge tensor
            return self.post(e)

    class SABlock(nn.Module):
        """Set-abstraction downsample: for each coarse centre, max over the
        group_k nearest FINER-level points of MLP(feature, relative position).
        Decomposed like LeanEdgeConv: per-point lin_h is computed at the fine
        level then gathered, only the (M,k,C_out) sum is materialised."""
        def __init__(self, c_in_, c_out):
            super().__init__()
            self.lin_h = nn.Linear(c_in_, c_out)
            self.lin_rel = nn.Linear(3, c_out, bias=False)
            self.post = nn.Sequential(nn.LayerNorm(c_out), nn.ReLU(),
                                      nn.Linear(c_out, c_out), nn.LayerNorm(c_out))

        def forward(self, h_fine, pos_fine, pos_coarse, gnbr):
            rel = pos_fine[gnbr] - pos_coarse.unsqueeze(1)   # (M,k,3)
            e = self.lin_h(h_fine)[gnbr] + self.lin_rel(rel)  # (M,k,C_out)
            e = torch.relu(e).amax(dim=1)                    # (M,C_out)
            return self.post(e)

    class FPBlock(nn.Module):
        """Feature propagation: 3-NN inverse-distance upsample of the coarse
        features + skip concat + pointwise MLP. All (N_fine, C) -- no k dim."""
        def __init__(self, c_coarse, c_skip, c_out):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(c_coarse + c_skip, c_out),
                                     nn.LayerNorm(c_out), nn.ReLU(),
                                     nn.Linear(c_out, c_out), nn.LayerNorm(c_out))

        def forward(self, h_coarse, h_skip, up_idx, up_w):
            up = (h_coarse[up_idx] * up_w.unsqueeze(-1)).sum(dim=1)
            return self.mlp(torch.cat([up, h_skip], dim=-1))

    class HierPointNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Linear(c_in, widths[0]), nn.ReLU())
            self.l0 = nn.ModuleList([LeanEdgeConv(widths[0])
                                     for _ in range(layers[0])])
            self.sa = nn.ModuleList([SABlock(widths[i], widths[i + 1])
                                     for i in range(L)])
            self.levels = nn.ModuleList([
                nn.ModuleList([LeanEdgeConv(widths[i + 1])
                               for _ in range(layers[i + 1])])
                for i in range(L)])
            # per-graph global context at the coarsest level: each point sees a
            # summary of ITS OWN graph (segment max-pool -- graphs never mix)
            self.gmlp = nn.Sequential(nn.Linear(2 * widths[L], widths[L]),
                                      nn.ReLU(), nn.Linear(widths[L], widths[L]),
                                      nn.LayerNorm(widths[L]))
            self.fp = nn.ModuleList([
                FPBlock(widths[i + 1], widths[i], widths[i]) for i in range(L)])
            self.refine = LeanEdgeConv(widths[0])
            drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.head = nn.Sequential(drop, nn.Linear(widths[0], N_CHANNELS))

        def forward(self, x, hier):
            pos_prev = x[:, :3]
            h = self.stem(x)
            for blk in self.l0:
                h = h + blk(h, hier["nbr0"])
            skips = [h]
            for i in range(L):
                h = self.sa[i](h, pos_prev, hier["pos"][i], hier["gnbr"][i])
                for blk in self.levels[i]:
                    h = h + blk(h, hier["lnbr"][i])
                pos_prev = hier["pos"][i]
                if i < L - 1:
                    skips.append(h)
            # segment-wise global max-pool (differentiable scatter_reduce; the
            # gradient flows to the argmax element, like torch.max)
            seg = hier["seg"]
            g = h.new_full((hier["n_graphs"], h.shape[1]), float("-inf"))
            g = g.scatter_reduce(0, seg.unsqueeze(-1).expand_as(h), h,
                                 reduce="amax", include_self=True)
            h = self.gmlp(torch.cat([h, g[seg]], dim=-1))
            for i in reversed(range(L)):
                up_idx, up_w = hier["up"][i]
                h = self.fp[i](h, skips[i], up_idx, up_w)
            h = h + self.refine(h, hier["nbr0"])
            return self.head(h)

    _feats = (["xyz"] + (["normal"] if use_normals else []) +
              (["curv"] if use_curvature else []) +
              (["concav"] if use_concavity else []) +
              (["edgedist"] if use_edge_dist else []))
    meta = {"backbone": "hierpoint",
            "input_features": "_".join(_feats), "c_in": c_in,
            "normals": use_normals, "curvature": use_curvature,
            "concavity": use_concavity, "edge_dist": use_edge_dist,
            "widths": list(widths), "ratios": list(ratios),
            "k_levels": list(k_levels), "layers": list(layers),
            "group_k": group_k, "up_k": up_k, "k": k0, "dropout": dropout}
    model = HierPointNet()
    n_par = sum(p.numel() for p in model.parameters())
    meta["n_params"] = int(n_par)
    return model, meta


def hier_params_from_meta(meta):
    """The build_hierarchy() kwargs a checkpoint's meta implies -- single source
    of truth for train prep AND inference so the pooling structure matches."""
    return {"ratios": tuple(meta.get("ratios", HP_DEFAULTS["ratios"])),
            "k_levels": tuple(meta.get("k_levels", HP_DEFAULTS["k_levels"])),
            "group_k": int(meta.get("group_k", HP_DEFAULTS["group_k"])),
            "up_k": int(meta.get("up_k", HP_DEFAULTS["up_k"]))}
