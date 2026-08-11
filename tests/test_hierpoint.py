# Unit tests for the hierpoint backbone (hierpoint.py + cp_regressor dispatch).
# Mirrors the knngraph safety net in tests/test_cp.py: batched-vs-solo parity,
# structure invariants, registry round-trip, and the sign-policy regression.
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import cp_regressor as cpr           # noqa: E402
import hierpoint as hp               # noqa: E402


def _graph(n, seed):
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(n, 3)).astype(np.float64)
    nbr = cpr._knn_graph(V, 16)
    hier = hp.build_hierarchy(V)
    x = cpr._knn_feature_tensor(V, nbr, True, "cpu", True, True, True)
    return V, nbr, hier, x


@pytest.fixture(scope="module")
def model_meta():
    torch.manual_seed(0)
    model, meta = hp.build_hierpoint_regressor()
    model.eval()          # dropout off: parity tests need deterministic forwards
    return model, meta


@pytest.fixture(scope="module")
def graphs():
    return _graph(500, 0), _graph(233, 1)


def test_hierarchy_invariants(graphs):
    for V, nbr, hier, x in graphs:
        n = hier["n"]
        assert n[0] == len(V)
        assert all(n[i + 1] <= n[i] for i in range(3)), "levels must shrink"
        for li in range(3):
            assert hier["pos"][li].shape == (n[li + 1], 3)
            assert int(hier["gnbr"][li].max()) < n[li]
            assert int(hier["lnbr"][li].max()) < n[li + 1]
            ui, uw = hier["up"][li]
            assert ui.shape == (n[li], hp.HP_DEFAULTS["up_k"])
            assert int(ui.max()) < n[li + 1]
            assert torch.allclose(uw.sum(1), torch.ones(n[li]), atol=1e-5), \
                "upsample weights must sum to 1 per fine vertex"


def test_hierarchy_deterministic():
    V = np.random.default_rng(7).normal(size=(300, 3))
    h1 = hp.build_hierarchy(V)
    h2 = hp.build_hierarchy(V)
    for li in range(3):
        assert torch.equal(h1["pos"][li], h2["pos"][li])
        assert torch.equal(h1["gnbr"][li], h2["gnbr"][li])


def test_batched_forward_backward_parity(model_meta, graphs):
    """2 graphs collated into one disconnected batch must give the SAME outputs
    and gradients as two solo forwards -- exercises the per-level index offsets
    AND the segment-wise global max-pool (the only op that could mix graphs)."""
    model, _ = model_meta
    g1, g2 = graphs
    out1 = model(g1[3], hp.collate([g1[2]], [g1[1]]))
    out2 = model(g2[3], hp.collate([g2[2]], [g2[1]]))
    X = torch.cat([g1[3], g2[3]], 0)
    H = hp.collate([g1[2], g2[2]], [g1[1], g2[1]])
    out_b = model(X, H)
    assert (out_b[:500] - out1).abs().max().item() < 1e-5
    assert (out_b[500:] - out2).abs().max().item() < 1e-5

    model.zero_grad()
    (out_b[:500].square().mean() + out_b[500:].square().mean()).backward()
    gb = {k: p.grad.clone() for k, p in model.named_parameters()
          if p.grad is not None}
    model.zero_grad()
    o1 = model(g1[3], hp.collate([g1[2]], [g1[1]]))
    o2 = model(g2[3], hp.collate([g2[2]], [g2[1]]))
    (o1.square().mean() + o2.square().mean()).backward()
    for k, p in model.named_parameters():
        if p.grad is not None:
            assert (gb[k] - p.grad).abs().max().item() < 1e-5, k
    model.zero_grad()


def test_registry_roundtrip(model_meta):
    """_meta_to_config('hierpoint', meta) must rebuild the identical architecture
    (the resume / --init-from / load_checkpoint path)."""
    model, meta = model_meta
    cfg = cpr._meta_to_config("hierpoint", meta)
    m2, meta2 = cpr.build_regressor("hierpoint", cfg)
    m2.eval()
    m2.load_state_dict(model.state_dict())    # raises on any shape mismatch
    assert meta2["c_in"] == meta["c_in"]
    assert meta2["widths"] == meta["widths"]
    g = _graph(120, 3)
    a = model(g[3], hp.collate([g[2]], [g[1]]))
    b = m2(g[3], hp.collate([g[2]], [g[1]]))
    assert torch.equal(a, b)


def test_output_head_shape(model_meta):
    model, meta = model_meta
    g = _graph(64, 5)
    out = model(g[3], hp.collate([g[2]], [g[1]]))
    assert out.shape == (64, cpr.N_CHANNELS)
    assert meta["c_in"] == g[3].shape[1] == 9


def test_tiny_graph_does_not_crash(model_meta):
    """Margin slivers / tiny selftest parts: levels clamp to >=8 points and k
    pads by repetition -- the forward must survive a 20-vertex graph."""
    model, _ = model_meta
    g = _graph(20, 9)
    out = model(g[3], hp.collate([g[2]], [g[1]]))
    assert out.shape == (20, cpr.N_CHANNELS)


def test_sign_inv_prefix_default_off():
    """Regression for the v19 angle collapse: with the new sign_inv_prefixes
    default (empty), a wscaduniverse part must NOT be trained sign-invariant.
    The old hard-coded gate silently made 79% of training graphs 180-deg
    ambiguous while the eval metric stayed signed (~100 deg angle error)."""
    import inspect
    src = inspect.getsource(cpr.train_knngraph_regressor)
    assert 'startswith("wscaduniverse")' not in src, \
        "hard-coded wscad sign-invariance gate is back"
    sig = inspect.signature(cpr.train_knngraph_regressor)
    assert sig.parameters["sign_inv_prefixes"].default == ""


def test_hier_disk_cache(tmp_path):
    V = np.random.default_rng(11).normal(size=(400, 3))
    h1 = hp.build_hierarchy(V, cache_dir=str(tmp_path))
    files = list(tmp_path.glob("hier_*.npz"))
    assert len(files) == 1
    h2 = hp.build_hierarchy(V, cache_dir=str(tmp_path))   # cache hit
    for li in range(3):
        assert torch.equal(h1["gnbr"][li], h2["gnbr"][li])
        ui1, uw1 = h1["up"][li]; ui2, uw2 = h2["up"][li]
        assert torch.equal(ui1, ui2)
        assert torch.allclose(uw1, uw2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_gpu_memory_ceiling():
    """WDDM tripwire: fwd+bwd at the target cap (14000) and the fused chunk
    (2x14000) must stay far below the ~3.7 GB spill cliff of the 4 GB T1200."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, _ = hp.build_hierpoint_regressor()
    model = model.cuda().train()
    gs = [_graph(14000, s) for s in (21, 22)]
    X = torch.cat([g[3] for g in gs], 0).cuda()
    H = hp.collate([g[2] for g in gs], [g[1] for g in gs], device="cuda")
    out = model(X, H)
    out.square().mean().backward()
    peak_mb = torch.cuda.max_memory_reserved() / (1 << 20)
    assert peak_mb < 3200, f"peak {peak_mb:.0f} MB -- too close to the WDDM cliff"
    del X, H, out, model
    torch.cuda.empty_cache()
