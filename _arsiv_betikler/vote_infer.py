# -*- coding: utf-8 -*-
"""OYLAMA CIKARIMI + hakem skoru: oy -> kume -> CP, mevcut urunle AYNI sette karsilastir.

Oylama basligi (train_vote.py) her baglanti-verteksinden ait oldugu instance MERKEZINE bir offset
tahmin eder. Burada: oy = verteks + offset; oylari yaricap-kumelemesi ile grupla; her kume BIR CP.
Bu, elle-ayarli post-proc'un (min_v 30 / cluster_mm 5) yerini alir -- FBI teshisi model-FN'lerinin
%63'unde modelin ZATEN atesledigini, kaybin turetmede oldugunu gostermisti.

Kume filtresi: bir kumeyi CP saymak icin en az --min-votes oy. (min_v'nin ogrenilmis muadili:
kucuk-ama-tutarli bir fragment bile oylari AYNI merkeze yollarsa CP olur -- min_v 30'un attigi durum.)

Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe vote_infer.py \
    --ckpt results/seg_extra/vote_s2.pt --only-mfg WEI --only-parts $(cat _hw_r3.txt)
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusionnet as D, thesis_remesh, connector3d
from cad_eval import align_frames
from infer_step_cp import step_to_mesh, load_any
from big_arbiter import greedy, eligible, OP

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
NCLS = 5


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def raw_forward(model, meta, V, F, device):
    """(V, 5+3) HAM cikti. D.predict KULLANILAMAZ: o exp()/softmax'i TUM kanallara uygular ve
    argmax'i 8 kanal uzerinden alir -> offset kanallarini bozar ve sinif etiketini yanlislar.
    Oylama modeli cfg loss='ce' ile kuruldugu icin son aktivasyon yok: ilk 5 ham logit, son 3 offset."""
    ops = D.precompute_operators(V, F, meta["k_eig"], OP)
    ops = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in ops.items()}
    with torch.no_grad():
        out = D._forward(model, ops, D._model_input(ops, meta)).cpu().numpy()
    del ops
    D.free_gpu_memory()
    return np.asarray(out, float)


def cluster_votes(votes, weights, radius, min_votes):
    """Oylari yaricap-kumelemesiyle grupla (agirlikca azalan cekirdek secimi)."""
    if not len(votes):
        return np.zeros((0, 3)), np.zeros(0)
    order = np.argsort(-weights)
    centers = []; counts = []
    assigned = np.zeros(len(votes), bool)
    for i in order:
        if assigned[i]:
            continue
        d = np.linalg.norm(votes - votes[i], axis=1)
        member = (d <= radius) & (~assigned)
        if member.sum() < min_votes:
            assigned[i] = True          # cekirdek tek basina kaliyorsa at
            continue
        assigned |= member
        w = weights[member]
        centers.append((votes[member] * w[:, None]).sum(0) / max(w.sum(), 1e-9))
        counts.append(int(member.sum()))
    return (np.array(centers, float) if centers else np.zeros((0, 3)),
            np.array(counts, float) if counts else np.zeros(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--only-mfg", default="WEI")
    ap.add_argument("--only-parts", nargs="+", default=[])
    ap.add_argument("--vertex-conf", type=float, default=0.5, help="oy verecek verteks icin min baglanti olasiligi")
    ap.add_argument("--radius", type=float, default=4.0, help="oy kumeleme yaricapi (mm)")
    ap.add_argument("--min-votes", nargs="+", type=int, default=[3, 5, 8, 12, 20, 30],
                    help="kume basina min oy -- taranir (min_v'nin ogrenilmis muadili)")
    ap.add_argument("--remesh-target", type=int, default=6000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    model, meta, _ = load_any(a.ckpt, dev=a.device)
    os.environ["BA_ALLOW_SEEN"] = "1"
    raw = eligible()
    if a.only_mfg: raw = [p for p in raw if p[0] == a.only_mfg]
    if a.only_parts:
        keep = set(a.only_parts); raw = [p for p in raw if p[1] in keep]
    print(f"{len(raw)} parca | {os.path.basename(a.ckpt)} | oylama cikarimi (r={a.radius}mm)", flush=True)

    cache = []; t0 = time.time()
    for k, (mfg, pid, jf, stp) in enumerate(raw, 1):
        try:
            j = json.load(open(jf, encoding="utf-8-sig"))
            Vj = np.array([[p["X"], p["Y"], p["Z"]] for p in j["Graphic3d"]["Points"]], float)
            G = np.array([[c["Point"]["X"], c["Point"]["Y"], c["Point"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            Gd = np.array([[c["InsertDirection"]["X"], c["InsertDirection"]["Y"], c["InsertDirection"]["Z"]] for c in j.get("ConnectionPoints", [])], float)
            if not len(G): continue
            Gd = Gd / (np.linalg.norm(Gd, axis=1, keepdims=True) + 1e-9)
            Vr, Fr = step_to_mesh(stp)
            V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=a.remesh_target)
            V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
            out = raw_forward(model, meta, V, F, a.device)   # HAM cikti sart (asagiya bak)
            probs = _softmax(out[:, :NCLS])
            conn_p = probs[:, CE] + probs[:, CT]
            sel = conn_p >= a.vertex_conf
            R, t, _ = align_frames(Vr, Vj)
            tol = max(3.0, 0.06 * float(np.linalg.norm(Vj.max(0) - Vj.min(0))))
            cache.append({"V": V, "off": out[:, NCLS:], "sel": sel, "w": conn_p,
                          "R": R, "t": t, "G": G, "Gd": Gd, "tol": tol})
        except Exception:
            continue
        if k % 25 == 0: print(f"  {k}/{len(raw)}  {time.time()-t0:.0f}s", flush=True)

    print(f"\n=== OYLAMA SONUCU ({len(cache)} parca) ===")
    print(f"{'min_oy':>7s} | {'P':>6s} {'R':>6s} {'F1':>6s}")
    for mv in a.min_votes:
        T = Fp = Fn = 0
        for c in cache:
            V = c["V"]; sel = c["sel"]
            votes = V[sel] + c["off"][sel]
            centers, _ = cluster_votes(votes, c["w"][sel], a.radius, mv)
            P = (centers @ c["R"].T + c["t"]) if len(centers) else np.zeros((0, 3))
            tp, fp, fn = greedy(P, c["G"], c["tol"], c["Gd"], 40.0)
            T += tp; Fp += fp; Fn += fn
        p = T/max(T+Fp,1); r = T/max(T+Fn,1); f = 2*p*r/max(p+r,1e-9)
        print(f"{mv:7d} | {p:6.3f} {r:6.3f} {f:6.3f}", flush=True)
    print("\n  karsilastirma: urun recall_hard_s2 WEI = P0.597 R0.547 F1 0.571")


if __name__ == "__main__":
    main()
