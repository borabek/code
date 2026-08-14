# -*- coding: utf-8 -*-
"""C1: YON SOZLUGU + KAHIN SINAVI -- Rota C'nin ilk kapisi.

NEDEN ONCE BU: mevcut tespitle (0.7584) poz ve aci MUKEMMEL olsa robot da 0.7584 olurdu;
su an 0.5893. Yani tespite hic dokunmadan **0.169** pay duruyor -- 0.80 tespit hedefine
giden paydan (0.042) dort kat fazla. Robot metriginin kaderi burada.

NEDEN AYRIK, NEDEN REGRESYON DEGIL: R2'de "yanlis eksen" sinifini AUC 0.930 ile TANIDIK
ama surekli regresorle yonu yeniden kurunca robot HER ESIKTE dustu (-0.0256). Sinifi
tanimak duzeltebilmek degildir. Ayrik secim, regresyonun yapamadigini yapabilir: dogru
yon zaten FIZIKSEL olarak turetilebilen birkac adaydan BIRIYSE, is onu SECMEYE indirger.

SOZLUK KUCUK VE FIZIKSEL TUTULUR. Duzlemde 24 yonluk bir yelpaze koysaydim kahin
bedavaya kusursuz cikardi ve olcum anlamini yitirirdi. Her giris bir GEOMETRIK NESNEDEN
gelir:
    mevcut     dagitilan zincirin ciktisi (poz+aci+uye)
    ham        duzeltmesiz aday yonu
    uye_k      birlesimde ATILAN uye yonleri (uye_yonu_sec'in havuzu)
    brep       B-rep analitik silindir ekseni
    kanal      yerel kanal ekseni (ic noktalarin PCA'si)
    normal     yerel yuzey normali
    duzlem_A/B agiz duzlemindeki ANA ve YAN eksen (yarik yonu ve dikeyi)
Her biri ve ISARET ESLERI (+-) -> tipik 10-18 giris.

KILL (onceden yazildi): kahin robot < 0.70 ise Rota C KAPANIR, secici EGITILMEZ.
Gecerse ayrik secici egitilir (regresyon DEGIL).
"""
import io
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
os.environ["WG_ZENGIN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RMAX = 9.0
DMAX = 18.0


def _birim(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def yerel_yonler(V, p, d):
    """Mesh'ten turetilen yonler: kanal ekseni, yuzey normali, agiz duzlemi eksenleri."""
    out = {}
    rel = V - p
    t = rel @ d
    rad = np.linalg.norm(rel - t[:, None] * d, axis=1)
    ic = (rad <= RMAX) & (t <= 1.0) & (t >= -DMAX)
    if ic.sum() >= 12:
        Q = rel[ic] - rel[ic].mean(0)
        try:
            _, _, W = np.linalg.svd(Q, full_matrices=False)
            out["kanal"] = _birim(W[0])
        except Exception:
            pass
    yak = (rad <= RMAX) & (np.abs(t) <= 2.0)
    if yak.sum() >= 8:
        Q = rel[yak] - rel[yak].mean(0)
        try:
            _, _, W = np.linalg.svd(Q, full_matrices=False)
            # agiz duzlemindeki ANA ve YAN eksen (yarik yonu ve dikeyi)
            out["duzlem_A"] = _birim(W[0] - (W[0] @ d) * d)
            out["duzlem_B"] = _birim(W[1] - (W[1] @ d) * d)
            out["normal"] = _birim(W[2])
        except Exception:
            pass
    return {k: v for k, v in out.items() if v is not None}


def main():
    import gate_tezgah as T
    import olcum_kumesi
    import thesis_remesh
    import wire_gate
    from big_arbiter import eligible
    from infer_step_cp import step_to_mesh
    from sklearn.ensemble import RandomForestClassifier
    from gece_kilit import bekci

    bekci("c1")
    D = T.yukle()
    olcum_kumesi.rapor_bas(D["rap"])
    cfg = D["cfg"]
    stp = {p: s for m, p, jf, s in eligible()}
    X, y, pid, keep = D["X"], D["y"], D["pid"], D["keep"]
    Z = np.zeros((len(X), X.shape[1] * 2))
    for u in np.unique(pid):
        i = np.where(pid == u)[0]
        Z[i] = wire_gate.parca_ici(X[i], D["donusum"])
    gate = {"clf": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                          random_state=0).fit(Z[keep], y[keep]),
            "n_feat": Z.shape[1], "donusum": D["donusum"]}

    try:
        import brep_axes
    except Exception:
        brep_axes = None

    KOLLAR = ["mevcut", "ham", "uye", "brep", "kanal", "normal", "duzlem_A", "duzlem_B"]
    say = {k: 0 for k in KOLLAR}
    det, rob_mevcut, rob_kahin, rob_isar, gg = [], [], [], [], []
    boy = []
    t0 = time.time()
    from sina_kume import esle
    for kk, r in enumerate(D["DER"], 1):
        if kk % 25 == 0:
            print(f"  {kk}/{len(D['DER'])}  {time.time()-t0:.0f}s", flush=True)
        P = np.zeros((0, 3)); Pd = np.zeros((0, 3)); Pham = np.zeros((0, 3))
        SOZ = []
        if r["X"] is not None and r.get("XR") is not None:
            Xr = np.hstack([r["X"], r["XR"]])
            k = wire_gate.karar_maskesi(wire_gate.karar_skoru(gate, Xr))
            if k.any():
                P = r["P"][k].copy(); Pham = r["Pd"][k].copy(); Pd = Pham.copy()
                c = [{"point": P[i], "direction": Pd[i]} for i in range(len(P))]
                c = wire_gate.pose_duzelt(Xr[k], c)
                if cfg.get("robot_aci_secici"):
                    c = wire_gate.aci_duzelt(Xr[k], c)
                if cfg.get("robot_uye_secici") and r.get("UYE"):
                    c = wire_gate.uye_yonu_sec(Xr[k], c, r["UYE"])
                P = np.array([x["point"] for x in c], float)
                Pd = np.array([x["direction"] for x in c], float)
                # --- SOZLUK
                V = None
                try:
                    Vr, Fr = step_to_mesh(stp[r["pid"]])
                    V, _ = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
                    V = np.ascontiguousarray(V, float)
                except Exception:
                    V = None
                cyl = None
                if brep_axes is not None:
                    try:
                        cyl = brep_axes.cylinders(stp[r["pid"]])
                    except Exception:
                        cyl = None
                for i in range(len(P)):
                    d0 = _birim(Pd[i])
                    S = {"mevcut": d0, "ham": _birim(Pham[i])}
                    if r.get("UYE"):
                        n_u = 0
                        for lst in r["UYE"]:
                            for cc in (lst if isinstance(lst, (list, tuple)) else []):
                                if not isinstance(cc, dict):
                                    continue
                                pt = np.asarray(cc.get("point", P[i]), float)
                                dd = cc.get("direction")
                                if dd is None or np.linalg.norm(pt - P[i]) > 5.0:
                                    continue
                                b = _birim(dd)
                                if b is not None:
                                    S[f"uye{n_u}"] = b; n_u += 1
                                if n_u >= 4:
                                    break
                    if cyl is not None:
                        try:
                            ax = brep_axes.axis_at(P[i], d0, cyl)
                            if ax is not None:
                                S["brep"] = _birim(ax[0] if isinstance(ax, tuple) else ax)
                        except Exception:
                            pass
                    if V is not None and d0 is not None:
                        S.update(yerel_yonler(V, P[i], d0))
                    S = {a: b for a, b in S.items() if b is not None}
                    SOZ.append(S)
                    for a in S:
                        say[a.rstrip("0123456789") if a.startswith("uye") else a] = \
                            say.get(a.rstrip("0123456789") if a.startswith("uye") else a, 0) + 1
                    boy.append(len(S))
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        rj = "cok" if r["n"] >= 8 else "dusuk"
        det.append((rj,) + esle(P, Pd, G, Gd, r["diag"], 0.0, 180.0, True))
        rob_mevcut.append((rj,) + esle(P, Pd, G, Gd, r["diag"], 2.0, 10.0, False))
        # --- KAHIN: her aday icin sozlukten GT'ye EN YAKIN yonu sec
        Pk = Pd.copy()
        if len(P) and len(G) and SOZ:
            diff = P[:, None, :] - G[None, :, :]
            al = (diff * Gd[None, :, :]).sum(-1)
            pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
            pe = np.where(np.abs(al) > 40, np.inf, pe)
            for i in range(len(P)):
                b = int(np.argmin(pe[i])) if np.isfinite(pe[i]).any() else None
                if b is None:
                    continue
                en, ed = None, None
                for a, v in SOZ[i].items():
                    ang = abs(float(v @ Gd[b]))
                    if en is None or ang > en:
                        en, ed = ang, v
                if ed is not None:
                    Pk[i] = ed
        rob_kahin.append((rj,) + esle(P, Pk, G, Gd, r["diag"], 2.0, 10.0, False))
        rob_isar.append((rj,) + esle(P, Pk, G, Gd, r["diag"], 2.0, 10.0, False, isaretli=True))
        gg.append(r["geo"])

    print(f"\nsozluk boyutu: medyan {np.median(boy):.0f} | ort {np.mean(boy):.1f}")
    print("giris kapsami:", {k: v for k, v in sorted(say.items(), key=lambda x: -x[1])})
    t_ = T.f1w(det); rm = T.f1w(rob_mevcut); rk = T.f1w(rob_kahin); ri = T.f1w(rob_isar)
    print(f"\n{'olcu':<26}{'deger':>9}")
    print(f"{'tespit (degismez)':<26}{t_:>9.4f}")
    print(f"{'robot MEVCUT':<26}{rm:>9.4f}")
    print(f"{'robot KAHIN (sozluk)':<26}{rk:>9.4f}   fark {rk-rm:+.4f}")
    print(f"{'robot KAHIN isaretli':<26}{ri:>9.4f}")
    print(f"{'ust sinir (=tespit)':<26}{t_:>9.4f}")
    print(f"\nsozluk, yon boslugunun %{100*(rk-rm)/max(t_-rm,1e-9):.0f}'ini kapsiyor")
    gecti = rk >= 0.70
    print(f"\nKILL: kahin robot >= 0.70 -> {'GECTI, ayrik secici egitilir' if gecti else 'GECMEDI, Rota C KAPANIR'}")
    with io.open("results/c1_yon_sozlugu.json", "w", encoding="utf-8") as f:
        json.dump({"tespit": t_, "robot_mevcut": rm, "robot_kahin": rk,
                   "robot_kahin_isaretli": ri, "sozluk_medyan": float(np.median(boy)),
                   "kapsam": say, "gecti": bool(gecti)}, f, indent=1)
    print("makbuz -> results/c1_yon_sozlugu.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
