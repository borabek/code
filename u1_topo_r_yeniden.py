# -*- coding: utf-8 -*-
"""U1: gate egitim verisinin 4 TOPOLOJI sutununu BASKA BIR YARICAPLA yeniden uret.

NEDEN UCUZ: topoloji sutunlari YALNIZ mesh'ten ve aday noktasindan hesaplanir. Aday noktalari
(`pts`/`dirs`) ve parca kimlikleri (`pids`) npz'de ZATEN sakli; mesh de diskte onbellekli
(results/mesh_cache/<pid>.npz = remesh SONRASI V,F). Yani ag cikarimi, aday turetme ve etiketleme
hic tekrarlanmaz -- sadece 4 sutun yeniden hesaplanir. Tam gate_regrow ~80 dk, bu ~10 dk.

KANIT SART (kendi kendini dogrulayan tasarim): once R=6.0 ile yeniden uretip npz'deki MEVCUT
sutunlarla karsilastiriyorum. Eger yeniden uretim yolu (mesh kaynagi, kenar yapisi, cagri sirasi)
gate_regrow'unkiyle ayni degilse bu test PATLAR. Ayni cikarsa R=8/12 sutunlari da guvenilir.
Bu kontrol olmadan "yeni R daha iyi" demek, farkli bir MESH yolunun etkisini R'ye yazmak olurdu
([[tek-turetme-proxy-gecersiz]] dersinin ayni bicimi).

KULLANIM:
    python u1_topo_r_yeniden.py --dogrula          # yalniz R=6 tekrar-uretim testi
    python u1_topo_r_yeniden.py --r 12.0           # yeni npz uret
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BA_ALLOW_SEEN", "1")

NPZ = "results/gate_regrow_data_topo.npz"
MESH_CACHE = "results/mesh_cache"
TOPO_SUT = slice(18, 22)      # 13 taban + 5 fiziksel = 18, sonraki 4 topoloji


def _mesh(pid, stp=None):
    """Remesh SONRASI V,F. Onbellekte varsa oradan; yoksa STEP'ten uretip onbellege yaz."""
    f = os.path.join(MESH_CACHE, pid + ".npz")
    if os.path.exists(f):
        d = np.load(f)
        return np.ascontiguousarray(d["V"], np.float64), np.ascontiguousarray(d["F"], np.int64)
    if not stp:
        return None, None
    import thesis_remesh
    from infer_step_cp import step_to_mesh
    Vr, Fr = step_to_mesh(stp)
    V, F = thesis_remesh.remesh_uniform(Vr, Fr, target=6000)
    V = np.ascontiguousarray(V, np.float64); F = np.ascontiguousarray(F, np.int64)
    os.makedirs(MESH_CACHE, exist_ok=True)
    np.savez_compressed(f, V=V.astype(np.float32), F=F.astype(np.int32))
    return V, F


def hesapla(R, sinir=0, ilerleme=True):
    """Her aday icin 4 topoloji sutununu R yaricapiyla hesapla. (X_topo, islenen_pid_maskesi)"""
    import topo_feats

    d = np.load(NPZ, allow_pickle=True)
    pids = np.asarray(d["pids"]).astype(str)
    pts = np.asarray(d["pts"], float)
    dirs = np.asarray(d["dirs"], float)
    stp_of = {}
    try:
        from big_arbiter import eligible
        stp_of = {p: s for m, p, jf, s in eligible()}
    except Exception as e:
        print(f"  (uyari: STEP haritasi yok -> {type(e).__name__}; yalniz onbellek kullanilir)")

    benzersiz = list(dict.fromkeys(pids.tolist()))
    if sinir:
        benzersiz = benzersiz[:sinir]
    out = np.zeros((len(pids), 4), float)
    tamam = np.zeros(len(pids), bool)
    t0 = time.time(); atlanan = 0
    for k, pid in enumerate(benzersiz, 1):
        if ilerleme and k % 100 == 0:
            print(f"  {k}/{len(benzersiz)} parca | {time.time()-t0:.0f}s | atlanan {atlanan}",
                  flush=True)
        idx = np.where(pids == pid)[0]
        try:
            V, F = _mesh(pid, stp_of.get(pid))
            if V is None:
                atlanan += 1
                continue
            onb = topo_feats.icbukey_kenarlar(V, F)          # R'den BAGIMSIZ, parca basina 1 kez
            for i in idx:
                out[i] = topo_feats.topo_ozellik(V, F, pts[i], dirs[i], R=R, onbellek=onb)
            tamam[idx] = True
        except Exception as e:
            atlanan += 1
            if atlanan <= 3:
                print(f"  atlandi {pid}: {type(e).__name__}: {e}")
    print(f"  bitti: {len(benzersiz)-atlanan}/{len(benzersiz)} parca | {time.time()-t0:.0f}s")
    return out, tamam


def dogrula(sinir=60):
    """R=6.0 ile yeniden uret ve npz'deki mevcut sutunlarla karsilastir."""
    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float)
    eski = X[:, TOPO_SUT]
    yeni, tamam = hesapla(6.0, sinir=sinir)
    if not tamam.any():
        print("HIC parca islenemedi -- yeniden uretim yolu KULLANILAMAZ"); return False
    e, y = eski[tamam], yeni[tamam]
    fark = np.abs(e - y)
    tam = float((fark.max(1) < 1e-6).mean())
    yakin = float((fark.max(1) < 1e-3).mean())
    print(f"\n  karsilastirilan satir: {tamam.sum()} ({len(np.unique(np.asarray(d['pids'])[tamam]))} parca)")
    print(f"  BIREBIR ayni (<1e-6): {tam:.4f}")
    print(f"  yakin      (<1e-3): {yakin:.4f}")
    for j, nm in enumerate(["kon_oran", "kon_sayi", "kon_aci", "kon_cevre"]):
        print(f"    {nm:10s} maks fark {fark[:, j].max():.6g} | ort {fark[:, j].mean():.6g}")
    ok = tam > 0.99
    print(f"\n  KARAR: {'YOL DOGRULANDI' if ok else 'YOL FARKLI -- R sonuclari GUVENILMEZ'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dogrula", action="store_true")
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--sinir", type=int, default=0)
    a = ap.parse_args()
    if a.dogrula:
        dogrula(a.sinir or 60); return
    assert a.r > 0, "--r ver"
    X_topo, tamam = hesapla(a.r, sinir=a.sinir)
    d = np.load(NPZ, allow_pickle=True)
    X = np.asarray(d["X"], float).copy()
    # ISLENEMEYEN parcalarin sutunlarini ESKISIYLE birak: karsilastirmayi R farkina odaklar,
    # kayip parca etkisini karistirmaz.
    X[tamam, TOPO_SUT] = X_topo[tamam]
    out = f"results/gate_regrow_data_topo_r{a.r:g}.npz".replace(".npz", ".npz")
    kayit = {k: d[k] for k in d.files}
    kayit["X"] = X
    np.savez_compressed(out, **kayit)
    print(f"\nyazildi -> {out} | guncellenen satir {int(tamam.sum())}/{len(X)}")


if __name__ == "__main__":
    main()
