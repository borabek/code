# -*- coding: utf-8 -*-
"""P6 OZNITELIK CIKARIMI: (konum x yon) secenek tablosu.

Dagitilan urun bir adayi TEK yonle puanliyor. Bu betik her adayi
`yon_bankasi.secenekler` ile cogaltir ve HER SECENEK icin oznitelik satiri uretir.

OZNITELIK BLOKLARI (toplam 92 sutun):
  A  58  havuz oznitelikleri (`_tam_oz`)      -- KONUM baglami, adayin kendi yonuyle
  B   9  agiz tanimlayicilari (`_tan_hizali`) -- KONUM baglami, adayin kendi yonuyle
  C  16  yon bankasi olculeri                 -- SECENEGE ozel (kaynak, destek, hiza)
  D   9  agiz tanimlayicilari YENIDEN         -- SECENEK YONUYLE hesaplanir

D blogu isin atislari gerektirir (`girme` / `erisim` / `girme_kenar`); asil maliyet
oradadir ve bir parcanin TUM secenekleri TEK cagrida toplu atilir.

NEDEN A ve B secenek yonuyle YENIDEN hesaplanmiyor: 58 sutunun cogu konumsal
(segmentasyon olasiligi, topoloji, komsuluk) ve yeniden uretimi parca basina
saniyeler suruyor. A/B konum baglamini, C/D yon kararini tasir. Bu ayrim ayni
zamanda modele "konum ne kadar iyi" ile "bu yon dogru mu" sorularini AYRI verir.

Kullanim:
    python kos_p6_oznitelik.py d6            # 468 parca
    python kos_p6_oznitelik.py tam           # 2583 egitim parcasi
    python kos_p6_oznitelik.py d7            # 835 (SINAV -- yalniz kapida)

Devam edilebilir: cikti parca basina bir npz, VAR OLAN ATLANIR (pid anahtarli --
indeks anahtarli devam bu projede daha once sessizce yanlis parcayi atlamisti).
"""
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
sys.path.insert(0, ".")

OZ = "results/_tam_oz"
TAN = "results/_tan_hizali"
CIK = "results/_p6_oz"

# on -> (silindir onbellegi, aciklik onbellegi, mesh onbellegi)
KUME = {
    "tam": ("results/_brepegit_silindirler.pkl", "results/_brepegit_acikliklar.pkl",
            "results/_p1_olasilik_brepegit"),
    "d6": ("results/_d6_silindirler.pkl", "results/_d6_acikliklar.pkl",
           "results/_p1_olasilik"),
    "d7": ("results/_d7_silindirler.pkl", "results/_d7_acikliklar.pkl",
           "results/_p1_olasilik_d7"),
}
KAYNAKLAR = (0, 1)          # havuz kaynagi: 0 = tez `v_o`, 1 = B-rep agzi


def main():
    on = sys.argv[1] if len(sys.argv) > 1 else "d6"
    if on not in KUME:
        sys.exit(f"bilinmeyen kume: {on}")
    sil_y, ack_y, ob = KUME[on]
    import trimesh

    import agiz_tanimlayici
    import urun_genis
    import yon_bankasi as YB

    os.makedirs(CIK, exist_ok=True)
    cy = pickle.load(open(sil_y, "rb"))
    ac = pickle.load(open(ack_y, "rb"))
    dosyalar = sorted(f for f in os.listdir(OZ)
                      if f.startswith(on + "_") and f.endswith(".npz"))
    print(f"{on}: {len(dosyalar)} parca | cikti {CIK}", flush=True)

    t0 = time.time()
    yazilan = atlanan = bos = 0
    for i, f in enumerate(dosyalar, 1):
        pid = f[len(on) + 1:-4]
        hedef = f"{CIK}/{on}_{pid}.npz"
        if os.path.exists(hedef):
            atlanan += 1
            continue
        z = np.load(f"{OZ}/{f}")
        kay = np.asarray(z["kaynak"], int)
        m = np.isin(kay, KAYNAKLAR)
        if int(m.sum()) < 2:
            bos += 1
            continue
        T = np.asarray(np.load(f"{TAN}/{f}")["T"], float)
        if len(T) != len(z["X"]):
            print(f"  ! {pid}: HIZALAMA BOZUK, atlandi", flush=True)
            bos += 1
            continue
        A = np.asarray(z["X"], float)[m]
        B = T[m]
        P = np.asarray(z["P"], float)[m]
        D = np.asarray(z["D"], float)[m]
        mf = f"{ob}/{pid}.npz"
        if not os.path.exists(mf):
            bos += 1
            continue
        zz = np.load(mf)
        V = np.ascontiguousarray(zz["V"], np.float64)
        Fc = np.ascontiguousarray(zz["F"], np.int64)
        cyl = cy.get(str(pid))
        idx, YD, C = YB.secenekler(P, D, cyl, V)
        if not len(idx):
            bos += 1
            continue
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        mesh = trimesh.Trimesh(V, Fc, process=False)
        # D blogu: agiz tanimlayicilari SECENEK YONUYLE
        Dblok = urun_genis.tanimlayici(P[idx], YD, cyl, mesh, diag)
        X = np.hstack([A[idx], B[idx], C, Dblok]).astype(np.float32)
        np.savez_compressed(hedef, X=X, idx=idx.astype(np.int32),
                            YD=YD.astype(np.float32), P=P.astype(np.float32),
                            D=D.astype(np.float32))
        yazilan += 1
        if i % 25 == 0:
            hz = (time.time() - t0) / max(yazilan, 1)
            print(f"  {i}/{len(dosyalar)}  yazilan {yazilan} atlanan {atlanan} "
                  f"bos {bos}  {hz:.2f} s/parca", flush=True)
    print(f"\nBITTI: yazilan {yazilan} | atlanan {atlanan} | bos {bos} | "
          f"{time.time() - t0:.0f} s", flush=True)
    print(f"sutun: 58 (A) + 9 (B) + {len(YB.OZ_AD)} (C) + "
          f"{len(agiz_tanimlayici.AD)} (D)")


if __name__ == "__main__":
    main()
