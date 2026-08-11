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
CIK = os.environ.get("P6_CIK", "results/_p6_oz")

# HAVUZ KAYNAKLARI. Onbellek uc kaynagi da tasiyor:
#   0 segmentasyon (tezin `v_o`'su)   ~15 aday/parca
#   1 B-rep agzi                      ~85
#   2 MESH TEPESI                    ~354   <- ADAY_YOK kovasinin cevabi burada
# Mesh tepeleri uctan uca UC kez ZARAR vermisti, ama o olcumlerde secici tek
# yonlu ve zayifti. Yon bankasi + ortak siralayici ile yeniden acilir; kararı
# LOMO verir. `P6_KAYNAK=012` ile acilir.
# `P6_MESH_MAX` mesh adaylarini SEGMENTASYON OLASILIGINA gore ust sinira indirir
# (0 = sinirsiz); boylece secenek sayisi patlamaz.
_ks = os.environ.get("P6_KAYNAK", "01")
KAYNAKLAR = tuple(int(c) for c in _ks)
MESH_MAX = int(os.environ.get("P6_MESH_MAX", "250"))   # taban ust sinir
MESH_KAT = int(os.environ.get("P6_MESH_KAT", "4"))     # kaynak0+1 sayisinin kati
MESH_R = float(os.environ.get("P6_MESH_R", "2.5"))     # uzamsal seyreltme (mm)

# on -> (silindir onbellegi, aciklik onbellegi, mesh onbellegi)
KUME = {
    "tam": ("results/_brepegit_silindirler.pkl", "results/_brepegit_acikliklar.pkl",
            "results/_p1_olasilik_brepegit"),
    "d6": ("results/_d6_silindirler.pkl", "results/_d6_acikliklar.pkl",
           "results/_p1_olasilik"),
    "d7": ("results/_d7_silindirler.pkl", "results/_d7_acikliklar.pkl",
           "results/_p1_olasilik_d7"),
}
# (KAYNAKLAR yukarida cevre degiskeninden kuruluyor. Burada IKINCI bir tanim
#  vardi ve onu SESSIZCE eziyordu: `P6_KAYNAK=012` verilmesine ragmen mesh
#  adaylari havuza girmiyordu ve hicbir hata cikmiyordu.)


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
    # PARCALI KOSU: `P6_SHARD=i/n` -> yalniz indeksi n'e bolumunden kalani i olan
    # dosyalar. Cikti parca basina tek dosya oldugu ve VAR OLAN ATLANDIGI icin
    # paylar birbirinin isini bozmaz; 16 cekirdegi kullanmanin en ucuz yolu.
    sh = os.environ.get("P6_SHARD")
    if sh:
        i_, n_ = (int(x) for x in sh.split("/"))
        dosyalar = [f for k, f in enumerate(dosyalar) if k % n_ == i_]
        print(f"  PAY {i_}/{n_}", flush=True)
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
        mf = f"{ob}/{pid}.npz"
        if not os.path.exists(mf):
            bos += 1
            continue
        zz = np.load(mf)
        V = np.ascontiguousarray(zz["V"], np.float64)
        Fc = np.ascontiguousarray(zz["F"], np.int64)
        if 2 in KAYNAKLAR and int((kay == 2).sum()) > 0:
            # MESH ADAYLARINI SEYRELT. Kaynak 0/1 ASLA elenmez.
            #
            # OLCULDU (D6, 468 parca, YALNIZ KONUM recall'u / aday-parca):
            #   en yuksek olasilik 60     0.6362 / 160
            #   en yuksek olasilik 150    0.7163 / 214
            #   uzamsal 4.0mm, 2x120      0.8091 / 178
            #   uzamsal 2.5mm, 4x250      0.8713 / 251   <- SECILEN (diz)
            #   uzamsal 2.0mm, sinirsiz   0.9768 / 458
            # Yani KAPSAMA, GUVENI yeniyor: olasiligi en yuksek tepeler ayni
            # agiz cevresinde kumeleniyor ve digerleri bos kaliyor. Seyreltme
            # yuksek olasilikli tepeden baslar, MESH_R yaricapinda bastirir.
            import connector3d

            import havuz_seyrelt
            pb = np.mean([np.asarray(q, float) for q in zz["pbs"]], axis=0)
            pp = havuz_seyrelt.ppos(pb, connector3d.CABLE_ENTRY,
                                    connector3d.CONTACT)
            i2 = np.where(kay == 2)[0]
            P2 = np.asarray(z["P"], float)[i2]
            s2 = (pp[np.argmin(np.linalg.norm(
                P2[:, None, :] - V[None, :, :], axis=-1), axis=1)]
                if len(P2) * len(V) < 6e7 else np.zeros(len(P2)))
            n01 = int(np.isin(kay, [k for k in KAYNAKLAR if k != 2]).sum())
            tut = np.zeros(len(i2), bool)
            tut[havuz_seyrelt.seyrelt(P2, s2, n01, MESH_R, MESH_MAX,
                                      MESH_KAT)] = True
            m2 = m.copy()
            m2[i2] = tut
            m = m2
        A = np.asarray(z["X"], float)[m]
        B = T[m]
        P = np.asarray(z["P"], float)[m]
        D = np.asarray(z["D"], float)[m]
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
        # `kaynak` DE YAZILIR: boylece TEK cikarimdan hem (0,1) hem (0,1,2)
        # kolu egitilebilir ve iki kolu ayri ayri cikarmak gerekmez.
        # ATOMIK YAZIM: gecici dosyaya yaz, sonra yerine tasi. Iki isci ayni
        # parcaya denk gelirse (yuk dengesizligi yuzunden yardimci isci
        # eklendiginde olur) yarim yazilmis npz kalmaz.
        gec = f"{hedef}.{os.getpid()}.tmp"
        np.savez_compressed(gec, X=X, idx=idx.astype(np.int32),
                            YD=YD.astype(np.float32), P=P.astype(np.float32),
                            D=D.astype(np.float32),
                            kaynak=kay[m].astype(np.int8))
        os.replace(gec + ".npz" if os.path.exists(gec + ".npz") else gec, hedef)
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
