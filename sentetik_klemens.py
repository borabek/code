# -*- coding: utf-8 -*-
"""D1 — SENTETIK KLEMENS URETECI

NEDEN. Duvar temsilde: NIT tipi yogun parcada konum AUC 0.7053, gereken
0.944 (`results/konum_auc_d6.json`). Havuz kucultme kapandi (en iyi 1.33x,
hedef 5x), HPO tukendi, hedef-fonksiyonu kollari kapiyi gecemedi. Geriye
tek yol kaldi: modelin "hangi aciklik kablo girisi" sorusunu OGRENMESI --
yani VERI.

Gercek veri stogu bitti ([[wscad-data-lever-dead]], D8 kurulamaz). Ama
klemens GEOMETRISI parametriktir ve sentetik uretilebilir:
  * GT INSAATTAN gelir -> etiket hatasi SIFIR
  * marka kavrami tanimaz -> gorulmemis marka kosuluna dogal uyum
  * yogun aile (asil duvar) istenildigi kadar uretilebilir

NEGATIF YAPILAR SART. Yalniz giris ureten bir korpus, "hangi aciklik giris"
sorusunu ogretmez -- her aciklik giris olur. Bu yuzden her parcaya kasitli
CELDIRICI konur: montaj deligi, ray yuvasi, test noktasi, havalandirma
yarigi. Bunlar GT'ye GIRMEZ. Modelin ogrenmesi gereken ayrim tam budur.

ORNEKLEME. Gercek korpusun olculen dagilimina yakin tutulur; uydurma bir
dagilim sentetik veriyi gercekten uzaklastirir.
"""
import numpy as np
import trimesh

# --- gercek korpustan olculen araliklar (klemens ailesi)
KUTUP = (2, 30)             # kutup sayisi
ADIM = (3.5, 16.0)          # kutuplar arasi adim (mm)
GIRIS_CAP = (1.6, 8.0)      # kablo girisi capi (mm)
GOVDE_H = (12.0, 60.0)      # govde yuksekligi (mm)
GOVDE_D = (8.0, 45.0)       # govde derinligi (mm)
GIRIS_DERIN = (4.0, 18.0)   # giris kanal derinligi (mm)
EGIM = (0.0, 35.0)          # giris ekseninin yuzey normalinden sapmasi


def _rng(tohum):
    return np.random.default_rng(tohum)


def _silindir(yaricap, boy, merkez, eksen):
    T = trimesh.geometry.align_vectors([0, 0, 1], eksen)
    T[:3, 3] = merkez
    return trimesh.creation.cylinder(radius=yaricap, height=boy,
                                     transform=T, sections=24)


def uret(tohum=0):
    """bir sentetik klemens uretir.

    doner: (mesh, G, Gd, kunye)
      G  : kablo girisi agiz merkezleri (n,3)
      Gd : DISARI bakan eksenler (n,3)   -- GT sozlesmesi 1.000 disari
    """
    r = _rng(tohum)
    n_kutup = int(r.integers(*KUTUP))
    adim = float(r.uniform(*ADIM))
    cap = float(r.uniform(*GIRIS_CAP))
    # cap adimi asamaz; gercek klemenste giris adimin ~%60'ini gecmez
    cap = min(cap, 0.6 * adim)
    h = float(r.uniform(*GOVDE_H))
    dr = float(r.uniform(*GOVDE_D))
    derin = float(r.uniform(*GIRIS_DERIN))
    derin = min(derin, 0.45 * dr)
    genis = n_kutup * adim
    govde = trimesh.creation.box(extents=(genis, dr, h))

    cift_sira = bool(r.random() < 0.65)      # giris/cikis karsit yuzlerde
    egim = np.radians(float(r.uniform(*EGIM)))
    yon_egim = float(r.uniform(0, 2 * np.pi))

    kesiciler, G, Gd = [], [], []
    yuzler = [(+1, np.array([0.0, 1.0, 0.0]))]
    if cift_sira:
        yuzler.append((-1, np.array([0.0, -1.0, 0.0])))
    for isaret, n_dis in yuzler:
        # egimli eksen: yuzey normalinden EGIM kadar sapmis
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        eks = (np.cos(egim) * n_dis
               + np.sin(egim) * (np.cos(yon_egim) * e1
                                 + np.sin(yon_egim) * e2))
        eks = eks / np.linalg.norm(eks)
        for k in range(n_kutup):
            x = -genis / 2 + adim * (k + 0.5)
            z = float(r.uniform(-0.15, 0.15)) * h
            agiz = np.array([x, isaret * dr / 2, z])
            kesiciler.append(_silindir(cap / 2, 2 * derin,
                                       agiz - eks * (derin * 0.5), eks))
            # HAVSA (huni agiz). Ilk surumde duz silindirik delik vardi ve
            # segmentasyon modeli sentetik parcada GT agizlarina HIC
            # yerellesmiyordu (agizda ortanca 0.0001, zemin 0.0002 --
            # `results/sentetik_duman.json`). Gercek kablo girisleri teli
            # yonlendirmek icin HAVSALIDIR; modelin ogrendigi yerel imza
            # muhtemelen bu huni. Bu yuzden agiza koni eklenir.
            hv_h = min(0.35 * derin, 0.9 * cap)
            if hv_h > 0.2:
                T = trimesh.geometry.align_vectors([0, 0, 1], eks)
                T[:3, 3] = agiz - eks * (hv_h * 0.5)
                koni = trimesh.creation.cone(radius=cap * 0.9, height=hv_h,
                                             sections=24, transform=T)
                # koni tabani disarida olacak sekilde cevir
                T2 = trimesh.geometry.align_vectors([0, 0, 1], -eks)
                T2[:3, 3] = agiz - eks * (hv_h * 0.5)
                kesiciler.append(trimesh.creation.cone(
                    radius=cap * 0.9, height=hv_h, sections=24,
                    transform=T2))
                del koni, T
            G.append(agiz)
            Gd.append(eks)
            # SIKMA VIDASI: girise yaklasik dik, ustten
            if r.random() < 0.8:
                v_eks = np.array([0.0, 0.0, 1.0])
                v_cap = cap * float(r.uniform(0.7, 1.1))
                v_der = float(r.uniform(0.3, 0.6)) * h
                kesiciler.append(_silindir(
                    v_cap / 2, 2 * v_der,
                    np.array([x, isaret * dr * 0.25, h / 2 - v_der * 0.5]),
                    v_eks))

    # --- IC KAMARA. Gercek klemens ici bos bir kabuktur; giris kanali bir
    # kamaraya acilir. Duz dolu govde, modelin ogrendigi "kanal -> bosluk"
    # imzasini tasimaz. Kamara govdeden daha kucuk ve iceride durur.
    if r.random() < 0.85:
        kw = float(r.uniform(0.35, 0.6)) * dr
        kh = float(r.uniform(0.30, 0.55)) * h
        kamara = trimesh.creation.box(extents=(genis * 0.92, kw, kh))
        kamara.apply_translation([0.0, 0.0, float(r.uniform(-0.1, 0.2)) * h])
        kesiciler.append(kamara)

    # --- CELDIRICILER (GT'ye GIRMEZ)
    # Bunlar olmadan korpus "her aciklik giristir" ogretir; ogrenilmesi
    # gereken ayrim tam da bunlarla giris arasindadir.
    for _ in range(int(r.integers(1, 4))):        # montaj deligi (uctan)
        x = float(r.choice([-genis / 2 + adim * 0.4, genis / 2 - adim * 0.4]))
        kesiciler.append(_silindir(
            float(r.uniform(1.2, 3.0)), 2 * h,
            np.array([x, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])))
    if r.random() < 0.7:                          # ray yuvasi (alt)
        yw = float(r.uniform(0.25, 0.5)) * dr
        yh = float(r.uniform(0.10, 0.25)) * h
        yv = trimesh.creation.box(extents=(genis * 1.1, yw, yh))
        yv.apply_translation([0, 0, -h / 2 + yh / 2])
        kesiciler.append(yv)
    for _ in range(int(r.integers(0, 3))):        # test noktasi (ust)
        x = float(r.uniform(-genis / 2 + adim, genis / 2 - adim))
        kesiciler.append(_silindir(
            float(r.uniform(0.5, 1.2)), h * 0.5,
            np.array([x, 0.0, h / 2 - h * 0.15]), np.array([0.0, 0.0, 1.0])))

    parca = govde
    for c in kesiciler:
        try:
            parca = parca.difference(c)
        except Exception:                          # noqa: BLE001
            continue
    G = np.asarray(G, float).reshape(-1, 3)
    Gd = np.asarray(Gd, float).reshape(-1, 3)
    kunye = {"tohum": tohum, "kutup": n_kutup, "adim": adim, "cap": cap,
             "govde": [genis, dr, h], "derin": derin,
             "cift_sira": cift_sira, "egim_derece": float(np.degrees(egim)),
             "gt": len(G)}
    return parca, G, Gd, kunye


def duman_testi(n=6):
    """uretecin saglamligi: su gecirmez mi, GT agizda mi, celdirici GT'de mi."""
    ok = True
    for t in range(n):
        m, G, Gd, k = uret(t)
        k_cap = k["cap"]
        su = bool(m.is_watertight)
        # her GT, yuzeye YAKIN olmali (agiz yuzeydedir).
        # ESIK DUZELTMESI: ilk yazimda |d| <= 1.5 mm istemistim ve 6'nin
        # 4'u "kaldi" verdi. Kusur uretecte DEGIL iddiada: agiz merkezi
        # deligin BOSLUGUNDA durur, en yakin duvar CAP/2 kadar uzaktir
        # (8 mm'lik girişte 4 mm). Dogru esik cap/2 + pay.
        d = trimesh.proximity.ProximityQuery(m).signed_distance(G)
        yuzeyde = bool(np.all(np.abs(d) <= k_cap / 2.0 + 0.6))
        birim = bool(np.allclose(np.linalg.norm(Gd, axis=1), 1.0, atol=1e-6))
        print(f"  t={t} kutup={k['kutup']:>2} GT={k['gt']:>2} "
              f"cift={int(k['cift_sira'])} egim={k['egim_derece']:.0f} "
              f"| su gecirmez {int(su)} | GT yuzeyde {int(yuzeyde)} "
              f"| yon birim {int(birim)} | ucgen {len(m.faces)}")
        ok &= su and yuzeyde and birim
    print("DUMAN TESTI:", "GECTI" if ok else "KALDI")
    return ok


if __name__ == "__main__":
    duman_testi()
