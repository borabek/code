# -*- coding: utf-8 -*-
"""B-rep genisletilmis havuz testleri. Makbuz: results/brep_filtre_taramasi.json."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import brep_havuz  # noqa: E402


def sil(c, a, b, eksen=(1.0, 0.0, 0.0), r=1.5):
    return {"axis": list(eksen), "radius": r, "center": list(c),
            "mouth_a": list(a), "mouth_b": list(b)}


def test_iki_agiz_iki_aday_ZIT_yonlu():
    P, D = brep_havuz.brep_adaylari([sil((0, 0, 0), (-5, 0, 0), (5, 0, 0))], [])
    assert len(P) == 2
    assert np.allclose(D[0], -D[1])


def test_sifir_eksen_ATLANIR_patlamaz():
    P, _ = brep_havuz.brep_adaylari(
        [{"axis": [0, 0, 0], "mouth_a": [0, 0, 0], "mouth_b": [1, 0, 0]}], [])
    assert len(P) == 0


def test_dedupe_ayni_agzi_teke_indirir():
    c = [sil((0, 0, 0), (0, 0, 0), (9, 0, 0)), sil((0, 0, 0), (1, 0, 0), (9, 0, 0))]
    assert len(brep_havuz.brep_adaylari(c, [], dedupe_mm=3.0)[0]) < 4


def test_segmentasyon_adaylari_ASLA_elenmez():
    """Tezin cevabi havuzda bozulmadan durmali."""
    Ps = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    Ds = np.array([[0.0, 0, 1], [0.0, 0, 1]])
    P, D, kay = brep_havuz.birlesik_havuz(
        Ps, Ds, [sil((0, 0, 0), (0, 0, 0), (0.5, 0, 0))], [])
    assert np.allclose(P[:2], Ps) and np.allclose(D[:2], Ds)
    assert list(kay[:2]) == [0, 0]


def test_seg_adayina_yakin_brep_onerisi_ATILIR():
    Ps = np.array([[0.0, 0, 0]]); Ds = np.array([[0.0, 0, 1]])
    P, _, kay = brep_havuz.birlesik_havuz(
        Ps, Ds, [sil((0, 0, 0), (0.5, 0, 0), (0.9, 0, 0))], [], dedupe_mm=3.0)
    assert len(P) == 1 and list(kay) == [0]


def test_aciklik_merkezi_normal_yonuyle_eklenir():
    P, D = brep_havuz.brep_adaylari([], [{"center": [3, 3, 3], "normal": [0, 0, 2]}])
    assert np.allclose(P[0], [3, 3, 3]) and np.allclose(D[0], [0, 0, 1])


def test_bos_girdi_bos_havuz():
    P, D, kay = brep_havuz.birlesik_havuz(
        np.zeros((0, 3)), np.zeros((0, 3)), [], [])
    assert len(P) == 0 and len(D) == 0 and len(kay) == 0


def test_kaynak_isaretleri_dogru_sayida():
    Ps = np.array([[0.0, 0, 0]]); Ds = np.array([[1.0, 0, 0]])
    P, _, kay = brep_havuz.birlesik_havuz(
        Ps, Ds, [sil((50, 0, 0), (45, 0, 0), (55, 0, 0))], [])
    assert len(kay) == len(P) and kay.sum() == len(P) - 1


def test_meta_siralamasi_P_ile_BIREBIR():
    """Tanimlayicilar meta'dan okunacak; siralama kaymasi sessiz hata olurdu."""
    c = [sil((0, 0, 0), (-9, 0, 0), (9, 0, 0), r=1.5),
         sil((0, 40, 0), (-9, 40, 0), (9, 40, 0), r=2.5)]
    a = [{"center": [0, 80, 0], "normal": [0, 0, 1]}]
    P, D, met = brep_havuz.brep_adaylari(c, a, meta=True)
    assert len(met) == len(P) == len(D)
    for i, m in enumerate(met):
        if "radius" in m:
            assert np.allclose(P[i], m["mouth_a"]) or np.allclose(P[i], m["mouth_b"])
        else:
            assert np.allclose(P[i], m["center"])


def test_meta_kapaliyken_eski_imza_KORUNUR():
    out = brep_havuz.brep_adaylari([sil((0, 0, 0), (-9, 0, 0), (9, 0, 0))], [])
    assert len(out) == 2


def test_meta_bos_girdide_uclu_doner():
    P, D, met = brep_havuz.brep_adaylari([], [], meta=True)
    assert len(P) == 0 and len(D) == 0 and met == []


def _kutu_mesh():
    import trimesh
    m = trimesh.creation.box((20, 20, 20))
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def test_mesh_adayi_esigi_gecmeyeni_ALMAZ():
    V, F = _kutu_mesh()
    P, D = brep_havuz.mesh_adaylari(V, F, np.zeros(len(V)), esik=0.5)
    assert len(P) == 0 and len(D) == 0


def test_mesh_adayi_normali_BIRIM():
    V, F = _kutu_mesh()
    P, D = brep_havuz.mesh_adaylari(V, F, np.ones(len(V)), esik=0.5,
                                    dedupe_mm=0.0)
    assert len(P) == len(V)
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)


def test_mesh_seyreltme_aday_sayisini_DUSURUR():
    V, F = _kutu_mesh()
    p = np.ones(len(V))
    a = len(brep_havuz.mesh_adaylari(V, F, p, esik=0.5, dedupe_mm=0.0)[0])
    b = len(brep_havuz.mesh_adaylari(V, F, p, esik=0.5, dedupe_mm=25.0)[0])
    assert b < a


def test_tam_havuz_kaynak_isaretleri_0_1_2():
    V, F = _kutu_mesh()
    Ps = np.array([[100.0, 0, 0]])
    Ds = np.array([[1.0, 0, 0]])
    c = [sil((50, 0, 0), (45, 0, 0), (55, 0, 0))]
    P, D, kay = brep_havuz.tam_havuz(Ps, Ds, c, [], V=V, F=F,
                                     ppos=np.ones(len(V)))
    assert set(np.unique(kay).tolist()) <= {0, 1, 2}
    assert kay[0] == 0 and np.allclose(P[0], Ps[0])
    assert (kay == 2).sum() > 0


def test_tam_havuz_mesh_verilmezse_ESKI_davranis():
    Ps = np.array([[0.0, 0, 0]])
    Ds = np.array([[0.0, 0, 1]])
    c = [sil((0, 0, 0), (-9, 0, 0), (9, 0, 0))]
    a = brep_havuz.birlesik_havuz(Ps, Ds, c, [])
    b = brep_havuz.tam_havuz(Ps, Ds, c, [])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
