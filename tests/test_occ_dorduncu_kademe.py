# -*- coding: utf-8 -*-
"""DORDUNCU MESH KADEMESI (OCC BRepMesh) -- 2026-08-05, DataSet 6.

Uc gmsh kademesi de ayni yerde kiriliyordu: OCC->gmsh ithalinde TEL/ILMEK topolojisi
("Could not fix wire in surface 139", "The 1D mesh seems not to be forming a closed loop").
gmsh yuzeyi 2B parametre uzayinda mesh'lemek icin KAPALI tel ister; BRepMesh yuzeyi
dogrudan ucgenler ve bozuk tele tahammul eder.

KALIBRASYON (iki yolun da calistigi 4 parcada olculdu): bbox BIREBIR ayni, hacim farki
%0.01-0.3, ikisi de su gecirmez. Yani ayni KATI, yalnizca daha kaba tessellasyon --
ve `thesis_remesh.remesh_uniform(target=6000)` bunu zaten normalize ediyor.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer_step_cp as I


def test_kaynakla_ayni_konumdaki_tepeleri_birlestirir():
    """Yuz-yuz tessellasyon DIKISSIZ gelir: her yuzde ayri tepe. Kaynak bunu kapatir."""
    V = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],      # ucgen A
                  [0., 0, 0], [1, 0, 0], [0, 0, 1]])     # ucgen B, iki tepesi ORTAK
    F = np.array([[0, 1, 2], [3, 4, 5]])
    V2, F2 = I._kaynakla(V, F)
    assert len(V2) == 4, "6 tepe -> 4 tekil tepe olmali"
    assert len(F2) == 2, "iki ucgen de korunmali"


def test_kaynakla_cokmus_ucgeni_atar():
    """Kaynaktan sonra iki kosesi ayni olan ucgen gecersizdir."""
    V = np.array([[0., 0, 0], [1e-9, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2], [0, 2, 3]])
    _, F2 = I._kaynakla(V, F)
    assert len(F2) == 1, "cokmus ucgen atilmali"


def test_tamlik_esigi_OCC_yolunda_da_gecerli():
    """Yarim kabuk korpusa GIRMEZ -- gmsh yolundaki disiplin OCC'de de var."""
    import inspect
    src = inspect.getsource(I._occ_mesh)
    assert "MESH_TAMLIK" in src, "OCC kademesi tamlik kontrolunu ATLAMAMALI"
    assert "YARIM MESH" in src


def test_kademe_sirasi_OCC_EN_SON():
    """OCC yalnizca UC gmsh kademesi de dustugunde calisir -- calisan parcalarin
    mesh'i ve TUM onbellekler aynen korunmali."""
    import inspect
    src = inspect.getsource(I.step_to_mesh)
    i1 = src.index("heal=False")
    i2 = src.index("heal=True)")
    i3 = src.index("tol=1e-2")
    # rindex: OCC adi iki yerde gecer -- istege bagli OCC-ONCE rotasi (en ustte, varsayilan
    # kapali) ve YEDEK zincirin son kademesi. Burada YEDEK zincirdeki sira sinaniyor.
    i4 = src.rindex("_occ_mesh")
    assert i1 < i2 < i3 < i4, "yedek zincirde OCC kademesi EN SONDA olmali"


@pytest.mark.skipif(not os.path.exists("all_wscad_stp"), reason="STEP korpusu yok")
def test_gercek_parcada_OCC_gmsh_ile_AYNI_KATIYI_verir():
    """Kalibrasyon testi: iki yol da calisan bir parcada bbox ve hacim ortusmeli."""
    import glob
    import trimesh
    from korpus_kimlik import step_kimlik as SK
    S = {SK(s): s for s in glob.glob("all_wscad_stp/*.stp")}
    if "3024407" not in S:
        pytest.skip("kalibrasyon parcasi yok")
    V1, F1 = I._gmsh_mesh(S["3024407"], heal=False)
    V2, F2 = I._occ_mesh(S["3024407"])
    assert np.allclose(V1.max(0) - V1.min(0), V2.max(0) - V2.min(0), atol=0.05)
    m1 = trimesh.Trimesh(V1, F1, process=True)
    m2 = trimesh.Trimesh(V2, F2, process=True)
    assert m1.is_watertight and m2.is_watertight
    assert abs(m1.volume - m2.volume) / m1.volume < 0.01, "hacim %1'den fazla sapmamali"


def test_occ_once_VARSAYILAN_KAPALI():
    """Dondurulmus kumelerin mesh KAYNAGI degismemeli: rota acikca istenmedikce kapali."""
    import inspect
    import os as _os
    src = inspect.getsource(I.step_to_mesh)
    assert 'MESH_OCC_ONCE' in src
    assert _os.environ.get("MESH_OCC_ONCE") != "1", "test ortaminda rota KAPALI olmali"


def test_occ_once_ACIKKEN_OCC_ONCE_denenir(monkeypatch):
    cagri = []
    monkeypatch.setattr(I, "_occ_mesh", lambda p: cagri.append("occ") or ("V", "F"))
    monkeypatch.setattr(I, "_gmsh_mesh", lambda *a, **k: cagri.append("gmsh") or ("V", "F"))
    monkeypatch.setenv("MESH_OCC_ONCE", "1")
    I.step_to_mesh("x.stp")
    assert cagri == ["occ"], "rota acikken OCC ONCE calismali, gmsh hic cagrilmamali"


def test_occ_once_KAPALIYKEN_gmsh_once(monkeypatch):
    cagri = []
    monkeypatch.setattr(I, "_occ_mesh", lambda p: cagri.append("occ") or ("V", "F"))
    monkeypatch.setattr(I, "_gmsh_mesh", lambda *a, **k: cagri.append("gmsh") or ("V", "F"))
    monkeypatch.delenv("MESH_OCC_ONCE", raising=False)
    I.step_to_mesh("x.stp")
    assert cagri == ["gmsh"], "varsayilanda gmsh once calismali"


def test_kaynakla_sifir_alanli_ucgeni_atar():
    """Dogrusal koseli ucgen NaN normal uretir -- OCC yolunda temizlenmeli."""
    V = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0],   # UCU DE ayni dogruda
                  [0., 1, 0]])
    F = np.array([[0, 1, 2], [0, 1, 3]])
    V2, F2 = I._kaynakla(V, F)
    assert len(F2) == 1, "sifir alanli ucgen atilmali, saglam olan kalmali"
    # kaynak indisleri YENIDEN SIRALAR -- eski indisler beklenmez, NOKTALAR sinanir
    kalan = {tuple(p) for p in V2[F2[0]]}
    assert kalan == {(0., 0, 0), (1., 0, 0), (0., 1, 0)}
