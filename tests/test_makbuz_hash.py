"""P0-c: damga her makbuza gitmeli ve DEGISIKLIGI yakalamali."""
import io, json, os
import makbuz_hash


def test_damga_alanlari():
    d = makbuz_hash.damga()
    for k in ("kod", "model", "config", "python"):
        assert k in d
    assert d["kod"], "kod hash'leri bos"


def test_urun_zinciri_hash_var():
    d = makbuz_hash.damga()
    assert "robot_cp.py" in d["kod"] and "wire_gate.py" in d["kod"]


def test_degisiklik_YAKALANIR(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("a", encoding="utf-8")
    h1 = makbuz_hash._h(str(f))
    f.write_text("b", encoding="utf-8")
    assert makbuz_hash._h(str(f)) != h1, "icerik degisti ama hash ayni"


def test_olmayan_dosya_None():
    assert makbuz_hash._h("yok_boyle_bir_dosya_xyz") is None
