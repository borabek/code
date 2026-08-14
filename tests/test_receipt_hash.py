"""P0-c: damga each makbuza gitmeli and DEGISIKLIGI yakalamali."""
import io ,json ,os 
import receipt_hash 


def test_damga_alanlari ():
    d =receipt_hash .damga ()
    for k in ("kod","model","config","python"):
        assert k in d 
    assert d ["kod"],"kod hash'leri bos"


def test_urun_zinciri_hash_var ():
    d =receipt_hash .damga ()
    assert "robot_cp.py"in d ["kod"]and "wire_gate.py"in d ["kod"]


def test_degisiklik_YAKALANIR (tmp_path ):
    f =tmp_path /"x.txt"
    f .write_text ("a",encoding ="utf-8")
    h1 =receipt_hash ._h (str (f ))
    f .write_text ("b",encoding ="utf-8")
    assert receipt_hash ._h (str (f ))!=h1 ,"icerik degisti but hash same"


def test_olmayan_dosya_None ():
    assert receipt_hash ._h ("yok_boyle_bir_dosya_xyz")is None 
