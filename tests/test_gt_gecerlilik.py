# -*- coding: utf-8 -*-
"""GT GECERLILIK KAPISI (2026-08-04): puanlanamaz GT'li parca korpusa girmemeli.

AL vakasi: 26 parcanin InsertDirection'i HARFIYEN (0,0,0) ve tum CP'leri ayni noktada.
Urun onlarda "0.000" aliyordu -- model kotu oldugu icin degil, GT OLCULEMEZ oldugu icin.
"""
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _yaz(tmp, cps):
    p = os.path.join(tmp, "t.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"ConnectionPoints": cps}, f)
    return p


def _cp(x, y, z, dx, dy, dz):
    return {"Point": {"X": x, "Y": y, "Z": z},
            "InsertDirection": {"X": dx, "Y": dy, "Z": dz}}


def test_saglam_gt_gecer(tmp_path):
    from big_arbiter import gt_gecerli
    assert gt_gecerli(_yaz(str(tmp_path), [_cp(0, 0, 0, 0, 0, 1), _cp(5, 0, 0, 0, 0, 1)]))


def test_sifir_yon_reddedilir(tmp_path):
    """AL vakasinin BIRINCI yarisi."""
    from big_arbiter import gt_gecerli
    assert not gt_gecerli(_yaz(str(tmp_path), [_cp(0, 0, 0, 0, 0, 0), _cp(5, 0, 0, 0, 0, 0)]))


def test_cakisik_noktalar_reddedilir(tmp_path):
    """AL vakasinin IKINCI yarisi: 19 CP'nin hepsi ayni yerde -> tekil esleme imkansiz."""
    from big_arbiter import gt_gecerli
    assert not gt_gecerli(_yaz(str(tmp_path), [_cp(1, 2, 3, 0, 0, 1)] * 4))


def test_bos_ve_bozuk_reddedilir(tmp_path):
    from big_arbiter import gt_gecerli
    assert not gt_gecerli(_yaz(str(tmp_path), []))
    assert not gt_gecerli(os.path.join(str(tmp_path), "yok.json"))


def test_tek_cp_cakisik_sayilmaz(tmp_path):
    """Tek CP'li parca 'hepsi ayni noktada' sayilmamali -- yoksa gecerli parcalar dusar."""
    from big_arbiter import gt_gecerli
    assert gt_gecerli(_yaz(str(tmp_path), [_cp(1, 2, 3, 0, 0, 1)]))


def test_genis_except_yok():
    """Ilk surumde `except Exception` bir NameError'i yutmus ve korpusu 4432 -> 0 dusurmustu.
    O desen geri gelmesin."""
    import inspect
    from big_arbiter import gt_gecerli
    # YORUMLARI AT: docstring/yorumda "except Exception KULLANMA" yaziyor ve ilk surumde
    # test onu yakalayip yanlis yere ates etti. Yalnizca KOD satirlarina bakilir.
    kod = [x.split("#")[0] for x in inspect.getsource(gt_gecerli).splitlines()]
    kod = "\n".join(kod)
    assert "except Exception" not in kod, "genis except kendi yazim hatani veri bulgusu gosterir"
