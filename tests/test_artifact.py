# -*- coding: utf-8 -*-
"""ARTIFACT BUTUNLUGU -- makbuzdaki model ile DAGITILAN model ayni mi?

2026-08-01 DENETIM BULGUSU: `cp_config.current_product.wire_gate.md5` BAYATTI (config
669ba207..., gercek dosya eb49d070...) ve hicbir sey onu kontrol etmiyordu. Yani "su model
dagitildi" iddiasi DENETLENMEMIS bir iddiaydi; model sessizce degistirilebilir ve makbuz
eski modeli isaret etmeye devam edebilirdi.
"""
import hashlib
import json
import os
import sys

import pytest

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cfg():
    with open(os.path.join(KOK, "cp_config.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("alan", ["wire_gate", "highcp_router"])
def test_dagitilan_artifact_makbuzla_AYNI(alan):
    blok = _cfg().get("current_product", {}).get(alan)
    if not isinstance(blok, dict) or "md5" not in blok:
        pytest.skip(f"{alan} icin md5 damgasi yok")
    yol = os.path.join(KOK, blok.get("yol", f"results/{alan}.pkl"))
    if not os.path.exists(yol):
        pytest.skip(f"{yol} yok")
    assert _md5(yol) == blok["md5"], (
        f"{alan}: DAGITILAN DOSYA makbuzdaki damgayla UYUSMUYOR. "
        f"config {blok['md5'][:8]}... vs gercek {_md5(yol)[:8]}... "
        f"Model degistiyse makbuz da guncellenmelidir (manset.py --yaz / dagit betigi).")


def test_gate_hatasi_SESSIZ_gecmez():
    """Gate patlarsa urun HAM BIRLESIMI (kesinlik ~0.40) sessizce dondurmemeli."""
    with open(os.path.join(KOK, "robot_cp.py"), encoding="utf-8") as f:
        s = f.read()
    assert "_GATE_HATA" in s, "gate hatasi kaydedilmiyor"
    assert "wire-gate atlandi" not in s, "eski SESSIZ atlama mesaji hala var"
    assert 'c.get("_gate_hata")' in s, "gate hatali CP'ler REVIEW'a dusurulmuyor"
    assert "sys.exit(3)" in s, "gate hatasi cikis koduna yansimiyor"


def test_manset_gate_kimligi_ARTEFAKTLA_AYNI():
    """headline_F1.gate, DAGITILAN dosyayi anlatmali.

    2026-08-02 DENETIMI: bu alan ELLE yazilmisti ve "22 sutun + cokus yonlendirme (44 sutun)"
    diyordu; oysa dagitilan gate 116 sutundu ve yonlendirme GERI ALINMISTI. Yani urunun
    kimligini anlatan tek metin, urunu anlatmiyordu. Alan artik manset.py --yaz ile
    ARTEFAKTTAN uretiliyor; bu test bayatlarsa yakalar.
    """
    import pickle
    h = _cfg().get("current_product", {}).get("headline_F1", {})
    g = h.get("gate")
    if not isinstance(g, dict) or "md5" not in g:
        pytest.skip("gate kimligi henuz uretilmemis (manset.py --yaz)")
    yol = os.path.join(KOK, g.get("dosya", "results/wire_gate.pkl"))
    if not os.path.exists(yol):
        pytest.skip(f"{yol} yok")
    assert _md5(yol) == g["md5"], (
        f"manset gate kimligi BAYAT: config {g['md5'][:8]}... vs dosya {_md5(yol)[:8]}...")
    with open(yol, "rb") as f:
        d = pickle.load(f)
    assert int(g["n_feat"]) == int(d["n_feat"]), (
        f"sutun sayisi uyusmuyor: kimlik {g['n_feat']} vs artefakt {d['n_feat']}")
    assert g.get("donusum") == d.get("donusum"), "donusum uyusmuyor"
