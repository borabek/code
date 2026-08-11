# -*- coding: utf-8 -*-
"""KISMI etiketli dogrulamada metrik MASKELI olmali.

2026-08-07: dogrulama kumesi 20 -> 190 parcaya cikarilirken kismi etiketli parcalara
gecildi ve Conn-IoU 0.5878 -> 0.0995'e COKTU. Sebep maskesiz IoU: kismi etikette
yalniz CableEntry isaretli, modelin DOGRU tahmin ettigi Contact bolgeleri YANLIS
POZITIF sayiliyordu. Egitim kaybi maskeli oldugu icin metrik egitim hedefiyle
CELISIYORDU -- boyle bir sinyalle secim, Contact'i AZ tahmin edeni odullendirirdi.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _src():
    return io.open("train_seg_extra.py", encoding="utf-8").read()


def test_miou_kismi_ornekte_maskeli():
    s = _src()
    i = s.index("def miou"); j = s.index("\ndef ", i)
    g = s[i:j]
    assert 'd.get("partial_ce")' in g, "miou kismi ornegi ayirt etmiyor"
    assert "m = gc | pc" in g, "maske kurulmuyor"


def test_val_partial_bayragi_var_ve_isleniyor():
    s = _src()
    assert "--val-partial" in s, "bayrak yok"
    assert 'for s in va: s["partial_ce"] = True' in s, \
        "bayrak val orneklerine ISLENMIYOR -- maske hic devreye girmez"


def test_tam_etiketli_davranis_DEGISMEZ():
    """Tam etiketli orneklerde eski yol aynen korunmali (tez kiyasi bozulmasin)."""
    s = _src()
    i = s.index("def miou"); j = s.index("\ndef ", i)
    g = s[i:j]
    assert "for c in range(NCLS)" in g, "tam etiketli dal kaldirilmis"


def test_kismi_valda_secim_sinyali_EGITIM_HEDEFI():
    """Metrik ICAT ETME: kismi val'de secim, egitim kaybinin AYNISI olmali.

    Ilk duzeltme yetmedi -- Conn-IoU'nun birlesimi zaten (pc|gc) oldugu icin maske
    orada hicbir sey degistirmiyordu; acc %6'ya dustu cunku isaret CE, hedef CE+CT.
    """
    s = _src()
    i = s.index("def miou"); j = s.index("\ndef ", i)
    g = s[i:j]
    assert "kismi_kayip" in g, "maskeli BCE biriktirilmiyor"
    assert "NCLS_CE] + sm2[:, NCLS_CT]" in g, "hedef kanal CE+CT degil"
    assert "-np.mean(kismi_kayip)" in g, "secim sinyali negatif kayip olarak donmuyor"
