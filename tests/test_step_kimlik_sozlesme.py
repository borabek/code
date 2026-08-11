# -*- coding: utf-8 -*-
"""STEP adlandirma sozlesmeleri -- DataSet 6 SESSIZ COKUSUNE karsi bekci.

2026-08-05: DataSet 6'nin 1656 STEP'i JSON sozlesmesinde geldi. Eski `step_kimlik`
hepsini TEK anahtara ("ElectricalTerminal_ElectricalTerminal") indirgiyordu; `eligible()`
STEP sozlugunu bu anahtarla kurdugu icin 1655 dosya sessizce eziliyor ve korpus 4405'te
kaliyordu. HICBIR hata verilmedi -- yalnizca sayinin oynamamasi ele verdi.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from korpus_kimlik import kimlik, step_kimlik


def test_indirme_sozlesmesi_zaman_damgali():
    assert step_kimlik("wscaduniverse_3271055_2026-08-04-13-25-02.stp") == "3271055"


def test_indirme_sozlesmesi_pid_alt_cizgili():
    assert step_kimlik("wscaduniverse_KUT16_GY_2026-08-04-13-25-02.stp") == "KUT16_GY"


def test_json_sozlesmesi_duz():
    assert step_kimlik("EFX.561142_ElectricalTerminal_ElectricalTerminal.stp") == "561142"


def test_json_sozlesmesi_pid_alt_cizgili():
    assert step_kimlik("UTL.UUT-6S-HV-PP-GY_ElectricalTerminal_ElectricalTerminal.stp") \
        == "UUT-6S-HV-PP-GY"


def test_json_sozlesmesi_uretici_alt_cizgili():
    assert step_kimlik("A-B_N.1492-H4_ElectricalTerminal_ElectricalTerminal.stp") == "1492-H4"


def test_step_ve_json_kimligi_AYNI_pid_verir():
    """Eslesmenin kosulu: iki taraf ayni anahtar uzayina dusmeli."""
    for ad in ("EFX.561142", "UTL.UUT-6S-HV-PP-GY", "ELMEX.KUT16_GY", "A-B_N.1492-H4"):
        j = ad + "_ElectricalTerminal_ElectricalTerminal.json"
        s = ad + "_ElectricalTerminal_ElectricalTerminal.stp"
        assert step_kimlik(s) == kimlik(j)[1], ad


def test_JSON_sozlesmeli_stepler_TEK_anahtara_dusMEZ():
    """Asil cokus: hepsi ayni anahtara indirgeniyordu."""
    adlar = [f"MFG.{p}_ElectricalTerminal_ElectricalTerminal.stp" for p in ("A1", "A2", "A3")]
    assert len({step_kimlik(a) for a in adlar}) == 3
