# -*- coding: utf-8 -*-
"""Urun ciktisindaki PARCA ADI correct must be -- duman testinde yakalandi.

`robot_cp.py` part kimligini `basename.split("_")[1]` with cikariyordu; this only
indirme sozlesmesinde (`wscaduniverse_<pid>_<ts>.stp`) correct. JSON sozlesmeli
dosyalarda (`UPUN.016029_ElectricalTerminal_...stp`) part adi "ElectricalTerminal"
cikiyordu -- i.e. MUSTERIYE TESLIM EDILEN JSON'da wrong name.
"""
import io 
import os 
import sys 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))


def test_robot_cp_step_kimlik_kullaniyor ():
    s =io .open ("robot_cp.py",encoding ="utf-8").read ()
    assert "from corpus_identity import step_kimlik"in s ,"tek source kullanilmiyor"
    assert 'basename(path).split("_")[1]'not in s ,"eski hatali ayristirma geri gelmis"
    assert 'basename(s).split("_")[1]'not in s 


def test_iki_sozlesmede_de_dogru_ad ():
    from corpus_identity import step_kimlik 
    assert step_kimlik ("wscaduniverse_1001.2_2026-08-04-13-25-02.stp")=="1001.2"
    assert step_kimlik ("UPUN.016029_ElectricalTerminal_ElectricalTerminal.stp")=="016029"


def test_glb_cizici_de_step_kimlik_kullaniyor ():
    """Duman testi: JSON sozlesmeli dosyada GLB HIC uretilmiyordu ('STEP absent')."""
    s =io .open ("export_robot_glb.py",encoding ="utf-8").read ()
    assert "from corpus_identity import step_kimlik"in s 
    assert 'basename(s).split("_")[1]'not in s ,"eski hatali ayristirma geri gelmis"
