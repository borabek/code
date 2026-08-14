# -*- coding: utf-8 -*-
"""Olcum betikleri urunun TAM zincirini kosmali.

2026-08-07'de was found: butun D6/D7 olcumleri `derive_candidates` + gate with duruyordu;
urunun real yolu gate'ten after `pose_correct` + `pick_direction_from_dictionary` de kosuyor.
D6'da measured: 0.2103 -> 0.2251 -> 0.2469.
"""
import io 
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
import product_chain 


def test_bos_girdi_bos_doner ():
    P =np .zeros ((0 ,3 ));D =np .zeros ((0 ,3 ))
    P2 ,D2 =product_chain .tam_poz (None ,None ,None ,P ,D )
    assert len (P2 )==0 and len (D2 )==0 


def test_adim_duserse_GIRDI_aynen_doner ():
    """Uydurma YOK: a step patlarsa input degismeden geri gelmeli."""
    P =np .array ([[1. ,2 ,3 ]]);D =np .array ([[0. ,0 ,1 ]])
    P2 ,D2 =product_chain .tam_poz (None ,None ,None ,P ,D )# feats_for patlar
    assert np .allclose (P2 ,P )and np .allclose (D2 ,D )


def test_zincir_sirasi_pose_sonra_yon ():
    import inspect 
    src =inspect .getsource (product_chain .tam_poz )
    assert src .index ("pose_correct")<src .index ("pick_direction_from_dictionary"),"direction sozlugu, poz duzeltmesinin CIKTISI uzerine calismali"
