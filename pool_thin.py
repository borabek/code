# -*- coding: utf-8 -*-
"""MESH ADAYI SEYRELTME -- training de urun de BU fonksiyonu cagirir.

MEASURED (D6, 468 part; criterion YALNIZ KONUM recall'u / candidate-part):
    most high probability 60      0.6362 / 160
    most high probability 150     0.7163 / 214
    uzamsal 4.0mm, 2x120       0.8091 / 178
    uzamsal 2.5mm, 4x250       0.8713 / 251   <- SECILEN (maliyet array)
    uzamsal 2.0mm, sinirsiz    0.9768 / 458

KAPSAMA GUVENI YENIYOR: segmentasyon olasiligi most high vertices AYNI mouth
cevresinde kumeleniyor; "most iyi 60"i almak parcanin yarisini empty birakiyor.
Uzamsal seyreltme high olasilikli tepeden baslar and `r` yaricapinda bastirir.

Bu rule two places yazilirsa measurement urunden ayrisir (this projede two times became).
Tek source burasidir.
"""
import numpy as np 

R =2.5 # seyreltme yaricapi (mm)
TABAN =250 # at least this up to mesh adayi
KAT =4 # source 0+1 sayisinin kati up to da may be


def cap (n01 ,baseline =TABAN ,fold =KAT ):
    """Parca karmasikligiyla olceklenen upper boundary."""
    return max (int (baseline ),int (fold )*int (n01 ))


def seyrelt (P2 ,s2 ,n01 ,r =R ,baseline =TABAN ,fold =KAT ):
    """Mesh adaylarindan tutulacaklarin INDEKSLERI.

    `P2` mesh candidate konumlari, `s2` each birinin segmentasyon olasiligi,
    `n01` mesh DISI candidate count (seg + B-rep).
    """
    P2 =np .asarray (P2 ,float ).reshape (-1 ,3 )
    if not len (P2 ):
        return np .zeros (0 ,int )
    s2 =np .asarray (s2 ,float ).reshape (-1 )
    ust =cap (n01 ,baseline ,fold )
    sec =[]
    for j in np .argsort (-s2 ):
        if len (sec )>=ust :
            break 
        if sec and float (np .min (np .linalg .norm (P2 [sec ]-P2 [j ],axis =1 )))<r :
            continue 
        sec .append (int (j ))
    return np .asarray (sec ,int )


def ppos (avg_probs ,CE ,CT ):
    """Tepe basina 'CP olma' olasiligi -- havuzun own olcutuyle AYNI."""
    a =np .asarray (avg_probs ,float )
    return a [:,int (CE )]+a [:,int (CT )]
