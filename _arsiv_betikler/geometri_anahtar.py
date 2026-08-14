# -*- coding: utf-8 -*-
"""GEOMETRI ANAHTARI: ikiz parcalari saptayan imza -- URETICI BETIK (eksikti).

WHY YAZILDI: `results/_strict_geometry_keys.json` dosyasini **10 betik OKUYOR, none of them
YAZMIYOR** -- `_der_tam.pkl`'in provenance bosluguyla same state (turet.py that bosluk for
yazilmisti). Anahtar 1963 parcayi kapsiyor; corpus 4720'ye ciktigi for **2757 parcanin
anahtari YOK**.

BU WHY KRITIK: [[geometry-twin-leakage]] -- parcalarin %80'inin korpusta ikizi present.
Sizinti korumalari and `protocol.egitim_maskesi` GRUP uzerinden works; anahtari olmayan
part "own basina grup" sayilir. Yani new veride OLCUM KUMESININ IKIZI varsa egitime
girer and headline SESSIZCE SISER. Turetmeden ONCE kapatilmali.

ESKI FORMAT COZULDU but BIREBIR taklit EDILMIYOR:
    [8.1, 50.4, 71.8]|v12|f13|c19|p161|h5.2.10.2.0.0.0.0.0.0.0.0
    v = ceil(log2(vertex)) , f = ceil(log2(face)) , basta SIRALI boundary kutusu, probe histogram
Sinir kutusunda 0.1mm'lik deviation present (8.2 -> 8.1), i.e. old anahtar biraz FARKLI a mesh
surumunden uretilmis. Taklit etmek instead of YENI anahtar yazilir and **old GRUPLAMAYI
yeniden uretip uretmedigi SINANIR** (dogrulama below).

TASARIM: anahtar remesh gurultusune DAYANIKLI must be ([[robot-nondeterminism]]: pymeshlab
surecler arasi ~0.4mm oynuyor) but different parcalari AYIRMALI.
"""
import hashlib 
import io 
import json 
import math 
import os 
import sys 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

BBOX_KOVA =0.5 # mm -- remesh gurultusu ~0.4mm, kova ondan large must be
HACIM_KOVA =0.02 # doluluk orani kovasi


def key_ (V ,F ):
    """Rotasyon-bagimsiz, remesh-gurultusune dayanikli geometri imzasi."""
    V =np .asarray (V ,float )
    bb =np .sort (V .max (0 )-V .min (0 ))
    bbk =[round (float (x )/BBOX_KOVA )for x in bb ]
    nv ,nf =len (V ),len (np .asarray (F ))
    # DOLULUK: mesh hacmi / boundary kutusu hacmi -- same dis olcude different ic yapiyi separates
    try :
        import trimesh 
        m =trimesh .Trimesh (vertices =V ,faces =np .asarray (F ,np .int64 ),process =False )
        hac =abs (float (m .volume ))/max (float (np .prod (bb )),1e-9 )
    except Exception :
        hac =-1.0 
    hk =round (hac /HACIM_KOVA )if hac >=0 else -1 
    # MERKEZDEN UZAKLIK HISTOGRAMI (rotasyon-bagimsiz).
    # KABALASTIRILDI (first version 5/5 kacirilan ikizde SUCLUYDU): 12 kova + %1 yuvarlama
    # single kovada 1 PUANLIK tesselasyon gurultusuyle anahtari degistiriyordu
    # (h0.4.7... vs h0.5.7...). 8 kova + %5 step, gurultuye dayanikli.
    c =V .mean (0 )
    r =np .linalg .norm (V -c ,axis =1 )
    r =r /max (r .max (),1e-9 )
    h ,_ =np .histogram (r ,bins =8 ,range =(0 ,1 ))
    h =(np .round (100 *h /max (len (r ),1 )/5 )*5 ).astype (int )
    # TEPE/YUZ SAYISI ANAHTARDAN CIKARILDI: mesh yogunlugu TESSELASYON artefaktidir,
    # geometri not. Bir vakada same part `v12` and `v13` aliyordu (2'nin kuvveti sinirini
    # gecmis). bbox + doluluk + histogram ayirt etmeye yetiyor (dogrulama below).
    return f"b{bbk [0 ]}.{bbk [1 ]}.{bbk [2 ]}|d{hk }|h"+".".join (map (str ,h ))


def dogrula (n =300 ,seed =0 ):
    """YENI anahtar, ESKI gruplamayi yeniden uretiyor mu? (old parcalarda sinanir)"""
    import glob 
    from infer_step_cp import step_to_mesh 
    from corpus_identity import step_kimlik 
    old_ =json .load (io .open ("results/_strict_geometry_keys.json",encoding ="utf-8"))
    sm ={step_kimlik (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    ortak =[p for p in old_ if p in sm ]
    rng =np .random .RandomState (seed )
    sec =[ortak [i ]for i in rng .permutation (len (ortak ))[:n ]]
    new_ ={}
    for i ,p in enumerate (sec ,1 ):
        if i %50 ==0 :
            print (f"  {i }/{len (sec )}",flush =True )
        try :
            V ,F =step_to_mesh (sm [p ])
            new_ [p ]=key_ (V ,F )
        except Exception :
            pass 
    ok =[p for p in sec if p in new_ ]
    # CIFT bazinda uyum: same old grupta olanlar new anahtarda da same mi?
    ee =ey =ye =0 
    for i in range (len (ok )):
        for j in range (i +1 ,len (ok )):
            a ,b =ok [i ],ok [j ]
            e =old_ [a ]==old_ [b ]
            y =new_ [a ]==new_ [b ]
            if e and y :
                ee +=1 
            elif e and not y :
                ey +=1 
            elif y and not e :
                ye +=1 
    print (f"\nDOGRULAMA ({len (ok )} part, {len (ok )*(len (ok )-1 )//2 } cift):")
    print (f"  eski AYNI grup & yeni AYNI  : {ee }   (korunan ikizler)")
    print (f"  eski AYNI grup & yeni FARKLI: {ey }   <- IKIZ KACIRILDI (kotu)")
    print (f"  eski FARKLI & yeni AYNI     : {ye }   <- YANLIS BIRLESTIRME (kotu)")
    duy =ee /max (ee +ey ,1 )
    kes =ee /max (ee +ye ,1 )
    print (f"  duyarlilik {duy :.4f} | precision {kes :.4f}")
    gecti =duy >=0.95 and kes >=0.95 
    print (f"  KILL: ikisi de >=0.95 -> {'GECTI'if gecti else 'GECMEDI'}")
    return gecti ,{"ee":ee ,"ey":ey ,"ye":ye ,"duyarlilik":duy ,"precision":kes }


if __name__ =="__main__":
    import argparse 
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--dogrula",type =int ,default =300 )
    a =ap .parse_args ()
    g ,r =dogrula (a .dogrula )
    with io .open ("results/geometri_anahtar_dogrulama.json","w",encoding ="utf-8")as f :
        json .dump ({"gecti":bool (g ),**r ,"bbox_kova":BBOX_KOVA },f ,indent =1 )
    print ("receipt -> results/geometri_anahtar_dogrulama.json")
