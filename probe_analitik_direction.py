# -*- coding: utf-8 -*-
"""III. KOL -- YONU OGRENME, HESAPLA (analitik silindir ekseni)

WHY SIMDI. Yayilim kolunun darbogazi measured: uretilen konumda correct direction
MEVCUT (oracle 0.527) but model onu SECEMIYOR (0.077). Yani sorun direction
BILGISININ yoklugu not, SECIMI. B-rep agzinda direction whereas ANALITIKTIR --
silindirin ekseni. Ogrenmeye gerek absent.

ISARET SORUNU. Metrik ISARETLI angle kullaniyor; eksenin isareti whereas keyfi
(+u and -u same axis). Disari bakan direction, silindir MERKEZINDEN AGZA giden
vektorle belirlenir (`mouth_a`/`mouth_b` onbellekte present).

BU SONDA this tavani olcer: each GT for, konum kutusunda a B-rep agzi VE that
agzin ANALITIK yonu ISARETLI angle kutusunda mi?

  konum         : mouth, GT'nin konum kutusunda mi (lateral<=2, axial<=40)
  konum+axis   : ustune axis ISARETSIZ 10 derece inside mi
  konum+signed: ustune axis ISARETLI 10 derece inside mi (URUN OLCUSU)

`konum+signed` yuksekse direction HESAPLANABILIR and ogrenilmesi gereksizdir.
ISARETSIZ with ISARETLI arasindaki difference, sign belirleme sorununun buyuklugu.

D7'ye BAKILMAZ.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
import d6_record # noqa: E402

SIL =os .environ .get ("AY_SIL","results/_d6_silindirler.pkl")
MARKALAR =set (os .environ .get ("AY_MARKA","NIT,MOR,SUPU,UPUN").split (","))
YANAL ,EKSENEL =2.0 ,40.0 


def _birim (v ):
    v =np .asarray (v ,float )
    n =np .linalg .norm (v ,axis =-1 ,keepdims =True )
    return v /np .maximum (n ,1e-12 )


def agizlar (sil ):
    """(konum, disari_bakan_yon) ciftleri. Yon = merkezden agza."""
    P ,D =[],[]
    for c in sil :
        ax =np .asarray (c ["axis"],float )
        ax =ax /max (np .linalg .norm (ax ),1e-12 )
        center_ =np .asarray (c ["center"],float )
        for k in ("mouth_a","mouth_b"):
            m =c .get (k )
            if m is None :
                continue 
            m =np .asarray (m ,float )
            v =m -center_ 
            if np .linalg .norm (v )<1e-6 :
                continue 
                # DISARI BAKAN direction: eksenin, merkezden agza giden bilesenle
                # same signed hali
            direction =ax if float (v @ax )>0 else -ax 
            # ISARET SOZLESMESI: GT yonu govdenin ICINE mi DISARI mi bakiyor?
            # `kanonik` blogunda pozitiflerin `disa_bakis` degeri NEGATIFTI,
            # i.e. GT ICERI bakiyor may be. AY_TERS=1 with sinanir.
            if os .environ .get ("AY_TERS","0")=="1":
                direction =-direction 
            P .append (m )
            D .append (direction )
    if not P :
        return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
    return np .asarray (P ),np .asarray (D )


def main ():
    cy =pickle .load (open (SIL ,"rb"))
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =0 
    for pid ,r in kay .items ():
        if r .get ("mfg")not in MARKALAR :
            continue 
        G =np .asarray (r .get ("G",[]),float )
        if not len (G ):
            continue 
        sil =cy .get (str (pid ))
        if not sil :
            continue 
        AP ,AD =agizlar (sil )
        if not len (AP ):
            continue 
        Gn =_birim (np .asarray (r ["Gd"],float ))
        v =AP [:,None ,:]-G [None ,:,:]
        al =(v *Gn [None ,:,:]).sum (-1 )
        yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
        konum =(yan <=YANAL )&(np .abs (al )<=EKSENEL )# (mouth, gt)
        cos =AD @Gn .T 
        aci_i =np .degrees (np .arccos (np .clip (cos ,-1 ,1 )))# ISARETLI
        aci_s =np .degrees (np .arccos (np .clip (np .abs (cos ),-1 ,1 )))# unsigned
        a =ist [r ["mfg"]]
        a ["gt"].append (len (G ))
        a ["konum"].append (int (konum .any (0 ).sum ()))
        a ["axis"].append (int ((konum &(aci_s <=K .ACI )).any (0 ).sum ()))
        a ["signed"].append (int ((konum &(aci_i <=K .ACI )).any (0 ).sum ()))
        a ["mouth"].append (len (AP ))
        n +=1 
    print (f"{n } part | kabul: lateral<={YANAL } axial<={EKSENEL } aci<={K .ACI }\n")
    print (f"{'brand':<7}{'GT':>7}{'mouth/p':>8}{'KONUM':>9}{'+axis':>9}"
    f"{'+ISARETLI':>11}{'sign kaybi':>14}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =sum (a ["gt"])
        k =sum (a ["konum"])/max (g ,1 )
        e =sum (a ["axis"])/max (g ,1 )
        i =sum (a ["signed"])/max (g ,1 )
        out [m_ ]={"gt":g ,"konum":k ,"axis":e ,"signed":i ,
        "isaret_kaybi":e -i ,
        "agiz_parca":float (np .mean (a ["mouth"]))}
        print (f"{m_ :<7}{g :>7}{np .mean (a ['mouth']):>8.0f}{k :>9.3f}{e :>9.3f}"
        f"{i :>11.3f}{e -i :>14.3f}")
    json .dump ({"lateral":YANAL ,"axial":EKSENEL ,"aci":K .ACI ,
    "brand":out ,
    "not":"B-rep agzi + ANALITIK axis yonu (merkezden agza "
    "isaretlenmis). 'sign kaybi' = unsigned - signed. "
    "D7'ye BAKILMADI."},
    open ("results/analitik_direction.json","w"),indent =1 )
    print ("\nmakbuz -> results/analitik_direction.json")
    print ("OKUMA: +ISARETLI yuksekse direction HESAPLANABILIR, ogrenilmesi gereksiz.")


if __name__ =="__main__":
    main ()
