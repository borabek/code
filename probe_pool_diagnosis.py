# -*- coding: utf-8 -*-
"""HAVUZ mu SKOR mu -- DOKUMDEN teshis (new inference GEREKTIRMEZ).

Onceki measurement INVALID cikmisti: output recall'u (0.4682) HAVUZ recall'undan
(0.4242) BUYUK gorunuyordu -- yapisal as IMKANSIZ, because output havuzun
lower kumesidir. Bu betik before that CELISKIYI teshis eder (pool gercekten
ciktinin upper kumesi mi), after correct soruyu sorar:

  KACIRILAN GT'lerin kaci HAVUZDA ZATEN VAR (i.e. sorun SKOR/GATE),
  kaci havuzda HIC YOK (i.e. sorun TEMSIL/ADAY URETIMI)?

Havuz kapsamasi for Macar eslestirme DEGIL, **GT basina kapsama** olculur
(that GT'yi kabul kutusunda karsilayan HERHANGI a candidate present mi) -- pool
soruşturmasinin correct olcutu budur.
"""
import json 
import sys 

import numpy as np 

DOKUM =sys .argv [1 ]if len (sys .argv )>1 else "results/_dump_baseline.json"
YOL =sys .argv [2 ]if len (sys .argv )>2 else "measured_path"
YANAL ,EKSEN ,ACI =2.0 ,40.0 ,10.0 


def _birim (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def kapsama (P ,D ,G ,Gd ,signed ):
    """(n_GT,) bool: each GT for kabul kutusunda EN AZ BIR candidate present mi."""
    if not len (P )or not len (G ):
        return np .zeros (len (G ),bool )
    diff =P [:,None ,:]-G [None ,:,:]
    al =(diff *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
    c =D @Gd .T 
    an =np .degrees (np .arccos (np .clip (c if signed else np .abs (c ),-1 ,1 )))
    kabul =(np .abs (al )<=EKSEN )&(an <=ACI )&(pe <=YANAL )
    return kabul .any (0 )


def main ():
    rec_ =[r for r in json .load (open (DOKUM ))if r .get ("path")==YOL ]
    print (f"{DOKUM } / path={YOL } -> {len (rec_ )} part")
    ust_kume_ihlali =0 
    havuz_bos =0 
    say ={k :0 for k in ("gt","cikti","pool","havuz_var_cikti_yok",
    "havuz_yok")}
    for signed in (False ,True ):
        for k in say :
            say [k ]=0 
        ust_kume_ihlali =havuz_bos =0 
        for r in rec_ :
            G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
            if not len (G ):
                continue 
            Gd =_birim (r ["Gd"])
            P =np .asarray (r ["P"],float ).reshape (-1 ,3 )
            D =_birim (r ["D"])if len (P )else np .zeros ((0 ,3 ))
            hP =np .asarray (r .get ("havuz_P")or [],float ).reshape (-1 ,3 )
            hD =_birim (r ["havuz_D"])if len (hP )else np .zeros ((0 ,3 ))
            if not len (hP ):
                havuz_bos +=1 
            c_cik =kapsama (P ,D ,G ,Gd ,signed )
            c_hav =kapsama (hP ,hD ,G ,Gd ,signed )
            # YAPISAL CHECK: output havuzun lower kumesiyse, ciktinin
            # kapsadigi each GT'yi pool da kapsamali.
            ust_kume_ihlali +=int ((c_cik &~c_hav ).sum ())
            say ["gt"]+=len (G )
            say ["cikti"]+=int (c_cik .sum ())
            say ["pool"]+=int (c_hav .sum ())
            say ["havuz_var_cikti_yok"]+=int ((c_hav &~c_cik ).sum ())
            say ["havuz_yok"]+=int ((~c_hav ).sum ())
        ad ="ISARETLI"if signed else "unsigned"
        gt =max (say ["gt"],1 )
        print (f"\n--- {ad } kabul kutusu (lateral<={YANAL } axis<={EKSEN } "
        f"aci<={ACI }) ---")
        print (f"  GT total                        : {say ['gt']}")
        print (f"  HAVUZ kapsamasi (ceiling)          : {say ['pool']:5d} "
        f"({say ['pool']/gt :.4f})")
        print (f"  CIKTI kapsamasi                  : {say ['cikti']:5d} "
        f"({say ['cikti']/gt :.4f})")
        print (f"  havuzda VAR but ciktida YOK      : "
        f"{say ['havuz_var_cikti_yok']:5d} "
        f"({say ['havuz_var_cikti_yok']/gt :.4f})  <- SKOR/GATE kaybi")
        print (f"  havuzda HIC YOK                  : {say ['havuz_yok']:5d} "
        f"({say ['havuz_yok']/gt :.4f})  <- TEMSIL kaybi")
        print (f"  [yapisal kontrol] ciktida present/havuzda none: "
        f"{ust_kume_ihlali }  (0 OLMALI)")
        if havuz_bos :
            print (f"  UYARI: {havuz_bos } parcada pool BOS kaydedilmis")
        loss =say ["gt"]-say ["cikti"]
        if loss :
            print (f"  => kacan {loss } GT'nin "
            f"%{100 *say ['havuz_var_cikti_yok']/loss :.1f}'i SKOR, "
            f"%{100 *say ['havuz_yok']/loss :.1f}'i TEMSIL")


if __name__ =="__main__":
    main ()
