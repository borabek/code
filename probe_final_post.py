# -*- coding: utf-8 -*-
"""SON-ISLEM KOLLARI — cevrimdisi, prediction dokumunden

WHY BURASI. Sunum sayilarindaki EN BUYUK single loss here:

    tespit F1            0.7878     (deligi buldu)
    robot, axis olcutu  0.5764     -> -0.2114  (axis 10 dereceden sapmis)
    robot, ISARETLI      0.4839     -> -0.0925  (180 derece TERS)

Ikisi de SAF GEOMETRI and ikisi de SON-ISLEM with duzeltilebilir OLABILIR.
Egitim gerektirmez; dokum thanks to saniyeler inside olculur.

KOLLAR:
  baseline        : dokunma
  disari       : yonu govdeden DISARI bakacak sekilde cevir
                 (GT sozlesmesi %100 disari -- `signed-direction-selector`)
  eksen_snap   : yonu parcanin ANA EKSENLERINDEN most yakinina oturt
  snap_disari  : ikisi birden
  gt_isaret    : sign GT'den (UST SINIR -- isaretten ne up to kaybettigimizi
                 olcer, DAGITILAMAZ)

GOVDE MERKEZI: dokumde mesh absent; prediction edilen CP'lerin weight merkezi
vekil as is used. Klemenste girisler karsit yuzlerde oldugu for this
centre body merkezine yakindir. Vekil ZAYIFSA arm haksiz yere duser --
that is why `gt_isaret` UST SINIRI de olculur: aradaki difference vekilin
kusurunu gosterir.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import canonical_d7 as K # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

DOKUM =os .environ .get ("SI_DOKUM","results/_tahmin_dokumu.json")
YOL =os .environ .get ("SI_YOL","saha")


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def ana_eksenler (P ):
    """parcanin ana eksenleri (PCA) + axis hizali three direction."""
    if len (P )<3 :
        return np .eye (3 )
    Q =P -P .mean (0 )
    _ ,_ ,Vt =np .linalg .svd (Q ,full_matrices =False )
    return np .vstack ([Vt ,np .eye (3 )])


def uygula (P ,D ,arm ,G =None ,Gd =None ,mesh_merkez =None ,yerel_n =None ,
halka_n =None ):
    if not len (P ):
        return P ,D 
    D =_birim (D )
    merkez =P .mean (0 )
    if arm in ("eksen_snap","snap_disari"):
        E =_birim (ana_eksenler (P ))
        # each yonu, |cos| most large which is ana eksene oturt (ISARETSIZ eslesme)
        c =np .abs (D @E .T )
        j =np .argmax (c ,axis =1 )
        yeni =E [j ]*np .sign (np .sum (D *E [j ],axis =1 ))[:,None ]
        D =_birim (yeni )
    if arm =="disari_mesh"and mesh_merkez is not None :
        r =P -np .asarray (mesh_merkez ,float )[None ,:]
        sign =np .sign (np .sum (D *r ,axis =1 ))
        sign [sign ==0 ]=1.0 
        D =D *sign [:,None ]
    if arm =="disari_normal"and yerel_n is not None and len (yerel_n ):
        N =_birim (np .asarray (yerel_n ,float ))
        sign =np .sign (np .sum (D *N ,axis =1 ))
        sign [sign ==0 ]=1.0 
        D =D *sign [:,None ]
    if arm in ("disari","snap_disari"):
        r =P -merkez [None ,:]
        sign =np .sign (np .sum (D *r ,axis =1 ))
        sign [sign ==0 ]=1.0 
        D =D *sign [:,None ]
    if arm =="halka_disari"and halka_n is not None and len (halka_n ):
        H =_birim (np .asarray (halka_n ,float ))
        if len (H )==len (D ):
            sign =np .sign (np .sum (D *H ,axis =1 ))
            sign [sign ==0 ]=1.0 
            D =D *sign [:,None ]
    if arm =="halka_kesin"and halka_n is not None and len (halka_n ):
        H =_birim (np .asarray (halka_n ,float ))
        if len (H )==len (D ):
            aci =np .degrees (np .arccos (np .clip (np .sum (D *H ,axis =1 ),-1 ,1 )))
            _e =float (os .environ .get ("SI_ESIK","150"))
            D =np .where ((aci >=_e )[:,None ],-D ,D )
    if arm in ("normal_kesin","mesh_kesin"):
        ref =(np .asarray (yerel_n ,float )if arm =="normal_kesin"
        and yerel_n is not None and len (yerel_n )else None )
        if ref is None and arm =="mesh_kesin"and mesh_merkez is not None :
            ref =P -np .asarray (mesh_merkez ,float )[None ,:]
        if ref is not None and len (ref )==len (D ):
            R =_birim (ref )
            cos =np .clip (np .sum (D *R ,axis =1 ),-1 ,1 )
            aci =np .degrees (np .arccos (cos ))
            # ESIK CEVREDEN: `normal_kesin` 150 derecede HIC tetiklenmedi
            # (full +0.0000). Esik taranabilir must be.
            _esik =float (os .environ .get ("SI_ESIK","150"))
            cevir =aci >=_esik 
            D =np .where (cevir [:,None ],-D ,D )
    if arm =="parca_modal"and len (D )>=3 :
    # parcadaki cogunluk yonune uy: most large signed kumeyi bul,
    # ona 90 dereceden extra ters olanlari cevir
        c =np .clip (D @D .T ,-1 ,1 )
        oy =(np .degrees (np .arccos (c ))<=90.0 ).sum (1 )
        ana =D [int (np .argmax (oy ))]
        # only ana yone DIK OLMAYANLARI (i.e. same eksende olanlari) cevir
        hiz =np .abs (np .clip (D @ana ,-1 ,1 ))
        ters =(D @ana <0 )&(hiz >0.5 )
        D =np .where (ters [:,None ],-D ,D )
    if arm =="gt_isaret"and G is not None and len (G ):
    # UST SINIR: each tahmini, EN YAKIN GT'nin isaretine cevir
        d =np .linalg .norm (P [:,None ,:]-G [None ,:,:],axis =-1 )
        j =np .argmin (d ,axis =1 )
        sign =np .sign (np .sum (D *_birim (Gd )[j ],axis =1 ))
        sign [sign ==0 ]=1.0 
        D =D *sign [:,None ]
    return P ,D 


def main ():
    d =[r for r in json .load (open (DOKUM ))if r ["yol"]==YOL ]
    if not d :
        sys .exit (f"{DOKUM } icinde '{YOL }' yok")
        # YENI KOLLAR (2026-08-13): dokum residual GERCEK body bilgisi tasiyor.
        # `disari_mesh`  : mesh MERKEZINDEN disari (prediction ortalamasi not)
        # `disari_normal`: YEREL YUZEY NORMALIYLE same yone (most correct vekil)
        # SECICI ISARET KURALLARI (2026-08-13). "HEP disari cevir" ZARAR verdi
        # (-0.027..-0.113), i.e. modelin isaretleri COGUNLUKLA DOGRU and error
        # AZINLIKTA. O halde correct rule "hep cevir" not "EMIN OLUNCA cevir".
        #   normal_kesin : only yerel normalle 150 dereceden extra celisiyorsa
        #   mesh_kesin   : only mesh merkeziyle 150 dereceden extra celisiyorsa
        #   parca_modal  : parcadaki COGUNLUGUN isaretine uy (tutarlilik)
        # HALKA NORMALI (2026-08-13). Teshis: most yakin tepenin normali deligin
        # DUVAR normalidir and eksene DIKTIR (median 88.9 derece) -- that yuzden
        # `disari_normal` −0.0266 verdi. Dogru referans, mouth CEVRESINDEKI
        # halkadan (3-8mm) alinan face normalidir.
    KOLLAR =("baseline","disari_normal","halka_disari","halka_kesin",
    "mesh_kesin","gt_isaret")
    agg ={k :collections .Counter ()for k in KOLLAR }
    for r in d :
        P0 =np .asarray (r ["P"],float ).reshape (-1 ,3 )
        D0 =np .asarray (r ["D"],float ).reshape (-1 ,3 )
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        dg =float (r ["diag"])
        for arm in KOLLAR :
            P ,D =uygula (P0 .copy (),D0 .copy (),arm ,G ,Gd ,
            r .get ("mesh_merkez"),r .get ("yerel_normal"),
            r .get ("halka_normal"))
            c =agg [arm ]
            for ad ,im in (("signed",True ),("unsigned",False )):
                tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,K .YANAL ,K .ACI ,
                False ,signed =im )[:3 ]
                c [ad +"_tp"]+=tp ;c [ad +"_fp"]+=fp ;c [ad +"_fn"]+=fn 
            tp ,fp ,fn =match_hungarian (P ,D ,G ,Gd ,dg ,0.0 ,180.0 ,True )[:3 ]
            c ["tespit_tp"]+=tp ;c ["tespit_fp"]+=fp ;c ["tespit_fn"]+=fn 

    def f1 (c ,on ):
        return (2 *c [on +"_tp"]/
        max (2 *c [on +"_tp"]+c [on +"_fp"]+c [on +"_fn"],1 ))

    print (f"{len (d )} part | yol={YOL }\n")
    print (f"{'arm':<14}{'tespit':>9}{'unsigned':>11}{'ISARETLI':>10}"
    f"{'fark':>9}")
    tab =f1 (agg ["baseline"],"signed")
    out ={}
    for arm in KOLLAR :
        c =agg [arm ]
        r ={"tespit":f1 (c ,"tespit"),"unsigned":f1 (c ,"unsigned"),
        "signed":f1 (c ,"signed")}
        out [arm ]=r 
        et =""
        if arm =="gt_isaret":
            et ="  (UST SINIR)"
        elif r ["signed"]>tab +0.01 :
            et ="  <- KAPI GECTI"
        print (f"{arm :<14}{r ['tespit']:>9.4f}{r ['unsigned']:>11.4f}"
        f"{r ['signed']:>10.4f}{r ['signed']-tab :>+9.4f}{et }")
    json .dump ({"yol":YOL ,"n_parca":len (d ),"sonuc":out ,
    "not":"Son-islem kollari, cevrimdisi. gt_isaret UST "
    "SINIRDIR (dagitilamaz). D7'ye BAKILMADI."},
    open (f"results/son_islem_{YOL }.json","w"),indent =1 )
    print (f"\nmakbuz -> results/son_islem_{YOL }.json")


if __name__ =="__main__":
    main ()
