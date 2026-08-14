# -*- coding: utf-8 -*-
"""III.2 -- ISARETI YEREL DIS NORMALDEN TURET

FINDING. Analitik silindir ekseni GT yonunu ZATEN tutuyor (unsigned ceiling
SUPU 0.612 / MOR 0.700 / UPUN 0.793). Geriye kalan single belirsizlik ISARET.

KURESEL SOZLESME CALISMIYOR: "part merkezinden disari" olcusu UPUN'da
0.130, "iceri" 0.743 veriyor; SUPU'da tersi. Ve GT olcumu showed ki UPUN
(0.597) and DIN (0.540) part ICINDE BILE karisik -- two yuzunde de giris
which is klemensler. Yani sign a MARKA ozelligi not, YEREL a feature.

BU SONDA isareti mouth noktasindaki MESH YUZEY NORMALINDEN turetir:
    disari = mesh'in that noktadaki dis normali
    direction    = eksenin, dis normalle POZITIF ic product veren hali

Kiyas: kuresel "disari" and kuresel "iceri" sozlesmeleriyle same tabloda.

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

SIL =os .environ .get ("YI_SIL","results/_d6_silindirler.pkl")
MESH_DIZ =os .environ .get ("YI_MESH","results/_p1_olasilik")
MARKALAR =set (os .environ .get ("YI_MARKA","NIT,MOR,SUPU,UPUN").split (","))
YANAL ,EKSENEL =2.0 ,40.0 


def _birim (v ):
    v =np .asarray (v ,float )
    return v /np .maximum (np .linalg .norm (v ,axis =-1 ,keepdims =True ),1e-12 )


def agiz_ve_eksen (sil ):
    """(mouth konumu, EKSEN (unsigned), silindir merkezi)."""
    P ,A ,C =[],[],[]
    for c in sil :
        ax =_birim (np .asarray (c ["axis"],float ))
        merkez =np .asarray (c ["center"],float )
        for k in ("mouth_a","mouth_b"):
            m =c .get (k )
            if m is None :
                continue 
            P .append (np .asarray (m ,float ))
            A .append (ax )
            C .append (merkez )
    if not P :
        return (np .zeros ((0 ,3 )),)*3 
    return np .asarray (P ),np .asarray (A ),np .asarray (C )


def main ():
    import trimesh 
    cy =pickle .load (open (SIL ,"rb"))
    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    ist =collections .defaultdict (lambda :collections .defaultdict (list ))
    n =atlanan =0 
    for pid ,r in kay .items ():
        if r .get ("mfg")not in MARKALAR :
            continue 
        G =np .asarray (r .get ("G",[]),float )
        sil =cy .get (str (pid ))
        if not len (G )or not sil :
            continue 
        mf =f"{MESH_DIZ }/{pid }.npz"
        if not os .path .exists (mf ):
            atlanan +=1 
            continue 
        z =np .load (mf )
        V =np .ascontiguousarray (z ["V"],np .float64 )
        F =np .ascontiguousarray (z ["F"],np .int64 )
        mesh =trimesh .Trimesh (V ,F ,process =False )
        AP ,AX ,AC =agiz_ve_eksen (sil )
        if not len (AP ):
            continue 
            # YEREL DIS NORMAL: mouth noktasina most yakin yuzeyin normali
        try :
            _ ,_ ,yuz =mesh .nearest .on_surface (AP )
            NN =_birim (mesh .face_normals [yuz ])
        except Exception :
            atlanan +=1 
            continue 
        Gn =_birim (np .asarray (r ["Gd"],float ))
        v =AP [:,None ,:]-G [None ,:,:]
        al =(v *Gn [None ,:,:]).sum (-1 )
        yan =np .linalg .norm (v -al [...,None ]*Gn [None ,:,:],axis =-1 )
        konum =(yan <=YANAL )&(np .abs (al )<=EKSENEL )

        def puan (D ):
            aci =np .degrees (np .arccos (np .clip (_birim (D )@Gn .T ,-1 ,1 )))
            return int ((konum &(aci <=K .ACI )).any (0 ).sum ())

        unsigned =np .degrees (np .arccos (
        np .clip (np .abs (AX @Gn .T ),-1 ,1 )))
        a =ist [r ["mfg"]]
        a ["gt"].append (len (G ))
        a ["konum"].append (int (konum .any (0 ).sum ()))
        a ["axis"].append (int ((konum &(unsigned <=K .ACI )).any (0 ).sum ()))
        # kuresel: merkezden agza
        d_kur =np .where (((AP -AC )*AX ).sum (1 ,keepdims =True )>0 ,AX ,-AX )
        a ["kuresel_disari"].append (puan (d_kur ))
        a ["kuresel_iceri"].append (puan (-d_kur ))
        # YEREL: mesh dis normaliyle same signed
        d_yer =np .where ((NN *AX ).sum (1 ,keepdims =True )>0 ,AX ,-AX )
        a ["yerel"].append (puan (d_yer ))
        a ["yerel_ters"].append (puan (-d_yer ))
        n +=1 
    print (f"{n } part islendi, {atlanan } atlandi\n")
    print (f"{'brand':<7}{'GT':>7}{'KONUM':>8}{'axis':>8}{'kur.disari':>11}"
    f"{'kur.iceri':>11}{'YEREL':>8}{'yerel ters':>11}")
    out ={}
    for m_ in sorted (ist ,key =lambda x :-sum (ist [x ]["gt"])):
        a =ist [m_ ]
        g =max (sum (a ["gt"]),1 )
        r ={k :sum (a [k ])/g for k in ("konum","axis","kuresel_disari",
        "kuresel_iceri","yerel",
        "yerel_ters")}
        r ["gt"]=g 
        out [m_ ]=r 
        print (f"{m_ :<7}{g :>7}{r ['konum']:>8.3f}{r ['axis']:>8.3f}"
        f"{r ['kuresel_disari']:>11.3f}{r ['kuresel_iceri']:>11.3f}"
        f"{r ['yerel']:>8.3f}{r ['yerel_ters']:>11.3f}")
    json .dump ({"brand":out ,
    "not":"Isaret kaynaklari: kuresel (merkezden agza) vs YEREL "
    "(mesh dis normali). 'axis' = unsigned ceiling. "
    "D7'ye BAKILMADI."},
    open ("results/yerel_isaret.json","w"),indent =1 )
    print ("\nmakbuz -> results/yerel_isaret.json")
    print ("OKUMA: YEREL, kuresel secenklerin ikisini de asiyorsa sign")
    print ("       GEOMETRIDEN cozulmus demektir (brand sozlesmesi gerekmez).")


if __name__ =="__main__":
    main ()
