# -*- coding: utf-8 -*-
"""CWT CEPHESI: D7 GT'sinin %37'si orada and F1 0.0466. Neden?

CWT 226 part / 1146 GT / 5.1 CP-part. Taban 0.0370, P6 0.0466 -- ikisi de
neredeyse sifir. NIT'te (D6) same desen vardi and orada kok why TEMSILDI:
havuzda cevabin yarisi olmasina despite model ranking uretemiyordu.

Bu betik CWT for AYNI three soruyu sorar and cevabi D7'ye BAKMADAN karar vermek
for not, NEREYE YATIRIM YAPILACAGINI bilmek for kullanir:

  1. HAVUZ: only konum / konum+direction recall'u ne?
  2. SIRALAMA: esikten bagimsiz recall@k, rastgeleye according to kac fold?
  3. YOGUNLUK: CWT parcalari NIT like mi (very candidate / few pozitif)?

D7 SINAV KUMESIDIR. Burada YALNIZCA DIAGNOSIS is done; no threshold/rule/arm
secimi this ciktilara bakilarak yapilmaz. Sonuc raporda "teshis" as gecer.
"""
import collections 
import json 
import os 
import sys 

import numpy as np 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import brep_pool # noqa: E402
import connector3d # noqa: E402
import thin_pool # noqa: E402
import canonical_d7 as K # noqa: E402
import direction_bank as YB # noqa: E402

YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
OB ="results/_p1_olasilik_d7"
OZ ="results/_tam_oz"
MARKALAR =os .environ .get ("CEPHE_MARKA","CWT,WIE").split (",")


def rec (P ,D ,G ,Gd ,direction =True ):
    if not len (P )or not len (G ):
        return 0 
    Gn =YB .birim (Gd )
    df =np .asarray (P ,float )[:,None ,:]-np .asarray (G ,float )[None ,:,:]
    al =(df *Gn [None ,:,:]).sum (-1 )
    yan =np .linalg .norm (df -al [...,None ]*Gn [None ,:,:],axis =-1 )
    ok =(yan <=YANAL )&(np .abs (al )<=EKSENEL )
    if direction :
        an =np .degrees (np .arccos (np .clip (YB .birim (D )@Gn .T ,-1.0 ,1.0 )))
        ok =ok &(an <=ACI )
    return int (ok .any (0 ).sum ())


def main ():
    kay =K .yukle ()
    agg =collections .defaultdict (collections .Counter )
    for f in sorted (os .listdir (OZ )):
        if not f .startswith ("d7_"):
            continue 
        pid =f [3 :-4 ]
        r =kay .get (pid )
        if r is None or not len (r .get ("G",[])):
            continue 
        if r ["mfg"]not in MARKALAR :
            continue 
        mf =f"{OB }/{pid }.npz"
        if not os .path .exists (mf ):
            continue 
        z =np .load (f"{OZ }/{f }")
        kk =np .asarray (z ["source"],int )
        P =np .asarray (z ["P"],float )
        D =np .asarray (z ["D"],float )
        zz =np .load (mf )
        V =np .ascontiguousarray (zz ["V"],np .float64 )
        Fc =np .ascontiguousarray (zz ["F"],np .int64 )
        pb =np .asarray (zz ["pbs"],float ).mean (0 )
        pp =thin_pool .ppos (pb ,connector3d .CABLE_ENTRY ,
        connector3d .CONTACT )
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        a =agg [r ["mfg"]]
        a ["part"]+=1 
        a ["gt"]+=len (G )
        m01 =np .isin (kk ,(0 ,1 ))
        a ["n01"]+=int (m01 .sum ())
        a ["nmesh"]+=int ((kk ==2 ).sum ())
        a ["konum01"]+=rec (P [m01 ],D [m01 ],G ,Gd ,direction =False )
        a ["yon01"]+=rec (P [m01 ],D [m01 ],G ,Gd ,direction =True )
        a ["konum012"]+=rec (P ,D ,G ,Gd ,direction =False )
        # TUM mesh tepeleri (seyreltmesiz) -- konum bilgisi meshte present mi?
        Pm ,Dm =brep_pool .mesh_adaylari (V ,Fc ,pp ,brep_pool .MESH_ESIK ,
        brep_pool .MESH_DEDUPE_MM )
        a ["nmesh_ham"]+=len (Pm )
        Pt =np .vstack ([P [m01 ],Pm ])if len (Pm )else P [m01 ]
        a ["konum_tam"]+=rec (Pt ,None ,G ,Gd ,direction =False )
        # direction bankasi with
        idx ,YD ,_ =YB .options (P ,D ,None ,V )
        a ["yon_banka"]+=rec (P [idx ],YD ,G ,Gd ,direction =True )
        a ["nsec"]+=len (idx )

    print (f"{'brand':<6}{'part':>6}{'GT':>7}{'CP/p':>6}{'n01/p':>7}"
    f"{'mesh/p':>8}{'mesh_ham':>9}")
    out ={}
    for m ,a in agg .items ():
        p =max (a ["part"],1 )
        g =max (a ["gt"],1 )
        o ={"part":a ["part"],"gt":a ["gt"],"cp_parca":a ["gt"]/p ,
        "n01_parca":a ["n01"]/p ,"mesh_parca":a ["nmesh"]/p ,
        "mesh_ham_parca":a ["nmesh_ham"]/p ,
        "konum_recall_01":a ["konum01"]/g ,
        "yon_recall_01":a ["yon01"]/g ,
        "konum_recall_012":a ["konum012"]/g ,
        "konum_recall_TAM_MESH":a ["konum_tam"]/g ,
        "yon_recall_BANKA":a ["yon_banka"]/g ,
        "secenek_parca":a ["nsec"]/p }
        out [m ]=o 
        print (f"{m :<6}{a ['part']:>6}{a ['gt']:>7}{o ['cp_parca']:>6.1f}"
        f"{o ['n01_parca']:>7.0f}{o ['mesh_parca']:>8.0f}"
        f"{o ['mesh_ham_parca']:>9.0f}")
    print (f"\n{'brand':<6}{'konum01':>9}{'yon01':>8}{'konum012':>10}"
    f"{'konumTAM':>10}{'yonBANKA':>10}")
    for m ,o in out .items ():
        print (f"{m :<6}{o ['konum_recall_01']:>9.4f}{o ['yon_recall_01']:>8.4f}"
        f"{o ['konum_recall_012']:>10.4f}"
        f"{o ['konum_recall_TAM_MESH']:>10.4f}"
        f"{o ['yon_recall_BANKA']:>10.4f}")
    json .dump ({"damga":receipt_hash .damga (),"sonuc":out ,
    "not":"CWT/WIE cephe TESHISI. D7 SINAV kumesidir; buradan "
    "hicbir threshold/rule/arm secimi YAPILMAZ, yalnizca nereye "
    "yatirim yapilacagi belirlenir."},
    open ("results/cwt_cephesi.json","w"),indent =1 )
    print ("\nmakbuz -> results/cwt_cephesi.json")


if __name__ =="__main__":
    main ()
