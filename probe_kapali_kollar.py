# -*- coding: utf-8 -*-
"""KAPANAN KOLLARI YOKLA: verdict kolun mu, sondanin mi kusuruydu?

Yelpaze ornegi (64 direction -> +0.0000 "OLU"; 256 direction -> +0.0676) showed ki
kapatma kararlari sondanin cozunurluguyle sinirli. Bu betik hafizada KAPALI
duran and `docs/KAPANAN_KOLLAR_DENETIMI.md`'de SUPHELI isaretlenen kollari
UCUZ TAVAN SONDALARIYLA yeniden yoklar. Yoklama uctan uca not TAVAN
duzeyindedir: "this arm havuza new DOGRU cevap katiyor mu?"

Yoklanan kollar:
  EKSEN     axis boyu ornekleme (silindir agzindan iceri t=0.05/0.15/0.30)
            kapanma gerekcesi: "tavani very few oynatti, 11x candidate". O measurement YON
            BANKASI OLMADAN yapilmisti.
  K21       periyodik yayma -- `order_stamp` with: capalardan sirayi uzat.
            kapanma gerekcesi: direction KAYNAKTAN KOPYALANIYORDU. Simdi direction
            kopyalanmiyor, only HAVUZDA VAR OLAN candidate one cikariliyor.

Olcut: yonlu recall (lateral<=2mm, signed angle<=10, axial<=40mm) and maliyet
(secenek/part). D6 on runs; D7'ye BAKMAZ.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import d6_record # noqa: E402
import order_stamp # noqa: E402
import direction_bank as YB # noqa: E402

YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
P6D ="results/_p6_oz_u25"
SIK_T =(0.05 ,0.15 ,0.30 )
N =int (os .environ .get ("SONDA_N","150"))
MARKA =os .environ .get ("KOL_MARKA","")


def rec (P ,D ,G ,Gd ):
    if not len (P )or not len (G ):
        return 0 
    Gn =YB .birim (Gd )
    df =np .asarray (P ,float )[:,None ,:]-np .asarray (G ,float )[None ,:,:]
    al =(df *Gn [None ,:,:]).sum (-1 )
    yan =np .linalg .norm (df -al [...,None ]*Gn [None ,:,:],axis =-1 )
    an =np .degrees (np .arccos (np .clip (YB .birim (D )@Gn .T ,-1.0 ,1.0 )))
    return int (((yan <=YANAL )&(np .abs (al )<=EKSENEL )&(an <=ACI ))
    .any (0 ).sum ())


def axis_sample (cyl ):
    P ,D =[],[]
    for c in cyl or []:
        a =np .asarray (c .get ("axis",[0 ,0 ,0 ]),float )
        n =np .linalg .norm (a )
        ma ,mb =c .get ("mouth_a"),c .get ("mouth_b")
        if n <1e-9 or ma is None or mb is None :
            continue 
        a =a /n 
        ma =np .asarray (ma ,float )
        mb =np .asarray (mb ,float )
        for t in SIK_T :
            P .append (ma +t *(mb -ma ));D .append (a )
            P .append (mb +t *(ma -mb ));D .append (-a )
    return (np .asarray (P ,float ).reshape (-1 ,3 ),
    np .asarray (D ,float ).reshape (-1 ,3 ))


def main ():
    cy =pickle .load (open ("results/_d6_silindirler.pkl","rb"))
    kay =d6_record .yukle ()
    fs =[f for f in sorted (os .listdir (P6D ))if f .startswith ("d6_")]
    if MARKA :
        fs =[f for f in fs if (kay .get (f [3 :-4 ])or {}).get ("mfg")==MARKA ]
    fs =fs [:N ]
    top =collections .Counter ()
    for i ,f in enumerate (fs ,1 ):
        pid =f [3 :-4 ]
        r =kay .get (pid )
        if r is None or not len (r .get ("G",[])):
            continue 
        z =np .load (f"{P6D }/{f }")
        idx =np .asarray (z ["idx"],int )
        YD =np .asarray (z ["YD"],float )
        P =np .asarray (z ["P"],float )
        G =np .asarray (r ["G"],float )
        Gd =np .asarray (r ["Gd"],float )
        Pb ,Db =P [idx ],YD 
        top ["gt"]+=len (G )
        top ["banka"]+=rec (Pb ,Db ,G ,Gd )
        top ["n_banka"]+=len (idx )

        # --- EKSEN kolu: axis boyu ornek konumlar + AYNI direction bankasi ---
        Pe ,De =axis_sample (cy .get (pid ))
        if len (Pe ):
            ie ,YDe ,_ =YB .secenekler (Pe ,De ,cy .get (pid ),np .zeros ((0 ,3 )))
            Pt =np .vstack ([Pb ,Pe [ie ]])
            Dt =np .vstack ([Db ,YDe ])
            top ["axis"]+=rec (Pt ,Dt ,G ,Gd )
            top ["n_eksen"]+=len (idx )+len (ie )
        else :
            top ["axis"]+=top ["banka"]-(top ["axis"]and 0 )
            top ["n_eksen"]+=len (idx )

            # --- K21 kolu: capalardan order damgalama (havuzdaki adayi one removes)
            # TAVAN olcumu: capalari GT'den ALMIYORUZ; havuzun most "order uyumlu"
            # adaylarini capa sayiyoruz -- real hatta capalar tahminden gelir.
        if len (Pb )>3 :
        # temsili capa: havuzda each other at most benzeyen 5 candidate
            sec =np .arange (0 ,len (Pb ),max (len (Pb )//5 ,1 ))[:5 ]
            X =order_stamp .oznitelik (Pb ,Db ,Pb [sec ],Db [sec ])
            top ["k21_isaretli"]+=int (X [:,0 ].sum ())
        if i %50 ==0 :
            print (f"  {i }/{len (fs )}",flush =True )

    g =max (top ["gt"],1 )
    p =max (len ([f for f in fs ]),1 )
    out ={"gt":top ["gt"],
    "banka":{"recall":top ["banka"]/g ,"secenek_parca":top ["n_banka"]/p },
    "axis":{"recall":top ["axis"]/g ,"secenek_parca":top ["n_eksen"]/p },
    "k21_isaretli_secenek_parca":top ["k21_isaretli"]/p }
    print (f"\nGT {top ['gt']} | part {p }"+(f" | brand {MARKA }"if MARKA else ""))
    print (f"{'arm':<10}{'yonlu recall':>14}{'secenek/part':>16}")
    print (f"{'BANKA':<10}{out ['banka']['recall']:>14.4f}"
    f"{out ['banka']['secenek_parca']:>16.0f}")
    print (f"{'+EKSEN':<10}{out ['axis']['recall']:>14.4f}"
    f"{out ['axis']['secenek_parca']:>16.0f}")
    print (f"\nEKSEN KAZANCI: "
    f"{out ['axis']['recall']-out ['banka']['recall']:+.4f} recall, "
    f"maliyet x{out ['axis']['secenek_parca']/max (out ['banka']['secenek_parca'],1 ):.2f}")
    print (f"K21: sira damgasi alan secenek/part "
    f"{out ['k21_isaretli_secenek_parca']:.0f}")
    json .dump ({"damga":makbuz_hash .damga (),"sonuc":out ,"brand":MARKA ,
    "not":"Kapanan kollarin TAVAN yoklamasi. D6; D7'ye BAKILMADI. "
    "K21 satiri temsili capalarla uretilmistir -- gercek "
    "hatta capalar tahminden gelir."},
    open ("results/kapali_kollar.json","w"),indent =1 )
    print ("receipt -> results/kapali_kollar.json")


if __name__ =="__main__":
    main ()
