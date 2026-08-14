# -*- coding: utf-8 -*-
"""KONUM and YONU AYIRMAK robot tavanini ne up to acar?

FINDING (`results/donusum_ayristir_*.json`): GT'lerin %3.6'sinda (D7'de %3.0)
YANAL olcutu a candidate sagliyor, ACI olcutunu BASKA a candidate. Ayrica VAL'de
%14.4'unde lateral TAMAM but angle wrong. Su an each candidate own (konum, direction) ciftini
KATI tasiyor -- oysa correct direction most zaman parcanin BASKA a yerinde already present.

Bu probe tavani olcer: konum a adaydan, direction YAKINDAKI kaynaklardan secilebilse
robot recall nereye cikar? Yon kaynaklari kademe kademe acilir:
  K0  own yonu (MEVCUT state -- baseline)
  K1  + `r` mm icindeki diger adaylarin yonleri
  K2  + parcadaki B-rep silindir eksenleri (+/-)
  K3  + parcanin baskin komsu yonu (tum adaylarin sign-hizali ortalamasi)

TAVAN olcumudur: real a selector bunun ALTINDA kalir. Ama ceiling acilmiyorsa
arm never kurulmaz.

TEZE SADIK: `v_o` KONUMLARI degismiyor; only hangi YON'un secilebilecegi
genisliyor. Tezin own (konum, direction) cifti always K0 as havuzda.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

import receipt_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import brep_pool # noqa: E402

YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
YON_R =float (os .environ .get ("YON_R","10.0"))# direction odunc alma yaricapi


def _birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    n =np .linalg .norm (V ,axis =1 ,keepdims =True )
    return V /np .maximum (n ,1e-12 )


def main ():
    rec_ =os .environ .get ("AYR_KAYIT","results/_der_yeni.pkl")
    cluster =os .environ .get ("AYR_KUME","results/val_set.json")
    silf =os .environ .get ("AYR_SIL","")
    pids ={str (p )for p in json .load (open (cluster ))["pidler"]}
    R =[r for r in pickle .load (open (rec_ ,"rb"))if str (r ["pid"])in pids ]
    cy =pickle .load (open (silf ,"rb"))if silf and os .path .exists (silf )else {}
    acf =os .environ .get ("AYR_AC","")
    ac =pickle .load (open (acf ,"rb"))if acf and os .path .exists (acf )else {}
    GENIS =os .environ .get ("AYR_GENIS","0")not in ("0","","false")
    SIK =os .environ .get ("AYR_SIK","0")not in ("0","","false")
    SIK_T =[float (x )for x in os .environ .get ("AYR_SIK_T","0.05,0.15,0.3").split (",")]
    print (f"kayit {rec_ } | cluster {cluster } | part {len (R )} | "
    f"silindir {len (cy )} | opening {len (ac )} | direction yaricapi {YON_R }mm | pool {'GENISLETILMIS'if GENIS else 'TEZ-SAF'}",flush =True )

    say =collections .Counter ()
    n_gt =0 
    for r in R :
        G =np .asarray (r .get ("G",[]),float )
        Gd =np .asarray (r .get ("Gd",[]),float )
        if not len (G ):
            continue 
        P =np .asarray (r ["P"],float )
        D =_birim (r ["Pd"])
        if SIK :
        # KONUM SIKLASTIRMA: silindir agzi TEK point veriyor; real CP agzin
        # birkac mm ilerisinde/gerisinde olabiliyor. Eksen along ORNEKLE.
        # Bu a TAVAN olcumudur -- havuzu buyutur, urun yolu DEGISMEZ.
            ek_p ,ek_d =[],[]
            for c in cy .get (str (r ["pid"]))or []:
                a =np .asarray (c ["axis"],float )
                na =np .linalg .norm (a )
                ma ,mb =c .get ("mouth_a"),c .get ("mouth_b")
                if na <1e-9 or ma is None or mb is None :
                    continue 
                a =a /na 
                ma =np .asarray (ma ,float );mb =np .asarray (mb ,float )
                for t in SIK_T :
                    ek_p .append (ma +t *(mb -ma ));ek_d .append (a )
                    ek_p .append (mb +t *(ma -mb ));ek_d .append (-a )
            if ek_p :
                P =np .vstack ([P ,np .asarray (ek_p ,float )])
                D =_birim (np .vstack ([D ,np .asarray (ek_d ,float )]))
        if GENIS :
        # KONUM havuzunu da genislet: KONUM YOK kovasi D7'de %65.8 with most
        # large loss; direction odunc almak that kovaya DOKUNMUYOR.
            P ,D ,_k =brep_pool .merged_pool (P ,D ,cy .get (str (r ["pid"])),
            ac .get (str (r ["pid"])))
            D =_birim (D )
        if not len (P ):
            n_gt +=len (G )
            continue 
        dg =float (r ["diag"])
        pid =str (r ["pid"])
        # part duzeyinde direction kaynaklari
        eks =[]
        for c in cy .get (pid )or []:
            a =np .asarray (c ["axis"],float )
            if np .linalg .norm (a )>1e-9 :
                eks .append (a )
        eks =_birim (eks )if eks else np .zeros ((0 ,3 ))
        center_ =np .asarray ([c ["center"]for c in (cy .get (pid )or [])
        if np .linalg .norm (np .asarray (c ["axis"],float ))>1e-9 ],
        float ).reshape (-1 ,3 )
        bask =D .mean (0 )if len (D )>1 else D [0 ]
        if len (D )>1 :
            B =D *np .sign (D @D [0 ])[:,None ]
            bask =B .mean (0 )
        bask =_birim ([bask ])[0 ]

        for j in range (len (G )):
            n_gt +=1 
            g ,gd =G [j ],Gd [j ]
            nn =np .linalg .norm (gd )
            if nn <1e-9 :
                say ["GT_YONU_BOZUK"]+=1 
                continue 
            u =gd /nn 
            v =P -g [None ]
            e =v @u 
            yan =np .linalg .norm (v -e [:,None ]*u [None ],axis =1 )
            konum_ok =(yan <=YANAL )&(np .abs (e )<=EKSENEL )
            if not konum_ok .any ():
                say ["KONUM YOK"]+=1 
                continue 
            idx =np .where (konum_ok )[0 ]

            def aci_ok (Dv ):
                if not len (Dv ):
                    return False 
                a =np .degrees (np .arccos (np .clip (_birim (Dv )@u ,-1.0 ,1.0 )))
                return bool ((a <=ACI ).any ())

            if aci_ok (D [idx ]):
                say ["K0 kendi yonu"]+=1 
                continue 
                # K1: yakindaki diger adaylarin yonleri
            bulundu =False 
            for i in idx :
                d =np .linalg .norm (P -P [i ],axis =1 )
                if aci_ok (D [d <=YON_R ]):
                    say ["K1 komsu adayin yonu"]+=1 
                    bulundu =True 
                    break 
            if bulundu :
                continue 
                # K2: yakindaki B-rep silindir eksenleri (+/-)
            if len (eks ):
                for i in idx :
                    if len (center_ ):
                        d =np .linalg .norm (center_ -P [i ],axis =1 )
                        yakin =eks [d <=YON_R ]
                    else :
                        yakin =eks 
                    if len (yakin )and (aci_ok (yakin )or aci_ok (-yakin )):
                        say ["K2 B-rep ekseni"]+=1 
                        bulundu =True 
                        break 
            if bulundu :
                continue 
                # K3: parcanin baskin yonu
            if aci_ok (bask [None ])or aci_ok (-bask [None ]):
                say ["K3 baskin direction"]+=1 
                continue 
            say ["YON HICBIR KAYNAKTA YOK"]+=1 

    print (f"\nTOPLAM GT: {n_gt }")
    for k ,v in say .most_common ():
        print (f"  {k :<26} {v :>6}  %{100 *v /max (n_gt ,1 ):.1f}")
    kum =0.0 
    print ("\nKADEMELI ROBOT RECALL TAVANI:")
    for k in ("K0 kendi yonu","K1 komsu adayin yonu","K2 B-rep ekseni",
    "K3 baskin direction"):
        kum +=say [k ]
        print (f"  {k :<26} {kum /max (n_gt ,1 ):.4f}")
    t =kum /max (n_gt ,1 )
    f1 =2 *kum /max (2 *kum +(n_gt -kum ),1 )
    print (f"\nrobot recall tavani {t :.4f} -> robot F1 tavani {f1 :.4f}")
    json .dump ({"damga":receipt_hash .damga (),"kayit":rec_ ,"cluster":cluster ,
    "yon_yaricapi":YON_R ,"n_gt":n_gt ,"sayim":dict (say ),
    "robot_recall_tavani":t ,"robot_f1_tavani":f1 ,
    "genisletilmis_havuz":GENIS ,
    "not":"Konum havuzdan, YON yakin kaynaklardan secilebilse ceiling. "
    "Mukemmel selector varsayimi -- gercek selector ALTINDA kalir."},
    open (os .environ .get ("AYR_CIKTI","results/konum_yon_ayirma.json"),
    "w"),indent =1 )
    print ("receipt yazildi")


if __name__ =="__main__":
    main ()
