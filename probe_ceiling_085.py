# -*- coding: utf-8 -*-
"""ROBOT TAVANINI 0.85'e TASI: konum and direction kaynaklarini kademe kademe zenginlestir.

DURUM (D7 brand-disi, `results/position_direction_sik_d7.json`): robot F1 tavani 0.7273.
Kalan loss: KONUM YOK %31.8, YON HICBIR KAYNAKTA YOK %11.0.
0.85 for recall 0.5714 -> 0.7391 is required (F1 = 2r/(1+r)).

KONUM kaynaklari kademeli:
  P0  segmentasyon `v_o` adaylari (TEZ)
  P1  + B-rep agizlari (silindir uclari + duzlemsel opening merkezleri)
  P2  + axis boyu ornekleme
  P3  + MESH TEPELERI  -- olculmustu ki GT'lerin %98.1'inin 2mm lateral
      komsulugunda vertex present ([[mesh-ceiling-not-lateral-model-hatasi]]); i.e. konum
      bilgisi meshte ZATEN VAR, sorun onu SECEBILMEK.

YON kaynaklari kademeli:
  Y0  adayin own yonu
  Y1  + 10mm icindeki komsu adaylarin yonleri
  Y2  + parcadaki B-rep silindir eksenleri (+/-)
  Y3  + parcanin ANA EKSENLERI (kutu eksenleri +/-, 6 direction) -- bedava, part
      hizasi most klemenste giris yonunu tasiyor

DURUSTLUK: bunlar TAVAN. P3 kolu part basina binlerce konum demektir, dagitilabilir
a sistem DEGIL; sorusu "bilgi meshte present mi" sorusudur, "urun bunu yapabilir mi"
sorusu DEGIL. Tez turetmesi no kolda degismiyor; `v_o` always P0'da.
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
import canonical_d7 as K # noqa: E402

YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 
YON_R =10.0 
OB ="results/_p1_olasilik_d7"
SIK_T =(0.05 ,0.15 ,0.3 )


def _birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    return V /np .maximum (np .linalg .norm (V ,axis =1 ,keepdims =True ),1e-12 )


def axis_sample (cyl ):
    P ,D =[],[]
    for c in cyl or []:
        a =np .asarray (c ["axis"],float )
        na =np .linalg .norm (a )
        ma ,mb =c .get ("mouth_a"),c .get ("mouth_b")
        if na <1e-9 or ma is None or mb is None :
            continue 
        a =a /na 
        ma =np .asarray (ma ,float )
        mb =np .asarray (mb ,float )
        for t in SIK_T :
            P .append (ma +t *(mb -ma ));D .append (a )
            P .append (mb +t *(ma -mb ));D .append (-a )
    return (np .asarray (P ,float ).reshape (-1 ,3 ),
    np .asarray (D ,float ).reshape (-1 ,3 ))


def main ():
    cy =pickle .load (open ("results/_d7_silindirler.pkl","rb"))
    ac =pickle .load (open ("results/_d7_acikliklar.pkl","rb"))
    kay =K .yukle (json .load (open ("results/d7_exam_set.json"))["pidler"])

    KOLLAR =[("P0+Y0  tez-saf",0 ,0 ),("P1+Y0  +B-rep mouth",1 ,0 ),
    ("P2+Y0  +axis ornek",2 ,0 ),("P2+Y1  +komsu yonu",2 ,1 ),
    ("P2+Y2  +silindir ekseni",2 ,2 ),("P2+Y3  +ana eksenler",2 ,3 ),
    ("P3+Y3  +MESH TEPELERI",3 ,3 )]
    say ={ad :collections .Counter ()for ad ,_ ,_ in KOLLAR }
    n_gt =0 
    missing =0 
    for pid ,r in sorted (kay .items ()):
        G =np .asarray (r .get ("G",[]),float )
        if not len (G ):
            continue 
        Gd =np .asarray (r ["Gd"],float )
        P0 =np .asarray (r ["P"],float ).reshape (-1 ,3 )
        D0 =_birim (r ["Pd"])if len (P0 )else np .zeros ((0 ,3 ))
        cyl =cy .get (str (pid ))
        Pb ,Db ,_k =brep_pool .merged_pool (P0 ,D0 ,cyl ,ac .get (str (pid )))
        Pe ,De =axis_sample (cyl )
        P2p =np .vstack ([Pb ,Pe ])if len (Pe )else Pb 
        D2p =_birim (np .vstack ([Db ,De ]))if len (De )else _birim (Db )
        f =f"{OB }/{pid }.npz"
        if os .path .exists (f ):
            V =np .asarray (np .load (f )["V"],float )
        else :
            V =np .zeros ((0 ,3 ))
            missing +=1 
            # direction kaynaklari
        eks =[]
        for c in cyl or []:
            a =np .asarray (c ["axis"],float )
            if np .linalg .norm (a )>1e-9 :
                eks .append (a )
        if eks :
            _e =_birim (eks )
            eks =np .vstack ([_e ,-_e ])# signed measurement for +/- ciftleri
        else :
            eks =np .zeros ((0 ,3 ))
        ana =np .zeros ((0 ,3 ))
        if len (V )>3 :
            Q =V -V .mean (0 )
            _a =_birim (np .linalg .svd (Q ,full_matrices =False )[2 ])
            ana =np .vstack ([_a ,-_a ])# signed measurement for +/- ciftleri
        pool ={0 :(P0 ,D0 ),1 :(Pb ,_birim (Db )),2 :(P2p ,D2p ),
        3 :(np .vstack ([P2p ,V ])if len (V )else P2p ,None )}
        n_gt +=len (G )
        for ad ,pk ,yk in KOLLAR :
            Pp ,Dp =pool [pk ]
            if not len (Pp ):
                say [ad ]["KONUM YOK"]+=len (G )
                continue 
            for j in range (len (G )):
                gd =Gd [j ]
                nn =np .linalg .norm (gd )
                if nn <1e-9 :
                    say [ad ]["GT BOZUK"]+=1 
                    continue 
                u =gd /nn 
                w =Pp -G [j ]
                e =w @u 
                yan =np .linalg .norm (w -e [:,None ]*u [None ],axis =1 )
                ok =(yan <=YANAL )&(np .abs (e )<=EKSENEL )
                if not ok .any ():
                    say [ad ]["KONUM YOK"]+=1 
                    continue 
                idx =np .where (ok )[0 ]

                def uy (Dv ):
                    """ISARETLI angle -- urunun metrigiyle AYNI.

                    Ilk version `np.abs(...)` with ISARETSIZ olcuyordu; this, ters
                    yone bakan a adayi correct sayar and tavani sisirir
                    ([[unsigned-oracle-artefakti]]: ~0.08). Yon kaynaklarinin
                    all of them already +/- ciftleriyle havuzda oldugu for signed
                    measurement source cesitliligini KAYBETTIRMEZ.
                    """
                    if Dv is None or not len (Dv ):
                        return False 
                    a =np .degrees (np .arccos (
                    np .clip (_birim (Dv )@u ,-1.0 ,1.0 )))
                    return bool ((a <=ACI ).any ())

                if Dp is not None and uy (Dp [idx [idx <len (Dp )]]):
                    say [ad ]["Y0 kendi"]+=1 
                    continue 
                if yk >=1 and Dp is not None :
                    bul =False 
                    for i in idx [idx <len (Dp )]:
                        d =np .linalg .norm (Pp [:len (Dp )]-Pp [i ],axis =1 )
                        if uy (Dp [d <=YON_R ]):
                            say [ad ]["Y1 komsu"]+=1 
                            bul =True 
                            break 
                    if bul :
                        continue 
                if yk >=2 and len (eks )and uy (eks ):
                    say [ad ]["Y2 silindir ekseni"]+=1 
                    continue 
                if yk >=3 and len (ana )and uy (ana ):
                    say [ad ]["Y3 ana axis"]+=1 
                    continue 
                say [ad ]["YON YOK"]+=1 

    print (f"D7 {n_gt } GT | mesh onbellegi olmayan part {missing }\n")
    print (f"{'arm':<28} {'recall':>8} {'F1 tavani':>10} {'KONUM YOK':>10} {'YON YOK':>9}")
    out ={}
    for ad ,_ ,_ in KOLLAR :
        c =say [ad ]
        # BASARI ANAHTARLARI ACIKCA SAYILIR.
        # ILK SURUM `k.startswith("Y")` diyordu and "YON YOK" da Y with basladigi
        # for BASARISIZLIGI basariya ekliyordu: Y1/Y2/Y3 kollarinda YON YOK
        # duserken recall SABIT kaliyordu (imkansiz) and P3 tavani 0.9989 like
        # SAHTE a number veriyordu. Kalip eslesmesiyle sayim YAPILMAZ.
        BASARI =("Y0 kendi","Y1 komsu","Y2 silindir ekseni","Y3 ana axis")
        tam =sum (c [k ]for k in BASARI )
        r =tam /max (n_gt ,1 )
        f1 =2 *r /(1 +r )
        out [ad ]={"recall":r ,"f1_tavani":f1 ,"kirilim":dict (c )}
        print (f"{ad :<28} {r :>8.4f} {f1 :>10.4f} "
        f"{c ['KONUM YOK']/max (n_gt ,1 ):>10.3f} {c ['YON YOK']/max (n_gt ,1 ):>9.3f}")
    en =max (out ,key =lambda k :out [k ]["f1_tavani"])
    print (f"\nEN YUKSEK: {en } -> F1 tavani {out [en ]['f1_tavani']:.4f}")
    print ("0.85 HEDEFI: "+("ULASILDI"if out [en ]["f1_tavani"]>=0.85 
    else f"ULASILMADI ({out [en ]['f1_tavani']:.4f})"))
    json .dump ({"damga":receipt_hash .damga (),"n_gt":n_gt ,"sonuc":out ,
    "not":"TAVAN olcumu, mukemmel selector. P3 kolu part basina binlerce "
    "konum -- dagitilabilir sistem DEGIL, 'bilgi meshte present mi' "
    "sorusunun cevabi. D7 brand-disi."},
    open ("results/ceiling_085.json","w"),indent =1 )
    print ("receipt -> results/ceiling_085.json")


if __name__ =="__main__":
    main ()
