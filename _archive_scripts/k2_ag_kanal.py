# -*- coding: utf-8 -*-
"""K2: AG TABANLI KANAL BETIMLEYICISI -- zayif yariyi hedefler.

GECENIN AYRISTIRMASI (K1):
    yuvarlak channel adaylari  %52.4  pozitif %46.9  F1 0.7638
    DIGER (kare/kelepce)     %47.6  pozitif %18.2  F1 0.5329   <- urunun zayif yarisi

Ve T4'un bosluk-grafi ozellikleri (single single AUC 0.69-0.71, measured_path most guclu marjinal
degerler) TAM DA BU YARIDA never hesaplanamiyor: all of them eseksenli B-rep SILINDIRINE dayali,
kare/yarik girisin silindiri absent.

BU KOL: same channel bilgisini AGDAN turetir -> %100 kapsam, kare agizlar DAHIL.
B-rep'e not mesh'e dayandigi for tez cizgisinde kalir (tezin own v_o turetmesi de
mesh boundary noktalarindan gelir).

OZELLIKLER (7), candidate (p,d) for yerel axial-radyal profilden:
    mg_derinlik   kanalin ne up to iceri gittigi (mm)
    mg_agiz_r     mouth yaricapi (mouth duzlemindeki noktalarin medyan radyal uzakligi)
    mg_yuvarlak   mouth radyal saciniminin degisim katsayisi (0 = daire, large = kare/yarik)
    mg_bosluk     mouth diskinde point OLMAYAN acisal dilimlerin orani
    mg_duvar      channel icindeki vertex point count (wall yogunlugu)
    mg_daralma    mouth yaricapi / derindeki radius (koni/kademe)
    mg_kapsam     mouth noktalarinin acisal kapsami (full kind mu, yarik mi)

Hepsi numpy with O(V) -- isin atmaya gerek absent, that is why 3217 candidate dakikalar surer.

KILL (onceden): candidate duzeyi +0.01. Ozellikle ZAYIF YARIDA (+0.02) beklenir; only
guclu yaride kazanip zayifta kaybederse arm KAPANIR.
"""
import io 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

AD =["mg_derinlik","mg_agiz_r","mg_yuvarlak","mg_bosluk","mg_duvar",
"mg_daralma","mg_kapsam"]
RMAX =9.0 # mm, yerel radius
DMAX =18.0 # mm, axial depth penceresi
NDILIM =16 # acisal dilim count


def kanal_profili (V ,P ,Pd ):
    """Her candidate for 7 channel sutunu. Yerel axial-radyal profil, O(V) numpy."""
    F =np .zeros ((len (P ),len (AD )))
    for i in range (len (P )):
        p =P [i ];d =Pd [i ]/(np .linalg .norm (Pd [i ])+1e-9 )
        rel =V -p 
        t =rel @d # axial (+ disari)
        rad =np .linalg .norm (rel -t [:,None ]*d ,axis =1 )
        yakin =(rad <=RMAX )&(t <=2.0 )&(t >=-DMAX )
        if yakin .sum ()<12 :
            continue 
        tt =t [yakin ];rr =rad [yakin ]
        rel_y =rel [yakin ]
        # AGIZ: |t| <= 1.5mm bandindaki points
        mouth =np .abs (tt )<=1.5 
        if mouth .sum ()>=6 :
            ra =rr [mouth ]
            F [i ,1 ]=float (np .median (ra ))
            F [i ,2 ]=float (np .std (ra )/max (np .mean (ra ),1e-6 ))
            # ACISAL KAPSAM and BOSLUK: mouth noktalarinin acisal dagilimi
            a_ =np .array ([1.0 ,0.0 ,0.0 ])
            if abs (float (d @a_ ))>0.9 :
                a_ =np .array ([0.0 ,1.0 ,0.0 ])
            u =np .cross (d ,a_ );u /=np .linalg .norm (u )+1e-9 
            v =np .cross (d ,u )
            th =np .arctan2 (rel_y [mouth ]@v ,rel_y [mouth ]@u )
            hist ,_ =np .histogram (th ,bins =NDILIM ,range =(-np .pi ,np .pi ))
            F [i ,3 ]=float ((hist ==0 ).mean ())
            F [i ,6 ]=float ((hist >0 ).mean ())
            # DERINLIK: channel icindeki (dar yaricapli) most deep point
        dar =rr <=max (F [i ,1 ],1.0 )*1.35 
        if dar .any ():
            F [i ,0 ]=float (max (0.0 ,-tt [dar ].min ()))
            F [i ,4 ]=float (dar .sum ())
            deep =dar &(tt <-max (F [i ,0 ]*0.6 ,1.0 ))
            if deep .sum ()>=4 :
                F [i ,5 ]=float (F [i ,1 ]/max (np .median (rr [deep ]),1e-6 ))
    return F 


def main ():
    import gate_bench as T 
    import measure_set 
    import thesis_remesh 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from night_kilit import guard 

    guard ("k2")
    d4 =np .load ("results/t4_gap.npz",allow_pickle =True )
    BG =d4 ["BG"];RY0 =d4 ["RY"]
    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    stp ={p :s for m ,p ,jf ,s in eligible ()}

    RX ,RY ,RG ,RP ,MG =[],[],[],[],[]
    t0 =time .time ()
    for k ,r in enumerate (D ["DER"],1 ):
        if k %25 ==0 :
            print (f"  {k }/{len (D ['DER'])}  {time .time ()-t0 :.0f}s",flush =True )
        if r ["X"]is None or r .get ("XR")is None or not len (r ["G"]):
            continue 
        X58 =np .hstack ([r ["X"],r ["XR"]])
        P =np .asarray (r ["P"],float );Pd =np .asarray (r ["Pd"],float )
        try :
            Vr ,Fr =step_to_mesh (stp [r ["pid"]])
            V ,_ =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,float )
            FM =kanal_profili (V ,P ,Pd )
        except Exception as e :
            print (f"    {r ['pid']}: {type (e ).__name__ }")
            FM =np .zeros ((len (P ),len (AD )))
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *float (r ["diag"]))
        yy =np .zeros (len (P ),int );up ,ug =set (),set ()
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
        for b in range (len (G ))):
            if not np .isfinite (d_ )or d_ >tt or a_ in up or b_ in ug :
                continue 
            up .add (a_ );ug .add (b_ );yy [a_ ]=1 
        for i in range (len (P )):
            RX .append (X58 [i ]);MG .append (FM [i ]);RY .append (yy [i ])
            RG .append (r ["geo"]);RP .append (r ["pid"])
    RX =np .array (RX );MG =np .array (MG );RY =np .array (RY )
    RG =np .array (RG );RP =np .array (RP )
    assert len (RY )==len (RY0 )and (RY ==RY0 ).all (),"label hizasi bozuk"
    tip =(BG [:,0 ]>0 )
    print (f"\n{len (RY )} candidate | mg kapsami {float ((MG !=0 ).any (1 ).mean ()):.1%} "
    f"(B-rep grafinin kapsami {float ((BG !=0 ).any (1 ).mean ()):.1%})")

    from t1_manufacturer_out import auc_mw 
    print (f"\n{'column':<14}{'AUC tum':>9}{'AUC zayif':>11}{'sifir-disi':>11}")
    for j ,a in enumerate (AD ):
        v =MG [:,j ]
        az =auc_mw (v [~tip ],RY [~tip ].astype (bool ))if (~tip ).any ()else 0.5 
        print (f"{a :<14}{auc_mw (v ,RY .astype (bool )):>9.3f}{az :>11.3f}{float ((v !=0 ).mean ()):>11.1%}")

    def karar (o ):
        m =np .zeros (len (o ),bool )
        for u in np .unique (RP ):
            i =RP ==u ;v =o [i ]
            m [i ]=(v >=0.5 *max (v .max (),1e-9 ))&(v >=0.25 )
        return m 

    def f1m (m ,msk =None ):
        s =np .ones (len (RY ),bool )if msk is None else msk 
        tp =int ((RY .astype (bool )&m &s ).sum ());fp =int ((~RY .astype (bool )&m &s ).sum ())
        fn =int ((RY .astype (bool )&~m &s ).sum ())
        p =tp /max (tp +fp ,1 );r =tp /max (tp +fn ,1 )
        return 2 *p *r /max (p +r ,1e-9 )

    def oof (M ):
        o =np .zeros (len (RY ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,RY ,RG ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [tr ],RY [tr ]).predict_proba (M [te ])[:,1 ]
        return o 

    o0 =oof (RX );o1 =oof (np .hstack ([RX ,MG ]));o2 =oof (np .hstack ([RX ,BG ,MG ]))
    m0 ,m1 ,m2 =karar (o0 ),karar (o1 ),karar (o2 )
    print (f"\n{'arm':<20}{'TUM':>9}{'yuvarlak':>10}{'ZAYIF':>9}")
    for ad ,m in (("58 baseline",m0 ),("58+network-channel",m1 ),("58+bg+network",m2 )):
        print (f"{ad :<20}{f1m (m ):>9.4f}{f1m (m ,tip ):>10.4f}{f1m (m ,~tip ):>9.4f}")
    d_tum =f1m (m1 )-f1m (m0 );d_zayif =f1m (m1 ,~tip )-f1m (m0 ,~tip )
    print (f"\nfark: TUM {d_tum :+.4f} | ZAYIF YARI {d_zayif :+.4f}")
    gecti =d_tum >=0.01 and d_zayif >0 
    print (f"KARAR: {'SINYAL VAR -> uctan uca sinava'if gecti else 'SINYAL YOK -> K2 KAPANIR'}")
    np .savez ("results/k2_ag_kanal.npz",MG =MG ,RY =RY ,RG =RG ,RP =RP ,tip =tip )
    with io .open ("results/k2_ag_kanal.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":f1m (m0 ),"ag_kanal":f1m (m1 ),"bg_ve_ag":f1m (m2 ),
        "zayif_taban":f1m (m0 ,~tip ),"zayif_yeni":f1m (m1 ,~tip ),
        "fark_tum":float (d_tum ),"fark_zayif":float (d_zayif ),
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/k2_ag_kanal.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
