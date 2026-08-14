# -*- coding: utf-8 -*-
"""E1: TOPLULUK SINAVI -- projenin EN IYI checkpointleri uctan uca never sinanmamisti.

DENETIM BULGUSU: dagitilan ensemble 1x k_eig64 (val Conn-IoU 0.6481) + 3x k_eig96
(0.6378/0.6629/0.6597). Diskte duran but DAGITILMAYAN two checkpoint projenin EN YUKSEK
val Conn-IoU'suna sahip: keig128_s1 = 0.7079, keig128_s2 = 0.6949.

k_eig 128 kolu TEK TOHUMLA oldurulmustu and that seed (keig128_s0, val 0.6237) ailenin
EN KOTUSUYDU. s1 and s2 uctan uca HIC olculmedi.

BU BETIK ikisini AYNI kod yolundan gecirip karsilastirir. Her kolun gate'i KENDI
adaylariyla egitilir (parite sarti: different ensemble = different candidate dagilimi = gate yeniden
uydurulmali; hafizadaki `gate-refit-minv4` dersi).

Karsilastirma measurement kumesi ICINDE grup-caprazdir (each two arm for AYNI split), because
B kolunun training korpusu turetilmedi. Bu, kollari ADIL siralar; mutlak value dagitilan
headline degildir.

KILL: tespit +0.01 VE grup bootstrap GA'si sifiri dislamali. Uretici-disi DUSMEMELI.
"""
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

KOL ={"A_DAGITILAN":"results/_der_tam.pkl","B_keig128_s1":"results/_der_B.pkl"}


def main ():
    import measure_set 
    import wire_gate 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    SON ,ROWS ={},{}
    for ad ,yol in KOL .items ():
        if not os .path .exists (yol ):
            print (f"{ad }: {yol } YOK, atlaniyor");continue 
        DER ,rap =measure_set .cluster (yol )
        if ad ==list (KOL )[0 ]:
            measure_set .rapor_bas (rap )
            # --- candidate duzeyi tablo
        RX ,RY ,RG ,RP ,RJ =[],[],[],[],[]
        for r in DER :
            if r ["X"]is None or r .get ("XR")is None :
                continue 
            X =np .hstack ([r ["X"],r ["XR"]])
            P =np .asarray (r ["P"],float )
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            yy =np .zeros (len (P ),int )
            if len (G )and len (P ):
                diff =P [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                tt =max (3.0 ,0.06 *float (r ["diag"]))
                up ,ug =set (),set ()
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
                for b in range (len (G ))):
                    if not np .isfinite (d_ )or d_ >tt or a_ in up or b_ in ug :
                        continue 
                    up .add (a_ );ug .add (b_ );yy [a_ ]=1 
            for i in range (len (P )):
                RX .append (X [i ]);RY .append (yy [i ]);RG .append (r ["geo"]);RP .append (r ["pid"])
            RJ .append (r )
        RX =np .array (RX );RY =np .array (RY );RG =np .array (RG );RP =np .array (RP )
        # --- part-ici z-skor (urunun donusumu)
        Z =np .zeros ((len (RX ),RX .shape [1 ]*2 ))
        for u in np .unique (RP ):
            i =np .where (RP ==u )[0 ]
            Z [i ]=wire_gate .within_part (RX [i ],"zskor")
            # --- GRUP-CAPRAZ gate (each arm KENDI adaylariyla)
        o =np .zeros (len (RY ))
        for tr ,te in GroupKFold (n_splits =5 ).split (Z ,RY ,RG ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (Z [tr ],RY [tr ]).predict_proba (Z [te ])[:,1 ]
            # --- uctan uca
        det ,rob ,gg =[],[],[]
        off =0 
        for r in RJ :
            n =len (r ["P"])
            sk =o [off :off +n ];off +=n 
            m =(sk >=0.5 *max (sk .max (),1e-9 ))&(sk >=0.25 )if n else np .zeros (0 ,bool )
            P =np .asarray (r ["P"],float )[m ]
            Pd =np .asarray (r ["Pd"],float )[m ]
            X =np .hstack ([r ["X"],r ["XR"]])[m ]
            if m .any ():
                c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                c =wire_gate .pose_correct (X ,c )
                c =wire_gate .angle_correct (X ,c )
                if r .get ("UYE"):
                    c =wire_gate .pick_member_direction (X ,c ,r ["UYE"])
                P =np .array ([x ["point"]for x in c ],float )
                Pd =np .array ([x ["direction"]for x in c ],float )
            rj ="cok"if r ["n"]>=8 else "dusuk"
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            det .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
            gg .append (r ["geo"])
        SON [ad ]={"tespit":f1w (det ),"robot":f1w (rob ),"candidate":int (len (RY )),
        "pozitif":float (RY .mean ()),
        "aday_recall":float (sum (int (RY [RP ==p ].sum ())for p in np .unique (RP ))/
        max (sum (r ["n"]for r in RJ ),1 ))}
        ROWS [ad ]=(det ,rob ,gg )
        s =SON [ad ]
        print (f"\n{ad }: tespit {s ['tespit']:.4f} | robot {s ['robot']:.4f} | "
        f"candidate {s ['candidate']} | candidate-recall {s ['aday_recall']:.4f}",flush =True )

    if len (ROWS )==2 :
        a ,b =list (ROWS )
        fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
        for i ,ad in ((0 ,"tespit"),(1 ,"robot")):
            _ ,lo ,hi =measure_set .grup_bootstrap (
            list (zip (ROWS [a ][i ],ROWS [b ][i ])),ROWS [a ][2 ],fn ,n =3000 )
            d =SON [b ][ad ]-SON [a ][ad ]
            print (f"\n{ad .upper ()}: {SON [a ][ad ]:.4f} -> {SON [b ][ad ]:.4f} ({d :+.4f})"
            f"  GA[{lo :+.4f},{hi :+.4f}] {'GERCEK'if (lo >0 or hi <0 )else 'noise'}")
            SON [f"{ad }_ga"]=[lo ,hi ]
        d =SON [b ]["tespit"]-SON [a ]["tespit"]
        gecti =d >=0.01 and SON ["tespit_ga"][0 ]>0 
        print (f"\nKILL: tespit +0.01 VE GA>0 -> {'GECTI'if gecti else 'GECMEDI'}")
        SON ["gecti"]=bool (gecti )
    with io .open ("results/e1_topluluk.json","w",encoding ="utf-8")as f :
        json .dump (SON ,f ,indent =1 )
    print ("receipt -> results/e1_topluluk.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
