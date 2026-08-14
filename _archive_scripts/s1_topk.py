# -*- coding: utf-8 -*-
"""S1: TOP-K DECISION KURALI -- gate'in SKORUNA not, KARARINA dokunan first arm.

WHY BU, WHY SIMDI: bugune up to denenen nine mekanizmanin DOKUZU DA gate'in
SKORLAMA tarafindaydi (feature secimi, donusum, model sinifi, weight, threshold degeri,
orgu, bosluk grafi, type yonlendirme, type-ici threshold). DECISION KURALININ KENDISI never
degismedi: still "part inside score >= 0.5*max VE >= 0.25".

O rule, klemensin EN TAHMIN EDILEBILIR ozelligini kullanmiyor: a parcanin KAC CP'si
oldugu geometriden large olcude okunabilir (kutup count x fold). Sayiyi prediction edip
TOP-K secmek, extra ateslemeyi and missing ateslemeyi AYNI ANDA hedefler -- threshold taramasi
bunu yapamaz, because threshold part basina sayiyi bilmez.

UC KOL KARSILASTIRILIR (all of them AYNI gate skorlariyla):
    URUN      goreli threshold (0.5*max, baseline 0.25)
    TOP-K     K part duzeyi a regresorle prediction edilir, most high K candidate alinir
    KARMA     top-K but skoru very low olanlar yine elenir (K and threshold birlikte)

K REGRESORU part duzeyi ozniteliklerle egitilir (candidate count, kutu boyutlari, hacim,
surface alani, candidate score dagilimi) and GRUP-CAPRAZ OOF with prediction edilir -- own parcasini
gormez.

KILL (onceden yazildi): detection +0.01 VE grup bootstrap GA'si sifiri disliyor.
Uretici-disi DUSMEMELI (this havuzda havuzlanmisi iyilestiren each arm WEI-disini cokertti).
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def parca_oz (P ,X ,score ,diag ):
    """Parca duzeyi features -- K'yi prediction etmek for."""
    s =np .sort (score )[::-1 ]
    q =lambda a :float (np .percentile (score ,a ))if len (score )else 0.0 
    return [float (len (P )),float (diag ),
    float (np .ptp (P [:,0 ]))if len (P )else 0.0 ,
    float (np .ptp (P [:,1 ]))if len (P )else 0.0 ,
    float (np .ptp (P [:,2 ]))if len (P )else 0.0 ,
    float (score .mean ()),float (score .std ()),q (90 ),q (75 ),q (50 ),q (25 ),
    float ((score >=0.5 ).sum ()),float ((score >=0.3 ).sum ()),
    float ((score >=0.5 *max (score .max (),1e-9 )).sum ()),
    float (s [0 ]-s [min (1 ,len (s )-1 )])if len (s )>1 else 0.0 ,
    float (X [:,:13 ].mean (0 ).sum ()),float (np .median (X [:,5 ]))if X .shape [1 ]>5 else 0.0 ]


def main ():
    import gate_bench as T 
    import measure_set 
    import wire_gate 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor 
    from sklearn .model_selection import GroupKFold 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    X ,y ,pid ,keep =D ["X"],D ["y"],D ["pid"],D ["keep"]
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u in np .unique (pid ):
        i =np .where (pid ==u )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],y [keep ]),
    "n_feat":Z .shape [1 ],"donusum":D ["donusum"]}

    # --- part duzeyi tablo
    R ,OZ ,KGT ,GG ,PID =[],[],[],[],[]
    for r in D ["DER"]:
        if r ["X"]is None or r .get ("XR")is None :
            continue 
        Xr =np .hstack ([r ["X"],r ["XR"]])
        sk =wire_gate .decision_score (gate ,Xr )
        P =np .asarray (r ["P"],float )
        R .append ({"r":r ,"X":Xr ,"score":sk ,"P":P })
        OZ .append (parca_oz (P ,Xr ,sk ,float (r ["diag"])))
        KGT .append (len (r ["G"]));GG .append (r ["geo"]);PID .append (r ["pid"])
    OZ =np .array (OZ ,float );KGT =np .array (KGT ,float );GG =np .array (GG )
    print (f"\n{len (R )} part | GT sayisi: medyan {np .median (KGT ):.0f} "
    f"ort {KGT .mean ():.1f} max {KGT .max ():.0f}")

    # --- K REGRESORU (grup-capraz OOF: own parcasini GORMEZ)
    Kp =np .zeros (len (OZ ))
    for tr ,te in GroupKFold (n_splits =5 ).split (OZ ,KGT ,GG ):
        Kp [te ]=RandomForestRegressor (n_estimators =400 ,min_samples_leaf =2 ,n_jobs =-1 ,
        random_state =0 ).fit (OZ [tr ],KGT [tr ]).predict (OZ [te ])
    error =np .abs (Kp -KGT )
    print (f"K tahmini: MAE {error .mean ():.2f} | tam isabet {float ((np .round (Kp )==KGT ).mean ()):.1%}"
    f" | +-1 icinde {float ((error <=1 ).mean ()):.1%}")
    # baseline karsilastirmasi: mevcut rule kac candidate seciyor
    n_urun =[]
    for d_ in R :
        sk =d_ ["score"]
        n_urun .append (int (((sk >=0.5 *max (sk .max (),1e-9 ))&(sk >=0.25 )).sum ()))
    n_urun =np .array (n_urun ,float )
    print (f"URUN kuralinin sectigi: MAE {np .abs (n_urun -KGT ).mean ():.2f} "
    f"(K regresoru {error .mean ():.2f})")

    def puanla (rule_ ,kk =None ):
        det ,rob ,gg =[],[],[]
        for n_ ,d_ in enumerate (R ):
            r =d_ ["r"];sk =d_ ["score"];P =d_ ["P"];Xr =d_ ["X"]
            if rule_ =="urun":
                m =(sk >=0.5 *max (sk .max (),1e-9 ))&(sk >=0.25 )
            elif rule_ =="topk":
                K =int (max (1 ,round (kk [n_ ])))
                m =np .zeros (len (sk ),bool )
                m [np .argsort (sk )[::-1 ][:K ]]=True 
            else :# KARMA
                K =int (max (1 ,round (kk [n_ ])))
                m =np .zeros (len (sk ),bool )
                m [np .argsort (sk )[::-1 ][:K ]]=True 
                m &=sk >=0.15 
            Pp =P [m ];Pd =np .asarray (r ["Pd"],float )[m ]
            if m .any ():
                c =[{"point":Pp [i ],"direction":Pd [i ]}for i in range (len (Pp ))]
                c =wire_gate .pose_correct (Xr [m ],c )
                c =wire_gate .angle_correct (Xr [m ],c )
                if r .get ("UYE"):
                    c =wire_gate .pick_member_direction (Xr [m ],c ,r ["UYE"])
                Pp =np .array ([x ["point"]for x in c ],float )
                Pd =np .array ([x ["direction"]for x in c ],float )
            rj ="very"if r ["n"]>=8 else "low"
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            det .append ((rj ,)+esle (Pp ,Pd ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (Pp ,Pd ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ))
            gg .append (r ["geo"])
        return det ,rob ,gg 

    print (f"\n{'rule':<14}{'detection':>9}{'robot':>9}{'onceki farka GA':>26}")
    d0 ,r0 ,gg =puanla ("urun")
    print (f"{'URUN':<14}{T .f1w (d0 ):>9.4f}{T .f1w (r0 ):>9.4f}")
    SON ={"urun":{"detection":T .f1w (d0 ),"robot":T .f1w (r0 )}}
    fn =lambda rows :T .f1w ([q for _ ,q in rows ])-T .f1w ([p for p ,_ in rows ])
    for ad in ("topk","karma"):
        d1 ,r1 ,_ =puanla (ad ,Kp )
        _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (d0 ,d1 )),gg ,fn ,n =3000 )
        print (f"{ad .upper ():<14}{T .f1w (d1 ):>9.4f}{T .f1w (r1 ):>9.4f}"
        f"   [{lo :+.4f},{hi :+.4f}] {'GERCEK'if (lo >0 or hi <0 )else 'noise'}")
        SON [ad ]={"detection":T .f1w (d1 ),"robot":T .f1w (r1 ),"ga":[lo ,hi ]}

        # --- KAHIN: K MUKEMMEL bilinseydi
    dK ,rK ,_ =puanla ("topk",KGT )
    print (f"{'KAHIN (K=GT)':<14}{T .f1w (dK ):>9.4f}{T .f1w (rK ):>9.4f}   <- K mukemmel olsa")
    SON ["kahin_K"]={"detection":T .f1w (dK ),"robot":T .f1w (rK )}
    en =max (("topk","karma"),key =lambda a :SON [a ]["detection"])
    d =SON [en ]["detection"]-SON ["urun"]["detection"]
    gecti =d >=0.01 and SON [en ]["ga"][0 ]>0 
    print (f"\nKILL: detection +0.01 VE GA>0 -> {'GECTI'if gecti else 'GECMEDI'} ({en } {d :+.4f})")
    with io .open ("results/s1_topk.json","w",encoding ="utf-8")as f :
        json .dump ({"sonuc":SON ,"K_MAE":float (error .mean ()),
        "urun_MAE":float (np .abs (n_urun -KGT ).mean ()),
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/s1_topk.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
