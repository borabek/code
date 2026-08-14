# -*- coding: utf-8 -*-
"""Q2: POSE HEAD UCTAN UCA -- "kararsizsa dokunma" somutlastirilmis hali.

Q1 (fold-disi, 1068 matched candidate):
    YANAL  <=2mm  74.0% -> 77.6%   (+3.6 score, CALISIYOR)
    YON    <=10d  72.6% -> 73.4%   (+0.8) but MEDYAN 0.00 -> 2.36 deg = ZATEN DOGRU olanlari BOZUYOR

Yon kafasinin kusuru this gecenin four olu kolununkiyle same sinif: correction HERKESE uygulaniyor.
Fark this ki residual duzeltmenin BUYUKLUGU prediction ediliyor -- i.e. "only gerekliyse dokun" kurali
ogrenilmis a buyukluk uzerinden yazilabilir.

KOLLAR:
  A  correction YOK (baseline)
  B  only YANAL, esiksiz
  C  only YANAL, |prediction| >= threshold whereas (threshold taranir)
  D  YANAL + YON, ikisi de esikli

KILL (onceden yazili): robot-hazir >= +0.02 VE detection kaybi < 0.005, GA with kanitli.
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
DERZ ="results/_der_zengin.pkl"
MAKS_MM =3.0 


def local_frame (d ):
    d =np .asarray (d ,float );d =d /(np .linalg .norm (d )+1e-9 )
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (d @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (d ,a );u /=np .linalg .norm (u )+1e-9 
    return d ,u ,np .cross (d ,u )


class _Sabit :
    """Tum gruplar for same (sizintisiz) pose modeli."""
    def __init__ (self ,m ):self .m =m 
    def get (self ,_g ):return self .m 


def main ():
    import measure_set 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor 
    from sklearn .model_selection import GroupKFold 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    DER ,rap =measure_set .cluster (DERZ )
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    gk =measure_set .geo_anahtarlari ()
    tg ={r ["geo"]for r in DER }
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    zen =np .load ("results/zengin_parite.npz",allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tpid =np .array ([str (x )for x in zen ["pids"]])
    tgrp =np .array ([gk .get (p ,"absent:"+p )for p in tpid ])
    keep =~np .isin (tgrp ,list (tg ))

    def donustur (X ,pidler ):
        if not DON :
            return X 
        Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
        for u in np .unique (pidler ):
            i =np .where (pidler ==u )[0 ]
            Z [i ]=wire_gate .within_part (X [i ],DON )
        return Z 
    Mt =donustur (Xt ,tpid )
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Mt [keep ],ytr [keep ])
    gate ={"clf":clf ,"n_feat":Mt .shape [1 ],"donusum":DON }

    # BUYUK POSE VERISI (q3): 1068 -> 6330 row, 169 -> 991 grup. Ag cikarimi olmadan
    # full korpustan cikarildi (hizalama onbellekteki remesh with; 18 parcada maks difference 0.019mm).
    pv =np .load ("results/pose_veri.npz",allow_pickle =True )
    RX ,RY =np .asarray (pv ["X"],float ),np .asarray (pv ["Y"],float )
    RG =np .array ([str (x )for x in pv ["geo"]]);RPID =np .array ([str (x )for x in pv ["pid"]])
    # SIZINTI KAPISI: measurement parcalarinin geometri gruplari pose egitiminden de CIKARILIR.
    _disi =~np .isin (RG ,list (tg ))
    print (f"pose havuzu: {len (RY )} satir -> leakage disi {int (_disi .sum ())}")
    RX ,RY ,RG ,RPID =RX [_disi ],RY [_disi ],RG [_disi ],RPID [_disi ]

    # POSE modeli: measurement parcalarinin GEOMETRI GRUBU DISINDA kalanlarla egitilir.
    # Her measurement parcasi for own grubu haric egitilmis model is used (grup-disi).
    print (f"pose training havuzu: {len (RY )} candidate / {len (set (RG ))} grup",flush =True )
    # Olcum gruplari havuzdan CIKARILDIGI for single model sizintisizdir.
    _tek =RandomForestRegressor (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ).fit (RX ,RY )
    POSE =_Sabit (_tek )
    print (f"pose modeli hazir ({len (RY )} satir, measurement gruplari HARIC)",flush =True )

    def puanla (arm ,esik_mm =0.0 ,esik_deg =0.0 ,direction =False ):
        det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                X58 =np .hstack ([r ["X"],r ["XR"]])
                s =wire_gate .decision_score (gate ,X58 )
                k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    if arm !="A":
                        mdl =POSE .get (r ["geo"])
                        if mdl is not None :
                            pr =mdl .predict (X58 [k ])
                            for i in range (len (P )):
                                d ,u ,v =local_frame (Pd [i ])
                                dw =pr [i ,0 ]*u +pr [i ,1 ]*v 
                                n =float (np .linalg .norm (dw ))
                                if n >=esik_mm and n >0 :
                                    P [i ]=P [i ]+dw *(min (n ,MAKS_MM )/n )
                                if direction :
                                    g =d +pr [i ,2 ]*u +pr [i ,3 ]*v 
                                    ac =np .degrees (np .arccos (np .clip (
                                    abs (float (g @d ))/(np .linalg .norm (g )+1e-9 ),0 ,1 )))
                                    if ac >=esik_deg :
                                        Pd [i ]=g /(np .linalg .norm (g )+1e-9 )
            rj ="very"if r ["n"]>=8 else "low"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    KOL =[("A correction absent",dict ()),
    ("B lateral, esiksiz",dict (esik_mm =0.0 )),
    ("C lateral, >=0.5mm",dict (esik_mm =0.5 )),
    ("C lateral, >=1.0mm",dict (esik_mm =1.0 )),
    ("C lateral, >=1.5mm",dict (esik_mm =1.5 )),
    ("D lateral+direction >=10d",dict (esik_mm =1.0 ,esik_deg =10.0 ,direction =True )),
    ("D lateral+direction >=20d",dict (esik_mm =1.0 ,esik_deg =20.0 ,direction =True ))]
    print (f"\n{'arm':<22}{'detection':>10}{'ROBOT':>10}{'d_tespit':>10}{'d_robot':>10}")
    SON ,PARCA ={},{}
    for ad ,kw in KOL :
        det ,rob =puanla ("A"if ad .startswith ("A")else "X",**kw )
        SON [ad ]=(float (f1w (det )),float (f1w (rob )))
        PARCA [ad ]=(det ,rob ,[x ["geo"]for x in DER ])
        a =SON ["A correction absent"]
        print (f"{ad :<22}{SON [ad ][0 ]:>10.4f}{SON [ad ][1 ]:>10.4f}"
        f"{SON [ad ][0 ]-a [0 ]:>+10.4f}{SON [ad ][1 ]-a [1 ]:>+10.4f}",flush =True )

    a =SON ["A correction absent"]
    en =max ((k for k in SON if not k .startswith ("A")),key =lambda k :SON [k ][1 ])
    dr =SON [en ][1 ]-a [1 ];dt =SON [en ][0 ]-a [0 ]
    da ,_ ,g =PARCA ["A correction absent"];db ,rb ,_ =PARCA [en ]
    _ ,ra ,_ =PARCA ["A correction absent"]
    cift =list (zip (ra ,rb ))
    fn =lambda rows :f1w ([y for _ ,y in rows ])-f1w ([x for x ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (cift ,g ,fn ,n =2000 )
    print (f"\nEN IYI: {en } | robot {dr :+.4f} [{lo :+.4f}, {hi :+.4f}] | detection {dt :+.4f}")
    gecti =dr >=0.02 and dt >-0.005 and lo >0 
    print (f"KILL: robot >= +0.02 VE detection kaybi < 0.005 VE GA sifiri dislamali -> "
    f"{'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/q2_pose_end_to_end.json","w",encoding ="utf-8")as f :
        json .dump ({"kollar":{k :list (v )for k ,v in SON .items ()},"en_iyi":en ,
        "d_robot":float (dr ),"d_tespit":float (dt ),
        "ga":[float (lo ),float (hi )],"gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/q2_pose_end_to_end.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
