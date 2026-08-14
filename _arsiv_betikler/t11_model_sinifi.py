# -*- coding: utf-8 -*-
"""T11: model SINIFI manufacturer transferini degistiriyor mu?

T10 OLCTU: scale-bagimsizlastirma transferi duzeltmedi (+0.0015 / +0.0032 = noise). Yani
manufacturer imzasi mutlak olcekten gelmiyor.

CERCEVE DUZELTMESI (onemli): elimizde YALNIZ IKI manufacturer present. "Uretici-disi" testi fiilen
"TEK a manufacturer with egit, baskasinda dene" demek -- mumkun which is EN ZOR ayar. Gercek dagitimda
ucuncu a manufacturer gelirse model IKISIYLE egitilmis olacak. O yuzden 0.4736 KOTUMSER a lower boundary.

BU BETIK: transfer, MODEL SINIFI with degisiyor mu? Kayitli bulgu (2026-07-29): ExtraTrees
aile-disi +0.012 verirken manufacturer-disi -0.061 kaybetmisti -> urunde RandomForest. Ama that measurement
13 sutunluydu; simdi 18 column and GORELI ESIK present. Ayrica never denenmemis kollar: more SIG
ormanlar (depth siniri), more large yaprak, lojistik regresyon.

Hipotez: more kisitli model = more few ezber = more iyi transfer. Bedeli tanidik veride.
KILL: a model, manufacturer-disi EN KOTU durumda RF'i gecmeli VE tanidik veride 0.02'den extra
kaybettirmemeli.
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier ,ExtraTreesClassifier ,HistGradientBoostingClassifier 
from sklearn .linear_model import LogisticRegression 
from sklearn .pipeline import make_pipeline 
from sklearn .preprocessing import StandardScaler 
from sklearn .model_selection import GroupKFold 


def main ():
    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    X =d ["X"][:,:18 ];y =d ["y"].astype (bool )
    mfg =np .array ([str (x )for x in d ["mfg"]]);pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    grp =np .array ([gk .get (p ,"yok:"+p )for p in pids ])
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    THR =float (cfg ["robot_wire_gate_threshold"])
    ORAN =float (cfg .get ("gate_goreli_oran",0.5 ));TABAN =float (cfg .get ("gate_goreli_taban",0.25 ))

    def karar (s ,p ):
        """DAGITILAN rule: part-ici goreli + mutlak baseline."""
        m =np .zeros (len (s ),bool )
        for u in np .unique (p ):
            i =p ==u ;v =s [i ]
            m [i ]=(v >=ORAN *max (v .max (),1e-9 ))&(v >=TABAN )
        return m 

    def f1 (m ,yy ):
        tp =int ((yy &m ).sum ());fp =int ((~yy &m ).sum ());fn =int ((yy &~m ).sum ())
        p_ =tp /max (tp +fp ,1 );r_ =tp /max (tp +fn ,1 )
        return 2 *p_ *r_ /max (p_ +r_ ,1e-9 )

    MOD ={
    "RF leaf3 (mevcut)":lambda :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,
    n_jobs =-1 ,random_state =0 ),
    "RF leaf20":lambda :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =20 ,
    n_jobs =-1 ,random_state =0 ),
    "RF depth 6":lambda :RandomForestClassifier (n_estimators =400 ,max_depth =6 ,
    n_jobs =-1 ,random_state =0 ),
    "RF depth 4":lambda :RandomForestClassifier (n_estimators =400 ,max_depth =4 ,
    n_jobs =-1 ,random_state =0 ),
    "ExtraTrees leaf3":lambda :ExtraTreesClassifier (n_estimators =400 ,min_samples_leaf =3 ,
    n_jobs =-1 ,random_state =0 ),
    "HistGB":lambda :HistGradientBoostingClassifier (random_state =0 ),
    "Lojistik":lambda :make_pipeline (StandardScaler (),
    LogisticRegression (max_iter =2000 ,random_state =0 )),
    }
    U =sorted (set (mfg ))
    print (f"{'model':<22}{'TANIDIK':>9}"+"".join (f"{'uret.'+u :>10}"for u in U )+f"{'EN KOTU':>10}")
    out ={}
    for ad ,mk in MOD .items ():
        o =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,grp ):
            o [te ]=mk ().fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
        tan =f1 (karar (o ,pids ),y )
        mv =[]
        for u in U :
            te =mfg ==u 
            s =mk ().fit (X [~te ],y [~te ]).predict_proba (X [te ])[:,1 ]
            mv .append (f1 (karar (s ,pids [te ]),y [te ]))
        print (f"{ad :<22}{tan :>9.4f}"+"".join (f"{x :>10.4f}"for x in mv )+f"{min (mv ):>10.4f}",
        flush =True )
        out [ad ]={"tanidik":tan ,"manufacturer":mv ,"en_kotu":min (mv )}
    t =out ["RF leaf3 (mevcut)"]
    print ("\nKARAR (kill: en kotu > mevcut VE tanidik loss <= 0.02):")
    for ad ,v in out .items ():
        if ad .startswith ("RF leaf3"):
            continue 
        g =v ["en_kotu"]>t ["en_kotu"]and (t ["tanidik"]-v ["tanidik"])<=0.02 
        print (f"  {ad :<22} en kotu {v ['en_kotu']-t ['en_kotu']:+.4f} | tanidik "
        f"{v ['tanidik']-t ['tanidik']:+.4f} -> {'GECTI'if g else 'GECMEDI'}")
    json .dump (out ,open ("results/t11_model_sinifi.json","w"),indent =1 )
    print ("receipt -> results/t11_model_sinifi.json")


if __name__ =="__main__":
    main ()
