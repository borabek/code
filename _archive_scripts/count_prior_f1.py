# -*- coding: utf-8 -*-
"""Sayi onculu URUN metrigine ne kazandiriyor? (CP-F1, leakage-siz)

BAGLAM: metadata modu (real CP count biliniyorsa top-N) olculmus +0.025 veriyor but katalog
verisi gerektiriyor. count_prior.py showed ki geometri sayiyi +-1 dogrulukla %77 biliyor
(MAE 1.30, baseline 6.04). Bu betik, that TAHMINI sayiyla ne kadarini geri alabildigimizi olcer.

KOLLAR
  A) baseline        : gate esigi (urunun bugunku davranisi)
  B) top-N GERCEK : real CP count bilinseydi (TAVAN -- ulasilamaz, referans)
  C) top-N TAHMIN : geometriden prediction edilen number (YENI KALDIRAC)
  D) yumusak      : tahmine according to part basina threshold kaydirmasi (sert top-N'den more safe)

SIZINTI: hem gate skoru hem number tahmini AILE-DISI OOF. Bir part, kendisini (ya da ailesini)
gormemis modellerle skorlanir.

KILL (olcumden before): corpus-agirlikli CP-F1 katkisi < +0.02 -> olu.
"""
import os ,sys ,json 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
NPZ ="results/gate_regrow_data.npz"
W ={"low":0.895 ,"very":0.105 }


def main ():
    from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor 
    from sklearn .model_selection import GroupKFold 
    import count_prior 

    d =np .load (NPZ ,allow_pickle =True )
    X ,y ,votes ,groups =d ["X"],d ["y"],d ["votes"],d ["groups"]
    fams =d ["fams"]
    ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    fam_of ={}
    for g ,f in zip (groups ,fams ):
        fam_of .setdefault (int (g ),str (f ))
    gkey =np .array ([fam_of [int (g )]for g in groups ])

    # --- OOF gate skoru (aile-disi) ---
    oof =np .zeros (len (y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,gkey ):
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ])
        oof [te ]=clf .predict_proba (X [te ])[:,1 ]

        # --- OOF number tahmini (aile-disi, AYNI split mantigi) ---
    Pf ,pids =count_prior .part_features (X ,votes ,groups )
    tgt =np .array ([ngt_of .get (int (p ),0 )for p in pids ],float )
    keep =tgt >0 
    Pf ,pids ,tgt =Pf [keep ],pids [keep ],tgt [keep ]
    pg =np .array ([fam_of [int (p )]for p in pids ])
    poof =np .zeros (len (tgt ))
    for tr ,te in GroupKFold (n_splits =5 ).split (Pf ,tgt ,pg ):
        rg =RandomForestRegressor (n_estimators =500 ,min_samples_leaf =2 ,n_jobs =-1 ,
        random_state =0 ).fit (Pf [tr ],tgt [tr ])
        poof [te ]=rg .predict (Pf [te ])
    pred_n ={int (p ):int (max (1 ,round (v )))for p ,v in zip (pids ,poof )}

    def cp_f1 (keep_mask_fn ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        for g in np .unique (groups ):
            m =groups ==g 
            n_gt =int (ngt_of .get (int (g ),0 ))
            if n_gt <=0 :
                continue 
            k ="very"if n_gt >=8 else "low"
            sel =keep_mask_fn (int (g ),oof [m ],y [m ])
            tp =int ((y [m ][sel ]==1 ).sum ())
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=int (sel .sum ())-tp ;agg [k ][2 ]+=n_gt -tp 
        out ={}
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );r =T /max (T +Fn ,1 )
            out [k ]=2 *p *r /max (p +r ,1e-9 )
        out ["weighted"]=sum (W [k ]*out [k ]for k in W )
        return out 

    def arm_threshold (thr ):
        return lambda g ,s ,yy :s >=thr 

    def arm_topn (getn ):
        def f (g ,s ,yy ):
            n =min (getn (g ),len (s ))
            sel =np .zeros (len (s ),bool )
            if n >0 :
                sel [np .argsort (-s )[:n ]]=True 
            return sel 
        return f 

    def arm_soft (getn ,lo =0.20 ,hi =0.60 ):
        """Esigi part basina kaydir: beklenenden COK candidate varsa threshold yukselir."""
        def f (g ,s ,yy ):
            n =max (1 ,getn (g ))
            for thr in np .arange (lo ,hi +1e-9 ,0.05 ):
                if int ((s >=thr ).sum ())<=n :
                    return s >=thr 
            return s >=hi 
        return f 

    true_n ={int (g ):int (ngt_of .get (int (g ),0 ))for g in np .unique (groups )}
    arms ={
    "A baseline (threshold 0.35)":arm_threshold (0.35 ),
    "B top-N GERCEK":arm_topn (lambda g :true_n .get (g ,0 )),
    "C top-N TAHMIN":arm_topn (lambda g :pred_n .get (g ,1 )),
    "D yumusak TAHMIN":arm_soft (lambda g :pred_n .get (g ,1 )),
    }
    print (f"{len (true_n )} part | {len (y )} candidate\n")
    print (f"{'arm':<22}{'low-CP':>10}{'very-CP':>10}{'agirlikli':>12}{'difference':>10}")
    base =None ;res ={}
    for name ,fn in arms .items ():
        r =cp_f1 (fn );res [name ]=r 
        if base is None :
            base =r ["weighted"]
        print (f"{name :<22}{r ['low']:>10.4f}{r ['very']:>10.4f}{r ['weighted']:>12.4f}"
        f"{r ['weighted']-base :>+10.4f}")
    live ={k :v for k ,v in res .items ()if k .startswith (("C","D"))}
    best =max (live .items (),key =lambda kv :kv [1 ]["weighted"])
    gain =best [1 ]["weighted"]-base 
    print (f"\nEN IYI UYGULANABILIR: {best [0 ]}  ({gain :+.4f})")
    print (f"TAVAN (gercek number): {res ['B top-N GERCEK']['weighted']-base :+.4f}")
    print (f"KAPI (>= +0.02): {'GECTI'if gain >=0.02 else 'OLU'}")
    json .dump ({"arms":{k :v for k ,v in res .items ()},"best":best [0 ],"gain":gain ,
    "ceiling":res ["B top-N GERCEK"]["weighted"]-base ,
    "kill_passed":bool (gain >=0.02 )},
    open ("results/count_prior_f1.json","w"),indent =1 )
    print ("receipt -> results/count_prior_f1.json")


if __name__ =="__main__":
    main ()
