# -*- coding: utf-8 -*-
"""T10: mutlak mm sutunlarini OLCEK-BAGIMSIZ yapmak transferi fixes mi?

T9 OLCTU (manufacturer imzasi = column single basina "hangi manufacturer" sorusunu ne up to cevapliyor):
    bos_derinlik 0.641 | ce_frac 0.620 | depth 0.612 | flat 0.604 | size 0.603 | brep_r 0.584
    safe cekirdek: nn_dist 0.504, n_close 0.513, esesenli 0.515, nverts 0.520
Imzali 4 sutunu ATMAK most kotu durumu 0.2799 -> 0.3121 does (+0.032) but ATMAK bilgiyi de
goturuyor -- L2 dersi: ezberci damgasi single basina atmak for yeterli not.

HIPOTEZ: `size`, `depth`, `nn_dist`, `bos_derinlik` MUTLAK mm. Bir ureticinin klemensleri
tipik as belli boyutta oldugu for this sutunlar ureticiyi ELE VERIYOR. Parca capina
bolununce IMZA silinir, ORAN bilgisi kalir.

KARSI HIPOTEZ (same derecede ciddi): mutlak buyukluk sometimes GERCEKTEN correct olcektir -- tel capi
ureticiden bagimsiz as 1-3mm'dir. T1 already fiziksel sutunlarin transferi +0.047 IYILESTIRDIGINI
olctu. O yuzden `brep_r`'ye DOKUNULMUYOR; only part-olcegine bagli olanlar normalize ediliyor.

UC KOL: (a) mevcut, (b) normalize EDILMIS (instead of koy), (c) HER IKISI (model secsin).
KILL: a arm, manufacturer-disi EN KOTU durumda mevcudu gecmezse and familiar veride 0.02'den extra
kaybettirmezse alinmaz.
"""
import json ,os 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 


def main ():
    import wire_gate 
    d =np .load ("results/gate_regrow_data_fiz.npz",allow_pickle =True )
    X =d ["X"][:,:18 ].copy ();y =d ["y"].astype (bool )
    mfg =np .array ([str (x )for x in d ["mfg"]]);pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_strict_geometry_keys.json"))
    grp =np .array ([gk .get (p ,"absent:"+p )for p in pids ])
    THR =float (json .load (open ("cp_config.json",encoding ="utf-8"))["robot_wire_gate_threshold"])
    isim =list (wire_gate .FEAT_NAMES_13 )+list (wire_gate .FEAT_NAMES_FIZ )

    # part capi: mesh onbelleginden; otherwise that parcanin adaylarinin yayilimindan (backup)
    cap ={}
    for p in np .unique (pids ):
        f =f"results/mesh_cache/{p }.npz"
        if os .path .exists (f ):
            try :
                V =np .load (f )["V"]
                cap [p ]=float (np .linalg .norm (V .max (0 )-V .min (0 )))
                continue 
            except Exception :
                pass 
        P =d ["pts"][pids ==p ]
        cap [p ]=float (np .linalg .norm (P .max (0 )-P .min (0 )))if len (P )>1 else 50.0 
    capv =np .array ([max (cap [p ],1e-6 )for p in pids ])
    print (f"part capi: {sum (1 for p in np .unique (pids )if os .path .exists (f'results/mesh_cache/{p }.npz'))}"
    f"/{len (np .unique (pids ))} mesh onbelleginden | medyan {np .median (list (cap .values ())):.1f}mm")

    NORM =["size","depth","nn_dist","bos_derinlik"]# brep_r'ye DOKUNULMUYOR
    idx =[isim .index (n )for n in NORM ]
    Xn =X .copy ()
    for i in idx :
        Xn [:,i ]=X [:,i ]/capv 
    Xb =np .hstack ([X ,X [:,idx ]/capv [:,None ]])

    def olc (M ,ad ):
        o =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (M ,y ,grp ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [tr ],y [tr ]).predict_proba (M [te ])[:,1 ]
        def f1 (s ,yy ):
            m =s >=THR 
            tp =int ((yy &m ).sum ());fp =int ((~yy &m ).sum ());fn =int ((yy &~m ).sum ())
            p_ =tp /max (tp +fp ,1 );r_ =tp /max (tp +fn ,1 )
            return 2 *p_ *r_ /max (p_ +r_ ,1e-9 )
        defn =f1 (o ,y )
        mv =[]
        for u in sorted (set (mfg )):
            te =mfg ==u 
            s =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (M [~te ],y [~te ]).predict_proba (M [te ])[:,1 ]
            mv .append (f1 (s ,y [te ]))
        print (f"{ad :<30}{defn :>9.4f}"+"".join (f"{x :>10.4f}"for x in mv )+f"{min (mv ):>10.4f}")
        return {"familiar":defn ,"manufacturer":mv ,"en_kotu":min (mv )}

    U =sorted (set (mfg ))
    print (f"\n{'arm':<30}{'TANIDIK':>9}"+"".join (f"{'uret.'+u :>10}"for u in U )+f"{'EN KOTU':>10}")
    out ={}
    out ["mevcut (mutlak)"]=olc (X ,"mevcut (mutlak)")
    out ["normalize (instead of)"]=olc (Xn ,"normalize (instead of)")
    out ["each ikisi (22 column)"]=olc (Xb ,"each ikisi (22 column)")

    t =out ["mevcut (mutlak)"]
    print ("\nKARAR:")
    for ad ,v in out .items ():
        if ad =="mevcut (mutlak)":
            continue 
        gecti =v ["en_kotu"]>t ["en_kotu"]and (t ["familiar"]-v ["familiar"])<=0.02 
        print (f"  {ad :<24} en kotu {v ['en_kotu']-t ['en_kotu']:+.4f} | familiar "
        f"{v ['familiar']-t ['familiar']:+.4f} -> {'GECTI'if gecti else 'GECMEDI'}")
    json .dump (out ,open ("results/t10_scale_bagimsiz.json","w"),indent =1 )
    print ("receipt -> results/t10_scale_bagimsiz.json")


if __name__ =="__main__":
    main ()
