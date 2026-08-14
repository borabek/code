# -*- coding: utf-8 -*-
"""T3: PARCA BASINA KESIMI OGREN (tespit tavanina giden most ucuz path).

T2 bosslugu ikiye boldu:
    ESIK kaybi   +0.0731 (%38) -- rule degisikligiyle alinabilir
    SIRALAMA     +0.1202 (%62) -- new bilgi is required

Ve kritik ayrinti: TEK a sabit threshold no sey vermiyor (most iyi global threshold +0.0039,
most iyi goreli ratio +0.0000). Yani loss "esigi wrong sectik"ten DEGIL, "each parcada correct
kesim different"dan geliyor. Kahin part-ici kesim +0.0731.

Bugun same desen IKI KEZ whereas yaradi: kurali sabit a sayiya not, parcanin KENDI sinyaline
baglamak (cokus yonlendirmesi) and ogrenilmis selector (yarik yonu). Burada da same sey:
parcanin SKOR DAGILIMINDAN kac candidate tutulacagini ogren.

OZELLIKLER (all of them calisma aninda, GT'siz):
    n_aday, maks, mean, std, medyan
    first-ikinci farki, first-ucuncu farki  (vertex ne up to ayrik?)
    threshold ustu numbers (0.25/0.35/0.5/0.7)
    skor dusus profili: sirali skorlarin ardisik farklarinin maksimumu and yeri (DOGAL KESIM)
    kosegen, very-CP router bayragi

TARGET: kahin K (F1'i at most yapan tutulan candidate count).
DEGERLENDIRME: geometri anahtarina according to GroupKFold, FOLD DISI prediction.

KILL (onceden yazili): tespit >= +0.02 VE DEV with VAL same yonde VE gorulmemis manufacturer
ortalamasi dusmeyecek. Ucu birden saglanmazsa DAGITILMAZ.
"""
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def match3 (P ,Pd ,G ,Gd ,diag ):
    hit =np .zeros (len (G ),bool );used =set ()
    if len (P )and len (G ):
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *diag )
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if d_ >tt or a_ in used or hit [b_ ]:
                continue 
            hit [b_ ]=True ;used .add (a_ )
    tp =int (hit .sum ())
    return tp ,len (P )-tp ,len (G )-tp 


def ozellik (s ,diag ,is_hi ):
    """Parcanin SKOR DAGILIMINDAN kesim ipuclari (GT'ye BAKMAZ)."""
    v =np .sort (np .asarray (s ,float ))[::-1 ]
    n =len (v )
    pad =lambda i :float (v [i ])if i <n else 0.0 
    dif =np .diff (v )*-1.0 if n >1 else np .array ([0.0 ])
    return [float (n ),float (v .max ()),float (v .mean ()),float (v .std ()),float (np .median (v )),
    pad (0 )-pad (1 ),pad (0 )-pad (2 ),pad (1 )-pad (2 ),
    float ((v >=0.25 ).sum ()),float ((v >=0.35 ).sum ()),
    float ((v >=0.50 ).sum ()),float ((v >=0.70 ).sum ()),
    float (dif .max ()),float (int (np .argmax (dif ))+1 )if n >1 else 1.0 ,
    float (diag ),float (bool (is_hi ))]


def main ():
    import robot_cp 
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import f1w 
    from sklearn .ensemble import RandomForestRegressor ,RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    with open ("results/_dev_val_kume.json",encoding ="utf-8")as f :
        kume_of =json .load (f )
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    keep =~np .isin (tr_grp ,list ({gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }))
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    rf0 =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [keep ],ytr [keep ])
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],dag .get ("donusum_z","zskor"))
    m ={"clf":rf0 (Xtr ),"clf_z":rf0 (Ztr ),"n_feat":Xtr .shape [1 ],
    "donusum":dag .get ("donusum"),"donusum_z":dag .get ("donusum_z","zskor")}
    mx =[float (m ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
    for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
    m ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))

    PAR =[]
    for r in DER :
        if r ["X"]is None or not len (r ["G"]):
            continue 
        s =np .asarray (wire_gate .decision_score (m ,r ["X"]),float )
        o =np .argsort (-s )
        PAR .append ({"pid":r ["pid"],"geo":gk .get (r ["pid"],"yok:"+r ["pid"]),
        "cluster":kume_of .get (r ["pid"],"dev"),"mfg":mfg_of .get (r ["pid"],"?"),
        "rj":"cok"if r ["n"]>=8 else "dusuk",
        "s":s [o ],"P":r ["P"][o ],"Pd":r ["Pd"][o ],
        "G":np .asarray (r ["G"],float ),"Gd":np .asarray (r ["Gd"],float ),
        "diag":float (r ["diag"]),"hi":r ["n"]>=8 })
    print (f"{len (PAR )} part",flush =True )

    # KAHIN K: F1'i at most yapan tutulan candidate count (skor sirasina according to first K)
    for p in PAR :
        en ,enk =-1 ,1 
        for K in range (0 ,len (p ["s"])+1 ):
            tp ,fp ,fn =match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"])
            f =2 *tp /max (2 *tp +fp +fn ,1 )
            if f >en :
                en ,enk =f ,K 
        p ["kahin_K"]=enk 
        p ["X"]=ozellik (p ["s"],p ["diag"],p ["hi"])
    X =np .array ([p ["X"]for p in PAR ],float )
    yk =np .array ([p ["kahin_K"]for p in PAR ],float )
    grp =np .array ([p ["geo"]for p in PAR ])
    su_K =np .array ([int (wire_gate .decision_mask (p ["s"]).sum ())for p in PAR ],float )
    print (f"kahin K: medyan {np .median (yk ):.1f} | su anki kural K: medyan {np .median (su_K ):.1f} "
    f"| ort fark {np .mean (yk -su_K ):+.2f}",flush =True )

    oof =np .zeros (len (PAR ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,yk ,grp ):
        oof [te ]=RandomForestRegressor (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],yk [tr ]).predict (X [te ])

    def puanla (Ks ,alt =None ):
        rows =[]
        for p ,K in zip (PAR ,Ks ):
            if alt is not None and p ["mfg"]!=alt :
                continue 
            K =int (max (0 ,min (len (p ["s"]),round (K ))))
            rows .append ((p ["rj"],)+match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"]))
        return f1w (rows )if rows else float ("nan")

    def puanla_kume (Ks ,km ):
        rows =[]
        for p ,K in zip (PAR ,Ks ):
            if p ["cluster"]!=km :
                continue 
            K =int (max (0 ,min (len (p ["s"]),round (K ))))
            rows .append ((p ["rj"],)+match3 (p ["P"][:K ],p ["Pd"][:K ],p ["G"],p ["Gd"],p ["diag"]))
        return f1w (rows )if rows else float ("nan")

    a =puanla (su_K )
    kah =puanla (yk )
    yeni =puanla (oof )
    print (f"\n{'arm':<26}{'TESPIT':>9}{'DEV':>9}{'VAL':>9}{'PXC':>9}{'WEI':>9}")
    for ad ,K in (("A su anki kural",su_K ),("B ogrenilmis kesim",oof ),("F KAHIN K",yk )):
        print (f"{ad :<26}{puanla (K ):>9.4f}{puanla_kume (K ,'dev'):>9.4f}{puanla_kume (K ,'val'):>9.4f}"
        f"{puanla (K ,'PXC'):>9.4f}{puanla (K ,'WEI'):>9.4f}")
    d_ =yeni -a 
    dev_ok =puanla_kume (oof ,"dev")-puanla_kume (su_K ,"dev")
    val_ok =puanla_kume (oof ,"val")-puanla_kume (su_K ,"val")
    mfg_a =np .mean ([puanla (su_K ,"PXC"),puanla (su_K ,"WEI")])
    mfg_b =np .mean ([puanla (oof ,"PXC"),puanla (oof ,"WEI")])
    gecti =(d_ >=0.02 )and (dev_ok >0 )and (val_ok >0 )and (mfg_b >=mfg_a )
    print (f"\nKILL: tespit >= +0.02 VE DEV/VAL ayni yonde VE manufacturer ort. dusmeyecek")
    print (f"  tespit {d_ :+.4f} | DEV {dev_ok :+.4f} | VAL {val_ok :+.4f} | "
    f"manufacturer ort {mfg_b -mfg_a :+.4f} -> {'GECTI'if gecti else 'GECMEDI'}")
    print (f"  kahinin {d_ /max (kah -a ,1e-9 ):.0%}'i yakalandi (kahin {kah -a :+.4f})")
    with open ("results/t3_part_kesimi.json","w",encoding ="utf-8")as f :
        json .dump ({"su_an":float (a ),"ogrenilmis":float (yeni ),"kahin":float (kah ),
        "dev":float (dev_ok ),"val":float (val_ok ),
        "uretici_ort_fark":float (mfg_b -mfg_a ),"gecti":bool (gecti )},f ,indent =1 )
    print ("\nmakbuz -> results/t3_part_kesimi.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
