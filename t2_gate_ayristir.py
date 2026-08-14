# -*- coding: utf-8 -*-
"""T2: gate'in +0.1933'luk bosslugu NEREDE -- SIRALAMADA mi, ESIKTE mi?

Tavan merdiveni (t_tavan.json) sunu showed:
    this an tespit 0.7439  ->  kahin GATE 0.9372   (+0.1933)
    candidate tavani da 0.9372  -> i.e. ADAYLAR ZATEN YETERLI, sorun SECIMDE.

Ama "secim" two ayri seydir and tedavileri BAMBASKA:
    SIRALAMA : gate correct adaylara high skor veriyor mu? (kotuyse -> new BILGI is required)
    ESIK     : skorlar iyi but kesme yeri mi kotu? (kotuyse -> UCUZ, rule degisir)

Bu betik bosslugu ikiye boler:
    A) this anki rule            : goreli threshold (0.5 x part-maks, baseline 0.25)
    B) EN IYI GLOBAL threshold       : tum korpusta single sabit threshold, most iyisi secilir
    C) EN IYI GORELI ratio       : goreli ratio taranir
    D) KAHIN PARCA-ICI threshold     : each part for EN IYI kesim (skorlara bakip)
    E) KAHIN top-K              : parcadaki GT count bilinseydi, most high K candidate
    F) KAHIN GATE               : correct adaylarin TAMAMI (siralamadan bagimsiz)

D with F arasindaki difference = SIRALAMANIN kaybi (skorlar correct adaylari uste tasiyamiyor).
A with D arasindaki difference  = ESIGIN kaybi (ucuz kazanc alani).
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
    """TESPIT modu -> (tp, fp, fn)."""
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


def main ():
    import wire_gate 
    from sina_cluster import f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    ORAN =float (cfg ["gate_goreli_oran"]);TABAN =float (cfg ["gate_goreli_taban"])
    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"none:"+p )for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    keep =~np .isin (tr_grp ,list ({gk .get (r ["pid"],"none:"+r ["pid"])for r in DER }))
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    rf =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [keep ],ytr [keep ])
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],dag .get ("donusum_z","zskor"))
    m ={"clf":rf (Xtr ),"clf_z":rf (Ztr ),"n_feat":Xtr .shape [1 ],
    "donusum":dag .get ("donusum"),"donusum_z":dag .get ("donusum_z","zskor")}
    mx =[float (m ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
    for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
    m ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))

    PAR =[]
    for r in DER :
        if r ["X"]is None or not len (r ["G"]):
            continue 
        s =wire_gate .decision_score (m ,r ["X"])
        PAR .append ({"rj":"very"if r ["n"]>=8 else "dusuk","s":np .asarray (s ,float ),
        "P":r ["P"],"Pd":r ["Pd"],"G":np .asarray (r ["G"],float ),
        "Gd":np .asarray (r ["Gd"],float ),"diag":float (r ["diag"]),
        "ngt":len (r ["G"])})
    print (f"{len (PAR )} part | {sum (len (p ['s'])for p in PAR )} candidate",flush =True )

    def puanla (selector ):
        rows =[]
        for p in PAR :
            k =selector (p )
            P =p ["P"][k ]if k .any ()else np .zeros ((0 ,3 ))
            Pd =p ["Pd"][k ]if k .any ()else np .zeros ((0 ,3 ))
            rows .append ((p ["rj"],)+match3 (P ,Pd ,p ["G"],p ["Gd"],p ["diag"]))
        return f1w (rows )

    SON ={}
    SON ["A su anki rule"]=puanla (lambda p :wire_gate .decision_mask (p ["s"]))

    en ,en_e =-1 ,None 
    for e in np .arange (0.05 ,0.96 ,0.05 ):
        v =puanla (lambda p ,e =e :p ["s"]>=e )
        if v >en :
            en ,en_e =v ,float (e )
    SON [f"B en iyi GLOBAL threshold ({en_e :.2f})"]=en 

    en2 ,en_o =-1 ,None 
    for o in np .arange (0.1 ,0.96 ,0.05 ):
        v =puanla (lambda p ,o =o :(p ["s"]>=o *max (p ["s"].max (),1e-9 ))&(p ["s"]>=TABAN ))
        if v >en2 :
            en2 ,en_o =v ,float (o )
    SON [f"C en iyi GORELI ratio ({en_o :.2f})"]=en2 

    def kahin_esik (p ):
        """Bu part for EN IYI kesim (skorlara according to) -- kahin, calisma aninda bilinemez."""
        candidate =sorted (set (p ["s"].tolist ())|{0.0 })
        en_k ,en_m =-1 ,np .zeros (len (p ["s"]),bool )
        for e in candidate :
            k =p ["s"]>=e 
            P =p ["P"][k ]if k .any ()else np .zeros ((0 ,3 ))
            Pd =p ["Pd"][k ]if k .any ()else np .zeros ((0 ,3 ))
            tp ,fp ,fn =match3 (P ,Pd ,p ["G"],p ["Gd"],p ["diag"])
            f =2 *tp /max (2 *tp +fp +fn ,1 )
            if f >en_k :
                en_k ,en_m =f ,k 
        return en_m 
    SON ["D KAHIN part-ici threshold"]=puanla (kahin_esik )

    def kahin_topk (p ):
        k =np .zeros (len (p ["s"]),bool )
        k [np .argsort (-p ["s"])[:p ["ngt"]]]=True 
        return k 
    SON ["E KAHIN top-K (number bilinse)"]=puanla (kahin_topk )

    def kahin_gate (p ):
        tp ,fp ,fn =match3 (p ["P"],p ["Pd"],p ["G"],p ["Gd"],p ["diag"])
        hit =np .zeros (len (p ["G"]),bool );used =set ()
        diff =p ["P"][:,None ,:]-p ["G"][None ,:,:]
        al =(diff *p ["Gd"][None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*p ["Gd"][None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *p ["diag"])
        k =np .zeros (len (p ["s"]),bool )
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
        for a in range (len (p ["P"]))for b in range (len (p ["G"]))):
            if d_ >tt or a_ in used or hit [b_ ]:
                continue 
            hit [b_ ]=True ;used .add (a_ );k [a_ ]=True 
        return k 
    SON ["F KAHIN gate (tam secim)"]=puanla (kahin_gate )

    print (f"\n{'arm':<34}{'TESPIT F1':>11}{'A'  'ya per':>12}")
    a =SON ["A su anki rule"]
    for k ,v in SON .items ():
        print (f"{k :<34}{v :>11.4f}{(v -a ):>+12.4f}")

    d_ =SON ["D KAHIN part-ici threshold"];f_ =SON ["F KAHIN gate (tam secim)"]
    print (f"\nBOSLUGUN AYRISTIRMASI (toplam {f_ -a :+.4f}):")
    print (f"  ESIK kaybi   (A -> D) : {d_ -a :+.4f}  "
    f"({(d_ -a )/max (f_ -a ,1e-9 ):.0%}) -- kural degisikligiyle alinabilir")
    print (f"  SIRALAMA kaybi (D -> F): {f_ -d_ :+.4f}  "
    f"({(f_ -d_ )/max (f_ -a ,1e-9 ):.0%}) -- YENI BILGI gerekir")
    print (f"\n  gercekci threshold kollari: B {SON [f'B en iyi GLOBAL threshold ({en_e :.2f})']-a :+.4f} | "
    f"C {SON [f'C en iyi GORELI ratio ({en_o :.2f})']-a :+.4f}")
    print (f"  E (CP sayisi bilinse) : {SON ['E KAHIN top-K (number bilinse)']-a :+.4f}")
    with open ("results/t2_gate_ayristir.json","w",encoding ="utf-8")as f :
        json .dump (SON ,f ,indent =1 )
    print ("\nmakbuz -> results/t2_gate_ayristir.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
