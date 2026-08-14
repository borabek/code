# -*- coding: utf-8 -*-
"""U8: yonlendirilmis arm (R0.1) DAGITILAN D'yi gercekten domine ediyor mu?

U7 tablosu (uctan uca):
    arm     WEI-disi  PXC-disi  familiar  seri-ORT
    A ham     0.4832    0.7203   0.7410   +0.0000
    D         0.5702    0.6828   0.7390   -0.0075   <- SU AN DAGITIK
    R0.1      0.5682    0.7029   0.7439   -0.0029

R0.1, D'yi UC eksende geciyor and dorduncude (WEI) -0.0020 geride. Ama this numbers single
orneklemden; "domine ediyor" demeden before ESLESTIRILMIS BOOTSTRAP with bakilmali -- ozellikle
WEI'deki -0.0020 gercekten ihmal edilebilir mi, and PXC'deki +0.0201 real mi.

Bu betik A/D/R0.1'i part duzeyinde saklar and three karsilastirmayi da GA with gives.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
Q =0.10 


def main ():
    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    ORAN =float (cfg ["gate_goreli_oran"]);TABAN =float (cfg ["gate_goreli_taban"])
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"absent:"+p )for p in tr_pid ])
    tr_mfg =np .array ([str (x )for x in d ["mfg"]])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    tg ={gk .get (r ["pid"],"absent:"+r ["pid"])for r in DER }
    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in tr_pid [tr_mfg ==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (tr_mfg )}
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],"zskor")

    def olc (alt ,keep ):
        rf =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (M [keep ],ytr [keep ])
        ca ,cd =rf (Xtr ),rf (Ztr )
        mx =[float (ca .predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
        for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
        threshold =float (np .quantile (mx ,Q ))
        det ={"A":[],"D":[],"R":[]}
        direction =0 
        for r in alt :
            sa =sd =None 
            if r ["X"]is not None :
                sa =ca .predict_proba (r ["X"])[:,1 ]
                sd =cd .predict_proba (wire_gate .within_part (r ["X"],"zskor"))[:,1 ]
            very =(sa is not None )and (float (sa .max ())<threshold )
            direction +=int (very )
            for ad ,s in (("A",sa ),("D",sd ),("R",sd if very else sa )):
                P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
                if s is not None :
                    m =(s >=ORAN *max (float (s .max ()),1e-9 ))&(s >=TABAN )
                    if m .any ():
                        P =r ["P"][m ];Pd =r ["Pd"][m ]
                det [ad ].append (("very"if r ["n"]>=8 else "low",)
                +esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
        return det ,threshold ,direction 

    rng =np .random .default_rng (0 )

    def boot (a ,b ,n =4000 ):
        v =[]
        for _ in range (n ):
            i =rng .integers (0 ,len (a ),len (a ))
            v .append (f1w ([b [k ]for k in i ])-f1w ([a [k ]for k in i ]))
        v =np .array (v )
        return v .mean (),np .percentile (v ,2.5 ),np .percentile (v ,97.5 )

    SON ={}
    split =[(mad ,(tr_mfg !=k )&~np .isin (tr_grp ,list (tg )),
    [x for x in DER if x ["mfg"]==mad ])for k ,mad in kod .items ()]
    split =[b for b in split if len (b [2 ])>=10 ]
    split .append (("familiar",~np .isin (tr_grp ,list (tg )),DER ))
    for ad ,keep ,alt in split :
        det ,threshold ,direction =olc (alt ,keep )
        SON [ad ]=det 
        print (f"\n=== {ad } (n={len (alt )}, threshold={threshold :.3f}, yonlendirilen {direction }/{len (alt )}) ===")
        for k in ("A","D","R"):
            print (f"  {k }: {f1w (det [k ]):.4f}")
        for x ,y in (("A","D"),("A","R"),("D","R")):
            m ,lo ,hi =boot (det [x ],det [y ])
            print (f"  {x }->{y }: {m :+.4f}  [{lo :+.4f}, {hi :+.4f}]  "
            f"{'GERCEK'if (lo >0 or hi <0 )else 'noise'}")

    print ("\n=== HAKIMIYET: R, dagitilan D'yi each eksende geciyor mu? ===")
    ok =True 
    for ad in SON :
        m ,lo ,hi =boot (SON [ad ]["D"],SON [ad ]["R"])
        iyi =m >0 or (lo <0 <hi )# ya more iyi, ya farksiz
        ok &=iyi 
        print (f"  {ad :<10} D->R {m :+.4f} [{lo :+.4f}, {hi :+.4f}] -> "
        f"{'R at least D up to iyi'if iyi else 'R DAHA KOTU'}")
    print (f"\nSONUC: {'R, D YERINE GECEBILIR'if ok else 'R, D instead of GECEMEZ'}")
    with open ("results/u8_yonlendirme_verify.json","w",encoding ="utf-8")as f :
        json .dump ({ad :{k :float (f1w (v ))for k ,v in det .items ()}for ad ,det in SON .items ()}
        |{"hakimiyet":bool (ok ),"q":Q },f ,indent =1 )
    with open ("results/u8_parca.pkl","wb")as f :
        pickle .dump (SON ,f )
    print ("receipt -> results/u8_yonlendirme_verify.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
