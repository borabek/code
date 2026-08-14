# -*- coding: utf-8 -*-
"""Y7 — OZNITELIK SECIMI: 100+ column, sistematik never budanmadi

Secicinin feature matrisi zamanla buyudu (donusturulmus X + source
blogu). Gereksiz sutunlar noise carries and gorulmemis markada zarar
verebilir. Bu betik ONEM SIRALAMASINA according to budayip olcer.

ONEM: HGB'nin own `permutation_importance`si pahali; bunun instead of
single gecislik a vekil is used -- each sutunu KARISTIRIP (permute)
AUC dususune bakmak. Kat-disi, GT'siz not but fold inside kalir.

KOLLAR: all of them / first-%75 / first-%50 / first-%25 (onem sirasina according to)
KOSUL: TANIDIK brand (rastgele katlar) -- 2 gunluk hedef this.
KAPI: +0.01. D7'ye BAKILMAZ.
"""
import collections ,json ,os ,sys ,time 
import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_tam4")
sys .path .insert (0 ,".")
import canonical_d7 as K 
import p6_decision 
from run_p6_ortak import yukle 
from sina_cluster import match_hungarian 

NEG_KAT =12 ;ITER ,LR ,YAPRAK ,L2R =200 ,0.06 ,63 ,1.0 
NMS =5.0 ;KURAL =("goreli",0.85 ,0.20 )

def temel (d ):
    return np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])]).astype (np .float32 )

def yap ():
    return HistGradientBoostingClassifier (max_iter =ITER ,learning_rate =LR ,
    max_leaf_nodes =YAPRAK ,l2_regularization =L2R ,random_state =0 )

def auc (s ,y ):
    poz ,neg =s [y ==1 ],s [y ==0 ]
    if not len (poz )or not len (neg ):return 0.5 
    h =np .concatenate ([poz ,neg ]);r =np .argsort (np .argsort (h ))+1.0 
    return float ((r [:len (poz )].sum ()-len (poz )*(len (poz )+1 )/2.0 )/(len (poz )*len (neg )))

def main ():
    t0 =time .time ()
    data_ =yukle ("d6",0 )
    for d in data_ :
        d ["y"]=np .asarray (d ["y"],int );d ["_M"]=temel (d )
    n_sut =data_ [0 ]["_M"].shape [1 ]
    print (f"{len (data_ )} part | {n_sut } sutun",flush =True )
    # KAT TOHUMU (2026-08-13). Ilk kosu +0.0097 verdi but single tohumluydu
    # and noise bandindaydi. `Y7_TOHUM` with fold bolunmesi degistirilip
    # same arm tekrarlanir; three tohumda da pozitifse karar verilebilir.
    _tohum =int (os .environ .get ("Y7_TOHUM","1"))
    rng =np .random .default_rng (_tohum );pay =rng .permutation (len (data_ ))%3 

    # --- ONEM: a fold on permutasyon vekili
    ic =[i for i in range (len (data_ ))if pay [i ]!=0 ]
    dis =[i for i in range (len (data_ ))if pay [i ]==0 ]
    M =np .vstack ([data_ [i ]["_M"]for i in ic ])
    Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
    r2 =np .random .default_rng (0 )
    poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
    sec =np .concatenate ([poz ,r2 .choice (neg ,min (len (neg ),NEG_KAT *len (poz )),replace =False )])
    m0 =yap ().fit (M [sec ],Y [sec ])
    MD =np .vstack ([data_ [i ]["_M"]for i in dis ]);YD =np .concatenate ([data_ [i ]["y"]for i in dis ])
    taban_auc =auc (m0 .predict_proba (MD )[:,1 ],YD )
    onem =np .zeros (n_sut )
    for j in range (n_sut ):
        X2 =MD .copy ();X2 [:,j ]=r2 .permutation (X2 [:,j ])
        onem [j ]=taban_auc -auc (m0 .predict_proba (X2 )[:,1 ],YD )
    rank_ =np .argsort (-onem )
    print (f"baseline AUC {taban_auc :.4f} | onem hesaplandi ({time .time ()-t0 :.0f} s)",flush =True )
    print (f"  en onemli 5 sutun: {rank_ [:5 ].tolist ()}")
    print (f"  onemi <=0 olan sutun sayisi: {int ((onem <=0 ).sum ())}/{n_sut }")

    KOLLAR =[("hepsi",n_sut ),("ilk75",int (n_sut *0.75 )),
    ("ilk50",int (n_sut *0.50 )),("ilk25",int (n_sut *0.25 ))]
    agg ={k :collections .Counter ()for k ,_ in KOLLAR }
    for f_ in range (3 ):
        ic =[i for i in range (len (data_ ))if pay [i ]!=f_ ]
        dis =[i for i in range (len (data_ ))if pay [i ]==f_ ]
        for ad ,k in KOLLAR :
            sut =rank_ [:k ]
            M =np .vstack ([data_ [i ]["_M"][:,sut ]for i in ic ])
            Y =np .concatenate ([data_ [i ]["y"]for i in ic ])
            rr =np .random .default_rng (0 )
            poz ,neg =np .where (Y ==1 )[0 ],np .where (Y ==0 )[0 ]
            s_ =np .concatenate ([poz ,rr .choice (neg ,min (len (neg ),NEG_KAT *max (len (poz ),1 )),replace =False )])
            mm =yap ().fit (M [s_ ],Y [s_ ]);del M 
            for i in dis :
                d =data_ [i ]
                s =mm .predict_proba (d ["_M"][:,sut ])[:,1 ]
                P ,D =p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,KURAL ,nms_mm =NMS )
                tp ,fp ,fn =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],
                K .YANAL ,K .ACI ,False ,signed =True )[:3 ]
                c =agg [ad ];c ["tp"]+=tp ;c ["fp"]+=fp ;c ["fn"]+=fn 
        print (f"  fold{f_ } bitti ({time .time ()-t0 :.0f} s)",flush =True )

    def f1 (c ):return 2 *c ["tp"]/max (2 *c ["tp"]+c ["fp"]+c ["fn"],1 )
    last_ ={k :f1 (agg [k ])for k ,_ in KOLLAR }
    print (f"\n=== TABAN (hepsi) {last_ ['hepsi']:.4f} ===")
    for ad ,k in KOLLAR [1 :]:
        fark =last_ [ad ]-last_ ["hepsi"]
        print (f"  {ad :<8}({k :>3} sutun) {last_ [ad ]:.4f}   {fark :+.4f}"
        +("  <- KAPI GECTI"if fark >=0.01 else ""))
    json .dump ({"damga":makbuz_hash .damga (),"kat_tohumu":_tohum ,
    "n_sutun":n_sut ,
    "taban_auc":taban_auc ,"onemsiz_sutun":int ((onem <=0 ).sum ()),
    "toplam":last_ ,
    "not":"Oznitelik secimi, permutasyon onemi. TANIDIK brand "
    "(rastgele katlar). D7'ye BAKILMADI."},
    open (f"results/oznitelik_secimi_t{_tohum }.json","w"),indent =1 )
    print ("receipt -> results/oznitelik_secimi.json")

if __name__ =="__main__":
    main ()
