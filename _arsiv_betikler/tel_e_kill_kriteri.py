# -*- coding: utf-8 -*-
"""TEL-E: channel ozellikleri (TEL-B) GERCEKTEN ayirt ediyor mu? KILL KRITERI uygulanir.

RULE (plana baslarken yazildi, sonuca according to degistirilmez):
    Mevcut gate seti UZERINE ek AUC katkisi < 0.02 whereas this feature ailesi OLU ilan edilir
    and oyle raporlanir. "Ise yaramadi but biraz more ugrasayim" dongusu bastan closed.

OLCUM:
  1  Her channel ozelliginin TEK BASINA ayirt edici gucu (Mann-Whitney AUC, 0.5 = kor)
  2  TABAN: mevcut 13 gate ozelligiyle aile-disi CV AUC
  3  BIRLESIK: 13 + 10 channel ozelligi
  4  EK KATKI = birlesik - baseline   (kill kriteri buna bakar)
  5  Bootstrap confidence araligi -- small ornekte single sayiya guvenilmez
"""
import json 
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
from sklearn .metrics import roc_auc_score 

SRC ="results/tel_b_feats.npz"
KILL =0.02 


def oof_auc (X ,y ,groups ,seed =0 ):
    oof =np .zeros (len (y ))
    for tr ,te in GroupKFold (5 ).split (X ,y ,groups ):
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =seed )
        c .fit (X [tr ],y [tr ])
        oof [te ]=c .predict_proba (X [te ])[:,1 ]
    return roc_auc_score (y ,oof ),oof 


def boot_ci (y ,s ,n =2000 ,seed =0 ):
    rng =np .random .RandomState (seed );vals =[]
    idx =np .arange (len (y ))
    for _ in range (n ):
        b =rng .choice (idx ,len (idx ),replace =True )
        if len (np .unique (y [b ]))<2 :continue 
        vals .append (roc_auc_score (y [b ],s [b ]))
    return float (np .percentile (vals ,2.5 )),float (np .percentile (vals ,97.5 ))


def main ():
    z =np .load (SRC ,allow_pickle =True )
    Xc ,X13 ,y =z ["X"],z ["X13"],z ["y"]
    pid =np .array ([str (q )for q in z ["pid"]])
    names =[str (n )for n in z ["names"]]

    lock =json .load (open ("results/split_lock.json"))
    fams ={k :v .get ("family",f"nr:{k }")for k ,v in lock ["parts"].items ()}
    fam =np .array ([fams .get (q ,f"nr:{q }")for q in pid ])
    fid ={v :i for i ,v in enumerate (sorted (set (fam )))}
    G =np .array ([fid [v ]for v in fam ])

    print (f"veri: {len (y )} candidate ({int (y .sum ())} TP / {int ((y ==0 ).sum ())} FP), "
    f"{len (set (pid ))} part, {len (set (fam ))} aile\n")
    if y .sum ()<20 or (y ==0 ).sum ()<20 :
        print ("YETERSIZ EXAMPLE -- measurement yapilmadi");return 

        # --- 1. single basina guc -------------------------------------------------------------------
    from scipy .stats import mannwhitneyu 
    print ("1) KANAL OZELLIKLERI TEK BASINA (AUC 0.5 = kor):")
    rows =[]
    for j ,nm in enumerate (names ):
        a ,b =Xc [y ==0 ,j ],Xc [y ==1 ,j ]
        try :
            u ,_ =mannwhitneyu (a ,b ,alternative ="two-sided");auc =u /(len (a )*len (b ))
        except Exception :
            auc =0.5 
        rows .append ((abs (auc -0.5 ),auc ,nm ,float (np .median (a )),float (np .median (b ))))
    for s ,auc ,nm ,ma ,mb in sorted (rows ,reverse =True ):
        flag ="  <-- kayda deger"if abs (auc -0.5 )>=0.10 else ""
        print (f"   {nm :<18} AUC {auc :.3f}   FP med {ma :8.2f}   TP med {mb :8.2f}{flag }")

        # --- 2-4. baseline vs birlesik --------------------------------------------------------------
    a_base ,s_base =oof_auc (X13 ,y ,G )
    a_comb ,s_comb =oof_auc (np .hstack ([X13 ,Xc ]),y ,G )
    a_chan ,_ =oof_auc (Xc ,y ,G )
    lo_b ,hi_b =boot_ci (y ,s_base );lo_c ,hi_c =boot_ci (y ,s_comb )
    delta =a_comb -a_base 

    print (f"\n2) AILE-DISI CV AUC:")
    print (f"   TABAN (13 gate ozelligi)      {a_base :.4f}   [{lo_b :.3f}, {hi_b :.3f}]")
    print (f"   sadece kanal (10 ozellik)     {a_chan :.4f}")
    print (f"   BIRLESIK (23 ozellik)         {a_comb :.4f}   [{lo_c :.3f}, {hi_c :.3f}]")
    print (f"\n3) EK KATKI = {delta :+.4f}   (kill esigi {KILL })")

    verdict ="YASIYOR"if delta >=KILL else "OLU"
    print (f"\n>>> KANAL OZELLIKLERI: {verdict }")
    if delta <KILL :
        print ("    Kill kriteri geregi this aile OLU ilan edildi. Ek ugras YOK.")
        print ("    (Hipotez: tel girisi kontakta biter, alet agzi arm yuvasinda -- measured, tutmadi")
        print ("     ya da mevcut set this bilgiyi already baska yoldan iceriyor.)")

    json .dump ({"n_cand":int (len (y )),"n_tp":int (y .sum ()),"n_parts":len (set (pid )),
    "auc_base":a_base ,"auc_channel_only":a_chan ,"auc_combined":a_comb ,
    "delta":delta ,"kill_threshold":KILL ,"verdict":verdict ,
    "base_ci":[lo_b ,hi_b ],"combined_ci":[lo_c ,hi_c ],
    "standalone":{nm :auc for _ ,auc ,nm ,_ ,_ in rows }},
    open ("results/tel_e_kill.json","w"),indent =1 )
    print ("\nmakbuz -> results/tel_e_kill.json")


if __name__ =="__main__":
    main ()
