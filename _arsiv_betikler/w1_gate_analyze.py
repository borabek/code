# -*- coding: utf-8 -*-
"""DURUST per-mfg gate threshold degerlendirmesi: grouped 5-fold CV (threshold train-fold'da secilir, test-fold'da
olculur -> in-sample iyimserlik YOK). Karsilastir: (a) mevcut global 0.35, (b) CV-secilmis per-mfg threshold,
(c) in-sample optimal (upper-boundary referans). results/gate_scores.json'dan (w1_gate_collect ciktisi)."""
import json ,numpy as np 

rows =json .load (open ("results/gate_scores.json"))
THRS =np .arange (0.0 ,0.71 ,0.02 )


def prf (tp ,fp ,fn ):
    p =tp /max (tp +fp ,1 );r =tp /max (tp +fn ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )


def totalG (rs ):
    return sum ({r ["pid"]:r ["N"]for r in rs }.values ())# each part N'i a times


def metrics_at (rs ,thr ,gtot =None ):
    if gtot is None :gtot =totalG (rs )
    tp =sum (1 for r in rs if r ["ws"]>=thr and r ["tp"])
    fp =sum (1 for r in rs if r ["ws"]>=thr and not r ["tp"])
    return prf (tp ,fp ,gtot -tp )


def best_thr (rs ):
    g =totalG (rs );return max (THRS ,key =lambda t :metrics_at (rs ,t ,g )[2 ])


def cv_eval (rs ,K =5 ):
    """grouped K-fold: threshold train'de sec, test'te uygula; TP/FP/FN test-fold'lar along topla."""
    pids =sorted ({r ["pid"]for r in rs });np .random .seed (0 )
    perm =np .random .permutation (len (pids ));folds =np .array_split (perm ,K )
    TP =FP =FN =0 ;chosen =[]
    for f in folds :
        test_pids ={pids [i ]for i in f }
        tr =[r for r in rs if r ["pid"]not in test_pids ];te =[r for r in rs if r ["pid"]in test_pids ]
        if not tr or not te :continue 
        thr =best_thr (tr );chosen .append (thr )
        gte =totalG (te )
        TP +=sum (1 for r in te if r ["ws"]>=thr and r ["tp"])
        FP +=sum (1 for r in te if r ["ws"]>=thr and not r ["tp"])
        FN +=gte -sum (1 for r in te if r ["ws"]>=thr and r ["tp"])
    return prf (TP ,FP ,FN ),float (np .median (chosen )),chosen 


print (f"{len (rows )} pred | mfg dagilimi:",{m :len ({r ['pid']for r in rows if r ['mfg']==m })for m in {r ['mfg']for r in rows }})
report ={}
for mfg in ["WEI","PXC","ALL"]:
    rs =rows if mfg =="ALL"else [r for r in rows if r ["mfg"]==mfg ]
    if not rs :continue 
    g35 =metrics_at (rs ,0.35 );(cvp ,cvr ,cvf ),cvthr ,chosen =cv_eval (rs )
    isb =best_thr (rs );(ip ,ir ,iff )=metrics_at (rs ,isb )
    print (f"\n=== {mfg } ({len ({r ['pid']for r in rs })} part, {totalG (rs )} GT) ===")
    print (f"  global 0.35      : P={g35 [0 ]:.3f} R={g35 [1 ]:.3f} F1={g35 [2 ]:.3f}")
    print (f"  CV per-mfg threshold  : P={cvp :.3f} R={cvr :.3f} F1={cvf :.3f}  (CV-median thr={cvthr :.2f}, folds={[round (c ,2 )for c in chosen ]})")
    print (f"  in-sample optimal: P={ip :.3f} R={ir :.3f} F1={iff :.3f}  (thr={isb :.2f})  [ust-sinir referans]")
    print (f"  >>> CV KAZANC vs 0.35: {cvf -g35 [2 ]:+.3f}")
    report [mfg ]={"global_035_f1":g35 [2 ],"cv_f1":cvf ,"cv_thr_median":cvthr ,
    "insample_f1":iff ,"insample_thr":float (isb ),"cv_gain":cvf -g35 [2 ]}
json .dump (report ,open ("results/gate_thr_report.json","w"),indent =2 )
print ("\n-> results/gate_thr_report.json")
print ("DURUST NOT: CV kazanci gercek (in-sample degil). in-sample = ulasilamaz ust-sinir.")
