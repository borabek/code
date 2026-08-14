# -*- coding: utf-8 -*-
"""URUN ANALIZI (offline, f1_sweep_data.npz'den): per-mfg gate sweep + CP-count prior + failure taxonomy.
Hepsi OOF (leakage absent). Kullanici P0.5/P1.3/P1.4 maddeleri.
"""
import json ,numpy as np 
from sklearn .ensemble import GradientBoostingClassifier 
from sklearn .model_selection import GroupKFold 

d =np .load ("results/f1_sweep_data.npz",allow_pickle =True )
X ,v ,y ,grp ,mfg =d ["X"],d ["votes"],d ["y"],d ["groups"],d ["mfg"]
ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
# OOF gate skoru
oof =np .zeros (len (y ));gkf =GroupKFold (5 )
for tr ,te in gkf .split (X ,y ,grp ):
    c =GradientBoostingClassifier (n_estimators =200 ,max_depth =3 ,learning_rate =0.05 ).fit (X [tr ],y [tr ])
    oof [te ]=c .predict_proba (X [te ])[:,1 ]

def prf (mask ):
    tp =int ((y [mask ]==1 ).sum ());nk =int (mask .sum ())
    gt =sum (ngt_of [g ]for g in np .unique (grp [mask .nonzero ()[0 ]if mask .dtype ==bool else mask ]))
    return tp ,nk ,gt 

out ={}
# --- P1.3 per-manufacturer gate sweep ---
print ("=== P1.3 PER-MFG GATE SWEEP (vote>=1) ===")
out ["per_mfg_gate"]={}
for name ,m in (("WEI",mfg ==1 ),("PXC",mfg ==0 )):
    gt =sum (ngt_of [g ]for g in np .unique (grp [m ]))
    best =None ;rows =[]
    for th in [0.20 ,0.25 ,0.30 ,0.35 ,0.40 ,0.45 ,0.50 ]:
        k =m &(v >=1 )&(oof >=th );tp =int ((y [k ]==1 ).sum ())
        p =tp /max (k .sum (),1 );r =tp /max (gt ,1 );f =2 *p *r /max (p +r ,1e-9 )
        rows .append ((th ,p ,r ,f ))
        if best is None or f >best [3 ]:best =(th ,p ,r ,f )
    out ["per_mfg_gate"][name ]={"best_thr":best [0 ],"P":round (best [1 ],3 ),"R":round (best [2 ],3 ),"F1":round (best [3 ],3 )}
    print (f"  {name }: en iyi threshold {best [0 ]:.2f} -> P{best [1 ]:.3f} R{best [2 ]:.3f} F1 {best [3 ]:.3f}")
print (f"  (mevcut ortak threshold 0.35; per-mfg ayirmak F1 artirir mi -> yukaridaki en-iyiler)")

# --- P1.4 CP-count prior (metadata-assisted) ---
print ("\n=== P1.4 CP-COUNT PRIOR (manufacturer CP sayisi kadar en-guvenli CP tut) ===")
def count_prior (mask ):
    tp =nk =gt =0 
    for g in np .unique (grp [mask ]):
        idx =np .where ((grp ==g )&mask &(v >=1 ))[0 ]
        if not len (idx ):
            gt +=ngt_of [g ];continue 
        n =ngt_of [g ]
        keep =idx [np .argsort (-oof [idx ])[:max (n ,0 )]]# most high wire_score N tanesi
        tp +=int ((y [keep ]==1 ).sum ());nk +=len (keep );gt +=n 
    p =tp /max (nk ,1 );r =tp /max (gt ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )
out ["cp_count_prior"]={}
for name ,m in (("HEPSI",np .ones (len (y ),bool )),("WEI",mfg ==1 ),("PXC",mfg ==0 )):
    p ,r ,f =count_prior (m )
    out ["cp_count_prior"][name ]={"P":round (p ,3 ),"R":round (r ,3 ),"F1":round (f ,3 )}
    print (f"  {name }: P{p :.3f} R{r :.3f} F1 {f :.3f}  (metadata-assisted mod, base urun DEGIL)")

    # --- P0.5 failure taxonomy (urun config vote>=1 + gate 0.35) ---
print ("\n=== P0.5 FAILURE TAXONOMY (vote>=1 + gate 0.35) ===")
kept =(v >=1 )&(oof >=0.35 )
dropped =(v >=1 )&(oof <0.35 )
tool_fp =int (((y ==0 )&dropped ).sum ())# gate correct eledi (low wire_score, mfg-eslesmez)
unlisted_or_resid =int (((y ==0 )&kept ).sum ())# kept but mfg-eslesmez = listelenmemis tel VEYA residual tool
kept_tp =int (((y ==1 )&kept ).sum ())
dropped_wire =int (((y ==1 )&dropped ).sum ())# gate yanlislikla real teli attigi (mfg-eslesen)
n_gt =sum (ngt_of .values ())
missed_fn =n_gt -kept_tp # never yakalanmayan manufacturer CP (dropped_wire dahil)
out ["taxonomy"]={"n_gt":n_gt ,"kept_tp":kept_tp ,"tool_fp_dropped":tool_fp ,
"kept_fp_unlisted_or_residual":unlisted_or_resid ,
"dropped_real_wire":dropped_wire ,"total_fn":missed_fn }
print (f"  manufacturer CP toplam: {n_gt }")
print (f"  kept TP (dogru tel):                 {kept_tp }")
print (f"  tool_fp (gate ATTI, dusuk skor):     {tool_fp }   <- gate'in temizledigi tool aglari")
print (f"  kept FP (unlisted-wire VEYA artik):  {unlisted_or_resid }   <- gate'ten gecen ama mfg-eslesmez")
print (f"  dropped_real_wire (gate YANLIS atti):{dropped_wire }   <- gate maliyeti (gercek tel gitti)")
print (f"  toplam FN (kacan tel):               {missed_fn }")
print (f"  -> gate'in NET etkisi: {tool_fp } tool temizledi, {dropped_wire } gercek tel kaybetti (ratio {tool_fp }/{dropped_wire })")

json .dump (out ,open ("results/product_analysis_2026_07_24.json","w"),indent =1 )
print ("\n-> results/product_analysis_2026_07_24.json")
