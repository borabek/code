"""OTONOM KAPSAMA TRENDI: %98 precision'da kapsanan CP orani, data arttikca buyuyor mu?
Bu, 'full otonomiye giden path present mi' sorusunun dogrudan olcumu."""
import json ,sys ,numpy as np 
TARGET =float (sys .argv [1 ])if len (sys .argv )>1 else 0.98 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 
r =np .load ('results/rich_feats.npz',allow_pickle =True )
lock =json .load (open ('results/split_lock.json'));LOCK =set (lock ['locked_parts']);META =lock ['parts']
X13 ,XR ,Y ,G =r ['X13'],r ['XR'],r ['y'],r ['groups']
RICH =np .hstack ([X13 ,XR [:,0 :33 ]])
rp =[str (v )for v in r ['part_ids']];gseen =np .array ([int (r ['seen'][int (g )])for g in G ])
gpid =np .array ([rp [int (g )]for g in G ]);ngt =dict (zip (r ['grp_ids'].tolist (),r ['ngt'].tolist ()))
WORK =(gseen ==0 )&np .array ([p not in LOCK for p in gpid ])
fam =np .array ([META .get (p ,{}).get ('family',f'nr:{p }')for p in gpid ])
allf =sorted (set (fam ));fidx ={v :i for i ,v in enumerate (allf )}
FG =np .array ([fidx [v ]for v in fam ])
GT =int (sum (ngt [int (g )]for g in np .unique (G [WORK ])))

def load_extra (path ):
    z =np .load (path ,allow_pickle =True );zp =[str (v )for v in z ['part_ids']]
    XX =np .hstack ([z ['X13'],z ['XR'][:,0 :33 ]])
    zg =np .array ([zp [int (g )]for g in z ['groups']])
    zf =np .array ([META .get (p ,{}).get ('family',f'nr:{p }')for p in zg ])
    return XX ,z ['y'],zf ,len (zp ),int (z ['ngt'].sum ())
EX ={}
for tag ,path in (('seen220','results/rich_extra.npz'),('yeni220','results/rich_new.npz')):
    try :EX [tag ]=load_extra (path )
    except Exception :pass 

def oof (extras ):
    o =np .zeros (len (Y ));idx =np .where (WORK )[0 ]
    for tr ,te in GroupKFold (5 ).split (RICH [idx ],Y [idx ],FG [idx ]):
        te_f =set (FG [idx ][te ].tolist ())
        Xtr ,Ytr =[RICH [idx [tr ]]],[Y [idx [tr ]]]
        for t_ in extras :
            XX ,YY ,ZF ,_ ,_ =EX [t_ ]
            keep =np .array ([fidx .get (f ,-1 )not in te_f for f in ZF ])
            Xtr .append (XX [keep ]);Ytr .append (YY [keep ])
        c =RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 )
        c .fit (np .vstack (Xtr ),np .concatenate (Ytr ))
        o [idx [te ]]=c .predict_proba (RICH [idx ][te ])[:,1 ]
    return o 

print (f"%{int (TARGET *100 )} PRECISION'DA OTONOM KAPSAMA (veri arttikca)")
print ("ADAY-TAVAN (mukemmel gate ile ulasilabilecek max kapsama): %.0f%%"%(100 *Y [WORK ].sum ()/GT ))
print ("%-34s %8s %10s %12s"%("training havuzu","training GT","AUTO prec","CP kapsama"))
for nm ,ex in (('TABAN (541 part)',[]),('+ yeni 220',['yeni220']),('+ yeni + seen220',['yeni220','seen220'])):
    if any (t not in EX for t in ex ):continue 
    o =oof (ex );kept =WORK &(o >=0.34 )
    best =None 
    for t in np .arange (0.34 ,0.99 ,0.01 ):
        a =kept &(o >=t );n =int (a .sum ())
        if n >=20 and int (Y [a ].sum ())/n >=TARGET :best =(t ,n ,int (Y [a ].sum ())/n ,int (Y [a ].sum ())/GT );break 
    gt_tr =GT +sum (EX [t_ ][4 ]for t_ in ex )
    if best :print ("%-34s %8d %10.4f %11.0f%%"%(nm ,gt_tr ,best [2 ],100 *best [3 ]))
    else :print ("%-34s %8d      hedefe ulasilamadi"%(nm ,gt_tr ))
