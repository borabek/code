# -*- coding: utf-8 -*-
"""Op 5+ : high-CP OZEL SECICI -- gate feature'larina LATTICE sinyalleri ekle (consistency + grid-fit +
row/col sayilari), high-CP expanded veride leave-one-part-out RF egit -> secim darbogazini kapat.
Non-high-CP standart gate with. Overall metadata F1 -> 0.80?"""
import json ,numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

pool =json .load (open ("results/highcp_pool.json"))
d =np .load ("results/f1_sweep_data.npz",allow_pickle =True )
Xs ,ys ,grp =d ["X"],d ["y"],d ["groups"];ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
HICP =11 ;hicp ={g for g in ngt_of if ngt_of [g ]>=HICP }


def lattice_feats (P ,ws ):
    """each candidate for lattice sinyalleri: consistency, grid-fit, n_row, n_col, u/v-position."""
    Q =P -P .mean (0 );_ ,_ ,Vt =np .linalg .svd (Q ,full_matrices =False );u ,v =Vt [0 ],Vt [1 ]
    pu =P @u ;pv =P @v 
    span =max (pu .max ()-pu .min (),pv .max ()-pv .min (),1.0 );tol =0.04 *span 
    n =len (P );cons =np .zeros (n );nrow =np .zeros (n );ncol =np .zeros (n )
    for i in range (n ):
        sr =np .abs (pv -pv [i ])<tol ;sc =np .abs (pu -pu [i ])<tol 
        al =(sr |sc );al [i ]=False 
        cons [i ]=float (ws [al ].sum ());nrow [i ]=float (sr .sum ()-1 );ncol [i ]=float (sc .sum ()-1 )
    cons =cons /max (cons .max (),1e-9 )
    # grid-fit: HER two eksende (u,v) pitch detection -> most yakin grid cizgisine yakinlik
    hi =ws >=np .percentile (ws ,60 )
    def gridfit (pp ):
        g =np .zeros (n )
        if hi .sum ()>=3 :
            pos =np .sort (pp [hi ]);gaps =np .diff (pos );gaps =gaps [gaps >0.02 *span ]
            if len (gaps ):
                pitch =np .median (gaps );r =np .abs ((pp -pos [0 ])%pitch );r =np .minimum (r ,pitch -r )
                g =1.0 -r /(pitch /2 +1e-9 )
        return g 
    gfu =gridfit (pu );gfv =gridfit (pv )
    # yerel yogunluk: 8mm ici high-ws komsu agirligi
    dens =np .zeros (n )
    for i in range (n ):
        near =(np .abs (pu -pu [i ])<0.08 *span )&(np .abs (pv -pv [i ])<0.08 *span );near [i ]=False 
        dens [i ]=float (ws [near ].sum ())
    dens =dens /max (dens .max (),1e-9 )
    return np .stack ([cons ,gfu ,gfv ,dens ,nrow /max (nrow .max (),1 ),ncol /max (ncol .max (),1 ),
    (pu -pu .min ())/span ,(pv -pv .min ())/span ],1 )


pids =list (pool )
Xaug ={};yy ={}
for p in pids :
    P =np .array (pool [p ]["P"]);ws =np .array (pool [p ]["ws"]);Xb =np .array (pool [p ]["X"])
    Xaug [p ]=np .hstack ([Xb ,lattice_feats (P ,ws )]);yy [p ]=np .array (pool [p ]["y"])

    # non-high-CP standart OOF (degismedi)
oof =np .zeros (len (ys ))
for tr ,te in GroupKFold (5 ).split (Xs ,ys ,grp ):
    c =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (Xs [tr ],ys [tr ]);oof [te ]=c .predict_proba (Xs [te ])[:,1 ]
nh_tp =nh_nk =nh_gt =0 
for g in np .unique (grp ):
    if g in hicp :continue 
    idx =np .where (grp ==g )[0 ];nh_gt +=ngt_of [g ]
    if len (idx ):keep =idx [np .argsort (-oof [idx ])[:ngt_of [g ]]];nh_tp +=int ((ys [keep ]==1 ).sum ());nh_nk +=len (keep )

def f1 (tp ,nk ,gt ):p =tp /max (nk ,1 );r =tp /max (gt ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )

# high-CP OZEL selector: leave-one-part-out, lattice-augmented
hc_tp =hc_nk =hc_gt =0 
for p in pids :
    Xtr =np .vstack ([Xaug [q ]for q in pids if q !=p ]);ytr =np .concatenate ([yy [q ]for q in pids if q !=p ])
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =0 ).fit (Xtr ,ytr )
    s =clf .predict_proba (Xaug [p ])[:,1 ];N =pool [p ]["N"]
    keep =np .argsort (-s )[:N ];hc_tp +=int ((yy [p ][keep ]==1 ).sum ());hc_nk +=len (keep );hc_gt +=N 
hr =hc_tp /max (hc_gt ,1 );hf =f1 (hc_tp ,hc_nk ,hc_gt )[2 ]
o =f1 (nh_tp +hc_tp ,nh_nk +hc_nk ,nh_gt +hc_gt )
print (f"high-CP OZEL SECICI (lattice-augmented): recall {hr :.3f} F1 {hf :.3f}")
print (f"  (onceki: expanded-gate recall 0.607 F1 0.669)")
print (f"OVERALL metadata: P{o [0 ]:.3f} R{o [1 ]:.3f} F1 {o [2 ]:.3f} | 0.80 {'GECILDI!'if o [2 ]>=0.80 else 'kaldi '+str (round (0.80 -o [2 ],3 ))}")
