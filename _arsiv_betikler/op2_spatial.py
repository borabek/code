# -*- coding: utf-8 -*-
"""Op 2 (offline): position-enriched havuzda SPATIAL rerank. Op1 kanitladi: pozisyonsuz gap kapanmiyor.
Pozisyonla: each candidate for within-part SPATIAL sinyaller (symmetry-partner, grid-consistency, neighbor
score-consistency, local density) -> RF top-N OOF. Baseline'i (ALL 0.813) gecer mi, 0.85 yolu present mi?
DURUST: WEI and PXC ikisi de dusmemeli (win kriteri)."""
import warnings ;warnings .filterwarnings ("ignore")
import numpy as np 
from sklearn .ensemble import RandomForestClassifier 
from sklearn .model_selection import GroupKFold 

d =np .load ("results/f1_pool_pos.npz",allow_pickle =True )
X ,y ,grp ,mfg ,P =d ["X"],d ["y"],d ["groups"],d ["mfg"],d ["P"]
ngt_of =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
g2m ={g :int (mfg [grp ==g ][0 ])for g in np .unique (grp )}
HICP =11 ;hicp ={g for g in ngt_of if ngt_of [g ]>=HICP }
gk =list (GroupKFold (5 ).split (X ,y ,grp ))


def spatial_feats (Pp ,ws ):
    """within-part spatial sinyaller (frame-invariant). Kucuk N'de bile symmetry/neighbor works."""
    Pp =np .asarray (Pp ,float );ws =np .asarray (ws ,float );n =len (Pp )
    if n <2 :
        return np .zeros ((n ,6 ))
    Q =Pp -Pp .mean (0 )
    _ ,_ ,Vt =np .linalg .svd (Q ,full_matrices =False );u ,v =Vt [0 ],Vt [1 ]
    pu =Q @u ;pv =Q @v ;span =max (pu .max ()-pu .min (),pv .max ()-pv .min (),1.0 )
    D =np .linalg .norm (Pp [:,None ]-Pp [None ],axis =-1 );np .fill_diagonal (D ,1e9 )
    # 1) mirror-partner: u-eksende yansimasi (-pu) yakin a candidate present mi (terminaller simetrik)
    mir =np .zeros (n )
    for i in range (n ):
        refl =np .abs (pu +pu [i ])+np .abs (pv -pv [i ])# (-pu[i], pv[i]) hedef
        j =np .argmin (refl +np .where (np .arange (n )==i ,1e9 ,0 ))
        mir [i ]=float (refl [j ]<0.08 *span )
        # 2) neighbor score-consistency: most yakin komsunun ws'i
    nn =np .argmin (D ,1 );nn_ws =ws [nn ]
    # 3) grid-consistency: u ya da v ekseninde hizali high-ws komsu agirligi
    cons =np .zeros (n );tol =0.05 *span 
    for i in range (n ):
        al =((np .abs (pv -pv [i ])<tol )|(np .abs (pu -pu [i ])<tol ));al [i ]=False 
        cons [i ]=float (ws [al ].sum ())
    cons =cons /max (cons .max (),1e-9 )
    # 4) local density (8% span inside komsu ws)
    dens =np .array ([float (ws [(D [i ]<0.10 *span )].sum ())for i in range (n )]);dens =dens /max (dens .max (),1e-9 )
    # 5) centrality: part merkezine goreli konum (edge mi ici mi)
    cen =np .linalg .norm (Q ,axis =1 )/max (np .linalg .norm (Q ,axis =1 ).max (),1e-9 )
    # 6) nn-dist (normalize)
    nnd =D .min (1 )/span 
    return np .stack ([mir ,nn_ws ,cons ,dens ,cen ,nnd ],1 )


def topN (score ):
    tp ={"ALL":0 ,"WEI":0 ,"PXC":0 };gt ={"ALL":0 ,"WEI":0 ,"PXC":0 }
    for g in np .unique (grp ):
        if g in hicp :continue 
        idx =np .where (grp ==g )[0 ];N =ngt_of [g ];m ="WEI"if g2m [g ]==1 else "PXC"
        keep =idx [np .argsort (-score [idx ])[:N ]];t =int ((y [keep ]==1 ).sum ())
        for k in ("ALL",m ):tp [k ]+=t ;gt [k ]+=N 
    return {k :tp [k ]/max (gt [k ],1 )for k in tp }


RF =lambda s =0 :RandomForestClassifier (400 ,min_samples_leaf =3 ,n_jobs =-1 ,random_state =s )

def oof (feat ,seed =0 ):
    s =np .zeros (len (y ))
    for tr ,te in gk :s [te ]=RF (seed ).fit (feat [tr ],y [tr ]).predict_proba (feat [te ])[:,1 ]
    return s 

base =topN (oof (X ))
print (f"BASELINE (13 feat): ALL {base ['ALL']:.4f}  WEI {base ['WEI']:.4f}  PXC {base ['PXC']:.4f}")

# spatial feat'lari first-pass ws with hesapla (leakage absent: ws OOF)
first =oof (X )
SP =np .zeros ((len (y ),6 ))
for g in np .unique (grp ):
    idx =np .where (grp ==g )[0 ];SP [idx ]=spatial_feats (P [idx ],first [idx ])
Xsp =np .hstack ([X ,SP ])

for tag ,feat in [("13 + 6 spatial",Xsp )]:
    rs =[topN (oof (feat ,s ))for s in range (3 )]
    r ={k :np .mean ([x [k ]for x in rs ])for k in ("ALL","WEI","PXC")}
    win =r ['ALL']>base ['ALL']+0.002 and r ['WEI']>=base ['WEI']-0.005 and r ['PXC']>=base ['PXC']-0.005 
    print (f"{tag }: ALL {r ['ALL']:.4f}  WEI {r ['WEI']:.4f}  PXC {r ['PXC']:.4f}  ({r ['ALL']-base ['ALL']:+.4f})  {'WIN'if win else 'kill'}")
    print (f"  -> standart pool select-recall {base ['ALL']:.3f} -> {r ['ALL']:.3f}  (oracle ceiling 0.895)")
