# -*- coding: utf-8 -*-
"""Op 4 (offline): high-CP havuzda LATTICE/row-consistency re-ranking. Darbogaz secim: candidate recall
0.800 but top-N recall 0.570 -- real teller havuzda but gate top-N six siraliyor. High-CP terminaller
periyodik grid; ten-grid adaylari yukari siralamak top-N recall'i artirabilir.
Sinyal: each candidate for ROW/COLUMN-consistency = surface-eksenlerinde (u,v) hizali high-score komsu count.
Re-rank = ws + lambda * consistency. lambda taranir, top-N F1 karsilastirilir."""
import json ,numpy as np 

pool =json .load (open ("results/highcp_pool.json"))


def face_axes (P ):
    Q =P -P .mean (0 )
    _ ,_ ,Vt =np .linalg .svd (Q ,full_matrices =False )
    return Vt [0 ],Vt [1 ]# two most large varyans yonu = surface eksenleri


def consistency (P ,ws ,tol_frac =0.04 ):
    """each candidate for: u ya da v ekseninde hizali (same row/kolon) high-score komsu agirligi."""
    u ,v =face_axes (P )
    pu =P @u ;pv =P @v 
    span =max (pu .max ()-pu .min (),pv .max ()-pv .min (),1.0 );tol =tol_frac *span 
    n =len (P );c =np .zeros (n )
    for i in range (n ):
        same_row =np .abs (pv -pv [i ])<tol # same v -> row
        same_col =np .abs (pu -pu [i ])<tol # same u -> kolon
        aligned =(same_row |same_col );aligned [i ]=False 
        c [i ]=float ((ws [aligned ]).sum ())# hizali komsularin ws toplami
    return c /max (c .max (),1e-9 )


def f1_topN (rank_scores ,mode ="all"):
    tp =nk =gt =0 
    for pid ,d in pool .items ():
        ws =np .array (d ["ws"]);y =np .array (d ["y"]);N =d ["N"];gt +=N 
        sc =rank_scores (pid ,d )
        keep =np .argsort (-sc )[:N ];tp +=int ((y [keep ]==1 ).sum ());nk +=len (keep )
    p =tp /max (nk ,1 );r =tp /max (gt ,1 );return p ,r ,2 *p *r /max (p +r ,1e-9 )


print (f"high-CP pool: {len (pool )} part, {sum (d ['N']for d in pool .values ())} manufacturer CP")
# baseline: ws
ps =f1_topN (lambda pid ,d :np .array (d ["ws"]))
print (f"\nbaseline (ws top-N):        P{ps [0 ]:.3f} R{ps [1 ]:.3f} F1 {ps [2 ]:.3f}")
# lattice: ws + lambda*consistency
print ("lambda | P     R     F1")
best =(ps [2 ],0.0 )
cons_cache ={pid :consistency (np .array (d ["P"]),np .array (d ["ws"]))for pid ,d in pool .items ()}
for lam in [0.1 ,0.2 ,0.3 ,0.5 ,0.8 ,1.2 ]:
    def rk (pid ,d ,lam =lam ):return np .array (d ["ws"])+lam *cons_cache [pid ]
    p ,r ,f =f1_topN (rk )
    mark =" <--"if f >best [0 ]else ""
    if f >best [0 ]:best =(f ,lam )
    print (f"{lam :6.1f} | {p :.3f} {r :.3f} {f :.3f}{mark }")
print (f"\nEN IYI: lambda {best [1 ]} -> F1 {best [0 ]:.3f} (baseline ws {ps [2 ]:.3f})")
print (f"  -> lattice re-rank high-CP F1'i {'YUKSELTTI'if best [0 ]>ps [2 ]else 'YUKSELTMEDI'} (+{best [0 ]-ps [2 ]:.3f})")
