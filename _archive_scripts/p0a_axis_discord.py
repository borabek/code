# -*- coding: utf-8 -*-
"""P0-a: EKSEN HIPOTEZI TESTI (CPU, insan-etiketi YOK, mevcut pool).
Hipotez: wire-CP'ler with tool/vida acikliklari FARKLI EKSEN ailelerinde yasar (tel ONDEN, vida/pusher
USTTEN girer) -- gate'in 13 lokal feature'inda this bilgi YOK.
Test: each parcada adaylari EKSENE according to kumele (part-ici, frame-invariant) -> kumelerin TP/FP saflini
NULL modele (etiketleri part-ici permute) karsi olc.
MONEY METRIK: 'sifir-TP kumesinde yasayan FP kutlesi' = yapisal as oldurulebilir FP orani.
GO: saflik >> null (z>5) VE saf-FP kumelerinde FP kutlesi >=%40."""
import json ,sys 
import numpy as np 

POOL =sys .argv [1 ]if len (sys .argv )>1 else "results/wei_aggr_pool.json"
ANG =float (sys .argv [2 ])if len (sys .argv )>2 else 20.0 # same axis sayilma esigi (derece)
SIGNED ="--unsigned"not in sys .argv # direction isareti onemli mi (ten vs arka giris)
rng =np .random .RandomState (0 )


def cluster_axes (D ,ang_deg ,signed =True ):
    """greedy axis clustering: cosine benzerligi ang_deg inside olanlar same cluster."""
    thr =np .cos (np .deg2rad (ang_deg ))
    n =len (D );lab =-np .ones (n ,int );k =0 
    for i in range (n ):
        if lab [i ]>=0 :continue 
        c =D [i ];lab [i ]=k 
        for j in range (i +1 ,n ):
            if lab [j ]>=0 :continue 
            d =float (D [i ]@D [j ])
            if (d if signed else abs (d ))>=thr :lab [j ]=k 
        k +=1 
    return lab ,k 


def purity (lab ,y ):
    """agirlikli saflik: each kumede cogunluk-sinifin payi."""
    tot =0 
    for c in np .unique (lab ):
        m =lab ==c ;nt =int (y [m ].sum ());nf =int (m .sum ()-nt )
        tot +=max (nt ,nf )
    return tot /max (len (y ),1 )


d =json .load (open (POOL ))
obs_p =[];null_p =[];z_all =[]
fp_in_pure =0 ;fp_tot =0 ;tp_tot =0 
clust_stats =[]
for pid ,v in d .items ():
    D =np .array (v ["dir"],float );y =np .array (v ["y"],int )
    if len (y )<4 or y .sum ()==0 :continue 
    D =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-9 )
    lab ,k =cluster_axes (D ,ANG ,SIGNED )
    op =purity (lab ,y )
    # NULL: etiketleri part-ici permute (same cluster boyutlari, same TP count)
    nulls =[]
    for _ in range (200 ):
        yp =rng .permutation (y );nulls .append (purity (lab ,yp ))
    nm ,ns =float (np .mean (nulls )),float (np .std (nulls )+1e-9 )
    obs_p .append (op );null_p .append (nm );z_all .append ((op -nm )/ns )
    # money metrik: sifir-TP kumelerindeki FP
    for c in np .unique (lab ):
        m =lab ==c ;nt =int (y [m ].sum ());nf =int (m .sum ()-nt )
        if nt ==0 :fp_in_pure +=nf 
        fp_tot +=nf ;tp_tot +=nt 
    clust_stats .append ((k ,len (y )))

obs_p =np .array (obs_p );null_p =np .array (null_p );z_all =np .array (z_all )
kk =np .array ([c [0 ]for c in clust_stats ]);nn =np .array ([c [1 ]for c in clust_stats ])
print (f"POOL {POOL } | {len (obs_p )} part | candidate/part ort {nn .mean ():.1f} | axis-kumesi/part ort {kk .mean ():.1f}"
f" | threshold {ANG :.0f}deg {'signed'if SIGNED else 'unsigned'}")
print (f"\n=== EKSEN KUMELERI TP/FP AYIRIYOR MU (null-model testi) ===")
print (f"  gozlenen saflik : {obs_p .mean ():.3f}")
print (f"  NULL saflik     : {null_p .mean ():.3f}  (etiketler part-ici permute)")
print (f"  diff            : {obs_p .mean ()-null_p .mean ():+.3f}   average z = {z_all .mean ():+.2f}")
print (f"  parcalarin %{100 *np .mean (z_all >2 ):.0f}'inde z>2 (axis bilgi tasiyor)")
print (f"\n=== MONEY METRIK: yapisal as oldurulebilir FP ===")
print (f"  total FP {fp_tot } | total TP {tp_tot }")
print (f"  SIFIR-TP axis kumesinde yasayan FP: {fp_in_pure }/{fp_tot } = {fp_in_pure /max (fp_tot ,1 ):.3f}")
print (f"  -> this FP'ler hicbir gercek wire-CP with same ekseni PAYLASMIYOR = axis-kuralIyla oldurulebilir")
go1 =z_all .mean ()>5 ;go2 =fp_in_pure /max (fp_tot ,1 )>=0.40 
print (f"\nKARAR: saflik-sinyali {'GECTI'if go1 else 'ZAYIF'} (z {z_all .mean ():+.2f}, threshold >5) | "
f"saf-FP kutlesi {'GECTI'if go2 else 'DUSUK'} ({fp_in_pure /max (fp_tot ,1 ):.2f}, threshold >=0.40)")
print ("  IKISI DE GECTI -> P0-b/c'ye devam (cluster-havuzlu gate)."if (go1 and go2 )
else "  -> axis single basina zayif; P0-b'de face/konum with birlestir ya da durustce kapat.")
