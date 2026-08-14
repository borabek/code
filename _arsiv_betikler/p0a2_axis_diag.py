# -*- coding: utf-8 -*-
"""P0-a2: EKSEN hipotezini oldurmeden ONCE ALETI dogrula (FBI disiplini).
P0-a: saflik z=+0.09 (sinyal absent) AMA sifir-TP kumesinde FP %60.6 -> this baseline-orani artefakti may be
(adaylarin %85'i FP). Uc soru:
 (1) NULL-DUZELTILMIS money metrik: sifir-TP kumesindeki FP kutlesi SANSTAN high mi?
 (2) ALET calisiyor mu: TP yonleri part inside each other YOGUN mu (wire girisleri ortak axis paylasir
     beklentisi)? Yogun degilse ya very-yonlu urun ya YON TAHMINIMIZ GURULTU -> hipotez test EDILEMEMIS becomes.
 (3) TP yonleri FP yonlerinden more mi dense (ayirici guc present mi)?
CPU-only, mevcut pool."""
import json ,sys 
import numpy as np 
from p0a_axis_discord import cluster_axes 

POOL =sys .argv [1 ]if len (sys .argv )>1 else "results/wei_aggr_pool.json"
ANG =20.0 
rng =np .random .RandomState (0 )


def resultant (D ):
    """direction yogunlugu (mean resultant length): 1=all of them same direction, 0=dagilmis."""
    if len (D )<2 :return np .nan 
    return float (np .linalg .norm (D .mean (0 )))


d =json .load (open (POOL ))
obs_mass ,null_mass =[],[]
tp_res ,fp_res ,tp_span =[],[],[]
for pid ,v in d .items ():
    D =np .array (v ["dir"],float );y =np .array (v ["y"],int )
    if len (y )<4 or y .sum ()==0 :continue 
    D =D /(np .linalg .norm (D ,axis =1 ,keepdims =True )+1e-9 )
    lab ,k =cluster_axes (D ,ANG ,True )

    def zero_tp_fp (yy ):
        f =0 
        for c in np .unique (lab ):
            m =lab ==c ;nt =int (yy [m ].sum ())
            if nt ==0 :f +=int (m .sum ()-nt )
        return f 
    nf =int ((y ==0 ).sum ())
    if nf ==0 :continue 
    obs_mass .append (zero_tp_fp (y )/nf )
    null_mass .append (np .mean ([zero_tp_fp (rng .permutation (y ))for _ in range (100 )])/nf )
    tp_res .append (resultant (D [y ==1 ]));fp_res .append (resultant (D [y ==0 ]))
    tp_span .append (len (np .unique (lab [y ==1 ])))# TP kac ayri axis kumesine yayilmis

obs_mass =np .array (obs_mass );null_mass =np .array (null_mass )
tp_res =np .array (tp_res ,float );fp_res =np .array (fp_res ,float );tp_span =np .array (tp_span )
print (f"POOL {POOL } | {len (obs_mass )} part\n")
print ("=== (1) NULL-DUZELTILMIS money metrik ===")
print (f"  gozlenen: sifir-TP kumesinde FP kutlesi {obs_mass .mean ():.3f}")
print (f"  NULL    : {null_mass .mean ():.3f}  (ayni kumeler, etiket permute)")
print (f"  GERCEK FAZLA: {obs_mass .mean ()-null_mass .mean ():+.3f}"
+("  <- sansin USTUNDE, real yapisal FP kutlesi"if obs_mass .mean ()-null_mass .mean ()>0.05 
else "  <- SANS SEVIYESI: '%60 oldurulebilir' ARTEFAKT idi"))
print ("\n=== (2)+(3) ALET KONTROLU: direction tahminleri anlamli mi ===")
v_tp =tp_res [~np .isnan (tp_res )];v_fp =fp_res [~np .isnan (fp_res )]
print (f"  TP direction-yogunlugu (resultant): ort {v_tp .mean ():.3f} med {np .median (v_tp ):.3f}")
print (f"  FP direction-yogunlugu            : ort {v_fp .mean ():.3f} med {np .median (v_fp ):.3f}")
print (f"  TP kac axis-kumesine yayiliyor: ort {tp_span .mean ():.2f} (1.0 = hepsi tek eksende)")
print (f"  TP'nin tek-eksende oldugu part orani: {np .mean (tp_span ==1 ):.2f}")
if v_tp .mean ()>0.85 :
    print ("  -> TP yonleri COK YOGUN: alet calisiyor, wire girisleri ortak axis paylasiyor.")
    print ("     O HALDE ayirici guc yoklugu = FP'ler de AYNI eksende (same yuzde, tel deliklerinin next to).")
elif v_tp .mean ()>0.6 :
    print ("  -> TP yonleri ORTA dense: kismen ortak axis; alet muhtemelen calisiyor.")
else :
    print ("  -> TP yonleri DAGINIK: ya very-yonlu urunler ya YON TAHMINI GURULTU -> hipotez test EDILEMEDI.")
