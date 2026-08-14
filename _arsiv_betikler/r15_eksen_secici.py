# -*- coding: utf-8 -*-
"""R1: EKSEN SECICISI -- three adayi URET, aralarindan GEOMETRI sectirsin.

WHY SECICI: three single-mekanizma denemesi de dustu (measured 2026-08-05):
  channel ekseni        dik vakalarin %13'unu kurtariyor
  mouth duzlemi normali %40 (but CHECK %21 -- net kazanc ~%19)
  yakinlik-oncelikli B-rep  194'te +5, gorulmemiste NET -30 (REDDEDILDI)
Hicbiri single basina yetmiyor but UCU DE different durumlarda iyi. Secici, each candidate for
GEOMETRIK GUVEN skoru uretip most yuksegini takes -- GT'yi GORMEDEN.

SECIM OLCUTU (all of them GT'siz, calisma aninda hesaplanabilir):
  s1 mouth uyumu  : axis along olculen mouth ic capi ne up to DAR (real channel dar becomes)
  s2 channel boyu  : axis along govdeye girince ne up to ilerlenebiliyor (channel = uzun bosluk)
  s3 disari      : `outward_along_axis` with same isareti veriyor mu (tutarlilik)

Bu betik URUNU DEGISTIRMEZ. Kazanmazsa kod does not change.
"""
import sys ,os ,json ,io ,pickle 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,'.')
import numpy as np ,trimesh ,wire_gate ,thesis_remesh 
import brep_axes as BX 
from cp_geometry import channel_axis ,mouth_width ,ray_hits ,outward_along_axis 
from f2_12_veri_kolu import egit 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 

def candidates (mesh ,cyl ,p ,d0 ):
    """Uc axis adayi (unsigned unit vektor)."""
    out =[("mevcut",d0 )]
    try :
        a =channel_axis (mesh ,p ,fallback =None )
        if a is not None :out .append (("channel",np .asarray (a ,float ).reshape (3 )))
    except Exception :pass 
    if cyl is not None and len (cyl [0 ]):
        C =np .asarray (cyl [0 ],float );A =np .asarray (cyl [1 ],float )
        rel =p -C ;t =(rel *A ).sum (1 )
        dik =np .linalg .norm (rel -t [:,None ]*A ,axis =1 )
        j =int (np .argmin (dik ))
        if dik [j ]<=6.0 :out .append (("brep",A [j ]))
    V =np .asarray (mesh .vertices ,float )
    q =V [np .linalg .norm (V -p ,axis =1 )<=6.0 ]
    if len (q )>=8 :
        Q =q -q .mean (0 );_w ,vv =np .linalg .eigh (Q .T @Q )
        out .append (("duzlem",vv [:,0 ]))
    return [(n ,v /(np .linalg .norm (v )+1e-9 ))for n ,v in out ]

def confidence (mesh ,p ,v ):
    """GT'siz geometrik confidence. Yuksek = this axis real a channel."""
    try :
        ic ,ort =mouth_width (mesh ,p ,v )
    except Exception :
        ic =ort =0.0 
    if ic <=1e-6 :return -1.0 
    try :
        h1 =ray_hits (mesh ,p ,v ,max_mm =40.0 );h2 =ray_hits (mesh ,p ,-v ,max_mm =40.0 )
    except Exception :
        return -1.0 
        # KANAL = a tarafta uzun bosluk, diger tarafta body.
    ileri =float (h1 [0 ])if len (h1 )else 40.0 
    geri =float (h2 [0 ])if len (h2 )else 40.0 
    boy =max (ileri ,geri )
    # dar mouth + uzun channel = guclu channel isareti; genis "mouth" duz yuzeydir
    return float (boy /max (ic ,0.5 ))

def main ():
    sv =json .load (io .open ('results/d5_4_sinav_kumesi.json',encoding ='utf-8'))
    G6 ={r ['pid']:r for r in pickle .load (open ('results/_der_yeni_g6.pkl','rb'))}
    yol ={p :s for m ,p ,jf ,s in eligible ()}
    D =[G6 [p ]for p in sorted (set (sv ['pidler']))if p in G6 ][:70 ]
    m ,_ =egit ('results/zengin_parite_v3.npz')
    A_mev ,A_sec ,A_kah =[],[],[]
    for i ,r in enumerate (D ,1 ):
        if i %20 ==0 :print (f"  {i }/{len (D )}",flush =True )
        if r ['X']is None or r .get ('XR')is None :continue 
        M =np .hstack ([r ['X'],r ['XR']]).astype (float )
        if M .shape [1 ]*2 !=m ['n_feat']:continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (m ,M ))
        P =np .asarray (r ['P'],float )[k ];Pd =np .asarray (r ['Pd'],float )[k ]
        G =np .asarray (r ['G'],float );Gd =np .asarray (r ['Gd'],float )
        if not len (P )or not len (G ):continue 
        try :
            Vr ,Fr =step_to_mesh (yol [r ['pid']])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            mesh =trimesh .Trimesh (vertices =np .ascontiguousarray (V ,np .float64 ),
            faces =np .ascontiguousarray (F ,np .int64 ),process =False )
            cyl =BX .cylinders (yol [r ['pid']])
        except Exception :
            continue 
        d =P [:,None ,:]-G [None ,:,:];al =(d *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe2 =np .where (np .abs (al )<=40 ,pe ,np .inf )
        nP =Pd /(np .linalg .norm (Pd ,axis =1 ,keepdims =True )+1e-9 )
        tol =max (3.0 ,0.06 *float (r ['diag']));up ,ug =set (),set ()
        for dd ,a_ ,b_ in sorted ((pe2 [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
            if dd >tol or a_ in up or b_ in ug :continue 
            up .add (a_ );ug .add (b_ )
            g =Gd [b_ ]
            candidate =candidates (mesh ,cyl ,P [a_ ],nP [a_ ])
            aci ={n :np .degrees (np .arccos (min (1 ,abs (float (v @g )))))for n ,v in candidate }
            A_mev .append (aci ["mevcut"])
            sk =[(confidence (mesh ,P [a_ ],v ),n )for n ,v in candidate ]
            A_sec .append (aci [max (sk )[1 ]])
            A_kah .append (min (aci .values ()))# KAHIN: adaylarin EN IYISI
    Am ,As ,Ak =np .array (A_mev ),np .array (A_sec ),np .array (A_kah )
    print (f"\n{len (Am )} cift")
    print (f"{'':<22}{'angle<=10':>10}{'DIK>80':>9}{'median':>10}")
    for ad ,A in (("MEVCUT",Am ),("SECICI",As ),("KAHIN (ceiling)",Ak )):
        print (f"{ad :<22}{100 *(A <=10 ).mean ():>9.0f}%{100 *(A >80 ).mean ():>8.0f}%{np .median (A ):>10.1f}")
    print (f"\nGO: selector MEVCUT'u >=5 puan gecmeli -> "
    f"{'GECTI'if 100 *((As <=10 ).mean ()-(Am <=10 ).mean ())>=5 else 'GECMEDI'}")

if __name__ =="__main__":
    main ()
