# -*- coding: utf-8 -*-
"""R1b: OGRENILMIS EKSEN SECICISI -- EGITIM VERISI URET.

R15 olctu: three candidate between correct axis NEREDEYSE HEP VAR (kahin %84 angle<=10, dik %26 -> %2)
but ELLE yazdigim secim kurali MEVCUT'tan kotu (%62 vs %68). Yani candidates iyi, RULE kotu.
Cozum: kurali OGREN -- `c4_yon_secici`nin sign for yaptigini axis for yap.

TEZ SADAKATI: `v_o` KONUMU does not change (tezin mouth-ortasi tanimi aynen), 5 sinif, remesh, network
does not change. Yalniz YON tahmini etkilenir -- and urunun direction hatti tezin ham `v_o - v_s`'sini
ZATEN asmis durumda (channel_axis -> _snap_axis -> normal-kovaryans -> B-rep, four kademe).
Secici that zincire besinci kademe adds; tezin ekseni ADAYLARDAN BIRI as kalir and sonuc
tezin ham ekseniyle YAN YANA raporlanir.

SIZINTI: exam kumesi and ikizleri DISARIDA. Egitim only `_der_yeni_g6` parcalarindan.
"""
import sys ,os ,json ,io ,pickle ,time 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,'.')
import numpy as np ,trimesh ,wire_gate ,thesis_remesh 
import brep_axes as BX 
from cp_geometry import channel_axis ,mouth_width ,ray_hits 
from f2_12_veri_kolu import egit 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 
from r15_eksen_secici import candidates 

AD =["mevcut","kanal","brep","duzlem"]

def ozn (mesh ,p ,v ,d0 ):
    """Aday basina GT'siz features."""
    try :ic ,ort =mouth_width (mesh ,p ,v )
    except Exception :ic =ort =0.0 
    try :
        h1 =ray_hits (mesh ,p ,v ,max_mm =40.0 );h2 =ray_hits (mesh ,p ,-v ,max_mm =40.0 )
    except Exception :h1 =h2 =np .zeros (0 )
    ileri =float (h1 [0 ])if len (h1 )else 40.0 
    geri =float (h2 [0 ])if len (h2 )else 40.0 
    return [float (ic ),float (ort ),ileri ,geri ,max (ileri ,geri ),min (ileri ,geri ),
    float (len (h1 )),float (len (h2 )),
    float (abs (v @d0 )),# mevcut yonle uyum
    float (np .max (np .abs (v ))),# eksene hizali mi
    (max (ileri ,geri )/max (ic ,0.5 ))if ic >0 else -1.0 ]

def main ():
    sv =set (json .load (io .open ('results/d5_4_sinav_kumesi.json',encoding ='utf-8'))['pidler'])
    G6 ={r ['pid']:r for r in pickle .load (open ('results/_der_yeni_g6.pkl','rb'))}
    yol ={p :s for m ,p ,jf ,s in eligible ()}
    egt =[r for p ,r in sorted (G6 .items ())if p not in sv ][:900 ]
    m ,_ =egit ('results/zengin_parite_v3.npz')
    X ,Y ,AC =[],[],[]
    t0 =time .time ()
    for i ,r in enumerate (egt ,1 ):
        if i %50 ==0 :
            print (f"  {i }/{len (egt )} ({(time .time ()-t0 )/i :.1f}s/part) satir {len (X )}",flush =True )
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
            g =Gd [b_ ];ad =candidates (mesh ,cyl ,P [a_ ],nP [a_ ])
            for nm ,v in ad :
                aci =np .degrees (np .arccos (min (1 ,abs (float (v @g )))))
                X .append (ozn (mesh ,P [a_ ],v ,nP [a_ ])+[float (AD .index (nm ))])
                Y .append (1 if aci <=10 else 0 )
                AC .append (aci )
    X =np .array (X ,float );Y =np .array (Y );AC =np .array (AC )
    np .savez ("results/r16_secici_veri.npz",X =X ,y =Y ,aci =AC )
    print (f"\n{len (X )} satir | dogru candidate orani %{100 *Y .mean ():.0f} -> results/r16_secici_veri.npz")

if __name__ =="__main__":
    main ()
