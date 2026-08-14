# -*- coding: utf-8 -*-
"""R11: B-REP ANALITIK EKSENI, YUVARLANMADAN ONCE, GT'yi buluyor mu?

`_snap_axis` docstring'i (olculmus): manufacturer CP'lerinin %19.1'i 10 dereceden EGIK (43'e up to);
matched ciftlerin %18.2'sinde eksenimiz 15-90 derece sapiyor, most ~90. Yuvarlamayi kapatmak
DENENDI and KAYBETTI, because teshis this: "measured_path HAM eksenler de already eksene hizali --
channel_axis egimi GOREMIYOR". Yani suclu yuvarlama not, EKSEN OLCUMU.

`channel_axis` only X/Y/Z dener and isin-sondasi deligin koni acisindan (~11 derece) keskin
olamaz. B-REP SILINDIR EKSENI whereas CAD'de ANALITIK durur -- cozunurluk siniri YOK.
`brep_axes.axis_at` mevcut and cp_openings:370'te is used AMA after `_snap_axis`'ten
geciyor. Soru: yuvarlamadan ONCE B-rep ekseni GT'yi ne up to buluyor?

Bu betik OLCER, degistirmez.
"""
import sys ,os ,json ,io ,pickle ,collections 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,'.')
import numpy as np ,wire_gate 
from f2_12_data_kolu import egit 
from big_arbiter import eligible 
import measure_set as OK 
import brep_axes as BX 

G6 ={r ['pid']:r for r in pickle .load (open ('results/_der_yeni_g6.pkl','rb'))}
path ={p :s for m ,p ,jf ,s in eligible ()}
m ,_ =egit ('results/zengin_parite_v3.npz')

def kos (DER ,ad ,bound_ =90 ):
    cyl_cache ={}
    stat =collections .Counter ();acilar ={'mevcut':[],'brep':[]}
    n =0 
    for r in DER :
        if n >=bound_ :break 
        if r ['X']is None or r .get ('XR')is None :continue 
        M =np .hstack ([r ['X'],r ['XR']]).astype (float )
        if M .shape [1 ]*2 !=m ['n_feat']:continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (m ,M ))
        P =np .asarray (r ['P'],float )[k ];Pd =np .asarray (r ['Pd'],float )[k ]
        G =np .asarray (r ['G'],float );Gd =np .asarray (r ['Gd'],float )
        if not len (P )or not len (G ):continue 
        pid =r ['pid']
        if pid not in cyl_cache :
            try :cyl_cache [pid ]=BX .cylinders (path [pid ])
            except Exception :cyl_cache [pid ]=None 
        cyl =cyl_cache [pid ]
        n +=1 
        d =P [:,None ,:]-G [None ,:,:];al =(d *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe2 =np .where (np .abs (al )<=40 ,pe ,np .inf )
        nP =Pd /(np .linalg .norm (Pd ,axis =1 ,keepdims =True )+1e-9 )
        tol =max (3.0 ,0.06 *float (r ['diag']));up ,ug =set (),set ()
        for dd ,a_ ,b_ in sorted ((pe2 [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
            if dd >tol or a_ in up or b_ in ug :continue 
            up .add (a_ );ug .add (b_ )
            g =Gd [b_ ]
            a_mev =np .degrees (np .arccos (min (1 ,abs (float (nP [a_ ]@g )))))
            acilar ['mevcut'].append (a_mev )
            ax =None 
            if cyl is not None and len (cyl [0 ]):
                try :ax =BX .axis_at (P [a_ ],nP [a_ ],cyl )
                except Exception :ax =None 
            if ax is None :
                stat ['brep_YOK']+=1 ;acilar ['brep'].append (a_mev )# backup: mevcut
            else :
                v =np .asarray (ax [0 ]if isinstance (ax ,tuple )else ax ,float ).reshape (3 )
                v =v /(np .linalg .norm (v )+1e-9 )
                a_br =np .degrees (np .arccos (min (1 ,abs (float (v @g )))))
                acilar ['brep'].append (a_br );stat ['brep_VAR']+=1 
                if a_mev >80 and a_br <=10 :stat ['DIK_KURTARILDI']+=1 
                if a_mev <=10 and a_br >80 :stat ['IYIYI_BOZDU']+=1 
    A =np .array (acilar ['mevcut']);B =np .array (acilar ['brep'])
    bv =stat ['brep_VAR'];by =stat ['brep_YOK']
    print (f'=== {ad } (n={len (A )} cift, {bv } B-rep VAR / {by } YOK) ===')
    print (f'  aci<=10  MEVCUT %{100 *(A <=10 ).mean ():.0f}   B-REP %{100 *(B <=10 ).mean ():.0f}')
    print (f'  DIK>80   MEVCUT %{100 *(A >80 ).mean ():.0f}   B-REP %{100 *(B >80 ).mean ():.0f}')
    print (f"  DIK KURTARILAN {stat ['DIK_KURTARILDI']} | IYIYI BOZAN {stat ['IYIYI_BOZDU']}")

D194_e ,_ =OK .cluster ('results/_der_tam.pkl')
D194 =[G6 [p ]for p in sorted ({r ['pid']for r in D194_e })if p in G6 ]
sv =json .load (io .open ('results/d5_4_exam_set.json',encoding ='utf-8'))
SIN =[G6 [p ]for p in sorted (set (sv ['pidler']))if p in G6 ]
kos (D194 ,'194luk TANIDIK');kos (SIN ,'GORULMEMIS')
