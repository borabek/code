# -*- coding: utf-8 -*-
"""R1b: DIK vakalarda AGIZ DUZLEMI NORMALI GT eksenini buluyor mu?

R1a olctu: channel ekseni dikleri only %13 kurtariyor -> dikler KARE/YAY giris (channel absent).
Kare a aciklikta giris yonu = ACIKLIK DUZLEMININ NORMALI. Sonda: candidate cevresindeki yerel
surface yamasina PCA -> most small ozdegerin vektoru (duzlem normali) vs GT ekseni.
Ayrica two radius denenir (3mm/6mm) -- mouth kenari mi genis patch mi more iyi.
"""
import sys ,os ,json ,io ,pickle 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,'.')
import numpy as np 
import wire_gate ,thesis_remesh 
from f2_12_data_kolu import egit 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 

sv =json .load (io .open ('results/d5_4_exam_set.json',encoding ='utf-8'))
G6 ={r ['pid']:r for r in pickle .load (open ('results/_der_yeni_g6.pkl','rb'))}
D =[G6 [p ]for p in sorted (set (sv ['pidler']))if p in G6 ]
path ={p :s for m ,p ,jf ,s in eligible ()}
m ,_ =egit ('results/zengin_parite_v3.npz')
dik =[];duz =[]
for r in D :
    if r ['X']is None or r .get ('XR')is None :continue 
    M =np .hstack ([r ['X'],r ['XR']]).astype (float )
    if M .shape [1 ]*2 !=m ['n_feat']:continue 
    k =wire_gate .decision_mask (wire_gate .decision_score (m ,M ))
    P =np .asarray (r ['P'],float )[k ];Pd =np .asarray (r ['Pd'],float )[k ]
    G =np .asarray (r ['G'],float );Gd =np .asarray (r ['Gd'],float )
    if not len (P )or not len (G ):continue 
    d =P [:,None ,:]-G [None ,:,:];al =(d *Gd [None ,:,:]).sum (-1 )
    pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
    pe2 =np .where (np .abs (al )<=40 ,pe ,np .inf )
    tol =max (3.0 ,0.06 *float (r ['diag']));up ,ug =set (),set ()
    for dd ,a_ ,b_ in sorted ((pe2 [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
        if dd >tol or a_ in up or b_ in ug :continue 
        up .add (a_ );ug .add (b_ )
        nP =Pd [a_ ]/(np .linalg .norm (Pd [a_ ])+1e-9 )
        c =abs (float (nP @Gd [b_ ]))
        (dik if c <0.342 else duz ).append ((r ['pid'],P [a_ ],Gd [b_ ]))
print (f"dik {len (dik )} | duz {len (duz )}")
rng =np .random .RandomState (0 )
VC ={}
def normal_pca (pid ,p ,ry ):
    if pid not in VC :
        V ,F =step_to_mesh (path [pid ]);V ,_ =thesis_remesh .remesh_uniform (V ,F ,target =6000 )
        VC [pid ]=np .asarray (V ,float )
    V =VC [pid ]
    q =V [np .linalg .norm (V -p ,axis =1 )<=ry ]
    if len (q )<8 :return None 
    Q =q -q .mean (0 )
    w ,vv =np .linalg .eigh (Q .T @Q )
    return vv [:,0 ]
for ad ,cluster in (("DIK",dik ),("duz-kontrol",duz )):
    sec =[cluster [i ]for i in rng .permutation (len (cluster ))[:50 ]]
    for ry in (3.0 ,6.0 ):
        n =k10 =0 
        for pid ,p ,gd in sec :
            nr =normal_pca (pid ,np .asarray (p ,float ),ry )
            if nr is None :continue 
            n +=1 
            if abs (float (nr @gd ))>=np .cos (np .deg2rad (10 )):k10 +=1 
        print (f"  {ad :<12} r={ry }mm  measured_path {n :>3}  duzlem normali GT'yi buluyor: %{100 *k10 /max (n ,1 ):.0f}")
