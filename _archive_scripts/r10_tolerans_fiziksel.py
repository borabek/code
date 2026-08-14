# -*- coding: utf-8 -*-
"""R10: ROBOT TOLERANSI FIZIKSEL OLARAK TURETILEBILIR MI? (2mm KEYFI MI?)

Kod tabaninda 2mm'nin gerekcesi YOK -- `GEO_thesis_faithful_plan.md:44` only "Robot hedefi
<2mm" diye BEYAN ediyor. Fiziksel dogrusu: tel agizdan gecebiliyorsa robot takabilir, i.e.
konum toleransi = (mouth ic yaricapi - tel yaricapi) = BOSLUK.

`mouth_width` A0'da kalibre edildi (%1 deviation). Her matched CP'de agzin real ic capini
olcup, klemens teli capina according to BOSLUGU cikariyoruz. Sonra: this fiziksel toleransla
robot-hazir F1 kac?

BU SISIRME DEGIL. Sisirme, sonucu guzellestirmek for esigi gevsetmektir. Burada threshold
GEOMETRIDEN turetiliyor and part basina DEGISIYOR -- dar agizda 2mm'den SIKI may be.
"""
import sys ,os ,json ,io ,pickle 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,'.')
import numpy as np ,trimesh 
import wire_gate ,thesis_remesh 
from f2_12_data_kolu import egit 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 
from cp_geometry import mouth_width 

# IEC 60947-7-1 anma kesitleri -> iletken capi (mm). Klemens agzi TELDEN buyuktur.
# En yaygin klemens 2.5mm^2 (cap ~1.78mm); 1.5mm^2 -> 1.38mm; 4mm^2 -> 2.26mm.
TEL_CAP =1.78 

sv =json .load (io .open ('results/d5_4_exam_set.json',encoding ='utf-8'))
G6 ={r ['pid']:r for r in pickle .load (open ('results/_der_yeni_g6.pkl','rb'))}
D =[G6 [p ]for p in sorted (set (sv ['pidler']))if p in G6 ]
path ={p :s for m ,p ,jf ,s in eligible ()}
m ,_ =egit ('results/zengin_parite_v3.npz')

VC ={};rec_ =[]
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
    nP =Pd /(np .linalg .norm (Pd ,axis =1 ,keepdims =True )+1e-9 );cos =np .clip (nP @Gd .T ,-1 ,1 )
    tol =max (3.0 ,0.06 *float (r ['diag']));up ,ug =set (),set ()
    for dd ,a_ ,b_ in sorted ((pe2 [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
        if dd >tol or a_ in up or b_ in ug :continue 
        up .add (a_ );ug .add (b_ )
        try :
            if r ['pid']not in VC :
                V ,F =step_to_mesh (path [r ['pid']]);V ,F =thesis_remesh .remesh_uniform (V ,F ,target =6000 )
                VC [r ['pid']]=trimesh .Trimesh (vertices =np .ascontiguousarray (V ,np .float64 ),
                faces =np .ascontiguousarray (F ,np .int64 ),process =False )
            ic ,ort =mouth_width (VC [r ['pid']],G [b_ ],Gd [b_ ])
        except Exception :
            ic =ort =0.0 
        rec_ .append ((pe2 [a_ ,b_ ],np .degrees (np .arccos (abs (cos [a_ ,b_ ]))),float (ic ),float (ort )))
    if len (rec_ )>200 :break 

K =np .array (rec_ )
yan ,aci ,ic ,ort =K [:,0 ],K [:,1 ],K [:,2 ],K [:,3 ]
ok =ic >0.1 
print (f"measured_path {len (K )} cift | mouth capi olculebilen {ok .sum ()}")
print (f"\nAGIZ IC CAPI (mm): ortanca {np .median (ic [ok ]):.2f} | %25 {np .percentile (ic [ok ],25 ):.2f} | %75 {np .percentile (ic [ok ],75 ):.2f}")
bosluk =np .maximum ((ic -TEL_CAP )/2.0 ,0.0 )
print (f"FIZIKSEL BOSLUK (=(ic_cap-{TEL_CAP })/2): ortanca {np .median (bosluk [ok ]):.2f}mm | %25 {np .percentile (bosluk [ok ],25 ):.2f} | %75 {np .percentile (bosluk [ok ],75 ):.2f}")
print (f"  bosluk >= 2mm which: %{100 *(bosluk [ok ]>=2 ).mean ():.0f}   (ie 2mm bunlarda GEREKSIZ SIKI)")
print (f"  bosluk <  1mm which: %{100 *(bosluk [ok ]<1 ).mean ():.0f}   (ie 2mm bunlarda FAZLA GEVSEK)")
def f1 (msk ):
    tp =int (msk .sum ());return tp 
print (f"\n{'threshold':<34}{'passing double':>12}{'ratio':>8}")
for ad ,m_ in (("SABIT 2mm + 10deg (mevcut)",(yan <=2 )&(aci <=10 )),
("SABIT 3mm + 10deg",(yan <=3 )&(aci <=10 )),
("FIZIKSEL bosluk + 10deg",(yan <=np .maximum (bosluk ,0.5 ))&(aci <=10 )),
("FIZIKSEL bosluk + 15deg",(yan <=np .maximum (bosluk ,0.5 ))&(aci <=15 )),
("only FIZIKSEL bosluk",(yan <=np .maximum (bosluk ,0.5 )))):
    print (f"{ad :<34}{f1 (m_ ):>12}{100 *m_ .mean ():>7.0f}%")
