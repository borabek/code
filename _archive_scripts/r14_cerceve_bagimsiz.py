# -*- coding: utf-8 -*-
"""R14: CERCEVE RESIDUAL'I MODELDEN BAGIMSIZ OLC -- 0.24 mu 0.40 mi raporlanacak?

SORUN: exam kumesini "cerceve saglikli" diye filtrelediginde robot F1 0.2410 -> 0.4006
cikiyor. Ama filtre olcutu MODELIN ADAYLARINA bakiyordu (GT'nin most yakin adaya mesafesi),
i.e. sonucun KENDISIYLE filtreliyorduk -- DAIRESEL.

DOGRU OLCUT: `cad_eval.align_frames` own MESH-TO-MESH residual'ini returns (uctan uca
GT/mesh hizalamasi, modelden TAMAMEN bagimsiz). Sabah AL vakasinda this kullanildi: residual
0.13mm output and "hizalama bozuk" hipotezi CURUTULDU.

TEZ SADAKATI: this a OLCUM duzeltmesidir, urun degisikligi not. Tezin `v_o` tanimina,
5 sinifa, remesh'e dokunulmuyor. Yalnizca "hangi parcalarda GT guvenilir" sorusu MODELDEN
BAGIMSIZ yanitlanip headline that zemine oturtuluyor.
"""
import sys ,os ,json ,io ,pickle 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,'.')
import numpy as np ,wire_gate ,cad_eval 
from f2_12_data_kolu import egit 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 
import measure_set as OK 

def olc (model ,DER ,robot ):
    tp =fp =fn =0 ;ca =np .cos (np .deg2rad (10 ))
    for r in DER :
        if r ['X']is None or r .get ('XR')is None :continue 
        M =np .hstack ([r ['X'],r ['XR']]).astype (float )
        if M .shape [1 ]*2 !=model ['n_feat']:continue 
        k =wire_gate .decision_mask (wire_gate .decision_score (model ,M ))
        P =np .asarray (r ['P'],float )[k ];Pd =np .asarray (r ['Pd'],float )[k ]
        G =np .asarray (r ['G'],float );Gd =np .asarray (r ['Gd'],float );e =0 
        if len (P )and len (G ):
            d =P [:,None ,:]-G [None ,:,:];al =(d *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (d -al [...,None ]*Gd [None ,:,:],axis =-1 )
            if robot :
                nP =Pd /(np .linalg .norm (Pd ,axis =1 ,keepdims =True )+1e-9 )
                ok =(pe <=2.0 )&(nP @Gd .T >=ca )&(np .abs (al )<=40 )
            else :
                ok =(pe <=max (3.0 ,0.06 *float (r ['diag'])))&(np .abs (al )<=40 )
            up ,ug =set (),set ()
            for dd ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
                if not ok [a_ ,b_ ]or a_ in up or b_ in ug :continue 
                up .add (a_ );ug .add (b_ );e +=1 
        tp +=e ;fp +=len (P )-e ;fn +=len (G )-e 
    return 2 *tp /max (2 *tp +fp +fn ,1 )

sv =json .load (io .open ('results/d5_4_exam_set.json',encoding ='utf-8'))
G6 ={r ['pid']:r for r in pickle .load (open ('results/_der_yeni_g6.pkl','rb'))}
E ={p :(jf ,s )for m ,p ,jf ,s in eligible ()}
SIN =[G6 [p ]for p in sorted (set (sv ['pidler']))if p in G6 ]

# --- MESH-TO-MESH residual (MODELDEN BAGIMSIZ)
res ={}
for i ,r in enumerate (SIN ,1 ):
    if i %40 ==0 :print (f"  {i }/{len (SIN )}",flush =True )
    jf ,stp =E .get (r ['pid'],(None ,None ))
    if jf is None :continue 
    try :
        j =json .load (io .open (jf ,encoding ='utf-8-sig'))
        Vj =np .array ([[q ['X'],q ['Y'],q ['Z']]for q in j ['Graphic3d']['Points']],float )
        Vr ,_ =step_to_mesh (stp )
        _ ,_ ,rr =cad_eval .align_frames (Vr ,Vj )
        res [r ['pid']]=float (rr )
    except Exception :
        pass 
R =np .array (list (res .values ()))
print (f"\nMESH-TO-MESH residual ({len (R )} part): ortanca {np .median (R ):.2f}mm | "
f"%75 {np .percentile (R ,75 ):.2f} | %90 {np .percentile (R ,90 ):.2f} | maks {R .max ():.2f}")
for e in (0.5 ,1.0 ,2.0 ):
    print (f"  <{e }mm: {(R <e ).sum ()} part (%{100 *(R <e ).mean ():.0f})")

m ,_ =egit ('results/zengin_parite_v3.npz')
print (f"\n{'layer (MODELDEN BAGIMSIZ criterion)':<34}{'part':>7}{'detection':>10}{'ROBOT':>9}")
for ad ,e in (('all of them',99. ),('residual <2mm',2.0 ),('residual <1mm',1.0 ),('residual <0.5mm',0.5 )):
    alt =[r for r in SIN if res .get (r ['pid'],99. )<e ]
    if len (alt )<20 :continue 
    print (f"{ad :<34}{len (alt ):>7}{olc (m ,alt ,False ):>10.4f}{olc (m ,alt ,True ):>9.4f}")
with io .open ("results/r14_cerceve_bagimsiz.json","w",encoding ="utf-8")as f :
    json .dump ({"residual":res },f ,indent =1 )
print ("receipt -> results/r14_cerceve_bagimsiz.json")
