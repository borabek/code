# -*- coding: utf-8 -*-
"""FB-1: EMBEDDING UZAYININ TAVANI -- 0.80'in mumkun olup olmadigini soyleyen measurement.

DIAGNOSIS (adli type raporundan, bugun dogrulandi): tezin `Contact` sinifi tasarimi geregi
"Kontaktierung *bzw.* Werkzeugeinschub" -- tel girisi and alet agzi TEK SINIF. Yani backbone,
bizim ayirmak istedigimiz two seyi BIRLESTIRMEK so as to egitildi. Dokuz gate mekanizmasinin
same cepheye carpmasinin koku this.

AMA rapor own "backbone kor" tezini de curutmus (family-out AUC):
    only backbone embedding'leri   0.9272
    only 13 el-yapimi feature    0.8997
Backbone ayrimi TASIYOR and el-yapimini geciyor. Yani bilgi orada; biz onu 73 sutuna
sikistirirken kaybediyoruz.

BU OLCUM: more before EL-YAPIMI 73 sutunda yaptigim komsu-uyusmazligi analizini (uyusmazlik
%28.2 -> ceiling ~0.86) AYNEN embedding uzayinda tekrarlar.

OKUMA:
  embedding uyusmazligi BELIRGIN DUSUKSE -> ceiling yukselir, tespit 0.84 (robot 0.80'in
      gerektirdigi) ULASILABILIR, and path TEMSILDEN gecer (yardimci supervizyon).
  FARK YOKSA -> problem gercekten doymus; 0.80 mevcut bilgi rejiminde absent.

Embedding: `last_lin` oncesi 128 boyutlu vertex temsili, adayin AGIZ BOLGESINDE havuzlanir
(mean + maksimum = 256 size). Havuzlama bolgesi, adayin cevresindeki 6mm.
"""
import io 
import json 
import os 
import sys 
import time 

import numpy as np 
import torch 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

ONB ="results/fb1_embed.pkl"
YARICAP =6.0 


def embed_al (model ,meta ,V ,F ,dev ,op_cache ):
    """last_lin ONCESI 128-boyutlu vertex temsili."""
    import diffusionnet as D_ 
    ops =D_ .precompute_operators (V ,F ,meta ["k_eig"],op_cache )
    ops ={k :(v .to (dev )if hasattr (v ,"to")else v )for k ,v in ops .items ()}
    tut ={}

    def kanca (mod ,giren ,cikan ):
        tut ["z"]=giren [0 ].detach ()
    h =model .last_lin .register_forward_hook (kanca )
    try :
        with torch .no_grad ():
            D_ ._forward (model ,ops ,D_ ._model_input (ops ,meta ))
    finally :
        h .remove ()
    return tut ["z"].cpu ().numpy ()if "z"in tut else None 


def main ():
    import gate_bench as T 
    import measure_set 
    import pickle 
    import thesis_remesh 
    from big_arbiter import eligible 
    from infer_step_cp import load_any ,step_to_mesh 
    from sklearn .preprocessing import StandardScaler 
    from sklearn .neighbors import NearestNeighbors 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"

    if os .path .exists (ONB ):
        with open (ONB ,"rb")as f :
            EMB ,RY ,RG ,RP ,HX =pickle .load (f )
        print (f"onbellekten: {len (RY )} candidate",flush =True )
    else :
        models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
        EMB ,RY ,RG ,RP ,HX =[],[],[],[],[]
        t0 =time .time ()
        for k ,r in enumerate (D ["DER"],1 ):
            if k %20 ==0 :
                print (f"  {k }/{len (D ['DER'])}  {time .time ()-t0 :.0f}s",flush =True )
            if r ["X"]is None or r .get ("XR")is None or not len (r ["G"]):
                continue 
            try :
                Vr ,Fr =step_to_mesh (stp [r ["pid"]])
                V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
                Z =[]
                for model ,meta in models :
                    z =embed_al (model ,meta ,V ,F ,dev ,
                    f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}")
                    if z is not None :
                        Z .append (z )
                if not Z :
                    continue 
                Zm =np .mean (Z ,axis =0 )# ensemble ortalamasi (V,128)
            except Exception as e :
                print (f"    {r ['pid']}: {type (e ).__name__ }");continue 
            P =np .asarray (r ["P"],float )
            X58 =np .hstack ([r ["X"],r ["XR"]])
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            yy =np .zeros (len (P ),int )
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            tt =max (3.0 ,0.06 *float (r ["diag"]))
            up ,ug =set (),set ()
            for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
            for b in range (len (G ))):
                if not np .isfinite (d_ )or d_ >tt or a_ in up or b_ in ug :
                    continue 
                up .add (a_ );ug .add (b_ );yy [a_ ]=1 
            for i in range (len (P )):
                d =np .linalg .norm (V -P [i ],axis =1 )
                m =d <=YARICAP 
                if m .sum ()<3 :
                    m =d <=YARICAP *2 
                if m .sum ()<1 :
                    continue 
                EMB .append (np .concatenate ([Zm [m ].mean (0 ),Zm [m ].max (0 )]))# 256
                HX .append (X58 [i ]);RY .append (yy [i ]);RG .append (r ["geo"]);RP .append (r ["pid"])
        with open (ONB ,"wb")as f :
            pickle .dump ((EMB ,RY ,RG ,RP ,HX ),f )
        print (f"-> {ONB }",flush =True )

    EMB =np .array (EMB ,float );RY =np .array (RY )
    RG =np .array (RG );RP =np .array (RP );HX =np .array (HX ,float )
    print (f"\n{len (RY )} candidate | embedding {EMB .shape [1 ]} boyut | el-yapimi {HX .shape [1 ]} sutun")

    def uyusmazlik (S ,k =15 ):
        S =StandardScaler ().fit_transform (S )
        nn =NearestNeighbors (n_neighbors =40 ).fit (S )
        _ ,idx =nn .kneighbors (S )
        v =[]
        for a in range (len (RY )):
            j =[x for x in idx [a ][1 :]if RP [x ]!=RP [a ]][:k ]
            if len (j )<8 :
                continue 
            v .append (np .mean (RY [np .array (j )]!=RY [a ]))
        v =np .array (v )
        return float (v .mean ()),float ((v >=0.4 ).mean ())

    print (f"\n{'uzay':<34}{'uyusmazlik':>12}{'ayrilamaz':>11}{'ceiling':>9}")
    SON ={}
    for ad ,S in (("EL-YAPIMI (73 sutun)",HX ),
    ("EMBEDDING (256, havuzlanmis)",EMB ),
    ("IKISI BIRDEN",np .hstack ([HX ,EMB ]))):
        u ,a =uyusmazlik (S )
        SON [ad ]={"uyusmazlik":u ,"ayrilamaz":a ,"ceiling":1 -u /2 }
        print (f"{ad :<34}{u :>11.1%}{a :>11.1%}{1 -u /2 :>9.3f}")

    e =SON ["EL-YAPIMI (73 sutun)"]["ceiling"];z =SON ["EMBEDDING (256, havuzlanmis)"]["ceiling"]
    print (f"\nEMBEDDING TAVANI - EL-YAPIMI TAVANI = {z -e :+.3f}")
    print ()
    if z -e >=0.02 :
        print ("HUKUM: TEMSIL uzayi DAHA AYRILABILIR. ~0.86 ceiling EL-YAPIMI uzayin tavaniydi;")
        print ("       problem doymus DEGIL. Tespit 0.84 (robot 0.80'in sarti) ULASILABILIR")
        print ("       ve yol TEMSILDEN gecer -> FB-2 (tel-farkindalikli yardimci supervizyon).")
    elif z -e >=0.005 :
        print ("HUKUM: ZAYIF ama pozitif fark. Temsil biraz daha ayrilabilir; FB-2 makul ama")
        print ("       beklenti olculu tutulmali.")
    else :
        print ("HUKUM: TEMSIL DE AYNI TAVANDA. Backbone gercekten ayrimi tasimiyor ->")
        print ("       0.80 mevcut bilgi rejiminde YOK; tek yol yeni veri/yeni etiket.")
    with io .open ("results/fb1_embedding_tavani.json","w",encoding ="utf-8")as f :
        json .dump (SON ,f ,indent =1 ,ensure_ascii =False )
    print ("receipt -> results/fb1_embedding_tavani.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
