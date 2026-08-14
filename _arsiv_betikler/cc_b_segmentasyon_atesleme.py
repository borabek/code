# -*- coding: utf-8 -*-
"""CC-B: very-CP parcalarinda KACIRILAN GT noktalarinda segmentasyon ATESLIYOR MU?

LISTENIN YARISININ KADERI BUNA BAGLI:
  * ATESLIYOR  -> engel post-isleme/cozunurluk. min_v10 + adaptif hedef (CC-C) whereas yarar,
                  CC-G (segmentasyon yeniden egitimi, GUNLER) gereksiz.
  * ATESMIYOR  -> network kor. Vertex'i acmak whereas yaramaz (sinif gelmez); CC-C and CC-E large
                  olcude bosa gider and CC-G ANA IS becomes.

CC-A'yi TAMAMLAR: CC-A "that acikliklarda yeterli VERTEX absent" dedi (min_v=30 for %82.7).
Bu, gerek kosulun ihlali. CC-B yeter kosulu sorar: vertexler olsa AGDA SINIF present mi?

YONTEM: each GT agzi for cevresindeki vertexlerin CE/CT olasiligina bak. Bulunan GT'lerle
(robotun eslestirdigi) kacirilanlari KIYASLA -- mutlak number not FARK anlamli.
"""
import os ,sys ,json ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

OUT ="results/cc_b_atesleme.json"
HIGH_CP =8 
RADII =(2.0 ,3.0 ,4.0 )


def main (limit =60 ):
    import torch ,trimesh 
    import thesis_remesh ,robot_cp 
    import diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from cp_geometry import mouths_for 
    from scipy .spatial import cKDTree 

    lock =json .load (open ("results/split_lock.json"));LOCK =set (lock ["locked_parts"])
    cfg =json .load (open ("cp_config.json"))
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["current_product"]["checkpoints"]]

    parts =[]
    for mfg ,pid ,jf ,stp in eligible ():
        if pid in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >=HIGH_CP :parts .append ((mfg ,pid ,jf ,stp ,n ))
    rng =np .random .RandomState (0 )
    if len (parts )>limit :
        parts =[parts [i ]for i in sorted (rng .choice (len (parts ),limit ,replace =False ))]
    print (f"{len (parts )} cok-CP part ({HIGH_CP }+ CP)\n",flush =True )

    found ,missed =[],[]
    t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp ,ngt )in enumerate (parts ,1 ):
        try :
            cps =robot_cp .extract (models ,stp ,dev ,conf_auto =float (cfg .get ("robot_conf_auto",0.5 )))
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            acc =None 
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
            probs =acc /len (models )
            conn =probs [:,CE ]+probs [:,CT ]
            lab =probs .argmax (-1 )

            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            R ,t ,_ =align_frames (Vr ,Vj )
            Gm =(G -t )@R ;Gdm =Gd @R 
            body =trimesh .Trimesh (V ,F ,process =False )
            mouths ,_ =mouths_for (body ,Gm ,Gdm )

            # hangi GT BULUNDU (robotun adayiyla eslesen)
            hit =np .zeros (len (Gm ),bool )
            if cps :
                P =np .array ([c ["point"]for c in cps ],float )
                tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
                diff =P [:,None ,:]-Gm [None ,:,:]
                al =(diff *Gdm [None ,:,:]).sum (-1 )
                perp =np .linalg .norm (diff -al [...,None ]*Gdm [None ,:,:],axis =-1 )
                perp =np .where (np .abs (al )<=40.0 ,perp ,np .inf )
                used_p =set ()
                for d_ ,a_ ,b_ in sorted ((perp [a_ ,b_ ],a_ ,b_ )
                for a_ in range (len (P ))for b_ in range (len (Gm ))):
                    if d_ >tol or a_ in used_p or hit [b_ ]:continue 
                    hit [b_ ]=True ;used_p .add (a_ )

            tree =cKDTree (V )
            for b in range (len (Gm )):
                rec ={"pid":pid ,"mfg":mfg }
                for r in RADII :
                    idx =tree .query_ball_point (mouths [b ],r )
                    rec [f"n{r }"]=len (idx )
                    rec [f"conn_max{r }"]=float (conn [idx ].max ())if idx else 0.0 
                    rec [f"conn_mean{r }"]=float (conn [idx ].mean ())if idx else 0.0 
                    rec [f"cecT_frac{r }"]=float (np .isin (lab [idx ],[CE ,CT ]).mean ())if idx else 0.0 
                (found if hit [b ]else missed ).append (rec )
        except Exception as e :
            print (f"  {pid }: HATA {type (e ).__name__ } {str (e )[:50 ]}",flush =True );continue 
        if k %10 ==0 :
            print (f"  {k }/{len (parts )}  bulunan {len (found )} / kacirilan {len (missed )}  "
            f"{time .time ()-t0 :.0f}s",flush =True )

    if not missed :
        print ("kacirilan GT absent -- measurement anlamsiz");return 
    print (f"\n=== {len (found )} BULUNAN vs {len (missed )} KACIRILAN GT ===\n")
    print (f"{'criterion':<22}{'BULUNAN':>10}{'KACIRILAN':>11}   yorum")
    out ={}
    for r in RADII :
        for key ,lbl in [(f"n{r }",f"vertex sayisi r={r }"),
        (f"conn_max{r }",f"max CE+CT olas. r={r }"),
        (f"cecT_frac{r }",f"CE/CT vertex orani r={r }")]:
            a =np .array ([x [key ]for x in found ]);b =np .array ([x [key ]for x in missed ])
            out [key ]={"found_median":float (np .median (a )),"missed_median":float (np .median (b ))}
            print (f"{lbl :<22}{np .median (a ):>10.2f}{np .median (b ):>11.2f}")
        print ()

        # DECISION: kacirilanlarda baglanti olasiligi VAR MI
    b3 =np .array ([x ["conn_max3.0"]for x in missed ])
    a3 =np .array ([x ["conn_max3.0"]for x in found ])
    frac_fire =float ((b3 >=0.30 ).mean ())
    print (f"KACIRILAN GT'lerin %{100 *frac_fire :.0f}'inde yakinda CE+CT olasiligi >= 0.30 var")
    verdict =("ATESLIYOR -> engel POST-ISLEME/COZUNURLUK"if frac_fire >=0.5 else 
    "ATESMIYOR -> AG KOR, CC-G ana is")
    print (f">>> KARAR: {verdict }")
    out ["n_found"]=len (found );out ["n_missed"]=len (missed )
    out ["frac_missed_with_signal"]=frac_fire 
    out ["found_conn_max3_median"]=float (np .median (a3 ))
    out ["missed_conn_max3_median"]=float (np .median (b3 ))
    out ["verdict"]=verdict 
    json .dump (out ,open (OUT ,"w"),indent =1 )
    print (f"\nmakbuz -> {OUT }")


if __name__ =="__main__":
    main (int (sys .argv [1 ])if len (sys .argv )>1 else 60 )
