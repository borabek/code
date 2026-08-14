# -*- coding: utf-8 -*-
"""C ADIMI, 1/2: EKSEN data seti -- manufacturer InsertDirection'larini KOSE hedefine cevir.

WHY: elimizde 8534 manufacturer ConnectionPoint yonu present and model bunlari HIC gormedi. Eksen this an
tamamen geometrik sezgiyle cikariliyor. Bugun measured: axis yonu robot-hazir F1'in 0.109'unu
single basina tutuyor, and B-rep duzeltmesinden after bile CP'lerin %16'sinda axis 15 dereceden
extra sapiyor. Geometrinin cozemedigi kisim yuva/kelepce girisleri -- oralarda silindir absent,
i.e. analitik axis de absent. Ogrenmek single kalan path.

TASARIM
  * Hedef only manufacturer CP'sinin YAKININDAKI kosede tanimlidir; gerisi MASKELENIR. Boylece
    CAD sozde-etiketlerine never dokunmayiz (onlar insan GT'siyle however F1~0.46 ortusuyor).
  * Yon ISARETSIZ ogrenilir (kosinusun MUTLAK degeri): manufacturer sign konvansiyonu parcadan
    parcaya degisiyor, isareti already geometrik as (outward) belirliyoruz.
  * Cerceve: hedefler cad_eval.align_frames with MESH cercevesine tasinir -- training and inference
    same cercevede must be, otherwise ogrenilen sey noise becomes.

Cikti: results/axis_dataset/<part>.npz  (idx, target_dir) + results/axis_dataset/index.json
"""
import os ,sys ,json ,glob ,argparse 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
os .environ .setdefault ("BA_ALLOW_SEEN","1")
OUT ="results/axis_dataset"


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--radius",type =float ,default =4.0 ,
    help ="manufacturer CP'sine bu mesafedeki koseler hedef alir (mm)")
    ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--target",type =int ,default =6000 )
    a =ap .parse_args ()

    import thesis_remesh 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh 
    from big_arbiter import eligible 

    os .makedirs (OUT ,exist_ok =True )
    done ={os .path .basename (f )[:-4 ]for f in glob .glob (os .path .join (OUT ,"*.npz"))}
    parts =[]
    for mfg ,pid ,jf ,stp in eligible ():
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig")).get ("ConnectionPoints")or [])
        except Exception :
            continue 
        if n >0 :
            parts .append ((mfg ,pid ,jf ,stp ,n ))
    if a .limit :
        parts =parts [:a .limit ]
    print (f"{len (parts )} part | {len (done )} zaten hazir | yaricap {a .radius }mm",flush =True )

    index ,nv_tot ,npart =[],0 ,0 
    for k ,(mfg ,pid ,jf ,stp ,n )in enumerate (parts ,1 ):
        if pid in done :
            continue 
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],
            float )
            nrm =np .linalg .norm (Gd ,axis =1 ,keepdims =True )
            keep =(nrm [:,0 ]>1e-9 )
            G ,Gd =G [keep ],Gd [keep ]/(nrm [keep ]+1e-12 )
            if not len (G ):
                continue 
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =a .target )
            V =np .ascontiguousarray (V ,np .float64 )
            R ,t ,_ =align_frames (Vr ,Vj )
            Gm =(G -t )@R # manufacturer CP -> mesh cercevesi
            Gdm =Gd @R 

            d =np .linalg .norm (V [:,None ,:]-Gm [None ,:,:],axis =-1 )
            near =d .min (1 )<=a .radius 
            if int (near .sum ())<8 :
                continue 
            owner =d .argmin (1 )
            idx =np .where (near )[0 ].astype (np .int32 )
            tgt =Gdm [owner [near ]].astype (np .float32 )
            np .savez (os .path .join (OUT ,f"{pid }.npz"),idx =idx ,tgt =tgt ,
            n_verts =np .int32 (len (V )),mfg =np .str_ (mfg ),stp =np .str_ (stp ))
            index .append ({"part":pid ,"mfg":mfg ,"stp":stp ,
            "n_target_verts":int (len (idx )),"n_cp":int (len (Gm ))})
            nv_tot +=len (idx );npart +=1 
        except Exception :
            continue 
        if k %50 ==0 :
            print (f"  {k }/{len (parts )}  hazir {npart } part / {nv_tot } hedef kose",flush =True )

    old =[]
    if os .path .exists (os .path .join (OUT ,"index.json")):
        try :
            old =json .load (open (os .path .join (OUT ,"index.json")))["parts"]
        except Exception :
            old =[]
    allp ={p ["part"]:p for p in old }
    allp .update ({p ["part"]:p for p in index })
    json .dump ({"radius_mm":a .radius ,"remesh_target":a .target ,
    "n_parts":len (allp ),"parts":list (allp .values ())},
    open (os .path .join (OUT ,"index.json"),"w"),indent =1 )
    print (f"\nbitti: {npart } yeni part, {nv_tot } hedef kose | toplam {len (allp )} part")
    print (f"-> {OUT }/")


if __name__ =="__main__":
    main ()
