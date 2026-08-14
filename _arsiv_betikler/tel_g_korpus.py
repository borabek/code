# -*- coding: utf-8 -*-
"""TEL-G'yi korpusa kos: low-CP adaylari for B-rep ozellikleri.

Model KOSULMAZ -- candidate konumlari rich_mv10_all.npz'de already present; single gereken STEP'i acmak
(meshlemeden) and cerceveyi geri cevirmek. ~1.5s/part.

KILL KRITERI (baslamadan yazildi, sonuca according to degistirilmez):
    Mevcut 41 feature UZERINE ek CP-F1 katkisi < 0.02 whereas this aile OLU ilan edilir.
    (TEL-B dersi: kriter AUC'ye not URUN METRIGINE yazilir.)
"""
import os ,sys ,json ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/tel_g_brep_feats.npz"
HIGH_CP =8 


def main ():
    import wire_g_brep as B 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh 
    from big_arbiter import eligible 

    lock =json .load (open ("results/split_lock.json"));LOCK =set (lock ["locked_parts"])
    z =np .load ("results/rich_mv10_all.npz",allow_pickle =True )
    rp =[str (v )for v in z ["part_ids"]]
    pid =np .array ([rp [int (g )]for g in z ["groups"]])
    pos ,y =z ["pos"],z ["y"]
    seen ={rp [i ]:int (z ["seen"][i ])for i in range (len (rp ))}
    ngt_i ={int (g ):int (n )for g ,n in zip (z ["grp_ids"],z ["ngt"])}
    ngt ={rp [int (g )]:ngt_i [int (g )]for g in np .unique (z ["groups"])}

    el ={p :(jf ,s )for m ,p ,jf ,s in eligible ()}
    todo =[q for q in sorted (set (pid ))
    if seen .get (q ,1 )==0 and q not in LOCK and 0 <ngt .get (q ,0 )<HIGH_CP and q in el ]
    print (f"{len (todo )} dusuk-CP part",flush =True )

    feats ,keys ={},None 
    if os .path .exists (OUT ):
        z2 =np .load (OUT ,allow_pickle =True )
        for q ,arr in zip (z2 ["pid_part"],z2 ["blocks"]):
            feats [str (q )]=arr 
        print (f"  [devam] {len (feats )} part zaten islenmis",flush =True )

    t0 =time .time ()
    for k ,q in enumerate (todo ,1 ):
        if q in feats :
            continue 
        try :
            with open ("results/_telg_current.txt","w")as fh :
                fh .write (f"{q }\t{k }/{len (todo )}\t{time .strftime ('%H:%M:%S')}")
            jf ,stp =el [q ]
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[a [c ]for c in "XYZ"]for a in j ["Graphic3d"]["Points"]],float )
            Vr ,Fr =step_to_mesh (stp )
            R ,t ,_ =align_frames (Vr ,Vj )
            P =(pos [pid ==q ]-t )@R # JSON -> STEP cercevesi (see wire_g_brep)
            S =B .read_brep (stp )
            feats [q ]=B .feats_for_points (S ,P )
        except Exception as e :
            print (f"  {q }: HATA {type (e ).__name__ } {str (e )[:50 ]}",flush =True )
            continue 
        if k %50 ==0 :
            np .savez (OUT ,pid_part =np .array (list (feats .keys ())),
            blocks =np .array (list (feats .values ()),dtype =object ),
            names =np .array (B .FEAT_NAMES ))
            print (f"  {k }/{len (todo )}  {len (feats )} ok  {time .time ()-t0 :.0f}s",flush =True )

    np .savez (OUT ,pid_part =np .array (list (feats .keys ())),
    blocks =np .array (list (feats .values ()),dtype =object ),
    names =np .array (B .FEAT_NAMES ))
    tot =sum (len (v )for v in feats .values ())
    print (f"-> {OUT }  {len (feats )} part / {tot } candidate  {time .time ()-t0 :.0f}s")


if __name__ =="__main__":
    main ()
