# -*- coding: utf-8 -*-
"""PITSTOP-2: regime-ayrimli gate esigi, ICE-ICE (nested) secimle.

WHY ICE-ICE: skorlanan kumede threshold secmek, secim-testte sizintisidir. Bugun
very-CP'de correct yaptik (ayri tarama kumesi), here da yapiyoruz: threshold aile-disi
EGITIM katlarinda secilir, DISARIDA birakilan katta skorlanir.

KILL (olcumden ONCE yazildi):
  * low-CP F1 katkisi (ice-ice) < +0.02 -> OLU, urune girmez
  * VEYA very-CP'de -0.01'den extra loss -> REDDEDILIR

CIKTI: results/pitstop2_gate_nested.json (receipt)
ONBELLEK: results/pitstop2_cache/<pid>.npz  -- ikinci kosu bedava
"""
import os ,sys ,json ,glob 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CACHE ="results/pitstop2_cache"
GRID =(0.15 ,0.25 ,0.30 ,0.35 ,0.40 ,0.45 ,0.50 ,0.55 )
DEPLOYED ={"dusuk":0.35 ,"cok":0.25 }# this an urunde which is


def build_cache (n_low =40 ,n_high =24 ,seed =0 ):
    """Her part for V,F,mean-olasilik,candidate-CP'ler,GT'yi mesh cercevesinde saklar."""
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from json_dataset import family_key 

    os .makedirs (CACHE ,exist_ok =True )
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    cfg =json .load (open ("cp_config.json"))
    pp =cfg ["prediction_postproc"]
    dev ="cuda"if torch .cuda .is_available ()else "cpu"

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :
            continue 
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :
            continue 
        if n >0 :
            parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (seed )
    lo =[x for x in parts if x [4 ]<8 ]
    hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),min (n_low ,len (lo )),replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),min (n_high ,len (hi )),replace =False )])

    todo =[x for x in sel if not os .path .exists (f"{CACHE }/{x [1 ]}.npz")]
    print (f"secilen {len (sel )} part ({len (lo [:n_low ])} dusuk + {len (hi [:n_high ])} cok); "
    f"{len (todo )} tanesi cikarilacak, {len (sel )-len (todo )} onbellekte",flush =True )

    if todo :
        models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["current_product"]["checkpoints"]]
    for i ,(m ,pid ,jf ,stp ,n )in enumerate (todo ,1 ):
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            acc ,per =None ,[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pb =np .asarray (pb ,float )
                acc =pb if acc is None else acc +pb 
                # URUNDEKI turetme: promote only very-CP'de open
                per .append (cp_openings .connection_points (
                V ,F ,pb .argmax (-1 ),min_v =int (pp ["min_vertices"]),classes =(CE ,CT ),
                dedupe_mm =10.0 ,probs =pb ,vertex_conf =float (pp ["vertex_confidence_mask"]),
                ct_depth_min_mm =1.0 ,cluster_mm =float (pp ["cluster_mm"]),
                conn_promote =(0.25 if n >=8 else 0.0 )))
            cps =robot_cp ._vote2 (per ,min_votes =1 )
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            # TAM CP kaydi saklanir: wire_gate.feats_for area/insertion_depth_mm/_votes de okuyor.
            # Alt cluster saklamak 13 ozelligin 3'unu sessizce sifirliyor (measured: tablo tamamen kayiyor).
            import pickle 
            pickle .dump (cps ,open (f"{CACHE }/{pid }.cps.pkl","wb"))
            np .savez_compressed (
            f"{CACHE }/{pid }.npz",V =V ,F =F ,probs =acc /len (per ),
            G =(G -t )@R ,Gd =Gd @R ,n =n ,
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 )))),
            fam =family_key (pid ))
            print (f"  [{i }/{len (todo )}] {pid } n={n } candidate={len (cps )}",flush =True )
        except Exception as e :
            print (f"  [{i }/{len (todo )}] {pid } HATA {type (e ).__name__ }: {e }",flush =True )
    return [x [1 ]for x in sel ]


def load_cache (pids ):
    import pickle 
    out =[]
    for pid in pids :
        f =f"{CACHE }/{pid }.npz"
        if not (os .path .exists (f )and os .path .exists (f"{CACHE }/{pid }.cps.pkl")):
            continue 
        d =np .load (f ,allow_pickle =True )
        cps =pickle .load (open (f"{CACHE }/{pid }.cps.pkl","rb"))
        # DENETIM: uretim kodunun okudugu alanlar onbellekte HAYATTA MI?
        for c in cps :
            miss =[k for k in ("point","direction","area","insertion_depth_mm")if k not in c ]
            if miss :
                raise RuntimeError (f"{pid }: onbellekte eksik CP alani {miss } -- gate ozellikleri bozulur")
        out .append ({"pid":pid ,"V":d ["V"],"F":d ["F"],"probs":d ["probs"],"cps":cps ,
        "G":d ["G"],"Gd":d ["Gd"],"n":int (d ["n"]),"tol":float (d ["tol"]),
        "fam":str (d ["fam"]),"regime":"cok"if int (d ["n"])>=8 else "dusuk"})
    return out 


def _match (P ,G ,Gd ,tol ):
    """big_arbiter konvansiyonu: eksene dik distance + +-40mm axial pencere."""
    hit =np .zeros (len (G ),bool );used =set ()
    if len (P )and len (G ):
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        perp =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        perp =np .where (np .abs (al )<=40.0 ,perp ,np .inf )
        for d_ ,a_ ,b_ in sorted ((perp [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
            if d_ >tol or a_ in used or hit [b_ ]:
                continue 
            hit [b_ ]=True ;used .add (a_ )
    tp =int (hit .sum ())
    return tp ,len (P )-tp ,len (G )-tp 


def counts (rows ,thr ,gate_cache ):
    import wire_gate 
    T =Fp =Fn =0 
    for r in rows :
        key =(r ["pid"],thr )
        if key not in gate_cache :
            import copy 
            s_ =wire_gate .apply (r ["V"],r ["F"],r ["probs"],copy .deepcopy (r ["cps"]),3 ,1 ,
            threshold =thr ,top_n =None )
            gate_cache [key ]=np .array ([c ["point"]for c in s_ ],float )if s_ else np .zeros ((0 ,3 ))
        tp ,fp ,fn =_match (gate_cache [key ],r ["G"],r ["Gd"],r ["tol"])
        T +=tp ;Fp +=fp ;Fn +=fn 
    p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
    return T ,Fp ,Fn ,p ,rc ,2 *p *rc /max (p +rc ,1e-9 )


def nested (rows ,gate_cache ,k =5 ):
    """Aile-disi katlar: threshold EGITIM katlarinda secilir, DISARIDAKI katta skorlanir."""
    fams =sorted ({r ["fam"]for r in rows })
    if len (fams )<k :
        k =max (2 ,len (fams ))
    assign ={f :i %k for i ,f in enumerate (fams )}
    T =Fp =Fn =0 ;picks =[]
    for fold in range (k ):
        tr =[r for r in rows if assign [r ["fam"]]!=fold ]
        te =[r for r in rows if assign [r ["fam"]]==fold ]
        if not tr or not te :
            continue 
        best =max (GRID ,key =lambda t :counts (tr ,t ,gate_cache )[5 ])
        picks .append (best )
        a ,b ,c ,*_ =counts (te ,best ,gate_cache )
        T +=a ;Fp +=b ;Fn +=c 
    p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
    return 2 *p *rc /max (p +rc ,1e-9 ),picks ,p ,rc 


def main ():
    pids =build_cache ()
    rows =load_cache (pids )
    gc_ ={}
    print (f"\nyuklenen: {len (rows )} part "
    f"({sum (r ['regime']=='dusuk'for r in rows )} dusuk / "
    f"{sum (r ['regime']=='cok'for r in rows )} cok)\n",flush =True )

    rec ={"grid":list (GRID ),"deployed":DEPLOYED ,"regimes":{}}
    for reg in ("dusuk","cok"):
        sub =[r for r in rows if r ["regime"]==reg ]
        if not sub :
            continue 
        print (f"=== {reg .upper ()}-CP  ({len (sub )} part, {len ({r ['fam']for r in sub })} aile) ===")
        print (f"{'threshold':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>8}{'R':>8}{'F1':>8}")
        tbl ={}
        for thr in GRID :
            c =counts (sub ,thr ,gc_ );tbl [thr ]=c 
            mk ="  <- URUNDE"if abs (thr -DEPLOYED [reg ])<1e-9 else ""
            print (f"{thr :>7.2f}{c [0 ]:>5}{c [1 ]:>5}{c [2 ]:>5}{c [3 ]:>8.3f}{c [4 ]:>8.3f}{c [5 ]:>8.3f}{mk }")
        dep =tbl [DEPLOYED [reg ]][5 ]
        orc =max (tbl .values (),key =lambda c :c [5 ])
        nf1 ,picks ,np_ ,nr_ =nested (sub ,gc_ )
        print (f"\n  urundeki threshold ({DEPLOYED [reg ]:.2f})      F1 {dep :.4f}")
        print (f"  oracle (ayni kumede secim) F1 {orc [5 ]:.4f}   <- SIZINTILI, rapor edilmez")
        print (f"  ICE-ICE (aile-disi secim)  F1 {nf1 :.4f}  P {np_ :.3f} R {nr_ :.3f}  "
        f"secilenler {picks }")
        print (f"  DURUST KATKI: {nf1 -dep :+.4f}\n",flush =True )
        rec ["regimes"][reg ]={"n_parts":len (sub ),"table":{str (k ):list (v )for k ,v in tbl .items ()},
        "deployed_f1":dep ,"oracle_f1":orc [5 ],"nested_f1":nf1 ,
        "nested_picks":picks ,"honest_gain":nf1 -dep }

    if len (rec ["regimes"])==2 :
        w ={"dusuk":0.895 ,"cok":0.105 }
        dep_w =sum (w [k ]*v ["deployed_f1"]for k ,v in rec ["regimes"].items ())
        nes_w =sum (w [k ]*v ["nested_f1"]for k ,v in rec ["regimes"].items ())
        rec ["corpus_weighted"]={"deployed":dep_w ,"nested":nes_w ,"gain":nes_w -dep_w }
        print (f"KORPUS-AGIRLIKLI (%89.5 dusuk / %10.5 cok)")
        print (f"  urunde  {dep_w :.4f}   ->  ice-ice {nes_w :.4f}   ({nes_w -dep_w :+.4f})")
        kill =rec ["regimes"]["dusuk"]["honest_gain"]>=0.02 and rec ["regimes"]["cok"]["honest_gain"]>=-0.01 
        rec ["kill_criterion_passed"]=bool (kill )
        print (f"\nKAPI (dusuk >= +0.02 VE cok >= -0.01): {'GECTI'if kill else 'OLU'}")

    json .dump (rec ,open ("results/pitstop2_gate_nested.json","w"),indent =1 )
    print ("\nmakbuz: results/pitstop2_gate_nested.json")


if __name__ =="__main__":
    main ()
