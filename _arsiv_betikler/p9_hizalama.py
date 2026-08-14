# -*- coding: utf-8 -*-
"""P9 HIZALAMA OLCUMU (P6b with AYNI test kumesi, seed 202) -- gecenin kapanis count.

KOLLAR (all of them sizintisiz: test aileleri gate egitiminden cikarilmis, esikler BUYUK veride
aile-disi OOF with secilmis, test kumesine bakilmadan):
  A = MEVCUT  : geo-orient KAPALI (urunun bugunku hali) + rt2 gate
  B = HIZALI  : geo-orient OPEN (rt2 verisinin cikarildigi hal) + rt2 gate

Bulunan uyumsuzluk: rt2 verisi GEOMETRIK yonlendirmeyle cikarildi, but urun onu KAPALI
calistiriyor -> gate, egitimde gordugunden FARKLI candidate dagilimi skorluyor. Bu, this gece
+0.1273 odeyen "olculen != calisan" sinifinin aynisi.

Beklenen zincir: baseline 0.78 + P1 (+0.006/+0.013 two orneklemde) + full-corpus gate.
CIKAN SAYI NE ISE O.
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"low":0.895 ,"very":0.105 }
NPZ4 ="results/gate_regrow_data_rt2.npz"
NPZ5 ="results/gate_regrow_data_p5.npz"


def big_thr (npz ,test_fams ):
    """BUYUK veride aile-disi OOF with regime basina most iyi threshold (test ailelerine bakmadan)."""
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    d =np .load (npz ,allow_pickle =True )
    X ,y ,groups ,fams =d ["X"],d ["y"],d ["groups"],d ["fams"]
    keep =~np .isin (fams .astype (str ),list (test_fams ))
    X ,y ,groups ,fams =X [keep ],y [keep ],groups [keep ],fams [keep ]
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    gk =fams .astype (str )
    reg =np .array ([("very"if int (ngt .get (int (g ),0 ))>=8 else "low")for g in groups ])
    tot ={k :0 for k in ("low","very")}
    seen =set ()
    for g in groups :
        g =int (g )
        if g in seen :continue 
        seen .add (g )
        n =int (ngt .get (g ,0 ))
        if n >0 :tot ["very"if n >=8 else "low"]+=n 
    oof =np .zeros (len (y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,gk ):
        oof [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
    out ={}
    for k in ("low","very"):
        best =(0 ,0.40 )
        for thr in np .arange (0.20 ,0.71 ,0.05 ):
            m =reg ==k ;sel =(oof >=thr )&m 
            tp =int ((y [sel ]==1 ).sum ());fp =int (sel .sum ())-tp 
            p =tp /max (tp +fp ,1 );r =tp /max (tot [k ],1 )
            f1 =2 *p *r /max (p +r ,1e-9 )
            if f1 >best [0 ]:best =(f1 ,float (thr ))
        out [k ]=best [1 ]
    return out 


def main ():
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,wire_gate ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from json_dataset import family_key 
    from sklearn .ensemble import RandomForestClassifier 

    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (202 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),70 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),30 ,replace =False )])
    test_fams ={family_key (p [1 ])for p in sel }
    print (f"FINAL test {len (sel )} part (70 dusuk / 30 cok), {len (test_fams )} aile disarida",
    flush =True )

    # esikler: large veride, test ailelerine bakmadan (P8)
    thrA =big_thr (NPZ4 ,test_fams )
    thrB =big_thr (NPZ5 ,test_fams )
    print (f"esikler A(4uye) {thrA } | B/C(5uye) {thrB }",flush =True )

    # gate'ler: full data eksi test aileleri
    gates ={}
    for tag ,npz in (("A",NPZ4 ),("B",NPZ5 )):
        d =np .load (npz ,allow_pickle =True )
        keep =~np .isin (d ["fams"].astype (str ),list (test_fams ))
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (d ["X"][keep ],d ["y"][keep ])
        gates [tag ]=clf 
        print (f"  gate {tag }: {int (keep .sum ())} candidate ile egitildi",flush =True )
    pickle .dump ({"clf":gates ["B"],"feat_names":wire_gate .FEAT_NAMES },
    open ("results/wire_gate_p5_heldout.pkl","wb"))

    cache =[]
    for mfg ,pid ,jf ,stp ,n in sel :
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            pbs =[]
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pbs .append (np .asarray (pb ,float ))
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd /=np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 
            R ,t ,_ =align_frames (Vr ,Vj )
            cache .append (dict (V =V ,F =F ,pbs =pbs ,stp =stp ,n =n ,G =(G -t )@R ,Gd =Gd @R ,
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))))
        except Exception :
            continue 
        if len (cache )%25 ==0 :
            print (f"  test {len (cache )}",flush =True )
    print (f"test cache {len (cache )} part\n",flush =True )

    def members (r ,mode ):
        pbs =list (r ["pbs"])
        if mode >=5 :
            pbs .append (sum (r ["pbs"])/len (r ["pbs"]))
        if mode >=6 :
            pbs .append (np .median (np .stack (r ["pbs"]),axis =0 ))
        return pbs 

    def cands (r ,mode ):
        plist =members (r ,mode )
        der =lambda pr :[cp_openings .connection_points (
        r ["V"],r ["F"],pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        conn_promote =pr )for pb in plist ]
        base =robot_cp ._vote2 (der (0.0 ),min_votes =1 )
        is_hi =robot_cp ._highcp_router (r ["stp"],r ["V"],len (base ))
        return (robot_cp ._vote2 (der (0.25 ),min_votes =1 )if is_hi else base ),is_hi 

    def score (mode ,gtag ,thr ):
        clf =gates [gtag ]
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        for r in cache :
            cps ,is_hi =cands (r ,mode )
            keep =[]
            if cps :
                probs =sum (r ["pbs"])/len (r ["pbs"])
                X =wire_gate .feats_for (r ["V"],r ["F"],probs ,cps ,CE ,CT )
                sc =clf .predict_proba (X )[:,1 ]
                t =thr ["very"]if is_hi else thr ["low"]
                keep =[c for c ,s_ in zip (cps ,sc )if s_ >=t ]
            Q =np .array ([c ["point"]for c in keep ],float )if keep else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"];hit =np .zeros (len (G ),bool );used =set ()
            if len (Q )and len (G ):
                diff =Q [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
                    if d_ >r ["tol"]or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k ="very"if r ["n"]>=8 else "low"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (G )-tp 
        out ={}
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            out [k ]=2 *p *rc /max (p +rc ,1e-9 )
        out ["w"]=sum (W [k ]*out [k ]for k in W )
        return out 

    ARMS =(("A_MEVCUT_geoKAPALI",4 ,"A",thrA ,"0"),
    ("B_HIZALI_geoACIK",4 ,"A",thrA ,"1"))
    res ={}
    print (f"{'arm':<22}{'low':>9}{'very':>9}{'agirlikli':>11}{'difference':>10}")
    base =None 
    import importlib 
    for name ,mode ,gtag ,thr ,geo in ARMS :
        os .environ ["CP_GEO_ORIENT"]=geo 
        importlib .reload (cp_openings )
        r =score (mode ,gtag ,thr );res [name ]=r 
        if base is None :base =r ["w"]
        print (f"{name :<22}{r ['low']:>9.4f}{r ['very']:>9.4f}{r ['w']:>11.4f}{r ['w']-base :>+10.4f}",
        flush =True )
    dB =res ["B_P1_5uye"]["w"]-base 
    print (f"\nFINAL: urun {base :.4f} -> P1 {res ['B_P1_5uye']['w']:.4f}  ({dB :+.4f})")
    print ("DECISION (>= +0.005 and low-CP gerilemiyor): "
    +("URUNE GIRER"if dB >=0.005 and res ["B_P1_5uye"]["low"]>=res ["A_urun_4uye"]["low"]-0.005 
    else "GIRMEZ"))
    json .dump ({k :v for k ,v in res .items ()}|{"delta_B":dB ,
    "thrA":thrA ,"thrB":thrB },
    open ("results/p9_hizalama.json","w"),indent =1 )
    print ("receipt -> results/p9_hizalama.json")


if __name__ =="__main__":
    main ()
