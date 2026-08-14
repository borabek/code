# -*- coding: utf-8 -*-
"""ASKER LISTESI A1 + A2 -- single GPU kosusu, same inference paylasiliyor.

A1  ROUTER TAZELIK DENETIMI: router 1753-part, min_v=10 donemi union sayilariyla egitildi.
    min_v 4'e indi -> input dagilimi kaydi. Dogru girdiyle (promote'suz min_v=4 union count)
    still correct rota veriyor mu? N2b referansi: dogruluk 0.953, very-CP yakalama 0.875.
    -> dusmediyse DOGRULANDI (degisiklik absent); dustuyse rt2 dagilimiyla yeniden egitilir.

A2  min_v IZGARA KENARI: K4b izgarasi (4,6,10,15) 4'te durdu and 4 KENARDAYDI -- own
    "izgara kenari optimum whereas araligi genislet" kuralimin ihlali. 2 and 3 never olculmedi.
    TAVAN (onceden): low-CP candidate-R already ~0.95+ -> kazanc however very-CP'den, butcesi +0.033.
    Beklenen 0...+0.01. KILL: agirlikli < +0.01 -> urune girmez.

SIZINTI NOTU: dagitilan gate (wire_gate.pkl) this parcalari egitimde gordu -> SEVIYELER sismis.
A2 a A/B'dir: same gate, same parts, single difference min_v -> DELTA gecerli, seviye not.
"""
import os ,sys ,json ,copy 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"low":0.895 ,"very":0.105 }


def main ():
    import torch ,thesis_remesh ,cp_openings ,robot_cp ,wire_gate ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    cfg =json .load (open ("cp_config.json"));pp =cfg ["prediction_postproc"]
    VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    TL =float (cfg ["robot_wire_gate_threshold"]);TH =float (cfg ["robot_wire_gate_threshold_highcp"])
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]

    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if n >0 :parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (13 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),26 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),22 ,replace =False )])
    print (f"{len (sel )} part (26 dusuk / 22 cok)",flush =True )

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
            cache .append (dict (V =V ,F =F ,pbs =pbs ,stp =stp ,n =n ,
            G =(G -t )@R ,Gd =Gd @R ,
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))))
        except Exception :
            continue 
    print (f"cache {len (cache )} part\n",flush =True )

    def derive (r ,mv ,promote ,plist =None ):
        return [cp_openings .connection_points (
        r ["V"],r ["F"],pb .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        conn_promote =promote )for pb in (plist if plist is not None else r ["pbs"])]

        # ---------------- A1: router tazelik denetimi (correct input: promote'suz min_v=4 union) ----
    tp =fp =fn =tn =0 
    for r in cache :
        base =robot_cp ._vote2 (derive (r ,4 ,0.0 ),min_votes =1 )
        pred =robot_cp ._highcp_router (r ["stp"],r ["V"],len (base ))
        true =r ["n"]>=8 
        if true and pred :tp +=1 
        elif true and not pred :fn +=1 
        elif not true and pred :fp +=1 
        else :tn +=1 
    acc =(tp +tn )/max (len (cache ),1 )
    rec =tp /max (tp +fn ,1 )
    print ("=== A1 ROUTER (min_v=4, promote'suz union sayisiyla) ===")
    print (f"  dogruluk {acc :.3f}  (TP{tp } FP{fp } FN{fn } TN{tn }) | cok-CP yakalama {rec :.3f}")
    print (f"  N2b referansi (min_v=10 donemi): dogruluk 0.953, yakalama 0.875")
    a1_ok =acc >=0.90 and fp ==0 
    print (f"  SONUC: {'DOGRULANDI -- yeniden training gereksiz'if a1_ok else 'DUSTU -> yeniden egitilmeli'}\n",
    flush =True )

    # ---------------- A2: min_v 2/3/4, calisma mantigiyla (router + promote) -----------------
    def score (mv ,add_avg =False ):
        agg ={"low":[0 ,0 ,0 ],"very":[0 ,0 ,0 ]}
        v1_tp =n_tp =0 # eslesen GERCEKLERIN kaci single-oy (mekanizma kaniti)
        for r in cache :
            plist =r ["pbs"]+[sum (r ["pbs"])/len (r ["pbs"])]if add_avg else r ["pbs"]
            base =robot_cp ._vote2 (derive (r ,mv ,0.0 ,plist ),min_votes =1 )
            is_hi =robot_cp ._highcp_router (r ["stp"],r ["V"],len (base ))
            cps =robot_cp ._vote2 (derive (r ,mv ,0.25 ,plist ),min_votes =1 )if is_hi else base 
            # mekanizma sayaci: gate ONCESI adaylardan GT'ye oturanlarin oy dagilimi
            P0 =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            if len (P0 )and len (r ["G"]):
                diff =P0 [:,None ,:]-r ["G"][None ,:,:]
                al =(diff *r ["Gd"][None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*r ["Gd"][None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                mm_ =pe .min (axis =1 )<=r ["tol"]
                for i in np .where (mm_ )[0 ]:
                    n_tp +=1 
                    v1_tp +=int (cps [i ].get ("_votes",1 )<=1 )
            probs =sum (r ["pbs"])/len (r ["pbs"])
            s_ =wire_gate .apply (r ["V"],r ["F"],probs ,copy .deepcopy (cps ),CE ,CT ,
            threshold =(TH if is_hi else TL ),top_n =None )if cps else []
            Q =np .array ([c ["point"]for c in s_ ],float )if s_ else np .zeros ((0 ,3 ))
            G ,Gd =r ["G"],r ["Gd"];hit =np .zeros (len (G ),bool );used =set ()
            if len (Q )and len (G ):
                diff =Q [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
                    if d_ >r ["tol"]or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            t_ =int (hit .sum ());k ="very"if r ["n"]>=8 else "low"
            agg [k ][0 ]+=t_ ;agg [k ][1 ]+=len (Q )-t_ ;agg [k ][2 ]+=len (G )-t_ 
        out ={}
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            out [k ]=2 *p *rc /max (p +rc ,1e-9 )
        out ["w"]=sum (W [k ]*out [k ]for k in W )
        out ["v1_share"]=v1_tp /max (n_tp ,1 )
        return out 

    print ("=== A2 min_v izgara kenari (SEVIYE sismis, DELTA gecerli) ===")
    print (f"{'min_v':>7}{'low':>9}{'very':>9}{'agirlikli':>11}")
    res ={}
    for mv in (2 ,3 ,4 ):
        r =score (mv );res [mv ]=r 
        mk ="  <- urunde"if mv ==4 else ""
        print (f"{mv :>7}{r ['low']:>9.4f}{r ['very']:>9.4f}{r ['w']:>11.4f}{mk }",flush =True )
    best =max (res .items (),key =lambda kv :kv [1 ]["w"])
    d =best [1 ]["w"]-res [4 ]["w"]
    print (f"\nen iyi min_v {best [0 ]} -> fark {d :+.4f}")
    print (f"KAPI (>= +0.01): {'GECTI'if d >=0.01 and best [0 ]!=4 else 'OLU / degisiklik absent'}")
    # ---------------- A5: UZLASMA YUKSELTICI (5. uye = mean harita) ----------------------
    # K2: gate'in oldurdugu gercekler TEK-OY (votes AUC 0.86). Ortalama haritanin adaylari,
    # sinirda-real acikliklari 1 oydan 2 oya removes; single-model gurultusu ortalamada kaybolur
    # and oyu YUKSELMEZ. best_full/keig128'den farki: yabanci uye not, uyelerin own ortalamasi
    # -> uzlasmayi sulandirmaz, keskinlestirir. Mekanizma kaniti = single-oy real payinin dusmesi.
    print (chr (10 )+"=== A5 uzlasma yukseltici (4 uye vs 4+mean) ===")
    r4 =res [4 ]
    r5 =score (4 ,add_avg =True )
    print (f"{'arm':>16}{'low':>9}{'very':>9}{'agirlikli':>11}{'single-oy real':>15}")
    print (f"{'4 uye (urun)':>16}{r4 ['low']:>9.4f}{r4 ['very']:>9.4f}{r4 ['w']:>11.4f}"
    f"{100 *r4 ['v1_share']:>14.0f}%")
    print (f"{'4+ORT (5 uye)':>16}{r5 ['low']:>9.4f}{r5 ['very']:>9.4f}{r5 ['w']:>11.4f}"
    f"{100 *r5 ['v1_share']:>14.0f}%")
    d5 =r5 ["w"]-r4 ["w"]
    print (chr (10 )+f"  A5 fark: {d5 :+.4f}  (mekanizma: tek-oy gercek payi "
    f"%{100 *r4 ['v1_share']:.0f} -> %{100 *r5 ['v1_share']:.0f})")
    print (f"  NOT: gate 4-uyeli dagilimla egitildi; A5 kalici olacaksa gate YENIDEN uydurulmali."
    f" Bu measurement ILK SINYAL.")

    # ---------------- A6: models-arasi ANLASMAZLIK ozelligi (fast AUC) --------------------
    from k7_dip_metal import auc as _auc 
    from scipy .spatial import cKDTree 
    pos ,neg =[],[]
    for r in cache :
        cps =robot_cp ._vote2 (derive (r ,4 ,0.0 ),min_votes =1 )
        if not cps :continue 
        P0 =np .array ([c ["point"]for c in cps ],float )
        lab =np .zeros (len (P0 ),int )
        if len (r ["G"]):
            diff =P0 [:,None ,:]-r ["G"][None ,:,:]
            al =(diff *r ["Gd"][None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*r ["Gd"][None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            used =set ();hit =np .zeros (len (r ["G"]),bool )
            for dd ,a ,b in sorted ((pe [a ,b ],a ,b )for a in range (len (P0 ))for b in range (len (r ["G"]))):
                if dd >r ["tol"]or a in used or hit [b ]:continue 
                used .add (a );hit [b ]=True ;lab [a ]=1 
        tree =cKDTree (r ["V"])
        conn =[pb [:,CE ]+pb [:,CT ]for pb in r ["pbs"]]
        for i in range (len (P0 )):
            idx =tree .query_ball_point (P0 [i ],4.0 )
            if not idx :continue 
            vals =[float (np .mean (c [idx ]))for c in conn ]
            (pos if lab [i ]else neg ).append (float (np .std (vals )))
    a6 =_auc (np .array (pos ),np .array (neg ))if pos and neg else float ("nan")
    print (chr (10 )+"=== A6 models-arasi anlasmazlik (CE+CT std) ===")
    print (f"  TP medyan {np .median (pos ):.3f} | FP medyan {np .median (neg ):.3f} | AUC {a6 :.3f}")
    print (f"  KAPI (|AUC-0.5| anlamli ve votes'un otesinde olmali; referans votes 0.86)")

    json .dump ({"a1":{"acc":acc ,"recall":rec ,"ok":bool (a1_ok )},
    "a2":{str (k ):{kk :vv for kk ,vv in v .items ()}for k ,v in res .items ()},
    "a2_best":best [0 ],"a2_delta":d ,
    "a5":{"r4":r4 ,"r5":r5 ,"delta":d5 },
    "a6_auc":float (a6 )},
    open ("results/a1_a2_asker.json","w"),indent =1 )
    print (chr (10 )+"receipt -> results/a1_a2_asker.json")


if __name__ =="__main__":
    main ()
