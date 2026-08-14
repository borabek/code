# -*- coding: utf-8 -*-
"""M: KONUM and EKSEN for ORACLE tavanlari -- madde 9/10'u kurmadan before.

SORU: konum refiner (madde 9) and coklu-hipotez B-rep snap (madde 10) kurulursa robot-hazir
nereye cikabilir? Otopsi "kusursuz gate + kusursuz axis = 0.8136" olctu but KONUMU kusursuz
varsaymadi. Eger konum da kusursuzken ceiling 0.75'in altindaysa, that two madde 0.75'i ACAMAZ and
bunu KURMADAN before bilmek is required.

YONTEM -- ORACLE (GT kullanir, upper sinirdir, urun degildir):
  konum kusursuz : each candidate, esleseceği GT'nin EKSEN CIZGISINE dik as tasinir -> lateral 0
  axis kusursuz : candidate yonu GT yonune esitlenir
Esleme gevsek olcutle (tespit) is done; so "already bulunmus" adaylarin kalitesi olculur.
Bulunamamis GT'ler (recall kaybi) HICBIR oracle with kurtarilmaz -- ayri satirda gosterilir.

Bu a KALDIRAC DEGIL, a TAVAN olcumudur. Sonucu: madde 9/10'a yatirim is done mi.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"cok":0.105 }


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from big_arbiter import eligible 

    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"dusuk":float (cfg ["robot_wire_gate_threshold"]),
    "cok":float (cfg ["robot_wire_gate_threshold_highcp"])}
    cache =pickle .load (open ("results/_h_probs.pkl","rb"))
    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    X =d ["X"];y =d ["y"]
    pids =np .array ([str (x )for x in d ["pids"]])
    gk =json .load (open ("results/_geometry_keys.json"))

    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
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
    rng =np .random .RandomState (202 )
    lo =[x for x in parts if x [4 ]<8 ]
    hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),70 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),30 ,replace =False )])
    tg ={gk .get (p [1 ],"yok:"+p [1 ])for p in sel }
    Gg =np .array ([gk .get (p ,"yok:"+p )for p in pids ])
    keep =~np .isin (Gg ,list (tg ))
    cols =None 
    try :
        import pickle as _pk 
        cols =_pk .load (open ("results/wire_gate.pkl","rb")).get ("cols")
    except Exception :
        pass 
    Xk =X [keep ][:,cols ]if cols else X [keep ]
    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Xk ,y [keep ])
    print (f"gate {int (keep .sum ())} candidate x {Xk .shape [1 ]} ozellik | esikler {THR }",flush =True )

    # URUN hatti with candidate uret (parite: robot_cp'nin own fonksiyonlari)
    DER =[]
    for r in cache :
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
        der =[cp_openings .connection_points (
        V ,F ,pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r ["stp"])for pb in plist ]
        base =robot_cp ._vote2 (der ,min_votes =1 )
        is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
        if is_hi :
            der2 =[cp_openings .connection_points (
            V ,F ,pb .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =pb ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
            conn_promote =0.25 ,step_path =r ["stp"])for pb in plist ]
            cps =robot_cp ._vote2 (der2 ,min_votes =1 )
        else :
            cps =base 
        kept =[]
        if cps :
            probs =sum (plist )/len (plist )
            Xc =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT )
            if cols :
                Xc =Xc [:,cols ]
            sc =clf .predict_proba (Xc )[:,1 ]
            t_ =THR ["cok"]if is_hi else THR ["dusuk"]
            kept =[c for c ,s_ in zip (cps ,sc )if s_ >=t_ ]
        DER .append (dict (
        P =np .array ([c ["point"]for c in kept ],float )if kept else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in kept ],float )if kept else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
    print (f"{len (DER )} part | gate sonrasi candidate "
    f"{sum (len (r ['P'])for r in DER )}\n",flush =True )

    def score (fix_pos =False ,fix_axis =False ,tol =2.0 ,am =10.0 ,pct =False ):
        agg ={"dusuk":[0 ,0 ,0 ],"cok":[0 ,0 ,0 ]}
        for r in DER :
            P =r ["P"].copy ();Pd =r ["Pd"].copy ()
            G ,Gd =r ["G"],r ["Gd"]
            if len (P )and len (G ):
            # ORACLE: before GEVSEK olcutle esle, after that esin GT'sine according to duzelt
                diff =P [:,None ,:]-G [None ,:,:]
                al0 =(diff *Gd [None ,:,:]).sum (-1 )
                pe0 =np .linalg .norm (diff -al0 [...,None ]*Gd [None ,:,:],axis =-1 )
                t0 =max (3.0 ,0.06 *r ["diag"])
                pe0m =np .where (np .abs (al0 )<=40.0 ,pe0 ,np .inf )
                pair ={}
                us ,ug =set (),set ()
                for d_ ,a_ ,b_ in sorted ((pe0m [a ,b ],a ,b )
                for a in range (len (P ))for b in range (len (G ))):
                    if d_ >t0 or a_ in us or b_ in ug :
                        continue 
                    us .add (a_ );ug .add (b_ );pair [a_ ]=b_ 
                for a_ ,b_ in pair .items ():
                    if fix_pos :# GT axis cizgisine DIK as tasi -> lateral error 0
                        rel =P [a_ ]-G [b_ ]
                        P [a_ ]=G [b_ ]+(rel @Gd [b_ ])*Gd [b_ ]
                    if fix_axis :
                        Pd [a_ ]=Gd [b_ ]
            hit =np .zeros (len (G ),bool );used =set ()
            if len (P )and len (G ):
                diff =P [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                an =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))
                tt =max (3.0 ,0.06 *r ["diag"])if pct else tol 
                pe =np .where ((np .abs (al )>40 )|(an >am ),np .inf ,pe )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )
                for a in range (len (P ))for b in range (len (G ))):
                    if d_ >tt or a_ in used or hit [b_ ]:
                        continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ())
            k ="cok"if r ["n"]>=8 else "dusuk"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (P )-tp ;agg [k ][2 ]+=len (G )-tp 
        o ={}
        for k ,(T ,Fp ,Fn )in agg .items ():
            p_ =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            o [k ]=2 *p_ *rc /max (p_ +rc ,1e-9 )
        return sum (W [k ]*o [k ]for k in W ),o ["dusuk"],o ["cok"]

    print (f"{'senaryo':<38}{'robot-hazir':>13}{'dusuk':>9}{'cok':>9}")
    rows =[
    ("MEVCUT",False ,False ),
    ("+ EKSEN kusursuz (oracle)",False ,True ),
    ("+ KONUM kusursuz (oracle, madde 9/10)",True ,False ),
    ("+ IKISI de kusursuz",True ,True ),
    ]
    res ={}
    for lab ,fp ,fa in rows :
        w ,l_ ,h_ =score (fix_pos =fp ,fix_axis =fa )
        res [lab ]=w 
        print (f"{lab :<38}{w :>13.4f}{l_ :>9.4f}{h_ :>9.4f}",flush =True )
    det ,dl ,dh =score (tol =0.0 ,am =180.0 ,pct =True )
    print (f"\n{'TESPIT (ayni candidates, gevsek criterion)':<38}{det :>13.4f}{dl :>9.4f}{dh :>9.4f}")
    print ("  ^ this row candidate+gate kalitesinin tavani: no konum/axis duzeltmesi bunu asamaz")
    base =res ["MEVCUT"]
    print (f"\nMADDE 9/10'un ORACLE degeri : {res ['+ KONUM kusursuz (oracle, madde 9/10)']-base :+.4f}")
    print (f"MADDE 8'in (axis) oracle degeri: {res ['+ EKSEN kusursuz (oracle)']-base :+.4f}")
    print (f"IKISI birlikte                  : {res ['+ IKISI de kusursuz']-base :+.4f}")
    print (f"\nSONUC: konum+axis KUSURSUZ olsa bile robot-hazir tavani "
    f"{res ['+ IKISI de kusursuz']:.4f}")
    json .dump ({k :float (v )for k ,v in res .items ()}|{"tespit":float (det )},
    open ("results/m_konum_tavani.json","w"),indent =1 )
    print ("receipt -> results/m_konum_tavani.json")


if __name__ =="__main__":
    main ()
