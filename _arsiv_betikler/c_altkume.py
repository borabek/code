# -*- coding: utf-8 -*-
"""C'yi ALT KUMEDE olc: B-rep'in (silindir VE duzlem) SUSTUGU adaylarda ureni geciyor mu?

C global as kaybetti (val >15d %30.2 vs urun %15.8). Ama urunun ekseni HER YERDE iyi not:
adaylarin ~%15'inde B-rep no surface eslestiremiyor and orada mevcut yontem (normal-kovaryans
/ yuvarlama) zayif. Soru: C full O ALT KUMEDE more mi iyi? Oyleyse C "global kazanan" not
"B-rep'in sustugu places konusan uye" as urune girer.

KILL: lower kumede C'nin >15d hatasi mevcut yontemi GECMEZSE C tamamen duser.
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def main ():
    import torch ,diffusionnet as D 
    import brep_axes as B 
    from big_arbiter import eligible 

    ck =torch .load ("results/axis_net/axis_net_best.pt",map_location ="cpu",weights_only =False )
    cfg ,meta =ck ["config"],ck ["meta"]
    model ,_ =D .build_diffusionnet (cfg ,n_classes =3 )
    model .load_state_dict (ck ["model"])
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    model =model .to (dev ).eval ()
    print (f"C yuklendi (val medyan {ck .get ('val_median_deg',-1 ):.2f}d)",flush =True )

    cache =pickle .load (open ("results/_tolerans_cache6.pkl","rb"))
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
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
    assert len (sel )==len (cache )

    # NOTE: C'nin training seti PXC+WAGO+SIE, dogrulama WEI. Bu 100 parcanin a kismi C'nin
    # EGITIMINDE may be -> leakage. O yuzden only WEI parcalarinda (C for gorulmemis)
    # rapor edilir; PXC ayri gosterilir but karara girmez.
    opc =f"results/step_infer/ops_k{meta ['k_eig']}"
    rows =[]
    for (m ,pid ,jf ,stp ,n ),r in zip (sel ,cache ):
        try :
            cyl =B .cylinders (stp );pl =B .planes (stp )
        except Exception :
            continue 
        V =np .ascontiguousarray (r ["V"],np .float64 );F =np .ascontiguousarray (r ["F"],np .int64 )
        Q ,Qd ,G ,Gd =r ["Q"],r ["Qd"],r ["G"],r ["Gd"]
        if not len (Q )or not len (G ):
            continue 
        try :
            with torch .no_grad ():
                ops =D .precompute_operators (V ,F ,meta ["k_eig"],opc )
                ops ={k :(v .to (dev )if hasattr (v ,"to")else v )for k ,v in ops .items ()}
                x =D ._model_input (ops ,meta )
                c =0.5 *(x .max (0 ).values +x .min (0 ).values )
                sc =float ((x .max (0 ).values -x .min (0 ).values ).norm ())or 1.0 
                out =D ._forward (model ,ops ,(x -c )/sc ).cpu ().numpy ()
        except Exception :
            continue 
            # eslesme
        diff =Q [:,None ,:]-G [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        t =max (3.0 ,0.06 *r ["diag"]);used =set ();hit =np .zeros (len (G ),bool )
        for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (G ))):
            if d_ >t or abs (al [a_ ,b_ ])>40 or a_ in used or hit [b_ ]:continue 
            hit [b_ ]=True ;used .add (a_ )
            i ,j =a_ ,b_ 
            c_ok =B .axis_at (Q [i ],Qd [i ],cyl ,max_off_mm =5.0 ,max_turn_deg =60.0 )is not None 
            p_ok =(not c_ok )and (B .axis_from_planes (Q [i ],Qd [i ],pl ,max_dist_mm =6.0 ,
            min_faces =4 ,flat_ratio =0.20 ,
            max_turn_deg =45.0 )is not None )
            # C'nin tahmini: CP'nin 4mm yakinindaki koselerin mean yonu
            dd =np .linalg .norm (V -Q [i ],axis =1 )
            near =dd <=4.0 
            if near .sum ()<3 :
                near =dd <=np .percentile (dd ,0.5 )
            v =out [near ]
            v =v /(np .linalg .norm (v ,axis =1 ,keepdims =True )+1e-9 )
            ref =v [0 ]
            v =v *np .sign ((v @ref ))[:,None ]# unsigned mean
            cdir =v .mean (0 );cdir /=np .linalg .norm (cdir )+1e-9 
            ang_prod =np .degrees (np .arccos (min (1.0 ,abs (float (np .dot (Qd [i ],Gd [j ]))))))
            ang_c =np .degrees (np .arccos (min (1.0 ,abs (float (np .dot (cdir ,Gd [j ]))))))
            rows .append ((m ,int (c_ok or p_ok ),ang_prod ,ang_c ))
    A =np .array ([(r [1 ],r [2 ],r [3 ])for r in rows ],float )
    mfg =np .array ([r [0 ]for r in rows ])
    print (f"\neslesen cift: {len (A )}  (WEI={int ((mfg =='WEI').sum ())} = C icin GORULMEMIS)")

    def show (mask ,lab ):
        if mask .sum ()<5 :
            print (f"  {lab :<34} n={int (mask .sum ())} (az)");return 
        pr ,cc =A [mask ,1 ],A [mask ,2 ]
        print (f"  {lab :<34} n={int (mask .sum ()):>4}  URUN >15d %{100 *(pr >15 ).mean ():5.1f} "
        f"(med {np .median (pr ):5.2f}d)   C >15d %{100 *(cc >15 ).mean ():5.1f} (med {np .median (cc ):5.2f}d)")

    wei =mfg =="WEI"
    det =A [:,0 ]==1 
    print ("\n=== WEI (C for gorulmemis -- DECISION BUNA according to) ===")
    show (wei &~det ,"B-rep SUSUYOR (asil soru)")
    show (wei &det ,"B-rep konusuyor (C girmemeli)")
    show (wei ,"WEI tumu")
    print ("\n=== PXC/diger (C'nin EGITIMINDE -- sizintili, bilgi amacli) ===")
    show (~wei &~det ,"B-rep SUSUYOR")
    print ("\nKILL: WEI + B-rep susan lower kumede C'nin >15d hatasi URUNUNKINDEN small must be.")


if __name__ =="__main__":
    main ()
