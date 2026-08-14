# -*- coding: utf-8 -*-
"""GEO-G1: FP/FN OTOPSISI -- KAPI 1.

SORU: GEO-BREP'in urettigi wrong pozitifler ACIKLANABILIR mi? (289 FP vs 69 TP idi)
Aciklanabilirse hedefli suzgec yazilabilir; dagitiksa F1 0.70 ulasilamaz and plan BURADA kapanir.

KAPI 1 (baslamadan yazildi): FP'lerin >= %60'i, <= 4 open geometrik imzali kumeye dusmeli.

IMZALAR (all of them olculebilir, ogrenilmis parametre absent):
  gecen      : deligin IKI ucu da serbest alana aciliyor (montaj/gecme deligi)
  eksen_dik  : ekseni, GT'lerin baskin eksenine ~dik  (vida/montaj yonu)
  cok_kucuk  : radius < 1.0 mm  (pim/percin)
  cok_buyuk  : radius > 3.0 mm  (body bosaltmasi/channel)
  sig        : channel derinligi < 3 mm (pah/yuva)
  only     : part inside same radius+eksende tekrari YOK (array not)
  diger      : hicbirine uymayan

Ayrica FN otopsisi: kacan GT'ler silindirik a yuzeyin yakininda MI? Degilse yarik/push-in
girisidir and recall tavani G2 (yarik adaylari) olmadan asilamaz.
"""
import os ,sys ,json ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/geo070_g1_otopsi.json"


def main (n_parts =60 ):
    import thesis_remesh ,geo_brep_cp ,tel_g_brep as B 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh 
    from big_arbiter import eligible 
    from cp_geometry import ray_hits 

    lock =json .load (open ("results/split_lock.json"));LOCK =set (lock ["locked_parts"])
    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        if 0 <n <8 :parts .append ((m ,p ,jf ,s ))
    rng =np .random .RandomState (0 )
    # G0: dev kumesi = first yari (deterministik), eval = ikinci yari (BU FAZDA DOKUNULMAZ)
    idx =rng .permutation (len (parts ))
    dev =[parts [i ]for i in idx [:n_parts ]]
    print (f"dev kumesi {len (dev )} part (eval yarisi bu fazda DOKUNULMADI)\n",flush =True )

    sig_count ={k :0 for k in ("gecen","eksen_dik","cok_kucuk","cok_buyuk","sig","only","diger")}
    n_fp =n_tp =0 
    fn_near_cyl =fn_total =0 
    rows =[]
    t0 =time .time ()
    for k ,(m ,pid ,jf ,stp )in enumerate (dev ,1 ):
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            mesh =(V ,F )
            cps =geo_brep_cp .detect (stp ,V ,F )
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            R ,t ,_ =align_frames (Vr ,Vj )
            Gm =(G -t )@R ;Gdm =Gd @R 
            P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 ))
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            hit =np .zeros (len (Gm ),bool );used =set ()
            if len (P ):
                diff =P [:,None ,:]-Gm [None ,:,:]
                al =(diff *Gdm [None ,:,:]).sum (-1 )
                perp =np .linalg .norm (diff -al [...,None ]*Gdm [None ,:,:],axis =-1 )
                perp =np .where (np .abs (al )<=40.0 ,perp ,np .inf )
                for d_ ,a_ ,b_ in sorted ((perp [a_ ,b_ ],a_ ,b_ )
                for a_ in range (len (P ))for b_ in range (len (Gm ))):
                    if d_ >tol or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            n_tp +=int (hit .sum ())

            # GT'lerin baskin ekseni (part ici cogunluk)
            dom =Gdm [0 ]if len (Gdm )else np .array ([1.0 ,0 ,0 ])
            if len (Gdm )>1 :
                sims =np .abs (Gdm @Gdm .T ).sum (1 )
                dom =Gdm [int (np .argmax (sims ))]

            for a_ ,c in enumerate (cps ):
                if a_ in used :
                    continue 
                n_fp +=1 
                p_ =np .asarray (c ["point"],float );d_ =np .asarray (c ["direction"],float )
                rad =float (c ["radius"]);dep =float (c ["depth"])
                # gecen mi: ters yonde de serbest alana ciksiyor mu
                far =float (np .linalg .norm (V .max (0 )-V .min (0 )))
                hin =ray_hits (mesh ,p_ ,d_ ,far )
                gecen =len (hin )<=1 
                dikey =abs (float (np .dot (d_ /(np .linalg .norm (d_ )+1e-9 ),dom )))<0.35 
                same =sum (1 for o in cps if o is not c and abs (o ["radius"]-rad )<0.2 
                and abs (float (np .dot (np .asarray (o ["direction"],float ),d_ )))>0.9 )
                if gecen :sig_count ["gecen"]+=1 
                elif dikey :sig_count ["eksen_dik"]+=1 
                elif rad <1.0 :sig_count ["cok_kucuk"]+=1 
                elif rad >3.0 :sig_count ["cok_buyuk"]+=1 
                elif dep <3.0 :sig_count ["sig"]+=1 
                elif same ==0 :sig_count ["only"]+=1 
                else :sig_count ["diger"]+=1 

                # FN otopsisi: kacan GT silindirik yuzeye yakin mi
            surf =B .read_brep (stp )
            cyl =[(s_ [3 ],s_ [4 ])for s_ in surf if s_ [0 ]=="Cylinder"]
            for b_ in range (len (Gm )):
                if hit [b_ ]:continue 
                fn_total +=1 
                gp =Gm [b_ ]
                near =any (np .linalg .norm (np .maximum (np .maximum (lo -gp ,gp -hi ),0 ))<=3.0 
                for lo ,hi in cyl )
                fn_near_cyl +=int (near )
        except Exception as e :
            print (f"  {pid }: {type (e ).__name__ } {str (e )[:45 ]}",flush =True )
        if k %20 ==0 :
            print (f"  {k }/{len (dev )}  {time .time ()-t0 :.0f}s",flush =True )

    print (f"\n=== FP OTOPSISI ({n_fp } FP / {n_tp } TP) ===")
    order =sorted (sig_count .items (),key =lambda x :-x [1 ])
    cum =0 
    for i ,(kk ,v )in enumerate (order ,1 ):
        pc =100 *v /max (n_fp ,1 )
        if i <=4 and kk !="diger":cum +=pc 
        print (f"  {kk :<12}{v :>6}  %{pc :5.1f}")
    print (f"\n  ILK 4 IMZANIN KAPSAMI (diger haric): %{cum :.1f}")
    print (f"  KAPI 1 (>=%60): {'GECTI'if cum >=60 else 'KALDI -> plan here KAPANIR'}")

    print (f"\n=== FN OTOPSISI ({fn_total } kacan GT) ===")
    pc_cyl =100 *fn_near_cyl /max (fn_total ,1 )
    print (f"  silindirik yuzeye 3mm'den yakin : {fn_near_cyl }/{fn_total }  %{pc_cyl :.1f}")
    print (f"  silindir YOK (yarik/push-in?)   : {fn_total -fn_near_cyl }  %{100 -pc_cyl :.1f}"
    f"   <- G2 yarik adaylarinin gerekcesi")

    json .dump ({"n_fp":n_fp ,"n_tp":n_tp ,"signatures":sig_count ,
    "top4_coverage_pct":cum ,"gate1_pass":cum >=60 ,
    "fn_total":fn_total ,"fn_near_cylinder":fn_near_cyl ,
    "fn_no_cylinder_pct":100 -pc_cyl ,"n_dev_parts":len (dev )},
    open (OUT ,"w"),indent =1 )
    print (f"\nmakbuz -> {OUT }")


if __name__ =="__main__":
    main (int (sys .argv [1 ])if len (sys .argv )>1 else 60 )
