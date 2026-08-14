# -*- coding: utf-8 -*-
"""F/2: UCTAN UCA manseti GEOMETRI gruplamasiyla yeniden olc.

F/1 gate duzeyinde olctu: family_key 0.7623 -> geometry_key 0.6475 (-0.1147). Cunku 1903
part only 765 geometri grubuna dusuyor and parcalarin %80.1'inin korpusta a IKIZI present.
"Test aileleri gate egitiminden cikarildi" derken gercekte only test PARCALARI cikariliyordu.

Bu betik same duzeltmeyi URUN metriginde yapar. Cikarim GEREKMEZ: results/_final_cache.pkl
gate-ONCESI adaylari and feature matrislerini tasiyor; degisen single sey gate'in hangi adaylarla
EGITILDIGI.
"""
import os ,sys ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"cok":0.105 }


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from json_dataset import family_key 
    from big_arbiter import eligible 

    cache =pickle .load (open ("results/_final_cache.pkl","rb"))
    d =np .load ("results/gate_regrow_data_rt2.npz",allow_pickle =True )
    X =d ["X"];y =d ["y"];groups =d ["groups"];fams =d ["fams"].astype (str )
    pids =np .array ([str (x )for x in d ["pids"]])
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    gk =json .load (open ("results/_geometry_keys.json"))

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
    test_pids =[p [1 ]for p in sel ]
    test_fams ={family_key (p )for p in test_pids }
    test_geo ={gk .get (p ,"yok:"+p )for p in test_pids }
    G_geo =np .array ([gk .get (p ,"yok:"+p )for p in pids ])

    keep_fam =~np .isin (fams ,list (test_fams ))
    keep_geo =~np .isin (G_geo ,list (test_geo ))
    print (f"gate havuzu {len (y )} candidate")
    print (f"  family_key ile disarida birakilan : {int ((~keep_fam ).sum ())}")
    print (f"  geometry_key ile disarida birakilan: {int ((~keep_geo ).sum ())} "
    f"(+{int ((~keep_geo ).sum ())-int ((~keep_fam ).sum ())} IKIZ)\n",flush =True )

    reg_all =np .array ([("cok"if int (ngt .get (int (g ),0 ))>=8 else "dusuk")for g in groups ])

    def build (keep ,gkey ):
        Xk ,yk ,gg =X [keep ],y [keep ],gkey [keep ]
        reg =reg_all [keep ]
        tot ={"dusuk":0 ,"cok":0 }
        for g in {int (g )for g in groups [keep ]}:
            n =int (ngt .get (g ,0 ))
            if n >0 :tot ["cok"if n >=8 else "dusuk"]+=n 
        o =np .zeros (len (yk ))
        for tr ,te in GroupKFold (n_splits =5 ).split (Xk ,yk ,gg ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (Xk [tr ],yk [tr ]).predict_proba (Xk [te ])[:,1 ]
        thr ={}
        for k in ("dusuk","cok"):
            b =(0.0 ,0.40 )
            for t in np .arange (0.20 ,0.71 ,0.05 ):
                m =reg ==k ;s_ =(o >=t )&m 
                tp =int ((yk [s_ ]==1 ).sum ());fp =int (s_ .sum ())-tp 
                p =tp /max (tp +fp ,1 );r =tp /max (tot [k ],1 )
                f =2 *p *r /max (p +r ,1e-9 )
                if f >b [0 ]:b =(f ,float (t ))
            thr [k ]=b [1 ]
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (Xk ,yk )
        return clf ,thr 

    def score (model ,TH ,tol ,am ,pct =False ):
        agg ={"dusuk":[0 ,0 ,0 ],"cok":[0 ,0 ,0 ]}
        for r in cache :
            Xc =r ["X13"]
            if Xc is None or not len (r ["P"]):
                Q =np .zeros ((0 ,3 ));Qd =np .zeros ((0 ,3 ))
            else :
                sc =model .predict_proba (Xc )[:,1 ]
                m =sc >=(TH ["cok"]if r ["is_hi"]else TH ["dusuk"])
                Q =r ["P"][m ];Qd =r ["Pd"][m ]
            Gt ,Gd =r ["G"],r ["Gd"]
            hit =np .zeros (len (Gt ),bool );used =set ()
            if len (Q )and len (Gt ):
                diff =Q [:,None ,:]-Gt [None ,:,:];al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                an =np .degrees (np .arccos (np .clip (np .abs (Qd @Gd .T ),0 ,1 )))
                tt =max (3.0 ,0.06 *r ["diag"])if pct else tol 
                pe =np .where ((np .abs (al )>40 )|(an >am ),np .inf ,pe )
                for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (Q ))for b in range (len (Gt ))):
                    if d_ >tt or a_ in used or hit [b_ ]:continue 
                    hit [b_ ]=True ;used .add (a_ )
            tp =int (hit .sum ());k ="cok"if r ["n"]>=8 else "dusuk"
            agg [k ][0 ]+=tp ;agg [k ][1 ]+=len (Q )-tp ;agg [k ][2 ]+=len (Gt )-tp 
        o ={};TP =FP =FN =0 
        for k ,(T ,Fp ,Fn )in agg .items ():
            p =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            o [k ]=2 *p *rc /max (p +rc ,1e-9 );TP +=T ;FP +=Fp ;FN +=Fn 
        return sum (W [k ]*o [k ]for k in W ),TP /max (TP +FP ,1 ),TP /max (TP +FN ,1 )

    print (f"{'gate bolmesi':<28}{'tespit':>9}{'yanal2':>9}{'ROBOT':>9}{'kesin':>8}{'recall':>8}")
    res ={}
    for lab ,keep ,gkey in (("family_key (ESKI)",keep_fam ,fams ),
    ("geometry_key (GERCEK)",keep_geo ,G_geo )):
        clf ,thr =build (keep ,gkey )
        det =score (clf ,thr ,0 ,180 ,True )
        lat =score (clf ,thr ,2.0 ,180 )
        rob =score (clf ,thr ,2.0 ,10 )
        res [lab ]=dict (tespit =det [0 ],yanal2 =lat [0 ],robot =rob [0 ],
        precision =det [1 ],recall =det [2 ],esikler =thr )
        print (f"{lab :<28}{det [0 ]:>9.4f}{lat [0 ]:>9.4f}{rob [0 ]:>9.4f}{det [1 ]:>8.3f}{det [2 ]:>8.3f}",
        flush =True )
    a =res ["family_key (ESKI)"];b =res ["geometry_key (GERCEK)"]
    print (f"\nUCTAN UCA SIZINTI BEDELI: tespit {b ['tespit']-a ['tespit']:+.4f}  "
    f"robot {b ['robot']-a ['robot']:+.4f}")
    json .dump (res ,open ("results/f_manset_geometri.json","w"),indent =1 ,default =str )
    print ("receipt -> results/f_manset_geometri.json")


if __name__ =="__main__":
    main ()
