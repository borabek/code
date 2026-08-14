# -*- coding: utf-8 -*-
"""T: IKI METRIGIN TAVAN MERDIVENI -- hedef 0.90 detection / 0.75 robot NEREDE kilitli?

Tavani yukseltmek for before NEREDE oldugunu and NEYIN kilitledigini bilmek is required. Bu betik
each two metrik for same merdiveni kurar and each basamakta KIM sinirliyor onu gosterir:

  0. SU AN                 : dagitilan urun (gate + goreli threshold + cokus yonlendirme)
  1. + KAHIN GATE          : candidates same, but GT with eslesenleri SECEBILSEYDIK (mukemmel karar)
  2. + KAHIN YON           : eslesenlerin acisi 0 olsaydi
  3. + KAHIN KONUM         : eslesenlerin lateral hatasi 0 olsaydi
  4. ADAY TAVANI           : a GT'nin YAKININDA never candidate present mi? (turetmenin recall tavani)

Merdivenin mantigi: 1. basamak GATE'in, 4. basamak TURETMENIN (network + cp_openings) tavanidir.
Ikisi arasindaki difference, gate'i mukemmellestirerek kazanilabilecek each seydir.

IMPORTANT: oracle sayilari ULASILABILIR TARGET DEGIL, UST SINIRDIR. Ayni data ten olculdugu
for iyimserdir. Ama a hedefin (0.90 / 0.75) hangi basamagin ten oldugunu SOYLER --
tavanin altindaysa calisilabilir, ustundeyse before ceiling yukseltilmelidir.
"""
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def _esle_sayilari (P ,Pd ,G ,Gd ,diag ,tol ,am ,pct ):
    """sina_cluster.esle with AYNI mantik, but esleme ciftlerini de returns."""
    hit =np .zeros (len (G ),bool );used =set ();ciftler =[]
    if len (P )and len (G ):
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        an =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))
        tt =max (3.0 ,0.06 *diag )if pct else tol 
        pe2 =np .where ((np .abs (al )>40 )|(an >am ),np .inf ,pe )
        for d_ ,a_ ,b_ in sorted ((pe2 [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if d_ >tt or a_ in used or hit [b_ ]:
                continue 
            hit [b_ ]=True ;used .add (a_ );ciftler .append ((a_ ,b_ ))
    tp =int (hit .sum ())
    return tp ,len (P )-tp ,len (G )-tp ,ciftler 


def main ():
    import wire_gate 
    from sina_cluster import f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
    with open ("results/_dev_val_cluster.json",encoding ="utf-8")as f :
        kume_of =json .load (f )
    d =np .load ("results/gate_regrow_data_topo.npz",allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"absent:"+p )for p in tr_pid ])
    Xtr =np .asarray (d ["X"],float );ytr =np .asarray (d ["y"])
    keep =~np .isin (tr_grp ,list ({gk .get (r ["pid"],"absent:"+r ["pid"])for r in DER }))
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    rf =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [keep ],ytr [keep ])
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],dag .get ("donusum_z","zskor"))
    m ={"clf":rf (Xtr ),"clf_z":rf (Ztr ),"n_feat":Xtr .shape [1 ],
    "donusum":dag .get ("donusum"),"donusum_z":dag .get ("donusum_z","zskor")}
    mx =[float (m ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
    for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
    m ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))
    print (f"gate kuruldu (sizintisiz) | cokus esigi {m ['esik_cokus']:.4f}\n",flush =True )

    BAS =["0 SU AN","1 +oracle GATE","2 +oracle YON","3 +oracle KONUM","4 ADAY TAVANI"]
    det ={b :[]for b in BAS };rob ={b :[]for b in BAS }
    for r in DER :
        rj ="very"if r ["n"]>=8 else "low"
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        Pt =r ["P"]if r ["X"]is not None else np .zeros ((0 ,3 ))
        Pdt =r ["Pd"]if r ["X"]is not None else np .zeros ((0 ,3 ))
        diag =float (r ["diag"])

        # 0 SU AN: urunun karar yolu
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if r ["X"]is not None :
            s =wire_gate .decision_score (m ,r ["X"])
            k =wire_gate .decision_mask (s )
            if k .any ():
                P ,Pd =Pt [k ],Pdt [k ]
        det ["0 SU AN"].append ((rj ,)+_esle_sayilari (P ,Pd ,G ,Gd ,diag ,0.0 ,180.0 ,True )[:3 ])
        rob ["0 SU AN"].append ((rj ,)+_esle_sayilari (P ,Pd ,G ,Gd ,diag ,2.0 ,10.0 ,False )[:3 ])

        # 1 KAHIN GATE: TUM candidates inside GT with eslesenleri sec (mukemmel karar)
        tp ,fp ,fn ,cift =_esle_sayilari (Pt ,Pdt ,G ,Gd ,diag ,0.0 ,180.0 ,True )
        sec =np .zeros (len (Pt ),bool )
        for a_ ,_ in cift :
            sec [a_ ]=True 
        Pk ,Pdk =Pt [sec ],Pdt [sec ]
        det ["1 +oracle GATE"].append ((rj ,)+_esle_sayilari (Pk ,Pdk ,G ,Gd ,diag ,0.0 ,180.0 ,True )[:3 ])
        rob ["1 +oracle GATE"].append ((rj ,)+_esle_sayilari (Pk ,Pdk ,G ,Gd ,diag ,2.0 ,10.0 ,False )[:3 ])

        # 2 +oracle YON: matched adaylarin yonu GT yonu is done
        Pd2 =Pdk .copy ()
        for i2 ,(a_ ,b_ )in enumerate (cift ):
            Pd2 [i2 ]=Gd [b_ ]
        det ["2 +oracle YON"].append ((rj ,)+_esle_sayilari (Pk ,Pd2 ,G ,Gd ,diag ,0.0 ,180.0 ,True )[:3 ])
        rob ["2 +oracle YON"].append ((rj ,)+_esle_sayilari (Pk ,Pd2 ,G ,Gd ,diag ,2.0 ,10.0 ,False )[:3 ])

        # 3 +oracle KONUM: matched adaylarin lateral sapmasi da sifirlanir (GT noktasina tasinir)
        P3 =Pk .copy ()
        for i3 ,(a_ ,b_ )in enumerate (cift ):
            P3 [i3 ]=G [b_ ]
        det ["3 +oracle KONUM"].append ((rj ,)+_esle_sayilari (P3 ,Pd2 ,G ,Gd ,diag ,0.0 ,180.0 ,True )[:3 ])
        rob ["3 +oracle KONUM"].append ((rj ,)+_esle_sayilari (P3 ,Pd2 ,G ,Gd ,diag ,2.0 ,10.0 ,False )[:3 ])

        # 4 ADAY TAVANI: each GT for a candidate VARSA sayilir (turetmenin recall tavani)
        n_es =len (cift )
        det ["4 ADAY TAVANI"].append ((rj ,n_es ,0 ,len (G )-n_es ))
        rob ["4 ADAY TAVANI"].append ((rj ,n_es ,0 ,len (G )-n_es ))

    print (f"{'basamak':<18}{'TESPIT F1':>11}{'ROBOT F1':>11}{'detection difference':>13}{'robot difference':>12}")
    onc_d =onc_r =None 
    OUT ={}
    for b in BAS :
        fd ,fr =f1w (det [b ]),f1w (rob [b ])
        print (f"{b :<18}{fd :>11.4f}{fr :>11.4f}"
        f"{(''if onc_d is None else f'{fd -onc_d :+.4f}'):>13}"
        f"{(''if onc_r is None else f'{fr -onc_r :+.4f}'):>12}")
        OUT [b ]={"detection":float (fd ),"robot":float (fr )}
        onc_d ,onc_r =fd ,fr 

    su_d =OUT ["0 SU AN"]["detection"];su_r =OUT ["0 SU AN"]["robot"]
    kg_d =OUT ["1 +oracle GATE"]["detection"];kg_r =OUT ["1 +oracle GATE"]["robot"]
    at_d =OUT ["4 ADAY TAVANI"]["detection"]
    print (f"\nHEDEFLERE GORE:")
    print (f"  TESPIT 0.90 : candidate tavani {at_d :.4f} -> "
    f"{'MUMKUN (ceiling ustunde)'if at_d >=0.90 else 'TAVANIN USTUNDE -- before TURETME recall'  'i artmali'}")
    print (f"                oracle GATE {kg_d :.4f} -> gate'i mukemmellestirmek "
    f"{'0.90 for YETER'if kg_d >=0.90 else 'YETMEZ'}")
    print (f"  ROBOT  0.75 : oracle direction+konum {OUT ['3 +oracle KONUM']['robot']:.4f} -> "
    f"{'MUMKUN'if OUT ['3 +oracle KONUM']['robot']>=0.75 else 'TAVANIN USTUNDE'}")
    print (f"\nNEREDE KILITLI:")
    print (f"  gate'i mukemmellestirme kazanci : detection {kg_d -su_d :+.4f} | robot {kg_r -su_r :+.4f}")
    print (f"  direction+konumu mukemmellestirme     : robot {OUT ['3 +oracle KONUM']['robot']-kg_r :+.4f}")
    print (f"  turetme recall tavani           : detection {at_d :.4f} (bunun ustune CIKILAMAZ)")
    with open ("results/t_ceiling.json","w",encoding ="utf-8")as f :
        json .dump (OUT ,f ,indent =1 )
    print ("\nmakbuz -> results/t_ceiling.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
