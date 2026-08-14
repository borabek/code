# -*- coding: utf-8 -*-
"""X1: IKI BOSLUGUN TAM AYRISTIRMASI -- to-do listesi bunun uzerine kurulacak.

TARGET: tespit 0.7584 -> 0.85 (+0.092) | robot-hazir 0.5893 -> 0.70 (+0.111)

SORULAR (all of them mevcut onbellekten, YENI CIKARIM YOK):
  A. TESPIT bosluğu: kahin gate with ceiling ne? Boslugun ne kadari FP-ATMA, ne kadari FN-KURTARMA?
  B. ROBOT bosluğu: TESPIT EDILMIS (eslesen) noktalarin kaci YANAL'dan, kaci EKSEN'den,
     kaci IKISINDEN birden dusuyor? -- 0.111 nereden gelecek, bunu this number soyler.
  C. ROBOT KAHINI: each eslesen point for UYE havuzundaki EN IYI direction secilebilseydi
     robot-hazir kac olurdu? (pick_member_direction'in tavani)
  D. Tespit tavani with robot tavani ARASINDAKI bagimlilik: gate mukemmel olsaydi robot kac?
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

W ={"low":0.895 ,"very":0.105 }


def f1 (tp ,fp ,fn ):
    p =tp /max (tp +fp ,1e-9 );r =tp /max (tp +fn ,1e-9 )
    return 2 *p *r /max (p +r ,1e-9 )


def agirlikli (det ):
    """regime -> (tp,fp,fn) toplami -> agirlikli F1."""
    A ={}
    for rj ,tp ,fp ,fn in det :
        a =A .setdefault (rj ,[0 ,0 ,0 ])
        a [0 ]+=tp ;a [1 ]+=fp ;a [2 ]+=fn 
    return sum (W [k ]*f1 (*v )for k ,v in A .items ()if k in W )


def main ():
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier 

    with io .open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    NPZ =cfg ["current_product"]["wire_gate"]["egitim_verisi"]
    DER ,rap =measure_set .cluster ("results/_der_tam.pkl")
    measure_set .rapor_bas (rap )
    gk =measure_set .geo_anahtarlari ();tg ={r ["geo"]for r in DER }

    zen =np .load (NPZ ,allow_pickle =True )
    Xt =np .hstack ([np .asarray (zen ["X22"],float ),np .asarray (zen ["XR"],float )])
    ytr =np .asarray (zen ["y"]);tp_ =np .array ([str (x )for x in zen ["pids"]])
    keep =~np .isin (np .array ([gk .get (p ,"absent:"+p )for p in tp_ ]),list (tg ))
    dag =wire_gate ._load (wire_gate .MODEL_PATH );DON =dag .get ("donusum")
    Z =np .zeros ((len (Xt ),Xt .shape [1 ]*2 ))
    for u in np .unique (tp_ ):
        i =np .where (tp_ ==u )[0 ]
        Z [i ]=wire_gate .within_part (Xt [i ],DON )
    gate ={"clf":RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Z [keep ],ytr [keep ]),
    "n_feat":Z .shape [1 ],"donusum":DON }

    D_urun ,D_kahin ,D_recall =[],[],[]
    yanal_hepsi ,aci_hepsi ,aci_kahin =[],[],[]
    R_urun ,R_kahin_yon ,R_kahin_gate =[],[],[]
    for r in DER :
        rj ="very"if r ["n"]>=8 else "low"
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        nG =len (G )
        if r ["X"]is None or r .get ("XR")is None or nG ==0 :
            D_urun .append ((rj ,0 ,0 ,nG ));D_kahin .append ((rj ,0 ,0 ,nG ))
            D_recall .append ((rj ,0 ,0 ,nG ));R_urun .append ((rj ,0 ,0 ,nG ))
            R_kahin_yon .append ((rj ,0 ,0 ,nG ));R_kahin_gate .append ((rj ,0 ,0 ,nG ))
            continue 
        X58 =np .hstack ([r ["X"],r ["XR"]])
        P =np .asarray (r ["P"],float );Pd =np .asarray (r ["Pd"],float )
        tol =max (3.0 ,0.06 *float (r ["diag"]))

        # --- ADAY <-> GT eslesmesi (lateral distance, +-40mm axial pencere)
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
        aci =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))

        def greedy (mask_ok ):
            up ,ug ,ciftler =set (),set (),[]
            for d_ ,a_ ,b_ in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))
            for b in range (nG )if mask_ok [a ,b ]):
                if not np .isfinite (d_ )or a_ in up or b_ in ug :
                    continue 
                up .add (a_ );ug .add (b_ );ciftler .append ((a_ ,b_ ))
            return ciftler 

        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X58 ))
        # URUNUN DUZELTME ZINCIRI (headline.py:208-216 with AYNI). Bunsuz olculen sey dagitilan
        # urun not, HAM candidate havuzudur -- first kosuda robot 0.4472 output, headline 0.5893.
        if k .any ()and cfg .get ("robot_pose_head",False ):
            _c =[{"point":P [i ],"direction":Pd [i ]}for i in np .where (k )[0 ]]
            _c =wire_gate .pose_correct (X58 [k ],_c )
            if cfg .get ("robot_aci_secici"):
                _c =wire_gate .angle_correct (X58 [k ],_c )
            if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                _c =wire_gate .pick_member_direction (X58 [k ],_c ,r ["UYE"])
            P =P .copy ();Pd =Pd .copy ()
            P [k ]=np .array ([x ["point"]for x in _c ],float )
            Pd [k ]=np .array ([x ["direction"]for x in _c ],float )
            diff =P [:,None ,:]-G [None ,:,:]
            al =(diff *Gd [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            aci =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))

            # --- A: TESPIT (urun / kahin gate / saf recall)
        m_tes =pe <=tol 
        m_urun =m_tes .copy ();m_urun [~k ]=False 
        c_urun =greedy (m_urun )
        D_urun .append ((rj ,len (c_urun ),int (k .sum ())-len (c_urun ),nG -len (c_urun )))
        c_hep =greedy (m_tes )# kahin gate: correct adaylari sec, gerisini at
        D_kahin .append ((rj ,len (c_hep ),0 ,nG -len (c_hep )))
        D_recall .append ((rj ,len (c_hep ),0 ,nG -len (c_hep )))

        # --- B: ROBOT (lateral<=2 VE angle<=10), URUNUN sectikleri ten
        m_rob =(pe <=2.0 )&(aci <=10.0 );m_rob [~k ]=False 
        c_rob =greedy (m_rob )
        R_urun .append ((rj ,len (c_rob ),int (k .sum ())-len (c_rob ),nG -len (c_rob )))

        # tespit edilmis ciftlerde HANGI criterion dusuyor?
        for a_ ,b_ in c_urun :
            yanal_hepsi .append (pe [a_ ,b_ ]);aci_hepsi .append (aci [a_ ,b_ ])
            uye =r .get ("UYE")
            en_iyi =aci [a_ ,b_ ]
            if uye :
                for lst in uye :
                    for c in (lst if isinstance (lst ,(list ,tuple ))else []):
                        d =c .get ("direction")if isinstance (c ,dict )else None 
                        if d is None :
                            continue 
                        d =np .asarray (d ,float )
                        pt =np .asarray (c .get ("point",P [a_ ]),float )
                        if np .linalg .norm (pt -P [a_ ])>5.0 :
                            continue 
                        v =np .degrees (np .arccos (np .clip (abs (float (d @Gd [b_ ])),0 ,1 )))
                        en_iyi =min (en_iyi ,v )
            aci_kahin .append (en_iyi )

            # --- C: YON KAHINI (lateral urunun, angle uye havuzunun EN IYISI)
        m_yk =(pe <=2.0 );m_yk [~k ]=False 
        cy =greedy (m_yk )
        iyi =0 
        for a_ ,b_ in cy :
            en_iyi =aci [a_ ,b_ ]
            uye =r .get ("UYE")
            if uye :
                for lst in uye :
                    for c in (lst if isinstance (lst ,(list ,tuple ))else []):
                        d =c .get ("direction")if isinstance (c ,dict )else None 
                        if d is None :
                            continue 
                        d =np .asarray (d ,float )
                        pt =np .asarray (c .get ("point",P [a_ ]),float )
                        if np .linalg .norm (pt -P [a_ ])>5.0 :
                            continue 
                        en_iyi =min (en_iyi ,np .degrees (np .arccos (
                        np .clip (abs (float (d @Gd [b_ ])),0 ,1 ))))
            iyi +=int (en_iyi <=10.0 )
        R_kahin_yon .append ((rj ,iyi ,int (k .sum ())-iyi ,nG -iyi ))

        # --- D: GATE KAHINI + robot olcutu
        m_gk =(pe <=2.0 )&(aci <=10.0 )
        cg =greedy (m_gk )
        R_kahin_gate .append ((rj ,len (cg ),0 ,nG -len (cg )))

    ya =np .array (yanal_hepsi );ac =np .array (aci_hepsi );ak =np .array (aci_kahin )
    print (f"\n{'='*66 }\nA. TESPIT BOSLUGU")
    print (f"  urun            {agirlikli (D_urun ):.4f}")
    print (f"  KAHIN GATE      {agirlikli (D_kahin ):.4f}   (mevcut adaylardan mukemmel secim)")
    tp_u =sum (x [1 ]for x in D_urun );fp_u =sum (x [2 ]for x in D_urun )
    fn_u =sum (x [3 ]for x in D_urun );tp_k =sum (x [1 ]for x in D_kahin )
    print (f"  urun: TP {tp_u } | FP {fp_u } | FN {fn_u }")
    print (f"  kahin TP {tp_k } -> gate'in KACIRDIGI dogru candidate: {tp_k -tp_u }")
    print (f"  hicbir adayin ulasamadigi GT (GERCEK recall duvari): {sum (x [3 ]for x in D_kahin )}")
    print (f"  -> boslugun {fp_u /(fp_u +(tp_k -tp_u ))*100 :.0f}%'i FP ATMA, "
    f"{(tp_k -tp_u )/(fp_u +(tp_k -tp_u ))*100 :.0f}%'i FN KURTARMA")

    print (f"\n{'='*66 }\nB. ROBOT BOSLUGU ({len (ya )} tespit edilmis cift)")
    print (f"  urun robot      {agirlikli (R_urun ):.4f}")
    y_ok =ya <=2.0 ;a_ok =ac <=10.0 
    print (f"  lateral <=2mm     {y_ok .mean ():6.1%}   (medyan {np .median (ya ):.2f}mm)")
    print (f"  aci   <=10deg   {a_ok .mean ():6.1%}   (medyan {np .median (ac ):.1f} deg)")
    print (f"  IKISI birden    {(y_ok &a_ok ).mean ():6.1%}")
    print (f"  yalniz YANAL dusuyor : {(~y_ok &a_ok ).mean ():6.1%}")
    print (f"  yalniz ACI  dusuyor  : {(y_ok &~a_ok ).mean ():6.1%}")
    print (f"  IKISI de dusuyor     : {(~y_ok &~a_ok ).mean ():6.1%}")
    print (f"\n  ACI KAHINI (uye havuzunun en iyisi): <=10deg {(ak <=10 ).mean ():6.1%} "
    f"(su an {a_ok .mean ():.1%}) -> uye havuzunda KALAN {(ak <=10 ).mean ()-a_ok .mean ():+.1%}")
    print (f"  lateral <=2 VE aci-kahin <=10: {(y_ok &(ak <=10 )).mean ():6.1%}")

    print (f"\n{'='*66 }\nC/D. ROBOT TAVANLARI")
    print (f"  urun                         {agirlikli (R_urun ):.4f}")
    print (f"  + mukemmel YON secimi        {agirlikli (R_kahin_yon ):.4f}")
    print (f"  + mukemmel GATE (direction dahil)  {agirlikli (R_kahin_gate ):.4f}")

    with io .open ("results/x1_bosluk.json","w",encoding ="utf-8")as f :
        json .dump ({"tespit_urun":agirlikli (D_urun ),"tespit_kahin":agirlikli (D_kahin ),
        "robot_urun":agirlikli (R_urun ),"robot_yon_kahin":agirlikli (R_kahin_yon ),
        "robot_gate_kahin":agirlikli (R_kahin_gate ),
        "yanal_ok":float (y_ok .mean ()),"aci_ok":float (a_ok .mean ()),
        "aci_kahin_ok":float ((ak <=10 ).mean ()),
        "yalniz_yanal":float ((~y_ok &a_ok ).mean ()),
        "yalniz_aci":float ((y_ok &~a_ok ).mean ()),
        "ikisi_de":float ((~y_ok &~a_ok ).mean ()),
        "gate_kacirdigi":int (tp_k -tp_u ),"urun_FP":int (fp_u ),
        "ulasilmaz_GT":int (sum (x [3 ]for x in D_kahin ))},f ,indent =1 )
    print ("\nmakbuz -> results/x1_bosluk.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
