# -*- coding: utf-8 -*-
"""S3: YARIK yonu kurali TUM eslesmelerde -- DUZELEN mi BOZULAN mi?

S2 umut verdi: dik sinifin 107 noktasindan 74'unun 6mm inside a yarik adayi present and 26'sinin
yonu GT'ye <=10 derece (medyan 26 derece, mevcut 90 derece instead of).

AMA BU YETMEZ. Calisma aninda hangi CP'nin yonunun bozuk oldugunu BILEMEYIZ; rule HERKESE
uygulanir. Parca-ici axis uzlasisi full that is why olmustu: dik noktalarda kazaniyordu but
correct olanlari bozunca NET -110 cikmisti.

Bu yuzden here TUM eslesmelere uygulanip DUZELEN and BOZULAN birlikte sayiliyor.

KURALLAR (all of them calisma aninda uygulanabilir -- GT'ye BAKMAZ):
  R1: 6mm inside yarik varsa HER ZAMAN yarigin yonunu al
  R2: only mevcut direction with yarik yonu 45 dereceden extra AYRISIYORSA al
  R3: same, threshold 60 derece

KILL (onceden yazili): net kazanc, eslesmelerin at least %2'si must be (robot-hazirda ~+0.013).
Altindaysa uctan uca measurement bile yapilmaz -- this kosuda candidate duzeyi ALTI times yaniltti.
"""
import collections 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
YAKIN_MM =6.0 
KURAL ={"R1 always":0.0 ,"R2 difference>45":45.0 ,"R3 difference>60":60.0 }


def main ():
    import geo_g2_yarik 
    import wire_g_brep as B 
    import wire_gate 
    from big_arbiter import eligible 
    from sklearn .ensemble import RandomForestClassifier 

    stp_of ={p :s for m ,p ,jf ,s in eligible ()}
    with open ("results/_u4_der.pkl","rb")as f :
        DER =pickle .load (f )
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

    HEDEF =collections .defaultdict (list )
    total_ =0 
    for r in DER :
        if r ["X"]is None :
            continue 
        s =wire_gate .decision_score (m ,r ["X"])
        k =wire_gate .decision_mask (s )
        if not k .any ():
            continue 
        P =r ["P"][k ];Pd =r ["Pd"][k ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        diff =P [:,None ,:]-G [None ,:,:]
        al =(diff *Gd [None ,:,:]).sum (-1 )
        pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
        an =np .degrees (np .arccos (np .clip (np .abs (Pd @Gd .T ),0 ,1 )))
        pe_t =np .where (np .abs (al )>40 ,np .inf ,pe )
        tt =max (3.0 ,0.06 *float (r ["diag"]))
        used ,hit =set (),set ()
        for dd ,a_ ,b_ in sorted ((pe_t [a ,b ],a ,b )
        for a in range (len (P ))for b in range (len (G ))):
            if dd >tt or a_ in used or b_ in hit :
                continue 
            used .add (a_ );hit .add (b_ )
            total_ +=1 
            HEDEF [r ["pid"]].append ({"p":P [a_ ].copy (),"pd":Pd [a_ ].copy (),
            "gd":Gd [b_ ].copy (),"angle":float (an [a_ ,b_ ])})
    n_dik =sum (1 for v in HEDEF .values ()for h in v if h ["angle"]>45 )
    print (f"{total_ } eslesme | dik sinif {n_dik } | {len (HEDEF )} part",flush =True )

    VF ={}
    for cluster in ("dev","val"):
        cf =f"results/_probs_{cluster }.pkl"
        if not os .path .exists (cf )and cluster =="dev":
            cf ="results/_h_probs.pkl"
        with open (cf ,"rb")as f :
            for r in pickle .load (f ):
                pid =os .path .basename (r ["stp"]).split ("_")[1 ]
                if pid in HEDEF :
                    VF [pid ]=(np .ascontiguousarray (r ["V"],np .float64 ),
                    np .ascontiguousarray (r ["F"],np .int64 ))
    print (f"{len (VF )}/{len (HEDEF )} parcanin mesh'i var",flush =True )

    say ={k :{"duzelen":0 ,"bozulan":0 ,"degismeyen":0 }for k in KURAL }
    yakin =yariksiz =0 
    for i_ ,(pid ,hedefler )in enumerate (HEDEF .items (),1 ):
        if i_ %25 ==0 :
            print (f"  {i_ }/{len (HEDEF )}",flush =True )
        if pid not in VF :
            continue 
        V ,F =VF [pid ]
        try :
            slots =geo_g2_yarik .detect_slots (B .read_brep (stp_of .get (pid )),(V ,F ))
        except Exception :
            continue 
        if not slots :
            yariksiz +=len (hedefler )
            continue 
        SP =np .array ([x ["point"]for x in slots ],float )
        SD =np .array ([x ["direction"]for x in slots ],float )
        for h in hedefler :
            dd =np .linalg .norm (SP -h ["p"],axis =1 )
            j =int (np .argmin (dd ))
            if dd [j ]>YAKIN_MM :
                continue 
            yakin +=1 
            a_yeni =float (np .degrees (np .arccos (np .clip (abs (float (SD [j ]@h ["gd"])),0 ,1 ))))
            fark =float (np .degrees (np .arccos (np .clip (abs (float (SD [j ]@h ["pd"])),0 ,1 ))))
            eski_ok =h ["angle"]<=10.0 
            yeni_ok =a_yeni <=10.0 
            for ad ,threshold in KURAL .items ():
                if fark <=threshold :
                    say [ad ]["degismeyen"]+=1 
                elif yeni_ok and not eski_ok :
                    say [ad ]["duzelen"]+=1 
                elif eski_ok and not yeni_ok :
                    say [ad ]["bozulan"]+=1 
                else :
                    say [ad ]["degismeyen"]+=1 

    print ("\n=== RESULT ===")
    print (f"{total_ } eslesme | {yakin } tanesinin {YAKIN_MM }mm icinde yarik adayi var "
    f"({yakin /max (total_ ,1 ):.1%}) | yariksiz parcada {yariksiz }")
    print (f"\n{'rule':<16}{'duzelen':>9}{'bozulan':>9}{'NET':>7}{'gecis':>10}{'robot kest.':>13}")
    en_iyi ,en_iyi_ad =0 ,None 
    for ad ,v in say .items ():
        net =v ["duzelen"]-v ["bozulan"]
        print (f"{ad :<16}{v ['duzelen']:>9}{v ['bozulan']:>9}{net :>+7}"
        f"{net /max (total_ ,1 ):>+10.2%}{0.4500 +0.673 *net /max (total_ ,1 ):>13.4f}")
        if net >en_iyi :
            en_iyi ,en_iyi_ad =net ,ad 
    bar =0.02 
    gecti =en_iyi_ad is not None and en_iyi /max (total_ ,1 )>=bar 
    print (f"\nKILL: net kazanc >= %{bar *100 :.0f} -> "
    f"{('GECTI: '+en_iyi_ad )if gecti else 'GECMEDI'}")
    if not gecti and en_iyi_ad :
        print (f"  en iyi arm {en_iyi_ad }: {en_iyi /max (total_ ,1 ):+.2%} (bar %{bar *100 :.0f})")
    with open ("results/s3_yarik_net.json","w",encoding ="utf-8")as f :
        json .dump ({"total":total_ ,"dik_sinif":n_dik ,"yakin":yakin ,
        "kurallar":say ,"en_iyi":en_iyi_ad ,"net":en_iyi ,
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/s3_yarik_net.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
