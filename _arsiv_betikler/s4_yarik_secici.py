# -*- coding: utf-8 -*-
"""S4: yarik yonunu NE ZAMAN alacagini OGREN (elle threshold instead of selector).

S3'te elle yazilmis esikler yikildi: dedektor eslesmelerin %80.4'unde atesliyor and
"always al" 41 duzeltip 228 bozuyor (net -187). Kusur fikirde not, KARARDA: sabit a
angle esigi "this yarik correct feature mi" sorusunu ayirt edemiyor.

Ayni sorunu bugun COKUS YONLENDIRMESINDE cozmustuk: kurali herkese uygulamak instead of NE ZAMAN
uygulanacagini a sinyale bagladik. Burada da same yaklasim, but sinyal OGRENILIYOR.

TAVAN ONCE SABITLENDI: kahin (only faydali oldugunda al) +41 point = gecis +%4.7 ->
robot-hazir 0.4500 -> ~0.4816. Yani this dalin MUTLAK tavani 0.48. Secici bunun a kismini takes.

OZELLIKLER (all of them calisma aninda hesaplanabilir, GT'ye BAKMAZ):
    distance        CP with yarik agzi arasi (mm)
    aralik        yarigin acikligi (mm)
    depth      yarik kanalinin derinligi (mm)
    alan          cifti olusturan duzlemlerin kucugunun alani
    fark_aci      yarik yonu with MEVCUT direction arasindaki angle
    brep_r        CP'nin B-rep silindir yaricapi (0 = silindir eslesmedi)
    n_slot        parcadaki yarik adayi count (yogunluk -> guvenilirlik dusuklugu)
    yakinlik_orani most yakin/ikinci most yakin yarik mesafesi (tekil mi, kalabalik mi)

TARGET (egitimde): yarik yonu GT'ye MEVCUT yondan more yakin mi?
DEGERLENDIRME: geometri anahtarina according to GroupKFold, FOLD DISI tahminle net duzelme.

KILL (onceden yazili): net kazanc >= eslesmelerin %2'si (robot-hazirda ~+0.013). Kahin %4.7,
i.e. selector kahinin at least %43'unu yakalamali. Altindaysa this dal KAPANIR.
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
OZ =["distance","aralik","depth","alan","fark_aci","brep_r","n_slot","yakinlik_orani"]


def main ():
    import geo_g2_yarik 
    import tel_g_brep as B 
    import wire_gate 
    from big_arbiter import eligible 
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 

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
    rf0 =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (M [keep ],ytr [keep ])
    Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
    for u in np .unique (tr_pid ):
        i =np .where (tr_pid ==u )[0 ]
        Ztr [i ]=wire_gate .within_part (Xtr [i ],dag .get ("donusum_z","zskor"))
    m ={"clf":rf0 (Xtr ),"clf_z":rf0 (Ztr ),"n_feat":Xtr .shape [1 ],
    "donusum":dag .get ("donusum"),"donusum_z":dag .get ("donusum_z","zskor")}
    mx =[float (m ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
    for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
    m ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))

    HEDEF =collections .defaultdict (list )
    total_ =0 
    BREP_SUT =13 
    for r in DER :
        if r ["X"]is None :
            continue 
        s =wire_gate .decision_score (m ,r ["X"])
        k =wire_gate .decision_mask (s )
        if not k .any ():
            continue 
        idx =np .where (k )[0 ]
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
            "gd":Gd [b_ ].copy (),"angle":float (an [a_ ,b_ ]),
            "brep_r":float (r ["X"][idx [a_ ],BREP_SUT ])})
    print (f"{total_ } eslesme | {len (HEDEF )} part",flush =True )

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

    SAT =[]
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
            continue 
        SP =np .array ([x ["point"]for x in slots ],float )
        SD =np .array ([x ["direction"]for x in slots ],float )
        SR =np .array ([x ["radius"]for x in slots ],float )
        SDe =np .array ([x ["depth"]for x in slots ],float )
        SA =np .array ([x ["area"]for x in slots ],float )
        for h in hedefler :
            dd =np .linalg .norm (SP -h ["p"],axis =1 )
            o =np .argsort (dd )
            j =int (o [0 ])
            if dd [j ]>YAKIN_MM :
                continue 
            ikinci =float (dd [o [1 ]])if len (o )>1 else 999.0 
            a_yeni =float (np .degrees (np .arccos (np .clip (abs (float (SD [j ]@h ["gd"])),0 ,1 ))))
            fark =float (np .degrees (np .arccos (np .clip (abs (float (SD [j ]@h ["pd"])),0 ,1 ))))
            SAT .append ({"pid":pid ,"geo":gk .get (pid ,"absent:"+pid ),
            "distance":float (dd [j ]),"aralik":float (SR [j ]*2.0 ),
            "depth":float (SDe [j ]),"alan":float (SA [j ]),
            "fark_aci":fark ,"brep_r":h ["brep_r"],
            "n_slot":float (len (slots )),
            "yakinlik_orani":float (dd [j ]/max (ikinci ,1e-6 )),
            "eski_aci":h ["angle"],"yeni_aci":a_yeni })

    print (f"\n{len (SAT )} (CP, yarik) cifti toplandi",flush =True )
    X =np .array ([[s [k ]for k in OZ ]for s in SAT ],float )
    old_ =np .array ([s ["eski_aci"]for s in SAT ])
    new_ =np .array ([s ["yeni_aci"]for s in SAT ])
    grp =np .array ([s ["geo"]for s in SAT ])
    y =(new_ <old_ ).astype (int )# TARGET: yarik yonu more mi yakin?

    duz_kahin =int (((new_ <=10 )&(old_ >10 )).sum ())
    boz_kahin =int (((old_ <=10 )&(new_ >10 )).sum ())
    print (f"KAHIN (yalniz faydaliyken al): +{duz_kahin } nokta = gecis "
    f"+{duz_kahin /max (total_ ,1 ):.2%} -> robot ~{0.4500 +0.673 *duz_kahin /max (total_ ,1 ):.4f}")
    print (f"  (her zaman al: +{duz_kahin } / -{boz_kahin } = net {duz_kahin -boz_kahin :+d})")
    print (f"  hedef dengesi: {y .mean ():.1%} pozitif")

    oof =np .zeros (len (y ))
    for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,grp ):
        oof [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =5 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]

    print (f"\n{'threshold':>6}{'alinan':>9}{'duzelen':>9}{'bozulan':>9}{'NET':>7}"
    f"{'gecis':>9}{'robot kest.':>13}")
    en_iyi ,en_iyi_esik =-10 **9 ,None 
    for threshold in (0.5 ,0.6 ,0.7 ,0.8 ,0.9 ):
        al =oof >=threshold 
        duz =int ((al &(new_ <=10 )&(old_ >10 )).sum ())
        boz =int ((al &(old_ <=10 )&(new_ >10 )).sum ())
        net =duz -boz 
        print (f"{threshold :>6.2f}{int (al .sum ()):>9}{duz :>9}{boz :>9}{net :>+7}"
        f"{net /max (total_ ,1 ):>+9.2%}{0.4500 +0.673 *net /max (total_ ,1 ):>13.4f}")
        if net >en_iyi :
            en_iyi ,en_iyi_esik =net ,threshold 
    bar =0.02 
    gecti =en_iyi /max (total_ ,1 )>=bar 
    print (f"\nKILL: net >= %{bar *100 :.0f} -> "
    f"{('GECTI (threshold '+str (en_iyi_esik )+')')if gecti else 'GECMEDI'}")
    print (f"  en iyi {en_iyi :+d} nokta = {en_iyi /max (total_ ,1 ):+.2%} | "
    f"kahinin {en_iyi /max (duz_kahin ,1 ):.0%}'i yakalandi")
    with open ("results/s4_yarik_secici.json","w",encoding ="utf-8")as f :
        json .dump ({"n_cift":len (SAT ),"toplam_eslesme":total_ ,"kahin":duz_kahin ,
        "en_iyi_net":int (en_iyi ),"en_iyi_esik":en_iyi_esik ,
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/s4_yarik_secici.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
