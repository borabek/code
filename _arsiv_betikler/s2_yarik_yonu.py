# -*- coding: utf-8 -*-
"""S2: DIK SINIFIN yonunu YARIK dedektoru veriyor mu?

S1 kapatici a sey showed: last direction 45 derecenin ten saptiginda zincirin HER halkasi da
sapmis (tez normali dahil, all of them %0). Yani correction, mevcut asamalari yeniden siralamakla
olmaz -- YENI BIR OZELLIK tespit etmek is required: telin girdigi YARIGIN kendisi.

geo_g2_yarik `detect_slots` full bunu ariyor: KARSILIKLI IKI DUZLEM arasindaki 1-6mm acikliklar
(yay-kelepceli / push-in terminal girisleri). Silindir arayan dedektor onlari GOREMEZ -- G1
otopsisi kacan GT'lerin %93.3'unun no silindire yakin olmadigini olcmustu.

SORU (single and net): dik sinifa giren each CP for, YAKININDA yonu GT with uyusan a yarik adayi
VAR MI? Varsa correction somut (that CP'nin yonunu yariktan al). Yoksa this sinif mevcut araclarla
cozulemez and bunu DURUSTCE raporlarim.

NOTE: yarik adayinin yonu de tasarim geregi koordinat eksenine hizali (`out[la] = sgn`).
Yani egik agizlari da cozmez; but DIK sinifta sorun egiklik not, YANLIS EKSEN secimi --
onu cozebilir.
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


def main ():
    import geo_g2_yarik 
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

    # 1) DIK sinifi topla (part basina)
    HEDEF =collections .defaultdict (list )
    total_ =0 
    for r in DER :
        if r ["X"]is None :
            continue 
        s =wire_gate .decision_score (m ,r ["X"]);k =wire_gate .decision_mask (s )
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
            if an [a_ ,b_ ]>45.0 :
                HEDEF [r ["pid"]].append ({"p":P [a_ ].copy (),"gd":Gd [b_ ].copy (),
                "angle":float (an [a_ ,b_ ]),"V":r .get ("V")})
    n_dik =sum (len (v )for v in HEDEF .values ())
    print (f"{total_ } eslesme | DIK sinif {n_dik } nokta, {len (HEDEF )} parcada",flush =True )

    # 2) O parcalarda yarik adaylari uret and yonleri karsilastir
    with open ("results/_u4_der.pkl","rb")as f :
        pass 
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
    print (f"{len (VF )}/{len (HEDEF )} parcanin mesh'i bulundu",flush =True )

    bulundu =uyan =0 
    aci_yeni ,aci_eski =[],[]
    hic_yarik_yok =0 
    for i_ ,(pid ,hedefler )in enumerate (HEDEF .items (),1 ):
        if i_ %10 ==0 :
            print (f"  {i_ }/{len (HEDEF )} part",flush =True )
        if pid not in VF :
            continue 
        V ,F =VF [pid ]
        try :
            import wire_g_brep as B 
            surf =B .read_brep (stp_of .get (pid ))
            slots =geo_g2_yarik .detect_slots (surf ,(V ,F ))
        except Exception as e :
            print (f"    {pid }: yarik dedektoru error {type (e ).__name__ }")
            continue 
        if not slots :
            hic_yarik_yok +=len (hedefler )
            continue 
        SP =np .array ([s ["point"]for s in slots ],float )
        SD =np .array ([s ["direction"]for s in slots ],float )
        for h in hedefler :
            dd =np .linalg .norm (SP -h ["p"],axis =1 )
            j =int (np .argmin (dd ))
            if dd [j ]>YAKIN_MM :
                continue 
            bulundu +=1 
            a =float (np .degrees (np .arccos (np .clip (abs (float (SD [j ]@h ["gd"])),0 ,1 ))))
            aci_yeni .append (a );aci_eski .append (h ["angle"])
            uyan +=int (a <=10.0 )

    print (f"\n=== SONUC ===")
    print (f"DIK sinif                        : {n_dik } nokta")
    print (f"  parcasinda HIC yarik bulunamadi: {hic_yarik_yok }")
    print (f"  {YAKIN_MM }mm icinde yarik adayi VAR   : {bulundu }")
    if bulundu :
        ay =np .array (aci_yeni );ae =np .array (aci_eski )
        print (f"  yarik yonu GT'ye <=10 deg      : {uyan } ({uyan /bulundu :.1%})")
        print (f"  yarik yonu medyan aci          : {np .median (ay ):.1f} deg "
        f"(mevcut direction: {np .median (ae ):.1f} deg)")
        print (f"\nKAZANC KESTIRIMI: {uyan } nokta duzelirse dik sinif "
        f"{n_dik } -> {n_dik -uyan }")
        print (f"  eslesmelerin {uyan /max (total_ ,1 ):.1%}'i kurtulur")
    else :
        print ("  -> yarik dedektoru this sinifa ULASAMIYOR")
    with open ("results/s2_yarik_yonu.json","w",encoding ="utf-8")as f :
        json .dump ({"dik_sinif":n_dik ,"yarik_var":bulundu ,"uyan":uyan ,
        "hic_yarik_yok":hic_yarik_yok ,"toplam_eslesme":total_ ,
        "ratio":(uyan /bulundu )if bulundu else None },f ,indent =1 )
    print ("\nmakbuz -> results/s2_yarik_yonu.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
