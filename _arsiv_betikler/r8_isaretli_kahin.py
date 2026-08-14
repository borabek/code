# -*- coding: utf-8 -*-
"""R8: ISARETLI KAHIN -- two direction sozlugunun GERCEK tavani (and a hukmun iptali).

FINDING (2026-08-04): direction tarafindaki TUM kahin olcumleri `abs(v . Gd)` with secim yapiyordu.
Yani kahin, 180 derece TERS a yonu "mukemmel" sayip seciyordu; after sonuc ISARETLI
metrikle olculunce low cikiyordu. Bu, bugun direction SECICISININ ETIKETINDE duzelttigim
hatanin AYNISI -- but this sefer KAHINDE, i.e. TAVAN olcumunde.

IKI SONUCU INVALID KILAR:
  1. `c1_yon_sozlugu` KILL'i ("kahin 0.6263 < 0.70 -> Rota C KAPANIR", gecti=false).
     O sozlukte FIZIKSEL girisler vardi: brep (B-rep analitik silindir ekseni), channel
     (ic noktalarin PCA ekseni), normal (yerel surface normali), duzlem_A/B (mouth
     duzleminin ana/yan ekseni), uye_k. Isaretli olcumu 0.5748 = TABANDAN DUSUK cikmisti
     -- but secimi yapan kahin ISARETSIZDI. Sozlukte correct cevap YOK demek DEGIL.
  2. R1'in "%87.5 direction kahini / %94.1 birlesik" sayilari da unsigned kahinle uretildi;
     dagitilan selector ISARETLI metrikte olculuyor. Ikisi karsilastirilamaz, i.e.
     "selector kahininin %39'unu yakaladi" ifadesi DAYANAKSIZ.

BU BETIK: two sozlugu de AYNI parcalarda kurar and UC tavani ISARETLI kahinle olcer:
    D1   dagitilan dictionary (mevcut, obb+-, uzlasi, yuz_uzlasi, dik+-, yuzn+-)
    C1   fiziksel dictionary  (brep, channel, normal, duzlem_A/B and sign esleri, uye_k)
    BIRLESIK
Her biri for hem ISARETLI hem ISARETSIZ kahin raporlanir ki artefakt gorunur olsun.

CIKTI: results/_r8_sozluk.pkl (selector egitimi for) + results/r8_isaretli_kahin.json
DAGITIM YOK.
"""
import io ,json ,os ,pickle ,sys ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import thesis_remesh ,wire_gate 
from big_arbiter import eligible 
from infer_step_cp import step_to_mesh 
from sina_cluster import esle ,f1w 
from d1_direction_distribute import sozluk_kur ,_birim 
from c1_yon_sozlugu import yerel_yonler 
from r5_konum_teshis import yon_uygula 

ONB ="results/_r8_sozluk.pkl"
R4 ="results/_r4_sozluk.pkl"


def c1_sozlugu (V ,p ,d0 ,cyl ,ba ,uye_liste ):
    """C1'in FIZIKSEL girisleri + ISARET ESLERI. Yon uretmez, geometriden turetir."""
    S =[]
    if cyl is not None and ba is not None :
        try :
            ax =ba .axis_at (p ,d0 ,cyl )
            if ax is not None :
                b =_birim (ax [0 ]if isinstance (ax ,tuple )else ax )
                if b is not None :
                    S +=[("brep",b ),("brep",-b )]
        except Exception :
            pass 
    if V is not None and d0 is not None :
        for a ,v in (yerel_yonler (V ,p ,d0 )or {}).items ():
            b =_birim (v )
            if b is not None :
                S +=[(a ,b ),(a ,-b )]
    n_u =0 
    for lst in (uye_liste or []):
        for cc in (lst if isinstance (lst ,(list ,tuple ))else []):
            if not isinstance (cc ,dict ):
                continue 
            pt =cc .get ("point");dd =cc .get ("direction")
            if pt is None or dd is None :
                continue 
            if np .linalg .norm (np .asarray (pt ,float )-p )>5.0 :
                continue 
            b =_birim (dd )
            if b is not None :
                S .append ((f"uye{n_u }",b ));n_u +=1 
            if n_u >=4 :
                break 
    return S 


def main ():
    DER ,gate ,ek =T2 .yukle ()
    stp ={p :s for m ,p ,jf ,s in eligible ()}
    YS =wire_gate ._load ("results/yon_secici.pkl")
    try :
        import brep_axes as ba 
    except Exception :
        ba =None 
    R4D =pickle .load (open (R4 ,"rb"))

    if os .path .exists (ONB ):
        PARCA =pickle .load (open (ONB ,"rb"))
        print (f"onbellekten: {len (PARCA )} part",flush =True )
    else :
        PARCA ={};t0 =time .time ()
        for kk ,r in enumerate (DER ,1 ):
            if kk %20 ==0 :
                print (f"  {kk }/{len (DER )}  {time .time ()-t0 :.0f}s",flush =True )
                pickle .dump (PARCA ,open (ONB ,"wb"))
            d_ =R4D .get (r ["pid"])
            if d_ is None :
                continue 
            P =d_ ["P"].copy ()
            Pd =yon_uygula (d_ ,YS )# DAGITILAN zincirin ciktisi = baseline
            V =None 
            try :
                Vr ,Fr =step_to_mesh (stp [r ["pid"]])
                V ,_ =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .ascontiguousarray (V ,float )
            except Exception :
                pass 
            cyl =None ;YUZN =[]
            if ba is not None :
                try :
                    cyl =ba .cylinders (stp [r ["pid"]])
                except Exception :
                    pass 
                try :
                    pl =ba .planes (stp [r ["pid"]])
                    if pl is not None and len (pl ):
                        _n =np .asarray (pl [1 ],float );_rr =np .asarray (pl [2 ],float )
                        for j in np .argsort (-_rr )[:6 ]:
                            b =_birim (_n [j ])
                            if b is not None :
                                YUZN .append (b )
                except Exception :
                    pass 
            SD ,UZ =sozluk_kur (V ,P ,Pd ,YUZN =YUZN )
            SC =[c1_sozlugu (V ,P [i ],_birim (Pd [i ]),cyl ,ba ,r .get ("UYE"))
            for i in range (len (P ))]
            PARCA [r ["pid"]]={"P":P ,"Pd":Pd ,"X":d_ ["X"],"SD":SD ,"SC":SC ,"UZ":UZ }
        pickle .dump (PARCA ,open (ONB ,"wb"))
        print (f"-> {ONB }",flush =True )

    kaps ={}
    for d_ in PARCA .values ():
        for S in d_ ["SC"]:
            for t2 ,_v in S :
                kaps [t2 ]=kaps .get (t2 ,0 )+1 
    print (f"\nC1 giris kapsami: {dict (sorted (kaps .items (),key =lambda x :-x [1 ]))}")
    print (f"sozluk boyu: D1 medyan "
    f"{np .median ([len (S )for d_ in PARCA .values ()for S in d_ ['SD']]):.0f} | "
    f"C1 medyan {np .median ([len (S )for d_ in PARCA .values ()for S in d_ ['SC']]):.0f}")

    def oracle_ (hangi ,isaretli_secim ):
        """hangi: 'D1'|'C1'|'BIRLESIK'. isaretli_secim: True whereas v.Gd, False whereas |v.Gd|."""
        rob ,gg =[],[]
        for r in DER :
            d_ =PARCA .get (r ["pid"])
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="cok"if r ["n"]>=8 else "dusuk"
            if d_ is None :
                P =np .zeros ((0 ,3 ));Pn =np .zeros ((0 ,3 ))
            else :
                P =d_ ["P"];Pn =d_ ["Pd"].copy ()
                if len (P )and len (G ):
                    diff =P [:,None ,:]-G [None ,:,:]
                    al =(diff *Gd [None ,:,:]).sum (-1 )
                    pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                    pe =np .where (np .abs (al )>40 ,np .inf ,pe )
                    for i in range (len (P )):
                        if not np .isfinite (pe [i ]).any ():
                            continue 
                        b =int (np .argmin (pe [i ]))
                        L =[]
                        if hangi in ("D1","BIRLESIK"):
                            L +=list (d_ ["SD"][i ])
                        if hangi in ("C1","BIRLESIK"):
                            L +=list (d_ ["SC"][i ])
                        en ,ed =None ,None 
                        for _t ,v in L :
                            s =float (v @Gd [b ])
                            s =s if isaretli_secim else abs (s )
                            if en is None or s >en :
                                en ,ed =s ,v 
                        if ed is not None :
                            Pn [i ]=ed 
            rob .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True ))
            gg .append (r ["geo"])
        return rob ,gg 

    t0 ,gg =oracle_ ("D1",True )
    # baseline: no kahin absent
    rob0 =[]
    for r in DER :
        d_ =PARCA .get (r ["pid"])
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        rj ="cok"if r ["n"]>=8 else "dusuk"
        P =d_ ["P"]if d_ else np .zeros ((0 ,3 ));Pn =d_ ["Pd"]if d_ else np .zeros ((0 ,3 ))
        rob0 .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True ))
    b =f1w (rob0 )
    print (f"\nTABAN (dagitilan zincir, ISARETLI metrik): {b :.4f}")
    print (f"\n{'sozluk':<12}{'ISARETLI kahin':>16}{'ISARETSIZ kahin':>17}{'artefakt':>11}")
    S ={"baseline":b }
    for h in ("D1","C1","BIRLESIK"):
        ri ,_ =oracle_ (h ,True )
        ru ,_ =oracle_ (h ,False )
        S [h ]={"signed":f1w (ri ),"unsigned":f1w (ru )}
        print (f"{h :<12}{f1w (ri ):>16.4f}{f1w (ru ):>17.4f}{f1w (ru )-f1w (ri ):>+11.4f}")
    print ("\n('artefakt' = unsigned kahinin SISIRDIGI miktar; secim ters yonu de dogru sayiyor)")
    d1i =S ["D1"]["signed"];bri =S ["BIRLESIK"]["signed"]
    print (f"\nDAGITILAN selector {b :.4f}; D1 sozlugunun ISARETLI tavani {d1i :.4f} "
    f"-> yakalanan pay %{100 *(b -0.5818 )/max (d1i -0.5818 ,1e-9 ):.0f} "
    f"(0.5818 = selector oncesi)")
    print (f"C1'in KATKISI (birlesik - D1): {bri -d1i :+.4f}")
    verdict =("C1 SOZLUGU YENIDEN ACILIR -- fiziksel girisler ISARETLI tavani yukseltiyor"
    if bri -d1i >=0.01 else 
    "C1 sozlugu ISARETLI tavana anlamli katki VERMIYOR -- kill gecerli kalir")
    print (f"\nHUKUM: {verdict }")
    with io .open ("results/r8_isaretli_kahin.json","w",encoding ="utf-8")as f :
        json .dump ({"baseline":b ,"sozlukler":S ,"c1_katkisi":bri -d1i ,
        "kapsam":kaps ,"verdict":verdict },f ,indent =1 )
    print ("receipt -> results/r8_isaretli_kahin.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
