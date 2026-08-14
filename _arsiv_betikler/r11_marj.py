# -*- coding: utf-8 -*-
"""R11: YON SECICISININ MARJI BAYAT MI? (robot kolu)

MARJ=0.05 degeri C4'te secildi: that kurulumda label ISARETSIZDI and selector measurement kumesinde
grup-capraz egitiliyordu. O gunden beri IKI sey degisti:
  * label ISARETLI became (training pozitif orani %31.4 -> %12.7) -> skor dagilimi kaydi
  * selector EGITIM KORPUSUNDA (154k row) egitilip DEPLOYED
Skor dagilimi kayinca ona bagli threshold bayatlar. Bu proje this cinsten bayatligi two times
yasadi ([[gate-refit-minv4]], [[topoloji-yaricapi-closed]]).

MARJIN ANLAMI: selector, `mevcut` yonden however alternatifin skoru mevcuttan MARJ up to
fazlaysa sapar. Buyuk marj = temkinli (few deviation), small marj = atak. Yon tarafinda
hasar asimetrisi konum tarafindaki like 1:20 DEGIL (wrong direction already wrong), but
yine de correct which is yonleri bozma riski present.

PROTOKOL: AYAR = dev + atanmamis (94 part), VERDICT = val (100 part). VAL'e before BAKILMAZ.
KILL (onceden yazildi): VAL fiziksel robotta +0.005 VE GA sifiri disliyor.
Tespit yapisal as does not change (tespit olcutu aciya bakmaz) -- yine de basilir.
"""
import io ,json ,os ,pickle ,sys 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import measure_set ,wire_gate 
from sina_cluster import esle ,f1w 
from d1_direction_distribute import satirla 

R4 ="results/_r4_sozluk.pkl"
MARJLAR =(0.00 ,0.02 ,0.05 ,0.10 ,0.15 ,0.20 ,0.30 )


def main ():
    DER ,gate ,ek =T2 .yukle ()
    PARCA =pickle .load (open (R4 ,"rb"))
    YS =wire_gate ._load ("results/yon_secici.pkl")
    assert YS is not None ,"results/yon_secici.pkl absent"
    print (f"dagitilan secicinin kayitli marji: {YS .get ('marj')}",flush =True )
    s3 =json .load (io .open ("results/split3.json",encoding ="utf-8"))
    val ={str (p )for p in s3 ["val"]["parts"]}

    # skorlari BIR KEZ hesapla; marj only DECISION kuralini changes
    SK ={}
    for r in DER :
        d_ =PARCA .get (r ["pid"])
        if d_ is None :
            continue 
        ax ,_ ,RJ =satirla (d_ ["X"],d_ ["SY"],d_ ["UZ"],d_ ["Pd"],None )
        if not ax :
            continue 
        pr =YS ["clf"].predict_proba (np .array (ax ,float ))[:,1 ]
        s ={}
        for n_ ,(i ,gi )in enumerate (RJ ):
            s .setdefault (i ,{})[gi ]=pr [n_ ]
        SK [r ["pid"]]=s 
    print (f"skor hazir: {len (SK )} part",flush =True )

    def kos (marj ,cluster ):
        rob ,det ,gg =[],[],[]
        for r in DER :
            if cluster is not None and r ["pid"]not in cluster :
                continue 
            d_ =PARCA .get (r ["pid"])
            G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
            rj ="very"if r ["n"]>=8 else "low"
            if d_ is None :
                P =np .zeros ((0 ,3 ));Pn =np .zeros ((0 ,3 ))
            else :
                P =d_ ["P"];Pn =d_ ["Pd"].copy ()
                s =SK .get (r ["pid"])
                if s is not None and marj is not None :
                    for i ,sc in s .items ():
                        g =max (sc ,key =sc .get )
                        if g !=0 and sc [g ]-sc .get (0 ,0.0 )>=marj :
                            Pn [i ]=d_ ["SY"][i ][g ][1 ]
            rob .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],2.0 ,10.0 ,False ,signed =True ))
            det .append ((rj ,)+esle (P ,Pn ,G ,Gd ,r ["diag"],0.0 ,180.0 ,True ))
            gg .append (r ["geo"])
        return rob ,det ,gg 

    ayar =None # dev+atanmamis = val DISI
    print (f"\n--- AYAR (val disi, {len ([r for r in DER if r ['pid']not in val ])} part) ---")
    print (f"{'marj':>7}{'robot(FIZ)':>13}{'tespit':>10}")
    AY ={}
    for m in MARJLAR :
        rb ,dt ,_ =kos (m ,{r ["pid"]for r in DER if r ["pid"]not in val })
        AY [m ]=f1w (rb )
        print (f"{m :>7.2f}{f1w (rb ):>13.4f}{f1w (dt ):>10.4f}")
    en =max (AY ,key =AY .get )
    print (f"\nAYAR kumesinde en iyi marj: {en :.2f} ({AY [en ]:.4f}); "
    f"dagitilan 0.05 ({AY [0.05 ]:.4f}) -> fark {AY [en ]-AY [0.05 ]:+.4f}")

    print (f"\n--- HUKUM (YALNIZ VAL, {len (val )} part; AYAR'da secilen TEK marj) ---")
    r0 ,d0 ,gg =kos (0.05 ,val )
    r1 ,d1 ,_ =kos (en ,val )
    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (r0 ,r1 )),gg ,fn ,n =3000 )
    print (f"{'marj':<16}{'robot(FIZ)':>13}{'tespit':>10}")
    print (f"{'0.05 (dagitilan)':<16}{f1w (r0 ):>13.4f}{f1w (d0 ):>10.4f}")
    print (f"{f'{en :.2f} (secilen)':<16}{f1w (r1 ):>13.4f}{f1w (d1 ):>10.4f}")
    dr =f1w (r1 )-f1w (r0 )
    print (f"\nVAL robot farki: {dr :+.4f}  GA[{lo :+.4f},{hi :+.4f}] "
    f"{'GERCEK'if (lo >0 or hi <0 )else 'noise'}")
    print (f"VAL tespit farki: {f1w (d1 )-f1w (d0 ):+.4f} (yapisal olarak 0 olmali)")
    hv0 ,hd0 ,_ =kos (0.05 ,None );hv1 ,hd1 ,_ =kos (en ,None )
    print (f"\n[bilgi] HAVUZLANMIS: robot {f1w (hv0 ):.4f} -> {f1w (hv1 ):.4f} "
    f"(HUKUM DEGIL: ayar kumesi havuzda)")
    gecti =dr >=0.005 and lo >0 
    print (f"\nKILL: VAL robot +0.005 VE GA>0 -> "
    f"{'GECTI -- marj guncellenir'if gecti else 'GECMEDI -- marj 0.05 KALIR'}")
    with io .open ("results/r11_marj.json","w",encoding ="utf-8")as f :
        json .dump ({"ayar":{str (k ):v for k ,v in AY .items ()},"secilen":en ,
        "val_mevcut":f1w (r0 ),"val_yeni":f1w (r1 ),"val_d":dr ,
        "ga":[lo ,hi ],"val_tespit_farki":f1w (d1 )-f1w (d0 ),
        "havuz_mevcut":f1w (hv0 ),"havuz_yeni":f1w (hv1 ),
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/r11_marj.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
