# -*- coding: utf-8 -*-
"""Y6: SIMETRI KANITI -- insan gozu GEREKTIRMEYEN, MODELDEN BAGIMSIZ GT-tamlik testi.

WHY BU: gorsel denetim two times konvansiyon sorununa carpti (manufacturer CP'si kanalin inside
and yonu iceri bakiyor; robotunki agizda and disari) and kullanici hakli as "bakarak
anlayamiyorum" dedi. Uzmanlik isteyen a yargiyi insana yuklemek instead of, NESNEL a criterion
kuruyorum.

FIKIR: klemens TEKRARLI a yapidir -- kutuplar sabit a ADIMLA dizilir. Ureticinin
LISTELEDIGI CP'lerden this adimi and yonu olcebiliriz. Eger robotun buldugu "wrong" point,
listelenmis a girisin TAM BIR KUTUP ADIMI otesindeyse, that point ureticinin own
oruntusunun a dugumundedir: i.e. orada AYNI cinsten a giris olmasi is required and manufacturer
onu listelememistir.

KRITIK: this test AGIN CIKTISINI HIC KULLANMAZ. Girdi only (a) ureticinin CP listesi and
(b) robotun buldugu noktanin KONUMU. Ne segmentasyon olasiligi, ne gate skoru, ne ozniteligi.
Yani "own FP'lerimi own modelimle correct ilan etme" dongusu YOK.

ADIM TAHMINI: parcanin GT noktalari arasindaki tum difference vektorleri; most sik gorulen kisa
vektor = kutup adimi. En few 3 GT is required; azsa part testten CIKAR (and orana katilmaz).

KILL/OKUMA: this a OLCUM duzeltmesidir. Cikan ratio, "313 FP'nin at least this kadari real
listelenmemis giristir" seklinde ALT SINIR as raporlanir -- oruntude olmayan a FP
de real may be, this test onu YAKALAMAZ.
"""
import io 
import json 
import os 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

TOL =1.5 # mm, orgu dugumune uzaklik toleransi
MIN_GT =3 


def adim_tahmini (G ):
    """GT noktalarindan KUTUP ADIMI vektorunu prediction et (most sik kisa difference vektoru)."""
    n =len (G )
    if n <MIN_GT :
        return None 
    D =[]
    for i in range (n ):
        for j in range (n ):
            if i ==j :
                continue 
            v =G [j ]-G [i ]
            L =np .linalg .norm (v )
            if 1.0 <L <60.0 :
                D .append (v )
    if len (D )<2 :
        return None 
    D =np .array (D )
    # most kisa difference uzunlugu civarindaki vektorler = komsu kutup adimi
    L =np .linalg .norm (D ,axis =1 )
    baseline =float (np .percentile (L ,10 ))
    m =np .abs (L -baseline )<=max (0.6 ,0.10 *baseline )
    if m .sum ()<2 :
        return None 
    A =D [m ]
    # sign birlestir (v and -v same step)
    ref =A [0 ]/(np .linalg .norm (A [0 ])+1e-9 )
    A =np .array ([v if float (v @ref )>=0 else -v for v in A ])
    return A .mean (0 )


def main ():
    import measure_set 

    with io .open ("results/fp_denetim.json",encoding ="utf-8")as f :
        FD =json .load (f )
    FP =FD ["hepsi"]
    DER ,_ =measure_set .cluster ("results/_der_tam.pkl")
    GT ={r ["pid"]:np .asarray (r ["G"],float )for r in DER }

    byp ={}
    for i ,f_ in enumerate (FP ):
        byp .setdefault (f_ ["pid"],[]).append (i )

    SONUC =[]
    atlanan_parca =0 
    for pid ,idxs in sorted (byp .items ()):
        G =GT .get (pid ,np .zeros ((0 ,3 )))
        s =adim_tahmini (G )
        if s is None :
            atlanan_parca +=1 
            for i in idxs :
                SONUC .append ({"idx":i ,"pid":pid ,"mfg":FP [i ]["mfg"],
                "regime":FP [i ]["regime"],"test":"yok","k":None ,
                "deviation":None })
            continue 
        step_ =float (np .linalg .norm (s ))
        for i in idxs :
            p =np .array (FP [i ]["nokta"],float )
            en_iyi ,en_k =None ,None 
            for g in G :
                v =p -g 
                t =float (v @s )/(step_ **2 +1e-12 )# kac step otede
                k =int (round (t ))
                if k ==0 :
                    continue 
                sap =float (np .linalg .norm (v -k *s ))# orgu dugumune uzaklik
                if en_iyi is None or sap <en_iyi :
                    en_iyi ,en_k =sap ,k 
            SONUC .append ({"idx":i ,"pid":pid ,"mfg":FP [i ]["mfg"],
            "regime":FP [i ]["regime"],
            "test":"orgude"if (en_iyi is not None and en_iyi <=TOL )else "degil",
            "k":en_k ,"deviation":en_iyi ,"adim_mm":step_ })

    test_edilen =[x for x in SONUC if x ["test"]!="yok"]
    orgude =[x for x in test_edilen if x ["test"]=="orgude"]
    print (f"\n{len (SONUC )} FP | test edilebilen {len (test_edilen )} "
    f"({atlanan_parca } part GT<{MIN_GT } -> disarida)")
    if test_edilen :
        p_ =len (orgude )/len (test_edilen )
        z =1.96 ;nn =len (test_edilen )
        d_ =1 +z *z /nn 
        m_ =(p_ +z *z /(2 *nn ))/d_ 
        s_ =z *np .sqrt (p_ *(1 -p_ )/nn +z *z /(4 *nn *nn ))/d_ 
        lo ,hi =max (0 ,m_ -s_ ),min (1 ,m_ +s_ )
        print (f"\nORGU DUGUMUNDE: {len (orgude )}/{len (test_edilen )} = {p_ :.1%} "
        f"(Wilson %95 GA {lo :.1%}-{hi :.1%})")
        print (f"  tolerans {TOL } mm | medyan deviation "
        f"{np .median ([x ['deviation']for x in orgude ])if orgude else float ('nan'):.2f} mm")
        for m in sorted ({x ["mfg"]for x in test_edilen }):
            a =[x for x in test_edilen if x ["mfg"]==m ]
            b =[x for x in a if x ["test"]=="orgude"]
            print (f"  {m }: {len (b )}/{len (a )} = {len (b )/len (a ):.1%}")
        for rj in ("dusuk","cok"):
            a =[x for x in test_edilen if x ["regime"]==rj ]
            b =[x for x in a if x ["test"]=="orgude"]
            if a :
                print (f"  {rj }-CP: {len (b )}/{len (a )} = {len (b )/len (a ):.1%}")

                # --- DUZELTILMIS KESINLIK (ALT SINIR)
        tp ,fp =FD ["tp"],FD ["fp"]
        pay_lo ,pay ,pay_hi =lo ,p_ ,hi 
        for ad ,q in (("alt sinir",pay_lo ),("nokta",pay ),("ust",pay_hi )):
            k0 =tp /(tp +fp );k1 =(tp +fp *q )/(tp +fp )
            print (f"  {ad :<10} ratio {q :.1%} -> precision {k0 :.4f} -> {k1 :.4f}")
    with io .open ("results/y6_simetri.json","w",encoding ="utf-8")as f :
        json .dump ({"tol_mm":TOL ,"min_gt":MIN_GT ,"sonuc":SONUC ,
        "test_edilen":len (test_edilen ),"orgude":len (orgude ),
        "not":("Bu test AGIN CIKTISINI KULLANMAZ: girdi yalniz manufacturer CP listesi "
        "+ robot noktasinin KONUMU. Sonuc bir ALT SINIRDIR -- oruntude "
        "olmayan a FP de real may be, this test onu yakalamaz.")},
        f ,indent =1 )
    print ("receipt -> results/y6_simetri.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
