# -*- coding: utf-8 -*-
"""P6 IKI KADEMELI: birinci gecisin tohumlarindan PERIYODIK YAPI ozniteligi.

RATIONALE. D6-ici LOMO'da measured: direction secimi cozuldu (kahin farki +0.0057) but
recall 0.177 / pool tavani 0.525. Aday-basina bilgi tukendi. Kullanilmamis bilgi
PARCA DUZEYINDE and measured: D6'da >=6 CP'li 1096 parcada GT'lerin **%90.8'i**
parcanin most sik OTELEME VEKTORUYLE baska a GT'ye ulasiyor.

KADEMELER
  1. P6 ortak siralayici (92 column) -> skor
  2. high skorlu secim = TOHUM -> `lattice` oznitelikleri (8 column)
  3. ikinci siralayici (100 column) -> nihai skor -> secim

SIZINTIYA KARSI IKI ONLEM
  * Tohumlar HER ZAMAN tahminden gelir, GT'den ASLA.
  * Egitim korpusunda birinci kademe skorlari MARKA-KATLI (out-of-fold) uretilir.
    Aksi halde tohumlar own training verisinde asiri iyi becomes, ikinci kademe
    gercekte olmayan a seed kalitesine according to ogrenir and sinavda coker.

AYAR VE OLCUM AYRIMI
  * Butun rule/threshold secimi `full` korpusunun MARKA KATLARINDA is done.
  * D6 (SUPU/UPUN/MOR/NIT/UTL/S+S/SE/ONV) TEMIZ OKUMADIR -- ayar for
    KULLANILMAZ.
  * D7 sinavdir and this betik ONA HIC BAKMAZ.
"""
import collections 
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 
from sklearn .ensemble import HistGradientBoostingClassifier 

import makbuz_hash 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
os .environ ["WG_ZENGIN"]="1"
os .environ .setdefault ("P6_DIZIN","results/_p6_oz_u25")
sys .path .insert (0 ,".")
import lattice # noqa: E402
import canonical_d7 as K # noqa: E402
import p6_decision # noqa: E402
import order_stamp # noqa: E402
import product_genis # noqa: E402
import direction_bank as YB # noqa: E402
from run_p6_ortak import yukle # noqa: E402
from sina_cluster import match_hungarian # noqa: E402

AB =p6_decision .AB 
C0 =AB 
# Esik izgarasi YUKARI open tutulur. Mesh havuzu acilinca darbogaz recall'dan
# KESINLIGE gecti (D6 ten okumasi: recall 0.285 -> 0.523 but precision
# 0.576 -> 0.322) and secilen rule izgaranin most upper degeri output. Sinirda kalan
# a optimum, bulunmamis optimum demektir.
KURALLAR =([("mutlak",e )for e in (0.10 ,0.20 ,0.30 ,0.40 ,0.50 ,0.60 ,0.70 ,
0.80 ,0.85 ,0.90 ,0.95 ,0.97 ,0.99 )]+
[("goreli",o ,t )for o in (0.30 ,0.50 ,0.70 ,0.85 ,0.95 )
for t in (0.05 ,0.20 ,0.40 ,0.60 )])
NMSLER =tuple (float (x )for x in 
os .environ .get ("P6_NMSLER","2.5,3.5,5.0").split (","))
# RULE KAHINI teshisi (disarida birakilan markada EN IYI rule) tarama
# maliyetini IKIYE katlar. D6'da gerekliydi (NIT cokusunun sebebini ayirmak
# for); `full` kosusunda varsayilan KAPALI.
KAHIN =os .environ .get ("P6_KAHIN","0")=="1"
# Kural aramasi kivrimin EGITIM parcalarinin a ORNEKLEMINDE is done: each rule
# tum parcalari gezip Macar eslemesi kosuyor and 42 rule x 2000 part a kolu
# dakikalarca bekletiyor. Ornekleme SECIMI degistirmez (kurallar arasi order
# birkac face parcada already kararli), only maliyeti dusurur.
ARAMA_N =int (os .environ .get ("P6_ARAMA_N","600"))
OLCUT =os .environ .get ("P6_OLCUT","makro")# rule secim olcutu
TOHUM_KURAL =("mutlak",0.60 )# seed ESIGI also taranmaz: high tutulur
TOHUM_NMS =5.0 
# IKINCI KADEME = KISA LISTE UZERINDE FP REDDEDICI.
# Ilk tasarimda ikinci kademe TUM secenekleri yeniden puanliyordu and only 8
# lattice sutunu ekliyordu -- measured, ZARAR verdi (-0.0363): birinci kademenin
# already cozdugu 2200 secenegin ezici cogunlugu apacik negatif and model kapasitesi
# oraya gidiyor. Dogru kurulum kaskad: birinci kademe RECALL for genis tarar,
# ikinci kademe only KISA LISTEYE bakar, birinci kademe skorunu da OZNITELIK
# as takes and hard negatifleri ayirmaya odaklanir.
KISA_ESIK =float (os .environ .get ("P6_KISA_ESIK","0.20"))
# NEGATIF ORANI 8 -> 12 (2026-08-12). Sistematik HPO first times yapildi
# (`run_improvement_sweep.py`, 20 yapilandirma, all of them UCTAN UCA robot F1,
# brand-disi katlar). Negatif orani TEK kazanan eksendi; ogrenme hizi,
# yaprak count and L2 notr ya da zararli output.
#   d6  (468 part, 4 fold) : 0.2994 -> 0.3135  (+0.0140)
#   full (2040 part, 5 fold): 0.3092 -> 0.3218  (+0.0126)   <- VERIFICATION
# Taban KASITLI as 8 secildi: taramanin first turu 6'dan olcuyordu and
# +0.0088 veriyordu; URETIM 8 kullandigi for dagitilan number 8'e according to
# measured. Egri 12-24 arasi duz a plato (0.3121-0.3135); 6 (0.3047) with
# 8 (0.2994) arasindaki ters donus fold gurultusunun ~+/-0.005 oldugunu
# gosteriyor, i.e. +0.0126'nin belirsizligi real.
NEG_KAT =int (os .environ .get ("P6_NEG_KAT","12"))
KOLLAR =tuple (os .environ .get ("P6_KOLLAR",
"TABAN,P6,P6_KAFES,P6_GEO").split (","))
A_SUT =58 # pool oznitelikleri (segmentasyon agindan turer)


ITER =int (os .environ .get ("P6_ITER","400"))


def yap (seed =0 ):
    return HistGradientBoostingClassifier (
    max_iter =ITER ,learning_rate =0.06 ,max_leaf_nodes =63 ,
    l2_regularization =1.0 ,random_state =seed )


def kendi (d ):
    """Aday basina TEK row: own direction secenegi (C blogunun k_kendi=1 satiri)."""
    return np .where (d ["X"][:,C0 ]==1.0 )[0 ]


def taban_satir (d ):
    """TABAN kolunun satirlari: own yonu VE mesh OLMAYAN candidates.

    Dagitilan urunun havuzunda mesh tepeleri YOK. Mesh'i baseline kolunda da
    birakmak, "new pool + new siralayici" kazancini tabana da yazmak olurdu
    and kiyas single degiskenli olmaktan cikardi.
    """
    k =kendi (d )
    return k [d ["kaynak"][d ["idx"][k ]]!=2 ]


ZORNEG =os .environ .get ("P6_ZORNEG","0")=="1"
TOHUM_N =int (os .environ .get ("P6_TOHUM_N","1"))


def alt_ornekle_zor (M ,Y ,s1 ,fold =NEG_KAT ,seed =0 ):
    """ZOR NEGATIF madenciligi: negatiflerin yarisi EN YUKSEK SKORLU olanlardan.

    Varsayilan ornekleme negatifleri RASTGELE seciyor; %98'i apacik negatif
    oldugu for model easy ornekle doluyor and karar sinirini hard bolgede
    ogrenemiyor. Burada negatif butcesinin yarisi birinci kademe skoruna according to
    EN YUKSEK negatiflerden, yarisi rastgeleden gelir -- so hem hard boundary
    hem baseline dagilimi temsil edilir.
    """
    if fold <=0 :
        return M ,Y 
    rng =np .random .default_rng (seed )
    poz =np .where (Y ==1 )[0 ]
    neg =np .where (Y ==0 )[0 ]
    n =min (len (neg ),fold *max (len (poz ),1 ))
    if s1 is None or not len (neg ):
        sec_neg =rng .choice (neg ,n ,replace =False )if len (neg )else neg 
    else :
        yari =n //2 
        rank_ =neg [np .argsort (-np .asarray (s1 ,float )[neg ])]
        zor =rank_ [:yari ]
        kalan =np .setdiff1d (neg ,zor ,assume_unique =False )
        rast =(rng .choice (kalan ,min (n -yari ,len (kalan )),replace =False )
        if len (kalan )else np .zeros (0 ,int ))
        sec_neg =np .concatenate ([zor ,rast ])
    sec =np .concatenate ([poz ,sec_neg ]).astype (int )
    rng .shuffle (sec )
    return M [sec ],Y [sec ]


def alt_ornekle (M ,Y ,fold =NEG_KAT ,seed =0 ):
    """Tum pozitifler + `fold` katı negatif. 3M satirlik egitimi kaldirilabilir
    kilar; threshold already sonradan taraniyor, baseline ratio degismesi zararsiz."""
    if fold <=0 :
        return M ,Y 
    rng =np .random .default_rng (seed )
    poz =np .where (Y ==1 )[0 ]
    neg =np .where (Y ==0 )[0 ]
    n =min (len (neg ),fold *max (len (poz ),1 ))
    sec =np .concatenate ([poz ,rng .choice (neg ,n ,replace =False )])
    rng .shuffle (sec )
    return M [sec ],Y [sec ]


def tohumla (d ,s ):
    """Birinci kademe skorlarindan TOHUM konum/yonleri."""
    return p6_decision .sec_ayrintili (d ["P"],d ["idx"],d ["YD"],s ,TOHUM_KURAL ,
    nms_mm =TOHUM_NMS )[:2 ]


def kafes_bloku (d ,s ):
    Pt ,Dt =tohumla (d ,s )
    return lattice .oznitelik (d ["P"][d ["idx"]],d ["YD"],Pt ,Dt )


def puanla (d ,s ,rule_ ,nms ,arm ):
    if arm =="TABAN":
        k =taban_satir (d )
        ci =d ["idx"][k ]# candidate indeksleri
        sk =s [k ]
        m =p6_decision .kabul_maskesi (sk ,rule_ )
        if not m .any ():
            return np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))
        P ,D =d ["P"][ci [m ]],d ["D"][ci [m ]]
        T =d ["X"][k ][m ][:,AB +len (YB .OZ_AD ):]
        n =np .ones (len (P ),bool )
        if nms >0 and len (P )>1 :
            import wire_gate 
            n =wire_gate .crowd_mask (P ,sk [m ])
        return P [n ],product_genis .isaret_duzelt (D [n ],T [n ])
    return p6_decision .sec (d ["P"],d ["idx"],d ["YD"],s ,rule_ ,nms_mm =nms )


def olc (data_ ,skor ,rule_ ,nms ,arm ):
    tp =fp =fn =0 
    tes =[]
    per =collections .defaultdict (lambda :[0 ,0 ,0 ])
    for d ,s in zip (data_ ,skor ):
        P ,D =puanla (d ,s ,rule_ ,nms ,arm )
        a ,b ,c =match_hungarian (P ,D ,d ["G"],d ["Gd"],d ["diag"],K .YANAL ,K .ACI ,
        False ,signed =True )[:3 ]
        tp +=a ;fp +=b ;fn +=c 
        q =per [d ["mfg"]]
        q [0 ]+=a ;q [1 ]+=b ;q [2 ]+=c 
        tes .append ((len (d ["G"]),)+match_hungarian (
        P ,D ,d ["G"],d ["Gd"],d ["diag"],max (3.0 ,0.06 *d ["diag"]),
        180.0 ,True )[:3 ])
    pm ={m :2 *q [0 ]/max (2 *q [0 ]+q [1 ]+q [2 ],1 )for m ,q in per .items ()}
    return {"robot":2 *tp /max (2 *tp +fp +fn ,1 ),"tespit":K .mikro (tes ),
    "TP":int (tp ),"FP":int (fp ),"FN":int (fn ),
    "recall":tp /max (tp +fn ,1 ),"precision":tp /max (tp +fp ,1 ),
    "makro":float (np .mean (list (pm .values ())))if pm else 0.0 ,
    "en_kotu":float (min (pm .values ()))if pm else 0.0 ,"brand":pm }


    # VARSAYILAN KAPALI (2026-08-12). Kanonik blok BASKA a betikte
    # (`run_merged_kol.py` / EK cercevesi) `full` brand-disi katlarinda
    # **+0.0151** vermisti. URETIM egiticisinde A/B was run and TERSI output:
    #   d6, secilen arm P6:  kanonik KAPALI 0.2932 -> OPEN 0.2794  (**-0.0138**)
    # Fark muhtemelen rule secimi: uretim fold inside rule ariyor and
    # ('mutlak', 0.97) seciyor; measurement betigim sabit ('goreli', 0.85, 0.20)
    # kullaniyordu. Yani olculen sey with dagitilacak sey AYNI DEGILDI.
    # LESSON: a kolu, DAGITILACAK kod yolunda yeniden olcmeden dagitma.
    # Kod duruyor; `P6_KANONIK=1` with acilir.
KANONIK =os .environ .get ("P6_KANONIK","0")=="1"


def canonical_block (d ):
    """KANONIK HIZALAMA blogu (10 column): parcayi KENDI PCA cercevesine oturtur.

    Olculdu 2026-08-12 (`full` brand-disi katlari): **+0.0151** -- that gun single
    degiskenli olculen 25 kolun UCTAN UCA gecen tekiydi. Donme/oteleme/scale
    degismezligi unit testli (`tests/test_kanonik_hizalama.py`).

    Mesh dizini kumeye according to degisir; ikisi de denenir and bulunamazsa candidate
    noktalarinin kendisi cerceve as is used (arm sessizce BOZULMAZ,
    only zayiflar).
    """
    import kanonik_hizalama as KH 
    P =np .asarray (d ["P"],float )
    idx =np .asarray (d ["idx"],int )
    V =None 
    for kok in ("results/_p1_olasilik_brepegit","results/_p1_olasilik"):
        mf =f"{kok }/{d ['pid']}.npz"
        if os .path .exists (mf ):
            V =np .asarray (np .load (mf )["V"],float )
            break 
    if V is None :
        V =P 
    return KH .oznitelik (P [idx ],d ["YD"],V ).astype (np .float32 )


def _kan (d ,n ):
    """kanonik blok ya da (n,0) -- kapaliysa no column eklenmez."""
    if not KANONIK or "_kan"not in d :
        return np .zeros ((n ,0 ),np .float32 )
    return np .asarray (d ["_kan"],np .float32 )


def oz (d ,arm ,kafes_blok =None ,s1 =None ):
    """Kolun feature matrisi.

    TABAN     : A+B, candidate basina single row, mesh HARIC (dagitilan rule)
    P6        : 92 + source gostergesi (3)
    P6_KAFES  : P6 + lattice (8) + birinci kademe skoru (1) -- YALNIZ kisa list
    """
    if arm =="TABAN":
        return p6_decision .donustur (d ["X"][taban_satir (d )][:,:AB ],"hepsi")
    X =np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])])
    if arm =="P6_GEO":
    # SEGMENTASYON OZNITELIKLERI ATILDI (first 58 column).
    # WHY: NIT'te rule kahini bile 0.0349 verdi -- i.e. loss esikte
    # DEGIL, skorda. Havuzun yonlu recall'u 0.5254 oldugu halde model that
    # markada bilgi uretemiyor. A blogu segmentasyon agindan turer and that network
    # NIT'e aktarilmiyorsa, ona dayanmayan a model DAHA IYI genellesebilir.
    # Kalan: mouth olculeri (9) + direction bankasi (16) + secenek yonuyle mouth
    # olculeri (9) + source (3) = 37 column, all of them GEOMETRIK/FIZIKSEL.
    # KANONIK blok SONA eklenir: P6_GEO sutunlari INDEKSLE diliyor
    # (A_SUT:AB and AB:), araya girmek that dilimleri sessizce bozardi.
        return np .hstack ([X [:,A_SUT :AB ],X [:,AB :],_kan (d ,len (X ))])
    if arm =="P6_KAFES":
        return np .hstack ([X ,kafes_blok ,np .asarray (s1 ,float )[:,None ],
        _kan (d ,len (X ))])
    return np .hstack ([X ,_kan (d ,len (X ))])


def kisa (s1 ):
    """Ikinci kademenin bakacagi satirlar."""
    return np .where (np .asarray (s1 ,float )>=KISA_ESIK )[0 ]


SIRA =os .environ .get ("P6_SIRA","1")=="1"


def sira_bloku (d ,s1 ):
    """Sira damgalama blogu (3 column). PARCA BASINA BIR KEZ is computed.

    ONCE `kafes_matris` inside hesaplaniyordu and each KAT x KOL for YENIDEN
    kosuyordu; a kosu 70 dakikada ilerlemedi. `lattice` blogu like a times
    hesaplanip tasinir.
    """
    Pt ,Dt =tohumla (d ,s1 )
    return order_stamp .oznitelik (d ["P"][d ["idx"]],d ["YD"],Pt ,Dt )


def kafes_matris (d ,kb ,s1 ,k ,sb =None ):
    """Ikinci kademe oznitelikleri, `k` satirlarinda.

    [92 donusturulmus | source 3 | lattice 8 | (order 3) | birinci kademe skoru 1]

    `order` blogu: birinci gecisin CAPALARINDAN sirayi uzatip havuzda karsiligi
    which is adaylari "order on" diye isaretler. `lattice` blogu a MESAFE
    olcusu gives; this whereas KABUL EDILMIS order uyeligi bayragidir -- two different
    soru, birlikte kullanilirlar. `P6_SIRA=0` with kapatilir.
    """
    X =np .hstack ([p6_decision .donustur (d ["X"]),
    p6_decision .kaynak_blok (d ["kaynak"][d ["idx"]])])
    par =[X [k ],np .asarray (kb )[k ]]
    if SIRA and sb is not None :
        par .append (np .asarray (sb )[k ])
    par .append (np .asarray (s1 ,float )[k ][:,None ])
    par .append (_kan (d ,len (X ))[k ])
    return np .hstack (par )


def egit (tr ,arm ,kafes_bloklar =None ,s1ler =None ,sira_bloklar =None ):
    Ms ,Ys =[],[]
    for i ,d in enumerate (tr ):
        if arm =="P6_KAFES":
            k =kisa (s1ler [i ])
            if not len (k ):
                continue 
                # float32'ye PARCA BASINA cevir: 2583 part x ~2000 secenek x 95
                # column float64 birikince vstack vertex bellegi 10 GB'a cikiyor.
            Ms .append (kafes_matris (
            d ,kafes_bloklar [i ],s1ler [i ],k ,
            None if sira_bloklar is None else sira_bloklar [i ]
            ).astype (np .float32 ))
            Ys .append (d ["y"][k ])
        else :
            Ms .append (oz (d ,arm ).astype (np .float32 ))
            Ys .append (d ["y"][taban_satir (d )]if arm =="TABAN"else d ["y"])
    if not Ms :
        return None 
    M =np .vstack (Ms )
    Y =np .concatenate (Ys )
    if ZORNEG and arm !="TABAN"and s1ler is not None :
        S1 =np .concatenate ([np .asarray (s ,float )[kisa (s )]if arm =="P6_KAFES"
        else np .asarray (s ,float )for s in s1ler ])
        M2 ,Y2 =alt_ornekle_zor (M ,Y ,S1 if len (S1 )==len (Y )else None )
    else :
        M2 ,Y2 =alt_ornekle (M ,Y )
    if TOHUM_N <=1 :
        return yap ().fit (M2 ,Y2 )
        # UC TOHUM ENSEMBLE: same data, different rastgelelik; skor ORTALAMASI alinir
    return [yap (t ).fit (*alt_ornekle (M ,Y ,seed =t ))for t in range (TOHUM_N )]


def _pp (m ,X ):
    """Tek model ya da ENSEMBLE listesi -- ikisi de works."""
    X =X .astype (np .float32 )
    if isinstance (m ,list ):
        return np .mean ([mm .predict_proba (X )[:,1 ]for mm in m ],axis =0 )
    return m .predict_proba (X )[:,1 ]


def skorla (m ,data_ ,arm ,kafes_bloklar =None ,s1ler =None ,sira_bloklar =None ):
    """Kolun skorlari. P6_KAFES'te KISA LISTE DISI satirlar 0 kalir --
    i.e. ikinci kademe birinci kademeyi EZEMEZ, only icinden selects."""
    out =[]
    for i ,d in enumerate (data_ ):
        if arm =="P6_KAFES":
            s =np .zeros (len (d ["X"]))
            k =kisa (s1ler [i ])
            if len (k ):
                X =kafes_matris (
                d ,kafes_bloklar [i ],s1ler [i ],k ,
                None if sira_bloklar is None else sira_bloklar [i ])
                s [k ]=_pp (m ,X )
            out .append (s )
            continue 
        p =_pp (m ,oz (d ,arm ))
        if arm =="TABAN":
            s =np .zeros (len (d ["X"]))
            s [taban_satir (d )]=p 
        else :
            s =p 
        out .append (np .asarray (s ,float ))
    return out 


def main ():
    t0 =time .time ()
    # KUME SECIMI. `full` = 9 brand / 2583 part (asil training and rule secimi).
    # `d6` = 8 brand / 468 part -- HIZLI YINELEME for. D6 this oturumda teshis
    # and arm secimi for YOGUN kullanildi, therefore TEMIZ OKUMA DEGILDIR;
    # temiz okuma only D7'dir and ona 3 okumalik butce with bakilir.
    # `P6_KUME` virgulle birden extra corpus alabilir: "full,d6".
    # WHY: D6 SINAV DEGIL, benim gelistirme kumem. Onu EGITIME katmak D7 for
    # tamamen mesru and brand cesitliligini 9 -> 17'ye removes. Bu projede
    # olculmustu: same part sayisinda KARISIK data single ureticiyi +0.0443 yener.
    # Kural secimi yine `full` markalarinin katlarinda is done.
    tr =[]
    for _k in os .environ .get ("P6_KUME","tam").split (","):
        tr +=yukle (_k .strip (),int (os .environ .get ("P6_TR","0")))
    for d in tr :
        d ["y"]=np .asarray (d ["y"],int )
    brand =collections .Counter (d ["mfg"]for d in tr )
    # KAT ESIGI. D6'da 60 esigi NIT'i (50 part) disarida birakiyordu -- oysa
    # NIT D6 GT'sinin %46'si and havuzun EN ZOR oldugu brand. Bir markanin never
    # fold olmamasi, kiyas sayilarinin that markayi HIC olcmemesi demek.
    KAT_MIN =int (os .environ .get ("P6_KAT_MIN","60"))
    katlar =[m for m ,n in brand .items ()if n >=KAT_MIN ]
    print (f"tam {len (tr )} part | markalar {dict (brand )}",flush =True )
    print (f"brand katlari (n>={KAT_MIN }): {katlar }  ({time .time ()-t0 :.0f} s)"
    f" | kapsanan part {sum (brand [m ]for m in katlar )}/{len (tr )}",
    flush =True )

    # --- 1) BIRINCI KADEME: brand-katli OOF skorlari -----------------------
    oof =[None ]*len (tr )
    for b in katlar :
        ic =[i for i ,d in enumerate (tr )if d ["mfg"]!=b ]
        dis =[i for i ,d in enumerate (tr )if d ["mfg"]==b ]
        m1 =egit ([tr [i ]for i in ic ],"P6")
        for i ,s in zip (dis ,skorla (m1 ,[tr [i ]for i in dis ],"P6")):
            oof [i ]=s 
        print (f"  OOF {b }: {len (dis )} part ({time .time ()-t0 :.0f} s)",
        flush =True )
    kucuk =[i for i ,s in enumerate (oof )if s is None ]
    if kucuk :# fold olusturamayan small markalar
        ic =[i for i in range (len (tr ))if i not in set (kucuk )]
        if not ic :# (only small kosularda becomes)
            rng =np .random .default_rng (0 )
            pay =rng .permutation (len (tr ))%3 
            for f_ in range (3 ):
                d_ =[i for i in range (len (tr ))if pay [i ]==f_ ]
                i_ =[i for i in range (len (tr ))if pay [i ]!=f_ ]
                m1 =egit ([tr [i ]for i in i_ ],"P6")
                for i ,s in zip (d_ ,skorla (m1 ,[tr [i ]for i in d_ ],"P6")):
                    oof [i ]=s 
            print ("  OOF: brand kati kurulamadi, 3 RASTGELE fold kullanildi "
            "(yalniz kucuk kosularda olur)",flush =True )
        else :
            m1 =egit ([tr [i ]for i in ic ],"P6")
            for i ,s in zip (kucuk ,skorla (m1 ,[tr [i ]for i in kucuk ],"P6")):
                oof [i ]=s 
            print (f"  OOF kucuk markalar: {len (kucuk )} part",flush =True )
    if not katlar :# kiyas katlari da otherwise rastgele boluruz
        rng =np .random .default_rng (1 )
        pay =rng .permutation (len (tr ))%3 
        for i ,d in enumerate (tr ):
            d ["_kat"]=f"fold{pay [i ]}"
        katlar =[f"fold{i }"for i in range (3 )]
        print (f"  KIYAS katlari rastgele: {katlar }",flush =True )
    else :
        for d in tr :
            d ["_kat"]=d ["mfg"]

    if KANONIK :
        for d in tr :
            d ["_kan"]=canonical_block (d )
        _kw =tr [0 ]["_kan"].shape [1 ]if tr else 0 
        # BOS BLOK SESSIZ NO-OP'A KARSI: this projede a blok "eklendi"
        # sanilip never dolmadan kosmustu. Genislik here YUKSEK SESLE
        # dogrulanir.
        assert _kw >0 ,"kanonik blok BOS -- oznitelik uretilmedi"
        print (f"kanonik blok {_kw } sutun ({time .time ()-t0 :.0f} s)",
        flush =True )

    kafes_tr =[kafes_bloku (d ,s )for d ,s in zip (tr ,oof )]
    sira_tr =([sira_bloku (d ,s )for d ,s in zip (tr ,oof )]if SIRA else None )
    kv =np .vstack (kafes_tr )
    print (f"lattice blogu {kv .shape } | lattice bulunan secenek orani "
    f"{kv [:,0 ].mean ():.3f} ({time .time ()-t0 :.0f} s)",flush =True )

    # --- 2) KAT ICINDE rule secimi + arm kiyasi ---------------------------
    top ={k :collections .Counter ()for k in KOLLAR }
    ayrinti ={}
    for b in katlar :
        ic =[i for i ,d in enumerate (tr )if d ["_kat"]!=b ]
        dis =[i for i ,d in enumerate (tr )if d ["_kat"]==b ]
        TR =[tr [i ]for i in ic ]
        TE =[tr [i ]for i in dis ]
        ayrinti [b ]={}
        for arm in KOLLAR :
            kf =(arm =="P6_KAFES")
            kb_tr =[kafes_tr [i ]for i in ic ]if kf else None 
            kb_te =[kafes_tr [i ]for i in dis ]if kf else None 
            # ZOR NEGATIF SESSIZ NO-OP'U (2026-08-12'de yakalandi):
            # `s1_tr` only kf (P6_KAFES) for doluyordu. `egit` icindeki
            # sart `ZORNEG and arm != "TABAN" and s1ler is not None` oldugu
            # for P6 kolunda ZORNEG HIC DEVREYE GIRMIYORDU -- bayrak
            # aciliyor, no sey degismiyordu. Kontrollu testte
            # ZORNEG=0 and =1 TP/FP/FN'e up to BIREBIR AYNI output.
            #
            # Skorlar each arm for ZATEN present; `egit`/`skorla` onlari only
            # P6_KAFES dalinda feature kurmak for kullaniyor, digerlerinde
            # dokunmuyor. Bu yuzden hepsine gecirmek GUVENLI and hard-negatif
            # yolunu ACAR.
            s1_tr =[oof [i ]for i in ic ]
            s1_te =[oof [i ]for i in dis ]
            sb_tr =([sira_tr [i ]for i in ic ]if kf and sira_tr else None )
            sb_te =([sira_tr [i ]for i in dis ]if kf and sira_tr else None )
            m =egit (TR ,arm ,kb_tr ,s1_tr ,sb_tr )
            if m is None :
                ayrinti [b ][arm ]={"robot":0.0 ,"TP":0 ,"FP":0 ,
                "FN":sum (len (d ["G"])for d in TE ),
                "kural":["yok"],"nms":0.0 }
                continue 
            s_tr =skorla (m ,TR ,arm ,kb_tr ,s1_tr ,sb_tr )
            s_te =skorla (m ,TE ,arm ,kb_te ,s1_te ,sb_te )
            ar =(np .random .default_rng (0 ).choice (len (TR ),ARAMA_N ,False )
            if ARAMA_N and len (TR )>ARAMA_N else np .arange (len (TR )))
            AR =[TR [i ]for i in ar ]
            AS =[s_tr [i ]for i in ar ]
            # RULE SECIM OLCUTU: training markalarinin MAKRO ortalamasi.
            # WHY MIKRO DEGIL: mikro, GT'si very which is markanin kuralini selects.
            # D6'da NIT GT'nin %46'si and NIT'te secilen threshold (0.97) HER SEYI
            # eliyor -> that markada F1 0.0016. Havuzda NIT'in cevabinin YARISI
            # (yonlu recall 0.5254) VAR; kaybeden rule, model not. Makro
            # criterion, single a markada COKMEYEN kurali tercih eder.
            def _puan (x ):
                r_ =olc (AR ,AS ,x [0 ],x [1 ],arm )
                return r_ ["makro"]if OLCUT =="makro"else r_ ["robot"]
            en =max (((r ,n )for r in KURALLAR for n in NMSLER ),key =_puan )
            r =olc (TE ,s_te ,en [0 ],en [1 ],arm )
            # RULE KAHINI (DIAGNOSIS, dagitilamaz): disarida birakilan markada EN
            # IYI rule ne verirdi? Fark buyukse loss RULE SECIMINDE, kucukse
            # MODELDE demektir.
            kah =(max (olc (TE ,s_te ,x ,n ,arm )["robot"]
            for x in KURALLAR for n in NMSLER )if KAHIN else 0.0 )
            for k in ("TP","FP","FN"):
                top [arm ][k ]+=r [k ]
            ayrinti [b ][arm ]=dict (r ,rule_ =list (en [0 ]),nms =en [1 ],
            kural_kahini =kah )
        a =ayrinti [b ]
        # KOLLAR cevre degiskeniyle degisebiliyor; satiri SABIT arm adlariyla
        # yazmak arm listesi kisaldiginda KeyError veriyordu.
        oz_ =" | ".join (f"{k } {a [k ]['robot']:.4f}"for k in KOLLAR if k in a )
        kh =a .get ("P6",{}).get ("kural_kahini",0.0 )
        print (f"  {b :<6} n={len (TE ):<4} {oz_ }"
        +(f"   [kural kahini P6 {kh :.4f}]"if kh else "")
        +f"   ({time .time ()-t0 :.0f} s)",flush =True )

    print (f"\n{'arm':<12} {'robot':>8} {'recall':>8} {'precision':>9} "
    f"{'TP':>7} {'FP':>7} {'FN':>7}")
    last_ ={}
    for arm in KOLLAR :
        c =top [arm ]
        f1 =2 *c ["TP"]/max (2 *c ["TP"]+c ["FP"]+c ["FN"],1 )
        rc =c ["TP"]/max (c ["TP"]+c ["FN"],1 )
        pr =c ["TP"]/max (c ["TP"]+c ["FP"],1 )
        last_ [arm ]={"robot":f1 ,"recall":rc ,"precision":pr ,**dict (c )}
        print (f"{arm :<12} {f1 :>8.4f} {rc :>8.4f} {pr :>9.4f} {c ['TP']:>7} "
        f"{c ['FP']:>7} {c ['FN']:>7}")
        # Farklar KOL LISTESINE according to uretilir; sabit arm adi yazmak list
        # kisaldiginda KeyError veriyordu (two times became).
    if "TABAN"in last_ :
        for k in KOLLAR :
            if k !="TABAN":
                print (f"\n{k :<9} - TABAN = "
                f"{last_ [k ]['robot']-last_ ['TABAN']['robot']:+.4f}",end ="")
        print ()

        # --- 3) NIHAI MODELLER (tum full) ---------------------------------------
    en_kol =max (KOLLAR ,key =lambda k :last_ [k ]["robot"])
    kural_sayim =collections .Counter (
    (tuple (ayrinti [b ][en_kol ]["kural"]),ayrinti [b ][en_kol ]["nms"])
    for b in katlar )
    rule_ ,nms =kural_sayim .most_common (1 )[0 ][0 ]
    print (f"\nSECILEN arm {en_kol } | kural {rule_ } | nms {nms } "
    f"(brand katlarinda en sik)")
    m1 =egit (tr ,"P6")
    paket ={"kademe1":m1 ,"arm":en_kol ,"kural":list (rule_ ),"nms":nms ,
    "zskor":"ab","AB":AB ,"tohum_kural":list (TOHUM_KURAL ),
    "tohum_nms":TOHUM_NMS ,"kisa_esik":KISA_ESIK ,"sira":SIRA }
    if en_kol =="P6_KAFES":
    # Ikinci kademe OOF skorlarindan egitilir: urunde birinci kademe skoru
    # gorulmemis parcadan gelecek, egitimde de oyle gelmeli.
        paket ["kademe2"]=egit (tr ,"P6_KAFES",kafes_tr ,oof ,sira_tr )
    with open ("results/p6_kademe2_model.pkl","wb")as f :
        pickle .dump (paket ,f )
    json .dump ({"damga":makbuz_hash .damga (),"toplam":last_ ,"brand":ayrinti ,
    "n_egitim":len (tr ),"katlar":katlar ,"secilen":en_kol ,
    "kural":list (rule_ ),"nms":nms ,"dizin":os .environ ["P6_DIZIN"],
    "not":"tam korpusunun MARKA KATLARINDA kural secimi + arm "
    "kiyasi. Tohumlar OUT-OF-FOLD skorlardan. D6 ve D7'ye "
    "BAKILMADI."},
    open ("results/p6_kademe2_tam.json","w"),indent =1 )
    print (f"receipt -> results/p6_kademe2_tam.json  ({time .time ()-t0 :.0f} s)")


if __name__ =="__main__":
    main ()
