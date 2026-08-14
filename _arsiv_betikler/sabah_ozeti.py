# -*- coding: utf-8 -*-
"""SABAH OZETI: gecenin butun makbuzlarini topla, ONCEDEN ILAN EDILEN kapilari
uygula, single sayfa rapor yaz.

WHY. Gece programi onlarca log and receipt uretiyor. Sabah bunlari elle okumak
hem slow hem de wrong okumaya open. Bu betik each makbuzu kapisiyla birlikte
gosterir and KARARI writes.

DUMAN MAKBUZU KORUMASI. Duman testleri (small ornek, minik parametre) same
file adlarina yaziyor and real sonuc sanilabiliyor -- this gece a times became.
`n_parca` esigin altindaysa receipt INVALID isaretlenir, count RAPOR EDILMEZ.

Kullanim:  python sabah_ozeti.py
"""
import glob 
import json 
import os 
import time 

KOK =os .path .dirname (os .path .abspath (__file__ ))
EN_AZ_PARCA =500 # bunun altindaki receipt DUMAN sayilir
KAPI_EK =0.01 # ek feature blogu kapisi
KAPI_HAVUZ =0.85 # yonlu pool recall kapisi
KAPI_D7 =0.10 # D7 OKUMA #2 for kumulatif kazanc kapisi


def oku (y ):
    try :
        with open (y ,encoding ="utf-8")as f :
            return json .load (f )
    except Exception :
        return None 


def sat (x ):
    return "-"if x is None else (f"{x :+.4f}"if isinstance (x ,float )else str (x ))


def ek_bloklar ():
    out =[]
    for y in sorted (glob .glob (os .path .join (KOK ,"results","ek_blok_*.json"))):
        d =oku (y )
        if not d :
            continue 
        ad =d .get ("blok",os .path .basename (y ))
        n =int (d .get ("n_parca",0 ))
        if n <EN_AZ_PARCA :
            out .append ((ad ,None ,None ,None ,f"GECERSIZ (duman, {n } part)"))
            continue 
        f =float (d .get ("fark",0.0 ))
        out .append ((ad ,float (d .get ("yok",0 )),float (d .get ("var",0 )),f ,
        "GECTI"if f >=KAPI_EK else "gecmedi"))
    return out 


def pool (dizin ="_p6_oz_tam3",cluster =None ):
    """Degerler KOKTE not `total` under: konum_recall / yonlu_recall /
    f1_tavani. (Ilk yazimda kokte aranmisti and rapor empty gosteriyordu.)

    Makbuz adi residual kumeyi de tasiyor; old (kumesiz) ada da bakilir."""
    adlar =([f"havuz_tavani_{dizin }_{cluster }.json"]if cluster else [])+[f"havuz_tavani_{dizin }_d6.json",f"havuz_tavani_{dizin }.json"]
    d =None 
    for a in adlar :
        d =oku (os .path .join (KOK ,"results",a ))
        if d :
            break 
    if not d :
        return None 
    t =d .get ("toplam",{})
    return (t .get ("konum_recall"),t .get ("yonlu_recall"),t .get ("f1_tavani"),
    t .get ("aday_parca"),bool (d .get ("kapi_a_gecti")))


def kademe2 ():
    out =[]
    for ad ,y in (("SIRA kapali (u25 baseline)","p6_kademe2_sira0.json"),
    ("SIRA acik (u25)","p6_kademe2_sira1.json"),
    ("tam3 TABAN (duz ayar)","p6_kademe2_tam3_taban.json"),
    ("B1 zor negatif (tam3)","p6_kademe2_B1_zorneg.json"),
    ("B6 ensemble (tam3)","p6_kademe2_B6_ensemble.json"),
    ("son kosu (uzerine yazilan)","p6_kademe2_tam.json")):
        d =oku (os .path .join (KOK ,"results",y ))
        if not d :
            continue 
        t =d .get ("toplam",{})
        kollar ={k :v .get ("robot")for k ,v in t .items ()
        if isinstance (v ,dict )and "robot"in v }
        out .append ((ad ,d .get ("dizin","?"),d .get ("secilen"),kollar ))
    return out 


def gece_fazlari ():
    y =os .path .join (KOK ,"results","_gece","ANA.log")
    if not os .path .exists (y ):
        return []
    with open (y ,encoding ="utf-8",errors ="replace")as f :
        return [s .strip ()for s in f 
        if "BITTI:"in s or "DUSTU:"in s ]


def main ():
    L =[]
    L .append ("# SABAH RAPORU -- "+time .strftime ("%Y-%m-%d %H:%M"))
    L .append ("")
    L .append ("Butun sayilar `tam` MARKA KATLARINDA (LOMO). D7 SINAVINA "
    "BAKILMADI. Manset metrik MIKRO robot F1.")
    L .append ("")

    L .append ("## 1. KAPI A -- tam-acik havuzun yonlu recall'u")
    # TUM pool makbuzlari single tabloda: new a corpus (for example ceiling-24
    # `tam4`) olculunce elle kod degistirmeden raporda gorunsun.
    ETIKET ={"_p6_oz_u25":"onceki pool",
    "_p6_oz_tam3":"tam-acik, ceiling 12",
    "_p6_oz_tam4":"tam-acik, TAVAN 24"}
    sat =[]
    for y in sorted (glob .glob (os .path .join (KOK ,"results",
    "havuz_tavani_*.json"))):
        d =oku (y )
        if not d :
            continue 
        t =d .get ("toplam",{})
        ad =os .path .basename (y )[len ("havuz_tavani_"):-len (".json")]
        dz ,_ ,km =ad .rpartition ("_")
        sat .append ((ETIKET .get (dz ,dz ),km ,t ,bool (d .get ("kapi_a_gecti"))))
    if not sat :
        L .append ("Henuz olculmedi (A2 fazi kosmadi).")
    else :
        L .append ("| pool | cluster | konum | **yonlu** | F1 tavani | candidate/part | KAPI A |")
        L .append ("|---|---|---|---|---|---|---|")
        for ad ,km ,t ,gec in sat :
            L .append (f"| {ad } | {km } | {t .get ('konum_recall',0 ):.4f} | "
            f"**{t .get ('yonlu_recall',0 ):.4f}** | "
            f"{t .get ('f1_tavani',0 ):.4f} | "
            f"{t .get ('aday_parca',0 ):.0f} | "
            f"{'GECTI'if gec else 'gecmedi'} |")
        L .append ("")
        L .append (f"KAPI A esigi: yonlu recall >= {KAPI_HAVUZ }.")
        L .append ("")
        L .append ("Kapi gecmezse pool genisletme kolu KAPANIR: ceiling "
        "yetmiyorsa selector ne kadar iyilesirse iyilessin hedefe "
        "ulasilamaz. TAVAN, mukemmel bir secicinin alacagi F1'dir -- "
        "VAAT DEGIL, UST SINIR.")
    L .append ("")

    L .append ("## 1b. SECENEK TAVANI (MAX_SEC) BAGLIYOR MU?")
    ms =sorted (glob .glob (os .path .join (KOK ,"results",
    "max_sec_sondasi*.json")))
    if not ms :
        L .append ("- measurement yok (`MS_MARKA=NIT python probe_max_sec.py`)")
    for y in ms :
        d =oku (y )
        if not d :
            continue 
        L .append (f"**orneklem: {d .get ('brand','?')} / {d .get ('n_parca')} "
        f"part, yelpaze {d .get ('yelpaze')}**")
        L .append ("")
        L .append ("| ceiling | yonlu recall | secenek maliyeti |")
        L .append ("|---|---|---|")
        for t ,v in sorted (d .get ("sonuc",{}).items (),key =lambda kv :int (kv [0 ])):
            L .append (f"| {t } | {v ['yonlu_recall']:.4f} | "
            f"{v ['maliyet_kat']:.2f}x |")
        L .append ("")
    if ms :
        L .append ("> Bugunku ceiling **12**. Tavan bagliyorsa direction kaynagi "
        "eklemek (yelpaze cozunurlugu) recall'u ARTIRMAZ -- yeni "
        "yonler tavana takilip mevcutlarin yerini alir. "
        "`YB_MAX_SEC` ile ayarlanir.")
    L .append ("")

    L .append ("## 2. EK OZNITELIK BLOKLARI (gate +0.01)")
    ACIKLAMA ={
    "kanonik":"parcayi KENDI ana eksenlerine oturtur (brand bagimsizlik)",
    "cluster":"candidates arasi rekabet: ayni adayin obur yonleri, 5mm rakip",
    "topoloji":"es-eksenli aile / dizi duzenliligi (yalniz mi, uye mi)",
    "simetri":"ayna simetri esi var mi (klemensler simetriktir)",
    "depth":"axis boyu yaricap profili (tel / vida / alet ayrimi)",
    "kafes_adet":"lattice adimindan BEKLENEN CP sayisi -> secim baskisi",
    "ozkalib":"part-ici oz-kalibrasyon (skor yuzdeligi, en iyiye fark)",
    }
    eb =ek_bloklar ()
    if not eb :
        L .append ("Henuz receipt yok.")
    else :
        L .append ("| blok | ne olcer | bloksuz | blokla | fark | karar |")
        L .append ("|---|---|---|---|---|---|")
        for ad ,y0 ,v0 ,f ,k in sorted (
        eb ,key =lambda r :(-(r [3 ]if r [3 ]is not None else -9 ))):
            L .append (f"| **{ad }** | {ACIKLAMA .get (ad ,'?')} | "
            f"{'-'if y0 is None else f'{y0 :.4f}'} | "
            f"{'-'if v0 is None else f'{v0 :.4f}'} | "
            f"{'-'if f is None else f'{f :+.4f}'} | {k } |")
    L .append ("")

    L .append ("## 3. KADEME2 KOL KIYASLARI")
    for ad ,dz ,sec ,kollar in kademe2 ():
        ks =", ".join (f"{k } {v :.4f}"for k ,v in sorted (kollar .items ()))
        L .append (f"- **{ad }** ({dz }) secilen={sec } -> {ks }")
    L .append ("")

    L .append ("## 3b. SECICI VERIMLILIGI -- 0.50 nereden gelebilir?")
    tv =pool ("_p6_oz_u25","tam")or pool ("_p6_oz_tam3","tam")
    ger =None 
    for ad ,_dz ,_sec ,kollar in kademe2 ():
        if "P6"in kollar :
            ger =kollar ["P6"]
            break 
    if tv and ger :
        ceiling =tv [2 ]
        verim =ger /max (ceiling ,1e-9 )
        L .append (f"- pool F1 TAVANI (`tam`, mukemmel selector): **{ceiling :.4f}**")
        L .append (f"- GERCEKLESEN (P6 kolu): **{ger :.4f}**")
        L .append (f"- **selector verimliligi = {verim :.1%}**")
        L .append ("")
        ger_tavan =0.50 /max (verim ,1e-9 )
        ger_verim =0.50 /max (ceiling ,1e-9 )
        L .append (f"0.50'ye iki yoldan gidilebilir:")
        L .append (f"1. **Havuzla:** verimlilik sabit kalirsa tavanin "
        f"**{ger_tavan :.4f}** olmasi gerekir"
        +("  -> 1.0'i asiyor, TEK BASINA IMKANSIZ"
        if ger_tavan >1.0 else ""))
        L .append (f"2. **Seciciyle:** ceiling sabit kalirsa verimliligin "
        f"**{ger_verim :.1%}** olmasi gerekir "
        f"({ger_verim /max (verim ,1e-9 ):.2f}x iyilesme)")
        L .append ("")
        L .append ("> Havuz kolu tek basina hedefe goturmuyor; SECICI kolu "
        "zorunlu. Bu, EK bloklarina ve candidate-kumesi modeline "
        "(D2) verilen onceligi belirler.")
    else :
        L .append ("- `tam` kumesinde ceiling olcumu henuz yok "
        "(`HT_ONLER=tam python probe_pool_tavani.py`)")
    L .append ("")

    L .append ("## 4. GECE FAZLARI")
    fz =gece_fazlari ()
    L .extend (f"- {s }"for s in fz )if fz else L .append ("- log yok")
    L .append ("")

    L .append ("## 4b. SAHA -- AUTO KATMANI (tier cokusu)")
    tc =oku (os .path .join (KOK ,"results","tier_cokusu_d7.json"))
    if not tc :
        L .append ("- measurement yok (`python probe_tier_cokusu.py`)")
    else :
        L .append (f"Dagitilan AUTO esigi = **{tc .get ('dagitilan_esik')}**")
        for ad ,k in tc .get ("kumeler",{}).items ():
            if k .get ("durum"):
                L .append (f"- `{ad }`: **{k ['durum']}** "
                f"({k ['n_isaret']} isaretin hepsi ayni skor)")
                continue 
            de =k .get ("dagitilan_esikte",{})
            L .append (f"- `{ad }`: AUTO payi **{de .get ('auto_pay',0 ):.4f}**, "
            f"precision **{de .get ('precision')or 0 :.4f}**"
            +("  <- REVIEW KATMANI BOS"if de .get ("review_bos")else ""))
        L .append ("")
        L .append ("> Gorulmemis markada robot HER isarete otonom guveniyor. "
        "Esigi yukseltmek kurtarmiyor (0.95'te bile precision ~0.47). "
        "Oneri: gorulmemis brand icin AUTO katmani KAPATILSIN.")
    L .append ("")

    L .append ("## 5. D7 OKUMA #2 KARARI")
    kaz =[f for _ ,_ ,_ ,f ,k in eb if f is not None and k =="GECTI"]
    top =sum (kaz )
    L .append (f"- kapiyi gecen blok sayisi: **{len (kaz )}**")
    L .append (f"- bu bloklarin toplam kazanci: **{top :+.4f}** "
    f"(gate +{KAPI_D7 :.2f})")
    L .append ("")
    if top >=KAPI_D7 :
        L .append ("**KARAR: D7 OKUMA #2 HAK EDILDI.** Yine de okuma ancak "
        "kanonik zincirle (`canonical_d7.mikro()`) yapilir.")
    else :
        L .append ("**KARAR: D7 OKUNMAZ.** Kumulatif kazanc kapinin altinda; "
        "okuma HARCANMAZ. Butcede kalan okuma sayisi degismez.")
    L .append ("")
    L .append ("> Kazanclar TOPLANARAK tahmin edilir; gercek birlesik kazanc "
    "genellikle DAHA AZ olur (bloklar ayni hatalari duzeltir). "
    "Toplam yalnizca KAPI kararidir, VAAT DEGILDIR.")

    metin ="\n".join (L )
    y =os .path .join (KOK ,"docs","SABAH_RAPORU.md")
    os .makedirs (os .path .dirname (y ),exist_ok =True )
    with open (y ,"w",encoding ="utf-8")as f :
        f .write (metin +"\n")
    print (metin )
    print (f"\n-> {y }")


if __name__ =="__main__":
    main ()
