# -*- coding: utf-8 -*-
"""OLCUM KUMESI -- puanlanacak parcalarin TEK and DENETLENEBILIR kaynagi.

2026-08-01 DENETIMI four ayri kusur buldu and dordu de here kapatiliyor:

1. **DEV/VAL ADLARI YANLIS BAGLIYDI.** `headline.py` kumeleri `results/_probs_{dev,val}.pkl`
   onbelleklerinden cikariyordu and "dev" aslinda ESKI `_h_probs.pkl` onbellegiydi. Gercek split
   `results/split3.json`'da (2026-07-31, keskin geometri anahtari). Iki atama %100 UYUSMUYORDU:
   benim "dev 97 / val 100" dedigim cluster, gercekte **54 DEV / 100 VAL / 40 ATANMAMIS / 3 LOCKED**.

2. **TEKRAR EDEN PARCA.** 200 row aslinda 197 part; three part (3214360, 2428950000,
   1010880000) two times sayiliyordu -- old DEV and VAL onbellekleri kesisiyor.

3. **LOCKED KIRLENMESI.** Uc LOCKED part DOGRUDAN puanlanmis; geometri grubu uzerinden
   total five LOCKED part four grupta kirlenmis. Geriye **95 grup-temiz LOCKED** kaliyor and
   bunlar TEK ATIS for saklanmalidir.

4. **PARCA BOOTSTRAP'I.** 197 part 174 GEOMETRI GRUBUNA dusuyor; part ornekleyen a
   bootstrap, same geometrinin ikizlerini bagimsiz sayarak confidence araligini DARALTIR.
   Dogru unit GRUP'kind.

Bu modul kumeyi kurar, kirliligi ROPORTAJ EDER and grup-bootstrap birimini gives.
Puanlama yapan each betik BURAYI cagirir; cluster tanimi a more kopyalanmaz.
"""
import collections 
import json 
import io 
import os 
import pickle 

import numpy as np 

SPLIT ="results/split3.json"
GEO ="results/_strict_geometry_keys.json"


def _oku (yol ):
    with io .open (yol ,encoding ="utf-8")as f :
        return json .load (f )


def bolme_atamasi ():
    """pid -> 'dev' | 'val' | 'locked' (split3.json, TEK dogruluk kaynagi)."""
    s3 =_oku (SPLIT )
    ata ={}
    for k in ("dev","val","locked"):
        for p in s3 [k ]["parts"]:
            ata [str (p )]=k 
    return ata 


def geo_anahtarlari ():
    return _oku (GEO )


def cluster (der_yolu ="results/_u4_der.pkl",locked_cikar =True ,tekrar_cikar =True ):
    """Puanlanacak kaydi dondur + kirlilik raporu.

    Doner: (KAYITLAR, rapor). Her kayda `cluster` (dev/val/atanmamis) and `geo` eklenir.
    """
    with open (der_yolu ,"rb")as f :
        DER =pickle .load (f )
    ata =bolme_atamasi ()
    gk =geo_anahtarlari ()

    gorulen ,temiz ,atilan_tekrar ,atilan_locked =set (),[],[],[]
    for r in DER :
        pid =str (r ["pid"])
        if tekrar_cikar and pid in gorulen :
            atilan_tekrar .append (pid )
            continue 
        gorulen .add (pid )
        b =ata .get (pid ,"atanmamis")
        if locked_cikar and b =="locked":
            atilan_locked .append (pid )
            continue 
        r =dict (r )
        r ["cluster"]=b 
        r ["geo"]=gk .get (pid ,"yok:"+pid )
        temiz .append (r )

        # KIRLILIK, PUANLAMADAN CIKARILANLARI DA KAPSAR (2026-08-02 denetimi).
        # Onceki version `kullanilan_geo`yu only KALAN parcalardan hesapliyordu; so
        # DOGRUDAN kullanilip cikarilan LOCKED parcalarin KENDI GRUBU "dokunulmamis" gorunuyor
        # and that parts yeniden TEMIZ sayiliyordu. Ama onlar KULLANILDI -- measurement onbelleginde
        # varlar and tarihsel as puanlandilar. Bir times dokunulan part kalici as kirlidir.
        # Etkisi: "temiz LOCKED 98" -> DOGRUSU 95.
    kullanilan_geo =({r ["geo"]for r in temiz }
    |{gk .get (p ,"yok:"+p )for p in atilan_locked }
    |{gk .get (p ,"yok:"+p )for p in atilan_tekrar })
    s3 =_oku (SPLIT )
    locked_kirli ={}
    for p in s3 ["locked"]["parts"]:
        g =gk .get (str (p ),"yok:"+str (p ))
        if g in kullanilan_geo :
            locked_kirli .setdefault (g ,[]).append (str (p ))
    locked_temiz =[str (p )for p in s3 ["locked"]["parts"]
    if gk .get (str (p ),"yok:"+str (p ))not in kullanilan_geo ]

    rapor ={
    "girdi_satir":len (DER ),
    "puanlanan_parca":len (temiz ),
    "atilan_tekrar":atilan_tekrar ,
    "atilan_locked":atilan_locked ,
    "kume_dagilimi":dict (collections .Counter (r ["cluster"]for r in temiz )),
    "geometri_grubu":len (kullanilan_geo ),
    "locked_kirli_grup":{g :v for g ,v in locked_kirli .items ()},
    "locked_kirli_parca":sorted ({p for v in locked_kirli .values ()for p in v }),
    "locked_temiz_n":len (locked_temiz ),
    "locked_temiz":locked_temiz ,
    }
    return temiz ,rapor 


def grup_bootstrap (satirlar ,gruplar ,f1_fn ,n =4000 ,seed =0 ):
    """GRUP birimli bootstrap. Parca not GEOMETRI GRUBU ornekler.

    WHY: 197 part 174 gruba dusuyor and gruplarin a kismi IKIZ parts iceriyor.
    Parca ornekleyen bootstrap ikizleri bagimsiz gozlem sayar -> confidence araligi SAHTE DARALIR.
    Grup ornekleyince bagimlilik korunur.
    """
    rng =np .random .default_rng (seed )
    grup_ind =collections .defaultdict (list )
    for i ,g in enumerate (gruplar ):
        grup_ind [g ].append (i )
    anahtar =list (grup_ind )
    v =[]
    for _ in range (n ):
        sec =rng .integers (0 ,len (anahtar ),len (anahtar ))
        idx =[i for k in sec for i in grup_ind [anahtar [k ]]]
        v .append (f1_fn ([satirlar [i ]for i in idx ]))
    v =np .asarray (v ,float )
    return float (v .mean ()),float (np .percentile (v ,2.5 )),float (np .percentile (v ,97.5 ))


def rapor_bas (rapor ):
    print (f"OLCUM KUMESI: {rapor ['girdi_satir']} satir -> {rapor ['puanlanan_parca']} part "
    f"({rapor ['geometri_grubu']} geometri grubu)")
    print (f"  cluster: {rapor ['kume_dagilimi']}")
    if rapor ["atilan_tekrar"]:
        print (f"  ATILAN TEKRAR ({len (rapor ['atilan_tekrar'])}): {rapor ['atilan_tekrar']}")
    if rapor ["atilan_locked"]:
        print (f"  ATILAN LOCKED ({len (rapor ['atilan_locked'])}): {rapor ['atilan_locked']}")
    print (f"  LOCKED kirlenmesi: {len (rapor ['locked_kirli_grup'])} grup / "
    f"{len (rapor ['locked_kirli_parca'])} part -> {rapor ['locked_kirli_parca']}")
    print (f"  GRUP-TEMIZ LOCKED: {rapor ['locked_temiz_n']} part (TEK ATIS icin saklanir)")




def locked_gruplari ():
    """95 grup-temiz LOCKED parcanin GEOMETRI GRUPLARI."""
    _ ,rap =cluster ("results/_der_tam.pkl")
    gk =geo_anahtarlari ()
    return {gk .get (p ,"yok:"+p )for p in rap ["locked_temiz"]}


def sinav_egitim_maskesi (pidler ):
    """FINAL SINAVI for training maskesi: LOCKED'in GEOMETRI GRUPLARINI disla.

    2026-08-03 BULGUSU (q2_sinav_butunlugu.py): dagitilan training verisi
    `zengin_parite_w2.npz` 95 grup-temiz LOCKED parcasinin **77'sini** iceriyor
    (grup uzerinden 80). Sinav dagitilan gate'le kosulsaydi 95 parcanin 80'i KIRLI
    olurdu and sonuc SISIK cikardi.

    WHY GOZDEN KACTI: "LOCKED'a dokunulmadi" hep "puanlamiyoruz" diye anlasildi.
    Olcum betikleri only OLCUM KUMESININ gruplarini egitimden atiyor; LOCKED'in
    gruplari no zaman atilmadi because LOCKED never puanlanmamisti.

    RULE: LOCKED on olculecek each gate, this maskeyle egitilmek ZORUNDADIR.
    Aksi halde exam GECERSIZDIR and single atislik hakki bosa gider.
    """
    import numpy as np 
    gk =geo_anahtarlari ()
    lg =locked_gruplari ()
    g =np .array ([gk .get (str (p ),"yok:"+str (p ))for p in pidler ])
    return ~np .isin (g ,list (lg ))


if __name__ =="__main__":
    sat ,rap =cluster ()
    rapor_bas (rap )
    with io .open ("results/measure_set.json","w",encoding ="utf-8")as f :
        json .dump (rap ,f ,indent =1 ,ensure_ascii =False )
    print ("receipt -> results/measure_set.json")
