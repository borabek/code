# -*- coding: utf-8 -*-
"""KUME BAGLAMI: "this option, parcanin DIGER secenekleri between nerede duruyor?"

SORUN. Bugunku selector NOKTASALDIR: each (konum, direction) secenegini single basina
puanlar. Oysa karar kurali GORELI (part-maksimumunun %85'i) and real soru
always karsilastirmalidir: "this candidate, same delige bakan digerlerinden iyi mi?",
"this direction, same adayin obur yonlerinden iyi mi?", "5mm otede more guclu a
rakip present mi?". Noktasal model bunlarin HICBIRINI goremez.

Bu, D2 (candidate-kumesi transformer) kolunun UCUZ yaklasimidir: dikkat mekanizmasi
instead of elle secilmis cluster ozetleri. Olculen selector verimliligi %36.5 and hedefe
1.62x is required; feature bloklari single basina bunu vermez but this blok
"candidates arasi baglam" hipotezini UCUZA sinar. Kazanc varsa D2'ye yatirim
gerekcelenir; otherwise hipotez zayiflar.

Sutunlar (8):
  log_secenek   ln(1 + parcadaki option count)
  log_aday      ln(1 + parcadaki AYRIK candidate count)
  aday_secenek  this adayin kac secenegi present
  aday_sira     this secenegin AYNI ADAY icindeki score order (0 = most iyi)
  aday_fark     score - (same adayin most high skoru)      [<= 0]
  en_iyi_uzak   parcanin EN IYI secenegine distance / kosegen
  en_iyi_aci    most iyi secenegin yonuyle angle (derece/180)
  rakip_5mm     5mm inside, skoru BUNDAN YUKSEK ayrik candidate count
"""
import numpy as np 

OZ_AD =["log_secenek","log_aday","aday_secenek","aday_sira","aday_fark",
"en_iyi_uzak","en_iyi_aci","rakip_5mm"]

RAKIP_R =5.0 # yerel rekabet yaricapi (mm) -- NMS with same scale


def _birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    return V /np .maximum (np .linalg .norm (V ,axis =1 ,keepdims =True ),1e-12 )


def oznitelik (P ,D ,idx ,s ,diag ):
    """(n, 8). `P`,`D` SECENEK konum/yonu; `idx` each secenegin candidate indeksi."""
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    D =_birim (D )
    idx =np .asarray (idx ,int ).ravel ()
    s =np .asarray (s ,float ).ravel ()
    n =len (P )
    X =np .zeros ((n ,len (OZ_AD )))
    if not n :
        return X 
    diag =max (float (diag ),1e-6 )

    X [:,0 ]=np .log1p (n )
    ayrik =np .unique (idx )
    X [:,1 ]=np .log1p (len (ayrik ))

    # --- ADAY ICI: same adaya ait options between order and difference
    # `idx` ARDISIK OLMAYABILIR; sirali gruplama for yeniden etiketle.
    _ ,ters =np .unique (idx ,return_inverse =True )
    n_ad =ters .max ()+1 
    en_iyi =np .full (n_ad ,-np .inf )
    np .maximum .at (en_iyi ,ters ,s )
    cnt_ =np .bincount (ters ,minlength =n_ad )
    X [:,2 ]=cnt_ [ters ]
    X [:,4 ]=s -en_iyi [ters ]

    # candidate ici order: before adaya, after skora according to sirala
    duz =np .lexsort ((-s ,ters ))
    rank_ =np .empty (n ,float )
    k =0 
    while k <n :
        j =k 
        a =ters [duz [k ]]
        while j <n and ters [duz [j ]]==a :
            j +=1 
        rank_ [duz [k :j ]]=np .arange (j -k )
        k =j 
    X [:,3 ]=rank_ 

    # --- PARCANIN EN IYI SECENEGI with iliski
    b =int (np .argmax (s ))
    X [:,5 ]=np .linalg .norm (P -P [b ],axis =1 )/diag 
    X [:,6 ]=np .degrees (np .arccos (np .clip (D @D [b ],-1 ,1 )))/180.0 

    # --- YEREL REKABET: 5mm inside skoru more high AYRIK candidate count
    # candidate basina most high score and temsili konum
    ad_p =np .zeros ((n_ad ,3 ))
    ad_p [ters ]=P # same adayin secenekleri same konumda
    if n_ad <=4000 :
        dm =np .linalg .norm (ad_p [:,None ,:]-ad_p [None ,:,:],axis =-1 )
        yakin =dm <=RAKIP_R 
        np .fill_diagonal (yakin ,False )
        daha_iyi =en_iyi [None ,:]>en_iyi [:,None ]
        X [:,7 ]=(yakin &daha_iyi ).sum (1 )[ters ]
    return X 
