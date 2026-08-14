# -*- coding: utf-8 -*-
"""SIRA DAMGALAMA: two-three capadan TUM SIRAYI head.

RATIONALE. Yogun klemenste (NIT 24 CP/part) candidate-basina ranking 12x'te
tikaniyor: 4467 secenek inside 24 dogruyu uste tasimak for ~50-100x gerek.
Ama measured ki GT'lerin **%90.8'i** parcanin most sik OTELEME VEKTORUYLE baska
a GT'ye ulasiyor. Yani kontaklarin yeri BIRBIRINDEN cikarilabilir.

Aday-basina karar instead of SIRA-DUZEYI karar: birkac YUKSEK GUVENLI capa bul,
onlardan oteleme vektorunu cikar, sirayi two yone correct uzat and each ongorulen
noktayi UCUZ FIZIK TESTIYLE dogrula (mouth orada gercekten present mi). Kabul
edilen noktanin YONU capadan KOPYALANMAZ -- direction always direction bankasindan
secilir (K2.1 full here error yapmisti: tespit +0.0126 but robot -0.0100).

Bu modul a ONERI URETICIDIR; ciktisi mevcut secimin UZERINE eklenir and
kabul/ret yine skor + NMS kurallarindan gecer.
"""
import numpy as np 

import lattice 

MIN_CAPA =2 # oteleme cikarmak for at least this up to capa
MAKS_ADIM =12 # sirayi kac step uzat (each two yone)
TOL_MM =1.0 # ongorulen point with pool adayi arasi kabul mesafesi


def _birim (V ):
    V =np .asarray (V ,float ).reshape (-1 ,3 )
    return V /np .maximum (np .linalg .norm (V ,axis =1 ,keepdims =True ),1e-12 )


def oneriler (P_capa ,D_capa ,P_havuz ,maks_adim =MAKS_ADIM ,tol =TOL_MM ):
    """Capalardan sirayi uzat, HAVUZDA KARSILIGI OLAN ongorulen noktalari don.

    Doner: (idx_havuz, kaynak_capa, step) -- `idx_havuz` havuzdaki candidate
    indeksleridir. YENI KONUM URETILMEZ: only havuzda ZATEN VAR OLAN but
    low skorlu candidates "order on" diye ONE CIKARILIR. Boylece konum
    dogrulugu havuzun garantisiyle sinirli kalir and uydurma point olusmaz.
    """
    P_capa =np .asarray (P_capa ,float ).reshape (-1 ,3 )
    P_havuz =np .asarray (P_havuz ,float ).reshape (-1 ,3 )
    if len (P_capa )<MIN_CAPA or not len (P_havuz ):
        return np .zeros (0 ,int ),np .zeros (0 ,int ),np .zeros (0 ,float )
    tler =lattice .otelemeler (P_capa )
    if not tler :
        return np .zeros (0 ,int ),np .zeros (0 ,int ),np .zeros (0 ,float )
    idx ,kayn ,adim =[],[],[]
    gorulen =set ()
    # ALT ADIMLAR: 1/2 VE 1/3. Birim test showed ki capalar 18mm arayla
    # dususe most sik oteleme 18mm cikiyor; half step 9mm gives but real step
    # 6mm'dir and aradaki kontaklar HIC ongorulmez. Ucte-a step this bosluga
    # bakar. (`lattice.ongorulen` de same duzeltmeyi carries.)
    kesir =sorted ({x /b for b in (1 ,2 ,3 )
    for x in range (1 ,b *maks_adim +1 )})
    for t ,_n in tler :
        for k in kesir :
            for sg in (1.0 ,-1.0 ):
                S =P_capa +sg *k *t 
                d =np .linalg .norm (P_havuz [:,None ,:]-S [None ,:,:],axis =-1 )
                j =np .argmin (d ,axis =1 )
                m =d [np .arange (len (P_havuz )),j ]
                for i in np .where (m <=tol )[0 ]:
                    if i in gorulen :
                        continue 
                    gorulen .add (i )
                    idx .append (int (i ))
                    kayn .append (int (j [i ]))
                    adim .append (float (sg *k ))
    return (np .asarray (idx ,int ),np .asarray (kayn ,int ),
    np .asarray (adim ,float ))


def oznitelik (P_havuz ,D_havuz ,P_capa ,D_capa ,**kw ):
    """Havuzdaki each candidate for (sira_uzerinde, step, capa_yon_uyumu).

    `lattice.feature` a MESAFE olcusu gives; this whereas KABUL EDILMIS a order
    uyeligi bayragidir -- ikisi different sorulari yanitlar and birlikte is used.
    """
    n =len (np .asarray (P_havuz ,float ).reshape (-1 ,3 ))
    X =np .zeros ((n ,3 ))
    i ,c ,a =oneriler (P_capa ,D_capa ,P_havuz ,**kw )
    if not len (i ):
        return X 
    X [i ,0 ]=1.0 
    X [i ,1 ]=np .abs (a )
    Dh =_birim (D_havuz )
    Dc =_birim (D_capa )
    X [i ,2 ]=np .abs ((Dh [i ]*Dc [c ]).sum (1 ))
    return X 
