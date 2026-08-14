# -*- coding: utf-8 -*-
"""B4: ACIKLIK-SEKILLI OTO-ETIKET -- disk instead of GERCEK kesit.

SORUN: `g5_mouth_label.agiz_etiketle` each CP'ye axis along DISK boyar (olculen
yaricapla). Kare / yarik / push-in girislerde this YANLIS geometridir: disk ya agzin
kosMelerini kacirir ya da cevresindeki govdeyi boyar. Sonuc, oz-tutarlilik kapisinin
that parcalari elemesi -- and full da that morfolojiler (SE %95 silindirsiz, NIT silindirleri
0.5mm pah) urunun most kotu oldugu yer.

COZUM: GT CP'sine most yakin B-rep acikliginin GERCEK POLIGONUNU kullan. Nokta, mouth
duzleminde poligonun ICINDE whereas and axis bandindaysa boyanir. Aciklik bulunamazsa
ESKI disk davranisina returns -- i.e. no part KAYBEDILMEZ.

TEZ DEGISMEZ: same 5 sinif, same `v_o` turetmesi, same ~6000 remesh. Degisen single sey
etiketin SEKLI and that da GT + STEP geometrisinden geliyor, uydurma not.
"""
import numpy as np 

EKSEN_ONCE =1.0 
EKSEN_SONRA =6.0 
PAY =1.15 
ESLESME_MM =4.0 # GT CP'si with opening merkezi arasi most large distance


def _icinde (Q2 ,poly2 ):
    """2B point-in-poligon (isin atma), vektorel."""
    n =len (poly2 )
    ic =np .zeros (len (Q2 ),bool )
    j =n -1 
    for i in range (n ):
        xi ,yi =poly2 [i ];xj ,yj =poly2 [j ]
        kes =((yi >Q2 [:,1 ])!=(yj >Q2 [:,1 ]))&(
        Q2 [:,0 ]<(xj -xi )*(Q2 [:,1 ]-yi )/(yj -yi +1e-12 )+xi )
        ic ^=kes 
        j =i 
    return ic 


def sekilli_etiketle (V ,mesh ,G ,Gd ,acikliklar ,CE ,HOUSING =0 ,disk_fallback =None ):
    """Doner: (labels, kac_CP_sekilli_boyandi)."""
    lab =np .full (len (V ),HOUSING ,np .int64 )
    nsek =0 
    A =acikliklar or []
    C =np .asarray ([a ["center"]for a in A ],float )if A else None 
    for p ,d in zip (G ,Gd ):
        nd =float (np .linalg .norm (d ))
        if nd <1e-9 :
            continue 
        d =d /nd 
        sec =None 
        if C is not None :
            dd =np .linalg .norm (C -p ,axis =1 )
            j =int (np .argmin (dd ))
            if dd [j ]<=ESLESME_MM and abs (float (np .asarray (A [j ]["normal"],float )@d ))>=0.7 :
                sec =A [j ]
        if sec is None :
            continue # this CP'yi disk yolu boyayacak
        m =np .asarray (sec ["center"],float )
        n =np .asarray (sec ["normal"],float )
        if float (n @d )<0 :
            n =-n 
        poly =np .asarray (sec ["poligon"],float )
        if len (poly )<3 :
            continue 
            # mouth duzleminde 2B baseline
        e1 =np .cross (n ,[0.0 ,0 ,1.0 ])
        if np .linalg .norm (e1 )<1e-6 :
            e1 =np .cross (n ,[0.0 ,1.0 ,0 ])
        e1 /=(np .linalg .norm (e1 )+1e-12 )
        e2 =np .cross (n ,e1 )
        rel =V -m 
        eks =rel @n 
        msk_band =(eks >=-EKSEN_ONCE )&(eks <=EKSEN_SONRA )
        if not msk_band .any ():
            continue 
        Q =rel [msk_band ]
        Q2 =np .column_stack ([Q @e1 ,Q @e2 ])/PAY 
        P2 =np .column_stack ([(poly -m )@e1 ,(poly -m )@e2 ])
        ic =_icinde (Q2 ,P2 )
        idx =np .where (msk_band )[0 ][ic ]
        if len (idx ):
            lab [idx ]=CE 
            nsek +=1 
    return lab ,nsek 
