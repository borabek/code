# -*- coding: utf-8 -*-
"""IKINCI KADEME: kisa listeye PAHALI FIZIKSEL mouth olculeri.

RATIONALE: this oturumda whereas yarayan single selector kolu C3'tu and orada kazanci saglayan
sey OGRENME DEGIL FIZIKTI -- iceri/disari isin mesafelerinin tutarli okunmasi
FP'yi %45 dusurdu. Ilk kademe tanimlayicilarinda only 9 olcu present and all of them
single isin / single radius duzeyinde.

Pahali olculeri TUM havuzda (100+ candidate/part) hesaplamak mumkun not; but gate
esigini gecen KISA LISTEDE (~10-30 candidate/part) mumkun. Bu modul that olculeri gives.

OLCULER (all of them isin tabanli, GT KULLANMAZ):
  profil_ort/std   mouth ekseninde different derinliklerde SERBEST YARICAP profili.
                   Tel kanali depth along OPEN kalir; vida deligi daralir,
                   dis acilmis delikte radius SALINIR.
  daralma          most dip serbest radius / agizdaki serbest radius
  dis_izi          radius profilinin salinim genligi (dis izi -> high)
  wall            mouth kenarindan DISARI, eksene dik yonde first yuzeye uzaklik
  koniklik         yaricabin derinlikle degisim egimi (pah/koni)
  karsi_agiz       axis along govdeyi gecip cikan isin present mi (hole mi kor mu)
  halka_duzlugu    mouth cevresindeki isin uzunluklarinin degisimi (duz mouth -> low)
  govde_orani      mouth merkezinden part merkezine uzaklik / part yarikosegeni

TEZE SADIK: only ADAY PUANLAMA; konum/direction uretimi and segmentasyon does not change.
"""
import numpy as np 

AD =["profil_ort","profil_std","daralma","dis_izi","duvar","koniklik",
"karsi_agiz","halka_duzlugu","govde_orani"]
DERINLIKLER =(0.5 ,1.5 ,3.0 ,5.0 ,8.0 )# mm, agizdan iceri
ISIN_SAYISI =8 # halka basina isin


def _birim (v ):
    n =np .linalg .norm (v )
    return v /n if n >1e-9 else None 


def _dik (ax ):
    a =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (float (ax @a ))>0.9 :
        a =np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (ax ,a )
    n =np .linalg .norm (u )
    if n <1e-12 :
        return np .array ([0.0 ,1.0 ,0.0 ]),np .array ([0.0 ,0.0 ,1.0 ])
    u =u /n 
    return u ,np .cross (ax ,u )


def _icerde (mesh ,O ,D ,uzak ):
    """Nokta MALZEMENIN ICINDE mi? Tek yonde kesisim count TEK whereas icerde.

    WHY GEREKLI: serbest radius olcerken isin full malzemeden baslarsa dis
    duvara up to gider and "channel genis" like gorunur. Ilk surumde full kutu,
    delikli kutudan DAHA OPEN measured (12.07 vs 2.00) and testi patlatti.
    Icerdeki ornek noktalarin serbest yaricapi SIFIR sayilir.
    """
    O =np .asarray (O ,float ).reshape (-1 ,3 )
    if not len (O ):
        return np .zeros (0 ,bool )
    d =np .asarray (D ,float ).reshape (-1 ,3 )
    _loc ,ir ,_t =mesh .ray .intersects_location (O ,d ,multiple_hits =True )
    say =np .zeros (len (O ),int )
    for r in ir :
        say [r ]+=1 
    return (say %2 )==1 


def _mesafe (mesh ,O ,D ,uzak ):
    out =np .full (len (O ),float (uzak ))
    if not len (O ):
        return out 
    loc ,ir ,_t =mesh .ray .intersects_location (np .asarray (O ,float ),
    np .asarray (D ,float ),
    multiple_hits =True )
    if len (loc ):
        d =np .linalg .norm (loc -np .asarray (O ,float )[ir ],axis =1 )
        for j ,r in enumerate (ir ):
            if d [j ]<out [r ]:
                out [r ]=d [j ]
    return out 


def tanimla (P ,D ,mesh ,diag ,merkez =None ):
    """(n, 9) matris. `D` DISARI bakan direction (sign duzeltmesinden SONRA)."""
    P =np .asarray (P ,float ).reshape (-1 ,3 )
    D =np .asarray (D ,float ).reshape (-1 ,3 )
    n =len (P )
    X =np .zeros ((n ,len (AD )))
    if n ==0 :
        return X 
    uzak =float (diag )
    if merkez is None :
        merkez =np .asarray (mesh .vertices ,float ).mean (0 )
    yari =0.5 *uzak 
    for i in range (n ):
        d =_birim (D [i ])
        if d is None :
            continue 
        u ,v =_dik (d )
        aci =np .linspace (0 ,2 *np .pi ,ISIN_SAYISI ,endpoint =False )
        yon_halka =np .array ([np .cos (a )*u +np .sin (a )*v for a in aci ])
        # DERINLIK PROFILI: each derinlikte, eksenden DISA correct isinlarla
        # serbest radius olculur (channel ne up to open).
        prof =[]
        for t in DERINLIKLER :
            nok =P [i ]-t *d 
            # Ornek point DOLU malzemedeyse orada channel YOKTUR -> serbest radius 0
            if _icerde (mesh ,nok [None ],d [None ],uzak )[0 ]:
                prof .append (0.0 )
                continue 
            O =np .repeat (nok [None ],ISIN_SAYISI ,axis =0 )
            prof .append (float (np .median (_mesafe (mesh ,O ,yon_halka ,uzak ))))
        prof =np .asarray (prof )
        X [i ,0 ]=prof .mean ()
        X [i ,1 ]=prof .std ()
        X [i ,2 ]=prof [-1 ]/max (prof [0 ],1e-6 )
        X [i ,3 ]=float (np .abs (np .diff (prof )).mean ())
        # DUVAR: agizdan DISA, eksene dik -> first yuzeye uzaklik
        O =np .repeat ((P [i ]+0.2 *d )[None ],ISIN_SAYISI ,axis =0 )
        X [i ,4 ]=float (np .median (_mesafe (mesh ,O ,yon_halka ,uzak )))
        # KONIKLIK: yaricabin derinlikle egimi
        t =np .asarray (DERINLIKLER )
        X [i ,5 ]=float (np .polyfit (t ,prof ,1 )[0 ])if len (t )>1 else 0.0 
        # KARSI AGIZ: govdeyi bastan sona gecen isin
        X [i ,6 ]=float (_mesafe (mesh ,(P [i ]-0.05 *d )[None ],(-d )[None ],
        uzak )[0 ]>=uzak *0.98 )
        # HALKA DUZLUGU: mouth cevresi isin uzunluklarinin degisimi
        O =np .repeat ((P [i ]+0.05 *d )[None ],ISIN_SAYISI ,axis =0 )
        h =_mesafe (mesh ,O ,yon_halka ,uzak )
        X [i ,7 ]=float (h .std ()/max (h .mean (),1e-6 ))
        X [i ,8 ]=float (np .linalg .norm (P [i ]-merkez )/max (yari ,1e-6 ))
    return X 
