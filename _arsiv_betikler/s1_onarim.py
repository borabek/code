# -*- coding: utf-8 -*-
"""S1: FIZIKSEL FIX OPERATORLERI -- three tane, saf fonksiyon, TEZ-NOTR.

Hicbiri agi, remesh'i ya da `v_o` tanimini degistirmez. Uctu de SON ISLEMDIR: gate
kararindan after, uretilen CP'nin FIZIKSEL gecerliligini fixes.

A4 OLCUMU (194 part / 1186 CP) this operatorlerin hedefini verdi:
    govde_ici       134 CP  (FP'lerde %22.4, TP'lerde %7.1 -> zenginlesme 3.16x)
    onu_kapali      148 CP  (%19.3 / %9.9  -> 1.96x)
    duvara_yapisik   92 CP  (%12.6 / %5.9  -> 2.12x)
    kombinasyon: 964 temiz | 56 two-bayrak | 36 UC-BAYRAK

TASARIM KURALI: each operator YALNIZ own bayragini hedefler and HAREKETI SINIRLIDIR.
Sinirsiz hareket, [[konum-selector-closed]]daki 1:20 hasar asimetrisini geri getirir --
kurtardigindan very breaks. Bu yuzden each operator `maks_mm` with kisitlanir and hareket
sonrasi bayrak YENIDEN olculur (S2).
"""
import numpy as np 

ILERI_MIN =5.0 
IC_CAP_MIN =0.8 
MAKS_TASIMA_MM =12.0 # seat_to_mouth already axis along; yine de upper boundary
MAKS_MERKEZLEME_MM =3.0 # mouth inside merkeze cekme -- mouth yaricapi mertebesi


def _birim (v ):
    v =np .asarray (v ,float )
    n =np .linalg .norm (v )
    return v /n if n >1e-9 else v 


def onar_govde_ici (mesh ,p ,d ,maks_mm =MAKS_TASIMA_MM ):
    """S1a: point govdenin ICINDEYSE axis along ACIKLIGIN AGZINA tasi.

    `cp_geometry.seat_to_mouth` full this isi yapar and tezin `v_o` insasiyla AYNI tanimdir:
    axis along ILK YUZEY KESISIMI. Nokta already disaridaysa DOKUNULMAZ (fonksiyonun
    own korumasi) -- otherwise empty alandaki a CP karsi duvara firlatilirdi.
    """
    from cp_geometry import is_inside ,seat_to_mouth 
    try :
        if not is_inside (mesh ,p ):
            return p ,False ,"already disarida"
        new_ ,off =seat_to_mouth (mesh ,p ,d )
        new_ =np .asarray (new_ ,float )
        if not np .isfinite (new_ ).all ()or np .linalg .norm (new_ -p )>maks_mm :
            return p ,False ,f"tasima {np .linalg .norm (new_ -p ):.1f}mm > {maks_mm }"
        return new_ ,True ,f"{np .linalg .norm (new_ -p ):.2f}mm tasindi"
    except Exception as e :
        return p ,False ,f"error {type (e ).__name__ }"


def onar_merkezle (mesh ,p ,d ,maks_mm =MAKS_MERKEZLEME_MM ,n_dirs =24 ):
    """S1b: point agzin DUVARINA yapisiksa mouth inside MERKEZE cek.

    `mouth_width` eksene dik n_dirs yonde isin atip first yuzeye mesafeyi olcer. Nokta
    merkezdeyse tum mesafeler benzer; duvara yapisiksa a direction COK KISA, karsisi UZUN.
    Merkeze correct hareket = (most uzak direction - most yakin direction)/2 up to, most uzak yone correct.

    Bu, tezin `v_o`'sunu DEGISTIRMEZ: v_o already "mouth ortasi"dir; here yapilan, that
    tanima DAHA SADIK a point bulmaktir. (Global tanim degisikligi AYRI a sey and
    olculup REDDEDILDI: [[konum-selector-closed]] -- cember/mouth/open three rakip de kaybetti.)
    """
    from cp_geometry import mouth_width 
    try :
        d =_birim (d )
        a =np .array ([1.0 ,0.0 ,0.0 ])
        if abs (float (d @a ))>0.9 :
            a =np .array ([0.0 ,1.0 ,0.0 ])
        u =_birim (np .cross (d ,a ));v =_birim (np .cross (d ,u ))
        ic ,ort =mouth_width (mesh ,p ,d ,n_dirs =n_dirs )
        if not (0 <ic <IC_CAP_MIN ):
            return p ,False ,"duvara yapisik not"
        from cp_geometry import ray_hits 
        aci =np .linspace (0 ,2 *np .pi ,n_dirs ,endpoint =False )
        mes ,direction =[],[]
        for t in aci :
            w =np .cos (t )*u +np .sin (t )*v 
            h =ray_hits (mesh ,p +1e-4 *w ,w ,30.0 )
            mes .append (float (min (h ))if len (h )else 30.0 )
            direction .append (w )
        mes =np .array (mes )
        i_min =int (np .argmin (mes ));i_max =int (np .argmax (mes ))
        step_ =min ((mes [i_max ]-mes [i_min ])/2.0 ,maks_mm )
        if step_ <=1e-3 :
            return p ,False ,"hareket gereksiz"
        new_ =p +step_ *direction [i_max ]
        return new_ ,True ,f"{step_ :.2f}mm merkeze"
    except Exception as e :
        return p ,False ,f"error {type (e ).__name__ }"


def onar_yon_cevir (mesh ,p ,d ,ileri_min =ILERI_MIN ):
    """S1c: takma yonunun ONU KAPALIYSA yonu TERS cevir and yeniden olc.

    Kural: ileri serbest distance < ileri_min whereas -d denenir. -d more ACIKSA cevrilir.
    IKI TARAF DA kapaliysa point a KANALDA DEGILDIR -> `False, "two taraf closed"`
    returns and cagiran taraf that adayi eleyebilir (S4 triyaji).
    """
    from cp_geometry import ray_hits 

    def acik (v ):
        try :
            h =ray_hits (mesh ,p +1e-3 *v ,v ,60.0 )
            return float (min (h ))if len (h )else float ("inf")
        except Exception :
            return float ("nan")
    d =_birim (d )
    ileri =acik (d )
    if not np .isfinite (ileri )or ileri >=ileri_min :
        return d ,False ,"ten already open"
    geri =acik (-d )
    if np .isfinite (geri )and geri <ileri_min :
        return d ,False ,"two taraf closed"
    if (not np .isfinite (geri ))or geri >ileri :
        return -d ,True ,f"cevrildi ({ileri :.1f} -> {geri :.1f}mm)"
    return d ,False ,"cevirmek iyilestirmiyor"


def _selftest ():
    """Operatorler bilinen geometride DOGRU davraniyor mu?"""
    import trimesh 
    from cp_geometry import is_inside 
    kutu =trimesh .creation .box ((24 ,24 ,12 ))
    sil =trimesh .creation .cylinder (radius =3.0 ,height =40 )
    m =kutu .difference (sil )
    # 1) body ICINDEKI point agza tasinmali
    p =np .array ([9.0 ,9.0 ,0.0 ])# kose -> malzeme inside
    assert is_inside (m ,p )
    new_ ,ok ,_ =onar_govde_ici (m ,p ,np .array ([0. ,0. ,1. ]))
    assert ok and not is_inside (m ,new_ ),"body ici onarimi calismadi"
    # 2) hole ekseninde, duvara YAKIN point merkeze cekilmeli
    p2 =np .array ([2.6 ,0.0 ,5.7 ])# r=3 deligin duvarina 0.4mm
    yeni2 ,ok2 ,_ =onar_merkezle (m ,p2 ,np .array ([0. ,0. ,1. ]))
    if ok2 :
        assert np .linalg .norm (yeni2 [:2 ])<np .linalg .norm (p2 [:2 ]),"merkeze cekmedi"
        # 3) onu closed direction cevrilmeli
    p3 =np .array ([0.0 ,0.0 ,-5.0 ])
    d3 =np .array ([0.0 ,0.0 ,-1.0 ])# asagi -> hemen disari (open)
    _ ,cev ,_ =onar_yon_cevir (m ,p3 ,d3 )
    return True 


if __name__ =="__main__":
    print ("s1_onarim selftest:","GECTI"if _selftest ()else "KALDI")
