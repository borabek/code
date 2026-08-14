# -*- coding: utf-8 -*-
"""STEP'in ANALITIK silindir eksenleri -- CP yonunu prediction etmek instead of OKUMAK for.

WHY: axis yonu bugun measured and robot-hazir F1'in 0.109'unu single basina that tutuyor
(lateral <=2mm 0.6043 -> axis sarti eklenince 0.4955). Mesh'ten axis kestirmenin two tavani present:
ray-probe hole koni acisindan (atan(r/L) ~ 11 derece) more keskin olamaz, normal-kovaryans whereas
only silindirik kanallarda tanimli (300 candidate "silindirik not" diye reddedildi).
Oysa STEP dosyasi silindir yuzeyleri ZATEN analitik tasiyor -- prediction etmeye gerek absent.

WHY gmsh, neden regex DEGIL: first deneme STEP metnini regex with okudu and CP'den silindire
distance medyan 21mm / %75'i 111mm output. Sebep MONTAJ DONUSUMU: AXIS2_PLACEMENT_3D koordinatlari
parcanin YEREL cercevesinde yazili. gmsh'in OCC cekirdegi donusumleri uygular; dogrulandi:
3211485 for global bbox [0,0,0]-[8.2,74.1,42.2], meshin bbox'i with BIREBIR same.

Eksen, yuzeyin ORNEKLENEN NORMALLERINDEN cikarilir (silindirde normaller eksene diktir), so
gmsh surumune ozgu a "axis dondur" API'sine bagimli kalinmaz. Olculdu: silindiriklik
(ev0/ev1) full silindirlerde 0.0000.
"""
import os ,json ,hashlib 
import numpy as np 

CACHE_DIR ="results/brep_cache"
import collections as _collections 
REJECT =_collections .Counter ()# tani: hangi gate kac times None dondurdu


def _oku (yol ):
    """JSON'u KAPATARAK oku. `json.load(open(x))` deseni file tanitici SIZDIRIR: this fonksiyon
    candidate basina cagriliyor and 200 parcalik a olcumde process tanitici tukenmesinden SESSIZCE
    became (2026-07-31, t8 kosusu; log binlerce ResourceWarning with doluydu)."""
    with open (yol ,"r",encoding ="utf-8")as f :
        return json .load (f )


def _yaz (yol ,veri ):
    with open (yol ,"w",encoding ="utf-8")as f :
        json .dump (veri ,f )


CYL_VER =2 # v2: radius/centre CEMBER OTURTMA with (v1 centroid kullaniyordu, YAY'larda wrong)


def _key (step_path ,ver =""):
    st =os .stat (step_path )
    h =hashlib .md5 (f"{os .path .abspath (step_path )}|{st .st_size }|{int (st .st_mtime )}".encode ()).hexdigest ()
    return os .path .join (CACHE_DIR ,h +ver +".json")


def _fit_circle (P ,ax ):
    """Eksene DIK duzlemde most small kareler cember oturtma -> (axis uzeri point, radius).

    WHY GEREKLI: STEP delikleri neredeyse always dikis cizgisinden BOLUNMUS silindirik
    yuzler as yazilir -- measured: a parcadaki silindirik yuzlerin %0'i full kind, u-araligi
    ceyrek (1.57) and half (3.14) turlarda yigiliyor. Yay uzerindeki noktalarin ORTALAMASI eksenin
    UZERINDE DEGILDIR; centroid'e which is medyan uzaklik da real radius degildir. Olculen sonuc:
    metin 0.25mm iken centroid tahmini 0.07mm (3.5 fold small).

    Bu, three yeri birden bozuyordu: robota giden `size_mm`, `axis_at`'in r_range filtresi (radius
    small kestirilince mesru silindirler eleniyordu) and `max_off_mm` distance testi (point eksende
    degildi). Cember oturtma yaya karsi bagisiktir: a yay bile merkezi and yaricapi belirler.
    """
    ax =ax /(np .linalg .norm (ax )+1e-12 )
    e1 =np .array ([1.0 ,0.0 ,0.0 ])
    if abs (ax @e1 )>0.9 :
        e1 =np .array ([0.0 ,1.0 ,0.0 ])
    e1 =e1 -(e1 @ax )*ax ;e1 /=np .linalg .norm (e1 )+1e-12 
    e2 =np .cross (ax ,e1 )
    o =P .mean (0 )
    rel =P -o 
    x =rel @e1 ;y =rel @e2 
    # Kasa cebirsel cember: x^2+y^2 + D x + E y + F = 0
    A =np .column_stack ([x ,y ,np .ones_like (x )])
    b =-(x **2 +y **2 )
    try :
        D ,E ,F =np .linalg .lstsq (A ,b ,rcond =None )[0 ]
    except np .linalg .LinAlgError :
        return o ,float (np .median (np .linalg .norm (rel -(rel @ax )[:,None ]*ax ,axis =1 )))
    cx ,cy =-D /2.0 ,-E /2.0 
    r2 =cx *cx +cy *cy -F 
    if not np .isfinite (r2 )or r2 <=0 :
        return o ,float (np .median (np .linalg .norm (rel -(rel @ax )[:,None ]*ax ,axis =1 )))
    ctr =o +cx *e1 +cy *e2 +((rel @ax ).mean ())*ax 
    return ctr ,float (np .sqrt (r2 ))


def cylinders (step_path ,use_cache =True ):
    """Doner: (N,3) axis-uzeri point, (N,3) unit axis, (N,) radius -- GLOBAL koordinatlarda.

    Sonuc diske onbelleklenir: gmsh with STEP okumak part basina saniyeler suruyor and same
    part measurement along defalarca geliyor.
    """
    ck =_key (step_path ,f"_c{CYL_VER }")
    if use_cache and os .path .exists (ck ):
        try :
            d =_oku (ck )
            return (np .array (d ["c"],float ).reshape (-1 ,3 ),
            np .array (d ["a"],float ).reshape (-1 ,3 ),
            np .array (d ["r"],float ))
        except Exception :
            pass 
    import gmsh 
    C ,A ,R =[],[],[]
    gmsh .initialize ()
    try :
        gmsh .option .setNumber ("General.Terminal",0 )
        gmsh .model .occ .importShapes (step_path )
        gmsh .model .occ .synchronize ()
        for dim ,tag in gmsh .model .getEntities (2 ):
            try :
                if gmsh .model .getType (dim ,tag )!="Cylinder":
                    continue 
                b =gmsh .model .getParametrizationBounds (dim ,tag )
                us =np .linspace (b [0 ][0 ],b [1 ][0 ],7 )[1 :-1 ]
                vs =np .linspace (b [0 ][1 ],b [1 ][1 ],5 )[1 :-1 ]
                pars =np .array ([[u ,v ]for u in us for v in vs ],float ).ravel ()
                if not len (pars ):
                    continue 
                N =np .asarray (gmsh .model .getNormal (tag ,pars ),float ).reshape (-1 ,3 )
                P =np .asarray (gmsh .model .getValue (dim ,tag ,pars ),float ).reshape (-1 ,3 )
                if len (N )<4 :
                    continue 
                M =N .T @N /len (N )
                ev ,evec =np .linalg .eigh (M )
                ax =evec [:,0 ]/(np .linalg .norm (evec [:,0 ])+1e-12 )
                ctr ,rad =_fit_circle (P ,ax )
                if not np .isfinite (rad )or rad <=1e-6 :
                    continue 
                C .append (ctr );A .append (ax );R .append (rad )
            except Exception :
                continue 
    finally :
    # finally SART: single meshlenemeyen STEP'ten after gmsh durumu bozulup TUM parcalari
    # surunduruyordu (gmsh-finalize-leak, 2026-07-28).
        try :
            gmsh .finalize ()
        except Exception :
            pass 
    C =np .asarray (C ,float ).reshape (-1 ,3 )
    A =np .asarray (A ,float ).reshape (-1 ,3 )
    R =np .asarray (R ,float )
    if use_cache :
        os .makedirs (CACHE_DIR ,exist_ok =True )
        try :
            _yaz (ck ,{"c":C .tolist (),"a":A .tolist (),"r":R .tolist ()})
        except Exception :
            pass 
    return C ,A ,R 


def axis_at (point ,seed_dir ,cyl ,max_off_mm =3.0 ,max_turn_deg =40.0 ,r_range =(0.5 ,12.0 ),
want_radius =False ):
    """CP'nin on durdugu silindiri bul and onun ANALITIK eksenini dondur (otherwise None).

    Eslestirme: CP, silindirin EKSEN CIZGISINE yakin must be (hole agzinda CP eksendedir).
    Birden extra candidate varsa seed yone most yakin axis secilir -- i.e. B-rep karari gives,
    seed only esitlik breaks.

    max_turn_deg: B-rep ekseni tohumdan this up to very sapiyorsa muhtemelen BASKA a silindire
    (komsu delige, vidaya) eslesmisizdir -> None. Kor correction yapilmaz.
    """
    C ,A ,R =cyl 
    if not len (C ):
        return (None ,None )if want_radius else None 
    p =np .asarray (point ,float )
    s =np .asarray (seed_dir ,float )
    ns =np .linalg .norm (s )
    if ns <1e-12 :
        return (None ,None )if want_radius else None 
    s =s /ns 
    rel =p -C 
    al =(rel *A ).sum (1 )
    off =np .linalg .norm (rel -al [:,None ]*A ,axis =1 )# axis cizgisine dik distance
    ok =(off <=max_off_mm )&(R >=r_range [0 ])&(R <=r_range [1 ])
    if not ok .any ():
        return (None ,None )if want_radius else None 
    cand =np .where (ok )[0 ]
    cos =np .abs (A [cand ]@s )
    j =cand [int (np .argmax (cos ))]
    ang =np .degrees (np .arccos (min (1.0 ,float (np .abs (A [j ]@s )))))
    if ang >max_turn_deg :
        return (None ,None )if want_radius else None 
    d =A [j ]
    d =d if float (np .dot (d ,s ))>=0 else -d 
    # Yaricap da ANALITIK: mouth genisligini isinla olcmek instead of silindirin own yaricapini
    # kullanmak more conclusive. Isinla measurement, point channel merkezinde degilse EN KUCUK mesafeyi
    # aliyor and 0.05mm like fiziksel olmayan degerler uretebiliyordu (1070018'de 4/13 CP).
    return (d ,float (R [j ]))if want_radius else d 


def planes (step_path ,use_cache =True ):
    """STEP'in DUZLEM yuzeyleri: (centre, normal, radius) -- GLOBAL koordinatlarda.

    WHY: `cylinders()` only silindirleri okuyor, but acikliklarin a kismi YUVA/KELEPCE
    girisi -- silindir not, DUZLEM duvarlardan olusan a channel. Olculdu: normal-kovaryans
    yontemi 300 adayi "silindirik not" diye reddetti and that adaylarda axis hatasi high kaldi.
    Bir yuva kanalinin ekseni de wall normallerinin HEPSINE diktir; i.e. same fizik, different
    surface tipi. gmsh 3211485'te 217 yuzeyin 171'ini Plane as veriyor.

    'radius' = surface ornek noktalarinin merkezden most large uzakligi; CP'nin SONSUZ duzleme
    not GERCEK yuze yakin olup olmadigini elemek for gerekli.
    """
    ck =_key (step_path ).replace (".json","_pl.json")
    if use_cache and os .path .exists (ck ):
        try :
            d =_oku (ck )
            return (np .array (d ["c"],float ).reshape (-1 ,3 ),
            np .array (d ["n"],float ).reshape (-1 ,3 ),
            np .array (d ["r"],float ))
        except Exception :
            pass 
    import gmsh 
    C ,N ,R =[],[],[]
    gmsh .initialize ()
    try :
        gmsh .option .setNumber ("General.Terminal",0 )
        gmsh .model .occ .importShapes (step_path )
        gmsh .model .occ .synchronize ()
        for dim ,tag in gmsh .model .getEntities (2 ):
            try :
                if gmsh .model .getType (dim ,tag )!="Plane":
                    continue 
                b =gmsh .model .getParametrizationBounds (dim ,tag )
                us =np .linspace (b [0 ][0 ],b [1 ][0 ],5 )[1 :-1 ]
                vs =np .linspace (b [0 ][1 ],b [1 ][1 ],5 )[1 :-1 ]
                pars =np .array ([[u ,v ]for u in us for v in vs ],float ).ravel ()
                if not len (pars ):
                    continue 
                P =np .asarray (gmsh .model .getValue (dim ,tag ,pars ),float ).reshape (-1 ,3 )
                nn =np .asarray (gmsh .model .getNormal (tag ,pars ),float ).reshape (-1 ,3 )
                if len (P )<3 :
                    continue 
                nrm =nn .mean (0 )
                ln =float (np .linalg .norm (nrm ))
                if ln <1e-9 :
                    continue 
                ctr =P .mean (0 )
                C .append (ctr );N .append (nrm /ln )
                R .append (float (np .linalg .norm (P -ctr ,axis =1 ).max ()))
            except Exception :
                continue 
    finally :
        try :
            gmsh .finalize ()
        except Exception :
            pass 
    C =np .asarray (C ,float ).reshape (-1 ,3 )
    N =np .asarray (N ,float ).reshape (-1 ,3 )
    R =np .asarray (R ,float )
    if use_cache :
        os .makedirs (CACHE_DIR ,exist_ok =True )
        try :
            _yaz (ck ,{"c":C .tolist (),"n":N .tolist (),"r":R .tolist ()})
        except Exception :
            pass 
    return C ,N ,R 


def axis_from_planes (point ,seed_dir ,pl ,max_dist_mm =6.0 ,min_faces =3 ,flat_ratio =0.35 ,
max_turn_deg =60.0 ):
    """Yuva kanalinin eksenini yakin DUZLEM duvarlarindan cikar (otherwise None).

    Kanal ekseni tum wall normallerine DIKTIR -> normal kovaryansinin most small ozvektoru.
    Terminalin DUZ ON YUZU de yakindir and normali eksene PARALEL oldugu for kovaryansi breaks;
    that is why seed yone ~paralel normalli yuzler ATILIR (silindir kolunda same tuzak yasandi:
    wrong half-uzay secimi olcumu bozup reddedilmesine path aciyordu).
    """
    C ,N ,R =pl 
    if not len (C ):
        return None 
    p =np .asarray (point ,float )
    s =np .asarray (seed_dir ,float )
    ns =float (np .linalg .norm (s ))
    if ns <1e-12 :
        return None 
    s =s /ns 
    d =np .linalg .norm (C -p ,axis =1 )
    near =d <=np .maximum (max_dist_mm ,R )# real yuze yakin mi (sonsuz duzleme not)
    # ON YUZ ELEMESI: normali eksene ~paralel which is duzlem channel duvari DEGIL, agzin kapagidir
    near &=np .abs (N @s )<0.80 
    if int (near .sum ())<min_faces :
        REJECT ["duzlem_yetersiz"]+=1 
        return None 
    n =N [near ]
    M =n .T @n /len (n )
    ev ,evec =np .linalg .eigh (M )
    if ev [1 ]<=1e-9 or ev [0 ]/max (ev [1 ],1e-12 )>flat_ratio :
        REJECT ["duzlem_kanal_degil"]+=1 
        return None 
    ax =evec [:,0 ]/(np .linalg .norm (evec [:,0 ])+1e-12 )
    if float (np .dot (ax ,s ))<0 :
        ax =-ax 
    if np .degrees (np .arccos (min (1.0 ,abs (float (np .dot (ax ,s ))))))>max_turn_deg :
        REJECT ["duzlem_tohumdan_uzak"]+=1 
        return None 
    REJECT ["duzlem_kabul"]+=1 
    return ax 


def axis_point (point ,seed_dir ,cyl ,max_off_mm =3.0 ,r_range =(0.5 ,12.0 ),max_turn_deg =60.0 ):
    """CP'yi eslesen silindirin EKSEN CIZGISI uzerine izdusur (axial konumu KORUYARAK).

    `axis_at` YONU returns; this NOKTAYI carries. Ikisi ayri: manufacturer ConnectionPoint'i acikligin
    ekseni UZERINDEDIR, but bizim noktamiz mesh kumesinin weight merkezidir and opening
    asimetrik ortuldugunde eksenden kayar.

    Secim kurali `axis_at` with AYNI must be -- baska a silindire izdusurmek noktayi
    tamamen baska a delige tasirdi.
    """
    C ,A ,R =cyl 
    if not len (C ):
        return None 
    p =np .asarray (point ,float )
    s =np .asarray (seed_dir ,float )
    ns =np .linalg .norm (s )
    if ns <1e-12 :
        return None 
    s =s /ns 
    rel =p -C 
    al =(rel *A ).sum (1 )
    off =np .linalg .norm (rel -al [:,None ]*A ,axis =1 )
    ok =(off <=max_off_mm )&(R >=r_range [0 ])&(R <=r_range [1 ])
    if not ok .any ():
        return None 
    cand =np .where (ok )[0 ]
    j =cand [int (np .argmax (np .abs (A [cand ]@s )))]
    if np .degrees (np .arccos (min (1.0 ,float (np .abs (A [j ]@s )))))>max_turn_deg :
        return None 
    return C [j ]+float ((p -C [j ])@A [j ])*A [j ]
