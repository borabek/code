# -*- coding: utf-8 -*-
"""CP GEOMETRI YARDIMCILARI -- tum measurement/gorsel araclarinin ORTAK kullandigi donusumler.

ANA GERCEK: URETICININ ConnectionPoint'i KONTAK YUVASINDADIR, govdenin ICINDE. Delik/mouth oradan
INSERT YONU along birkac mm oteded, yuzeydedir. Model AGZI prediction eder. GT'yi yuvada varsayan each
measurement wrong becomes (isaretci kati inside kalir, teshis wrong noktaya bakar, single yonlu axis
penceresi ters taraftaki agzi kacirir).

2026-07-28 KOK HATA VE COZUMU -- kopyalamadan before oku:
  trimesh'in contains() / ray.intersects_location() fonksiyonlarinin IKISI DE `rtree` paketine
  baglidir and this ortamda rtree KURULU DEGIL -> each cagri ModuleNotFoundError atiyordu. Eski kodda
  `except Exception: continue` bunu SESSIZCE yutuyor, fonksiyonlar no sey yapmadan girdiyi geri
  donduruyordu. Sonuc: yuva->mouth tasimasi never olmadi, disari-direction secimi never calismadi, and two
  fonksiyon da "ok" gorundu. Ayrica remesh'lenmis govdeler WATERTIGHT DEGIL -- contains()
  correct arac olmazdi already.
  Bu yuzden here DIS BAGIMLILIK YOK: Moller-Trumbore isin-ucgen kesisimi saf numpy with yazildi.
  Watertight olmayan meshlerde de works, sessizce failed OLMAZ.
"""
import numpy as np 


def _mesh_arrays (mesh ):
    """trimesh nesnesi ya da (V, F) ciftini kabul et."""
    if hasattr (mesh ,"vertices"):
        return np .asarray (mesh .vertices ,float ),np .asarray (mesh .faces ,np .int64 )
    V ,F =mesh 
    return np .asarray (V ,float ),np .asarray (F ,np .int64 )


def ray_hits (mesh ,origin ,direction ,max_mm =1e9 ,eps =1e-6 ):
    """Isin-ucgen kesisim mesafeleri (artan sirali). Saf numpy, bagimlilik absent.

    Moller-Trumbore. Doner: origin'den itibaren pozitif t mesafeleri (mm), <= max_mm olanlar.
    """
    V ,F =_mesh_arrays (mesh )
    o =np .asarray (origin ,float );d =np .asarray (direction ,float )
    n =float (np .linalg .norm (d ))
    if n <1e-12 or not len (F ):
        return np .zeros (0 )
    d =d /n 
    tri =V [F ]
    e1 =tri [:,1 ]-tri [:,0 ]
    e2 =tri [:,2 ]-tri [:,0 ]
    pv =np .cross (d ,e2 )
    det =np .einsum ("ij,ij->i",e1 ,pv )
    ok =np .abs (det )>1e-12 
    inv =np .zeros_like (det )
    inv [ok ]=1.0 /det [ok ]
    tv =o -tri [:,0 ]
    u =np .einsum ("ij,ij->i",tv ,pv )*inv 
    qv =np .cross (tv ,e1 )
    v =np .einsum ("j,ij->i",d ,qv )*inv 
    t =np .einsum ("ij,ij->i",e2 ,qv )*inv 
    m =ok &(u >=-1e-9 )&(v >=-1e-9 )&(u +v <=1 +1e-9 )&(t >eps )&(t <=max_mm )
    t =np .sort (t [m ])
    if len (t )<2 :
        return t 
        # ORTAK KENAR/KOSE TEKILLESTIRME: isin two ucgenin paylastigi kenardan gecerse AYNI kesisim two
        # times sayilir (kutu yuzeyinde measured: [15., 15.]). Bu, parite testini breaks -- icerideki point
        # 'disarida' gorunur. Ayni t degerleri single kesisime indirilir; 1e-7 mm'den yakin two AYRI surface
        # real geometride yoktur.
    return t [np .concatenate (([True ],np .diff (t )>1e-7 ))]


def _reach (mesh ,point ):
    """Noktadan meshi TAMAMEN gecmeye yetecek isin menzili.

    Sabit a 'span' YETMEZ: govdeden uzaktaki a point for isin, cikis yuzeyine varmadan
    kirpilir, kesisim count TEK gorunur and parite testi 'iceride' der (testte yakalandi).
    Menzil, noktanin uzakligini da icermek zorunda.
    """
    V ,_ =_mesh_arrays (mesh )
    c =(V .max (0 )+V .min (0 ))/2.0 
    span =float (np .linalg .norm (V .max (0 )-V .min (0 )))
    return float (np .linalg .norm (np .asarray (point ,float )-c ))+span *1.5 


def is_inside (mesh ,point ,probe =None ):
    """Parite testi: rastgele a isin single sayida surface kesiyorsa point ICERIDEDIR.

    Watertight olmayan meshlerde single isin yanilabilir -> 5 isinla oy cokluguna bakilir.
    """
    far =_reach (mesh ,point )
    rng =np .random .default_rng (0 )
    dirs =rng .normal (size =(5 ,3 ))
    dirs /=np .linalg .norm (dirs ,axis =1 ,keepdims =True )
    odd =sum (len (ray_hits (mesh ,point ,dd ,far ))%2 ==1 for dd in dirs )
    return odd >=3 


def seat_to_mouth (mesh ,seat ,direction ,max_mm =45.0 ,**_ ):
    """Uretici CP'si GOVDENIN ICINDEYSE onu ACIKLIGIN AGZINA tasi: axis along ILK YUZEY KESISIMI.

    Iki yone de isin at (insert yonunun isareti parcadan parcaya degisir), most yakin kesisimi sec.
    Doner: (agiz_noktasi, isaretli_offset_mm). Kesisim otherwise seat'i aynen returns (offset 0).

    IMPORTANT (2026-07-28 measurement): CP'nin ICERIDE olmasi PARCAYA GORE DEGISIR -- 3270115'te 16/16 iceride
    but 3273112'de 0/19, already yuzeyde. Bu yuzden before ICERIDE MI diye bakilir; disaridaki point
    OYNATILMAZ. (Yoksa empty alandaki a CP for isin karsi duvara carpar and point 36mm oteye
    firlatilirdi -- measured.)
    """
    d =np .asarray (direction ,float );n =float (np .linalg .norm (d ))
    seat =np .asarray (seat ,float )
    if n <1e-9 :
        return seat ,0.0 
    d =d /n 
    if not is_inside (mesh ,seat ):
        return seat ,0.0 
    best_t ,best_s =None ,1.0 
    for sgn in (+1.0 ,-1.0 ):
        h =ray_hits (mesh ,seat ,sgn *d ,max_mm )
        if len (h )and (best_t is None or h [0 ]<best_t ):
            best_t ,best_s =float (h [0 ]),sgn 
    if best_t is None :
        return seat ,0.0 
    return seat +best_s *d *best_t ,best_s *best_t 


_DIR_CACHE ={}


def _candidate_dirs (n_dirs =None ):
    """Aday axis yonleri (unsigned -> half kure yeter).

    n_dirs None/3 whereas ESKI davranis: only X, Y, Z. Daha buyukse Fibonacci kuresiyle
    equal dagilimli half kure ornegi uretilir; koordinat eksenleri always iceride tutulur
    ki eksene hizali durumlarda old sonuc birebir korunsun.
    """
    n =3 if n_dirs is None else int (n_dirs )
    if n <=3 :
        return [np .array ([1.0 ,0 ,0 ]),np .array ([0 ,1.0 ,0 ]),np .array ([0 ,0 ,1.0 ])]
    key =n 
    if key in _DIR_CACHE :
        return _DIR_CACHE [key ]
    m =2 *n # full kure uret, yarisini al
    i =np .arange (m )+0.5 
    phi =np .arccos (1 -2 *i /m )
    theta =np .pi *(1 +5 **0.5 )*i 
    D =np .stack ([np .cos (theta )*np .sin (phi ),np .sin (theta )*np .sin (phi ),np .cos (phi )],1 )
    D =D [D [:,2 ]>=0 ]# half kure (unsigned axis)
    D =np .vstack ([np .eye (3 ),D ])# koordinat eksenleri MUTLAKA denensin
    D /=np .linalg .norm (D ,axis =1 ,keepdims =True )
    out =[d for d in D ]
    _DIR_CACHE [key ]=out 
    return out 


def _refine_dirs (best ,k =12 ,spread_deg =12.0 ):
    """Kazanan yonun cevresinde ince tarama -- kaba izgara cozunurlugunu asmak for."""
    b =np .asarray (best ,float );b /=np .linalg .norm (b )+1e-12 
    tmp =np .array ([1.0 ,0 ,0 ])if abs (b [0 ])<0.9 else np .array ([0 ,1.0 ,0 ])
    u =np .cross (b ,tmp );u /=np .linalg .norm (u )+1e-12 
    v =np .cross (b ,u )
    out =[]
    for r in (0.5 ,1.0 ):
        a =np .radians (spread_deg )*r 
        for j in range (k ):
            t =2 *np .pi *j /k 
            d =np .cos (a )*b +np .sin (a )*(np .cos (t )*u +np .sin (t )*v )
            out .append (d /np .linalg .norm (d ))
    return out 


def channel_axis (mesh ,point ,fallback =None ,min_gain =1.6 ,n_dirs =None ):
    """Acikligin GERCEK eksenini GEOMETRIDEN bul (unsigned axis returns).

    WHY: yonu 'ham tahminin most large bileseni' with most yakin eksene yuvarlamak, ham prediction
    kararsizken TAMAMEN wrong eksene yuvarliyordu -- measured (2026-07-29): parcalarin yarisinda
    angle hatasi 0.0 derece, diger yarisinda 42-48 derece. Iki kutuplu imza = wrong axis secimi.

    FIZIKSEL OLCUT: a tel kanali, own ekseni along IKI TARAFTA DA open which is single eksendir
    (a taraf disari, diger taraf kanalin dibi). Dik eksenler hole yaricapi up to gidip duvara
    carpar. Bu yuzden each axis for min(ileri_serbest, geri_serbest) is computed; channel ekseni
    bunu maksimize eder.

    Kazanan, ikinciyi min_gain katindan extra gecemezse karar GUVENSIZ sayilir and fallback
    dondurulur (kor a prediction instead of mevcut degeri korumak more safe).

    n_dirs: denenecek candidate direction count. VARSAYILAN 3 = only X/Y/Z (old davranis).
    ROOT CAUSE BURADAYDI (2026-07-30): this fonksiyon only 3 koordinat eksenini deniyordu, i.e.
    EGIK a axis dondurmesi matematiksel as imkansizdi. Oysa manufacturer ConnectionPoint
    yonlerinin %19.1'i eksene 10 dereceden extra egik (8534 CP'de measured, most extra 43.1 derece;
    PXC %16.9, WEI %21.7) -- klemenste tel acili a huniden girer. Bu yuzden matched CP'lerin
    %18.2'sinde eksenimiz 15-90 derece sapiyordu. n_dirs > 3 verildiginde kure on real
    arama is done (kaba tarama + kazananin cevresinde ince tarama).
    """
    p =np .asarray (point ,float )
    reach =_reach (mesh ,p )
    scores =[]
    for e in _candidate_dirs (n_dirs ):
        hp =ray_hits (mesh ,p ,+e ,reach )
        hm =ray_hits (mesh ,p ,-e ,reach )
        dp =float (hp [0 ])if len (hp )else reach 
        dm =float (hm [0 ])if len (hm )else reach 
        # OPEN TARAF SARTI: real a tel kanali EN AZ BIR TARAFTAN open havaya cikar (that taraf
        # no yuzeyi kesmez). Yalnizca min(ileri, geri) bakmak bunu istemiyordu and dar a
        # oyugu "channel" sanabiliyordu; sonuc, okun kanalin karsi duvarina 7-13mm mesafede
        # bakmasiydi (measured 2026-07-29, PXC.2770943'te 3 isaretci).
        open_side =int (len (hp )==0 )+int (len (hm )==0 )
        scores .append ((open_side ,min (dp ,dm ),e ))
        # before OPEN TARAFI which is eksenler, after more DERIN channel
    scores .sort (key =lambda x :(-x [0 ],-x [1 ]))
    if scores [0 ][0 ]==0 :# no axis disari acilmiyor -> karar absent
        return None if fallback is None else np .asarray (fallback ,float )
    best ,second =scores [0 ][1 ],scores [1 ][1 ]
    if best <=1e-6 or best >=reach *0.999 :# ya no sey ya each sey open -> karar absent
        return None if fallback is None else np .asarray (fallback ,float )
    if scores [0 ][0 ]==scores [1 ][0 ]and second >1e-9 and best <min_gain *second :
        return None if fallback is None else np .asarray (fallback ,float )# net kazanan absent
    win =scores [0 ]
    if n_dirs is not None and int (n_dirs )>3 :# kazananin cevresinde INCE tarama
        for e in _refine_dirs (win [2 ]):
            hp =ray_hits (mesh ,p ,+e ,reach );hm =ray_hits (mesh ,p ,-e ,reach )
            dp =float (hp [0 ])if len (hp )else reach 
            dm =float (hm [0 ])if len (hm )else reach 
            cand =(int (len (hp )==0 )+int (len (hm )==0 ),min (dp ,dm ),e )
            if (cand [0 ],cand [1 ])>(win [0 ],win [1 ]):
                win =cand 
    return win [2 ]


def outward_along_axis (mesh ,point ,direction ,max_mm =None ,**_ ):
    """Eksen along +d / -d'den hangisi DISARI (empty tarafa) bakiyorsa onu dondur (unit vektor).

    Karari GERCEK GOVDE veriyor: each yonde onundeki surface kesisim SAYISINA bakilir. Az kesisim =
    that tarafta more few malzeme = disari. Beraberlikte first yuzeyi more UZAKTA which is (i.e. more genis
    empty alani which is) direction kazanir.

    WHY BOUNDING-BOX DEGIL: kutu testi, noktanin parcanin neresinde durduguna according to isareti ters
    cevirir; same yuzeydeki CP'ler zit yonlere savrulur (2026-07-28 gorsel hatasi).
    """
    d =np .asarray (direction ,float );n =float (np .linalg .norm (d ))
    if n <1e-9 :
        return np .array ([0.0 ,0.0 ,1.0 ])
    d =d /n 
    if max_mm is None :
        max_mm =_reach (mesh ,point )
    hp =ray_hits (mesh ,point ,+d ,max_mm )
    hm =ray_hits (mesh ,point ,-d ,max_mm )
    fp =float (hp [0 ])if len (hp )else np .inf 
    fm =float (hm [0 ])if len (hm )else np .inf 

    # IKI REJIM AYRI: point malzemenin ICINDEYSE 'disari' = KISA yoldan cikilan taraf. Serbest
    # alandaysa 'disari' = onunde more AZ malzeme / more COK bosluk which is taraf. Tek rule ikisine
    # birden uymuyor: icerideki point for 'more very bosluk' olcutu ters cevap veriyordu (test).
    if (len (hp )%2 ==1 )or (len (hm )%2 ==1 ):
        return +d if fp <=fm else -d 
    key_p =(len (hp ),-fp )
    key_m =(len (hm ),-fm )
    if key_p !=key_m :
        return +d if key_p <key_m else -d 

        # BERABERLIK -- and this NADIR DEGIL: ince terminalde a agizdan bakinca two direction de empty kalir
        # (ikisinde de 0 kesisim, ikisinde de fp=fm=inf). Eski rule here '<=' with HER ZAMAN +d
        # donuyordu; model yonu whereas ISARETSIZ a axis oldugu for sign KEYFI kaliyor and same
        # yuzeydeki oklar rastgele ters donuyordu (2026-07-29 gorsel hatasi, measured).
        # Karari MALZEME DAGILIMI gives: axis etrafindaki dar silindirde hangi tarafta more AZ
        # vertex varsa disarisi odur.
    V =np .asarray (mesh .vertices ,float )
    p0 =np .asarray (point ,float )
    rel =V -p0 
    al =rel @d 
    perp =np .linalg .norm (rel -al [:,None ]*d [None ,:],axis =1 )
    span =float (np .linalg .norm (V .max (0 )-V .min (0 )))
    r_probe =max (2.0 ,0.05 *span )
    l_probe =max (4.0 ,0.25 *span )
    near =perp <=r_probe 
    n_p =int ((near &(al >0 )&(al <=l_probe )).sum ())
    n_m =int ((near &(al <0 )&(al >=-l_probe )).sum ())
    if n_p !=n_m :
        return +d if n_p <n_m else -d 

        # Hala beraberse (real hole-boyu passing channel): deterministik last rule -- body
        # merkezinden UZAGA. Bu durumda two direction de fiziksel as gecerlidir.
    c =0.5 *(V .max (0 )+V .min (0 ))
    return +d if float ((p0 -c )@d )>=0 else -d 


def exit_length (mesh ,point ,direction ,margin =6.0 ,min_len =0.0 ,max_len =None ):
    """Noktadan `direction` along TUM malzemeyi gecip disari cikmak for gereken UZUNLUK.

    Isin uzerindeki EN UZAK surface kesisimi + margin. Boylece isaretci igne, full gerektigi up to
    uzar -- ne body inside kaybolur ne de parcadan uzun becomes.
    (Onceki cozum boyu prediction edip 1.6x'lik dongulerle buyutuyordu; 100mm'lik parcada 121mm igne
    uretti. Bu fonksiyon correct boyu TEK SEFERDE olcer.)
    """
    h =ray_hits (mesh ,point ,direction ,_reach (mesh ,point ))# fixed span DEGIL: uzak point kirpilirdi
    L =(float (h [-1 ])if len (h )else 0.0 )+margin 
    L =max (L ,min_len )
    return min (L ,max_len )if max_len else L 


def mouths_for (mesh ,seats ,directions ,**kw ):
    """Cok sayida GT for toplu yuva->mouth. Doner: (agizlar Nx3, offsetler N)."""
    if not len (seats ):
        return np .zeros ((0 ,3 )),np .zeros (0 )
    ms ,offs =[],[]
    for s ,d in zip (np .asarray (seats ,float ),np .asarray (directions ,float )):
        m ,o =seat_to_mouth (mesh ,s ,d ,**kw )
        ms .append (m );offs .append (o )
    return np .array (ms ),np .array (offs )


def mouth_width (mesh ,point ,axis ,n_dirs =24 ,max_mm =None ):
    """Agizdaki BOSLUGUN real genisligi -- channel eksenine DIK olculur.

    WHY: robot_cp eskiden size_mm'i segmentlenmis BOLGENIN alanindan turetiyordu
    (2*sqrt(alan/pi)). Contact sinifi tum kontak yuzeyini kapsadigi for this, 4mm'lik
    klemenslerde ~25mm "hole capi" veriyordu -- fiziksel as imkansiz. Robotun ihtiyaci
    which is sey telin gecegi acikligin genisligi, that yuzden OLCULUR: mouth noktasindan eksene dik
    yonlerde isin atilir and first yuzeye mesafeye bakilir.

    Doner: (ic_cap_mm, ort_cap_mm) = 2*min and 2*mean distance.
    Yuvarlak delikte ikisi esittir; yassi yuvada ic_cap DAR kenari gives (telin sigmasi
    gereken olcu), ort_cap whereas yuvanin genel acikligini.
    Hicbir yonde surface bulunmazsa (point govdenin disindaysa) (0.0, 0.0) returns -- cagiran
    taraf old tahminine geri donebilsin diye, sessizce uydurma a number URETILMEZ.
    """
    V ,_F =_mesh_arrays (mesh )
    if max_mm is None :
    # Menzil ACIKLIKTAN large must be. Kosegenin %25'i with was tried and 15mm'lik a hole
    # 0.00 dondurdu (isin karsi duvara ULASAMADI) -- i.e. "olcemedim" like gorunen silent
    # a kirpma. %50 real terminallerdeki most genis mouth for fazlasiyla yeterli.
        max_mm =0.50 *float (np .linalg .norm (V .max (0 )-V .min (0 )))
    a =np .asarray (axis ,float )
    na =float (np .linalg .norm (a ))
    if na <1e-12 :
        return 0.0 ,0.0 
    a =a /na 
    # eksene dik ortonormal baseline
    tmp =np .array ([1.0 ,0.0 ,0.0 ])if abs (a [0 ])<0.9 else np .array ([0.0 ,1.0 ,0.0 ])
    u =np .cross (a ,tmp );u /=np .linalg .norm (u )+1e-12 
    v =np .cross (a ,u )
    d =[]
    for k in range (n_dirs ):
        th =2.0 *np .pi *k /n_dirs 
        r =np .cos (th )*u +np .sin (th )*v 
        h =ray_hits (mesh ,point ,r ,max_mm )
        if len (h ):
            d .append (float (h [0 ]))
    if not d :
        return 0.0 ,0.0 
    d =np .asarray (d )
    return round (2.0 *float (d .min ()),2 ),round (2.0 *float (d .mean ()),2 )


import collections as _collections 
REJECT =_collections .Counter ()# tani: hangi gate kac times None dondurdu


def channel_axis_normals (mesh ,point ,seed_dir ,radius =6.0 ,min_faces =8 ,max_turn_deg =60.0 ,
flat_ratio =0.35 ):
    """Kanal eksenini DUVAR NORMALLERINDEN cikar -- ray-probe'dan very more keskin.

    WHY BU YONTEM: channel_axis() ray atarak min(ileri,geri) serbest mesafeyi maksimize eder.
    Bu criterion, hole own koni acisindan (atan(r/L)) more iyi cozemez: r=2.5mm L=15mm a yuvada
    eksenin +-9.5 derecelik konisi icindeki TUM yonler dibe same mesafede carpar, i.e. score DUZDUR.
    Sentetikte measured: egik kanalda remaining error 9.2 derece, hangi cozunurlukte taransa taransin.
    Gercek terminalde r~2mm L~10mm -> ~11 derece baseline boundary. Yetmez.

    SILINDIRIK KANALDA axis, wall normallerine DIKTIR. Yani normal kovaryans matrisinin
    EN KUCUK ozdegerine ait ozvektoru eksendir. Sentetik dogrulama: 0-43 derece egimde deviation
    0.00 derece; 0.05mm mesh gurultusuyle 0.03-0.07 derece.

    seed_dir: mevcut (kaba) direction. Yalnizca YUZ SECIMI for is used -- terminalin DUZ ON YUZU
    de CP'nin yakinindadir and normalleri eksene PARALEL oldugu for kovaryansi breaks. Bu yuzden
    only mouth duzleminin GERISINDEKI (channel icindeki) yuzler alinir. Sonuc seed'den
    max_turn_deg'den extra saparsa measurement guvenilmez sayilir and None returns (kor prediction absent).
    """
    V ,F =_mesh_arrays (mesh )
    p =np .asarray (point ,float )
    s =np .asarray (seed_dir ,float )
    ns =float (np .linalg .norm (s ))
    if ns <1e-12 or not len (F ):
        REJECT ["gecersiz_girdi"]+=1 
        return None 
    s =s /ns 
    tri =V [F ]
    c =tri .mean (1 )
    nrm =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
    a2 =np .linalg .norm (nrm ,axis =1 )# 2 x alan
    rel =c -p 
    inside =rel @s <=0.5 # mouth duzleminin gerisi = channel ici
    keep =(a2 >1e-12 )&(np .linalg .norm (rel ,axis =1 )<=radius )&inside 
    if int (keep .sum ())<min_faces :
        REJECT ["yuz_yetersiz"]+=1 
        return None 
    n =nrm [keep ]/a2 [keep ,None ]
    w =a2 [keep ]
    C =(n *w [:,None ]).T @n /w .sum ()
    ev ,evec =np .linalg .eigh (C )
    if ev [1 ]<=1e-9 or ev [0 ]/max (ev [1 ],1e-12 )>flat_ratio :
        REJECT ["silindirik_degil"]+=1 
        return None # normaller a duzleme yayilmiyor -> silindirik channel not
    d =evec [:,0 ]
    d =d /(np .linalg .norm (d )+1e-12 )
    if float (np .dot (d ,s ))<0 :
        d =-d 
    if np .degrees (np .arccos (min (1.0 ,abs (float (np .dot (d ,s ))))))>max_turn_deg :
        REJECT ["tohumdan_uzak"]+=1 
        return None # seed'den kopuk -> guvenilmez
    REJECT ["kabul"]+=1 
    return d 


def channel_axis_robust (mesh ,point ,seed_dir ,radius =8.0 ,min_faces =8 ,flat_ratio =0.35 ,
max_turn_deg =90.0 ):
    """Ekseni COK TOHUMLA olc and kazanani GEOMETRI sectirsin -- single tohuma guvenme.

    COZULEN TRAP (measured 2026-07-30): channel_axis_normals face secimini TOHUM yone according to
    does ("agzin gerisi = channel ici"). Tohum 90 derece yanlissa wrong half-uzay secilir,
    terminalin DUZ ON YUZU orneklemeye girer, normaller three boyuta yayilir and fonksiyon
    "silindirik not" deyip REDDEDER. Red whereas wrong tohumu oldugu like birakir.
    Kendi kendini besleyen a loop: wrong seed -> red -> wrong seed. Cok-CP parcalarda
    matched CP'lerin %22.3'u 45 dereceden extra sapiyordu and this ratio yaricaptan BAGIMSIZDI
    (radius taramasi: %22.1-22.3 arasi never kipirdamadi) -- i.e. reason komsu hole kirliligi
    not, full as this loop.

    YONTEM: candidate tohumlar = verilen seed + six koordinat yonu. Her biri for yuzler secilir
    and normal kovaryansi cikarilir. Kazanan, EN SILINDIRIK which is (ev0/ev1 orani most small) --
    i.e. karari seed not geometrinin kendisi gives. Hicbir candidate esigi gecemezse None returns
    (kor prediction uretmez).

    Dikdortgen yuva/kelepçe girisi de gecerlidir: two paralel wall + two three normalleri yine
    eksene DIKTIR, i.e. kovaryansin most small ozvektoru yine eksendir. Reddedilen sey silindir
    olmamasi not, ON YUZUN orneklemeye karismasiydi.
    """
    V ,F =_mesh_arrays (mesh )
    p =np .asarray (point ,float )
    s0 =np .asarray (seed_dir ,float )
    n0 =float (np .linalg .norm (s0 ))
    if n0 <1e-12 or not len (F ):
        return None 
    s0 =s0 /n0 
    tri =V [F ]
    c =tri .mean (1 )
    nrm =np .cross (tri [:,1 ]-tri [:,0 ],tri [:,2 ]-tri [:,0 ])
    a2 =np .linalg .norm (nrm ,axis =1 )
    rel =c -p 
    near =(a2 >1e-12 )&(np .linalg .norm (rel ,axis =1 )<=radius )
    if int (near .sum ())<min_faces :
        REJECT ["yuz_yetersiz"]+=1 
        return None 

    seeds =[s0 ]
    for k in range (3 ):
        e =np .zeros (3 );e [k ]=1.0 
        seeds +=[e ,-e ]
    best =None 
    for sd in seeds :
        keep =near &(rel @sd <=0.5 )
        if int (keep .sum ())<min_faces :
            continue 
        n =nrm [keep ]/a2 [keep ,None ]
        w =a2 [keep ]
        C =(n *w [:,None ]).T @n /w .sum ()
        ev ,evec =np .linalg .eigh (C )
        if ev [1 ]<=1e-9 :
            continue 
        ratio =ev [0 ]/ev [1 ]
        if best is None or ratio <best [0 ]:
            best =(ratio ,evec [:,0 ],sd )
    if best is None or best [0 ]>flat_ratio :
        REJECT ["silindirik_degil"]+=1 
        return None 
    d =best [1 ]/(np .linalg .norm (best [1 ])+1e-12 )
    if float (np .dot (d ,s0 ))<0 :
        d =-d 
    if np .degrees (np .arccos (min (1.0 ,abs (float (np .dot (d ,s0 ))))))>max_turn_deg :
        REJECT ["tohumdan_uzak"]+=1 
        return None 
    REJECT ["kabul"]+=1 
    return d 
