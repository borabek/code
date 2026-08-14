"""p5-v2: secenek manufacturer + bire-a etiketleme + ortak secim kisitlari."""
import numpy as np 
import p5v2_secenek as PS 
import p5v2_egit as PE 


def _cy (m ,a ,r ,boy =4.0 ):
    m =np .asarray (m ,float );a =np .asarray (a ,float )/np .linalg .norm (a )
    return {"axis":a ,"radius":r ,"mouth_a":m ,"mouth_b":m +a *boy }


def test_mevcut_ilk_null_son ():
    """TEZ SADAKATI: `v_o` HER ZAMAN 0 numarali secenek; NULL most probe."""
    P =np .array ([[0. ,0 ,0 ]]);D =np .array ([[0. ,0 ,1 ]])
    o =PS .secenekler (P ,D ,[_cy ([1 ,0 ,0 ],[1 ,0 ,0 ],2.0 )],[],50.0 )[0 ]
    assert np .allclose (o [0 ][0 ],P [0 ])and np .allclose (o [0 ][1 ],D [0 ])
    assert o [0 ][3 ]==-1 # MEVCUT kisit disi
    assert o [-1 ][0 ]is None # NULL


def test_ozn_uzunlugu ():
    P =np .array ([[0. ,0 ,0 ]]);D =np .array ([[0. ,0 ,1 ]])
    for s in PS .secenekler (P ,D ,[_cy ([1 ,0 ,0 ],[1 ,0 ,0 ],2.0 )],[],50.0 )[0 ]:
        assert len (s [2 ])==len (PS .OZ_AD )


def test_ayni_agiz_ayni_kimlik ():
    """Bir silindirin two isareti AYNI fiziksel agzi paylasir -> same kimlik."""
    P =np .array ([[0. ,0 ,0 ]]);D =np .array ([[0. ,0 ,1 ]])
    o =PS .secenekler (P ,D ,[_cy ([1 ,0 ,0 ],[1 ,0 ,0 ],2.0 )],[],50.0 )[0 ]
    kim =[s [3 ]for s in o if s [3 ]>=0 ]
    assert len (kim )>len (set (kim )),"sign ciftleri ayni kimligi paylasmali"


def test_etiket_BIRE_BIR ():
    """Iki candidate AYNI GT'yi saglasa bile YALNIZ BIRI pozitif must be (greedy DEGIL)."""
    P =np .array ([[0. ,0 ,0 ],[0.5 ,0 ,0 ]])
    D =np .array ([[0. ,0 ,1 ],[0. ,0 ,1 ]])
    G =np .array ([[0. ,0 ,0.5 ]]);Gd =np .array ([[0. ,0 ,1 ]])
    secs =PS .secenekler (P ,D ,[],[],50.0 )
    y =PE .etiketle (secs ,G ,Gd )
    assert sum (int (v .any ())for v in y )==1 ,"iki candidate da pozitif olmus (greedy)"


def test_secim_agzi_TEK_adaya_verir ():
    """Ayni agza two candidate talip; ORTAK secim only birine vermeli."""
    P =np .array ([[0. ,0 ,0 ],[0.6 ,0 ,0 ]])
    D =np .array ([[0. ,0 ,1 ],[0. ,0 ,1 ]])
    secs =PS .secenekler (P ,D ,[_cy ([1 ,0 ,0 ],[1 ,0 ,0 ],2.0 )],[],50.0 )
    skor =[np .full (len (o ),0.1 )for o in secs ]
    for s in skor :# mouth secenekleri MEVCUT'tan cazip
        s [1 :-1 ]=0.9 
    Pf ,Df =PE .sec (secs ,skor )
    kullanilan =[tuple (np .round (p ,3 ))for p in Pf ]
    assert len (kullanilan )==len (set (kullanilan )),"ayni mouth iki adaya verilmis"


def test_mevcut_GERCEK_fallback ():
    """Hicbir secenek cazip degilse candidate SILINMEZ, MEVCUT'ta kalir."""
    P =np .array ([[1. ,2 ,3 ]]);D =np .array ([[0. ,0 ,1 ]])
    secs =PS .secenekler (P ,D ,[],[],50.0 )
    skor =[np .array ([0.9 ,0.1 ])]# MEVCUT high, NULL low
    Pf ,_ =PE .sec (secs ,skor )
    assert len (Pf )==1 and np .allclose (Pf [0 ],P [0 ])


def test_NULL_adayi_atabilir ():
    P =np .array ([[1. ,2 ,3 ]]);D =np .array ([[0. ,0 ,1 ]])
    secs =PS .secenekler (P ,D ,[],[],50.0 )
    skor =[np .array ([0.1 ,0.9 ])]# NULL high
    Pf ,_ =PE .sec (secs ,skor )
    assert len (Pf )==0 


def test_siralayici_dogruyu_UST_SIRAYA_koyar ():
    """Pairwise siralayici, same candidate inside correct secenegi most high skorlamali."""
    import numpy as np 
    import p5v2_egit as PE 
    rng =np .random .RandomState (0 )
    veri =[]
    for _ in range (40 ):
    # 4 secenek; DOGRU which is 3. sutunu (distance) EN KUCUK which is
        F =rng .rand (4 ,len (PS .OZ_AD ))
        dogru =int (np .argmin (F [:,3 ]))
        secs =[[(np .zeros (3 ),np .array ([0. ,0 ,1 ]),list (F [k ]),-1 )for k in range (4 )]]
        y =[np .zeros (4 ,int )]
        y [0 ][dogru ]=1 
        veri .append ({"secs":secs ,"y":y })
    m =PE .Siralayici (n =60 ,leaf =2 ).fit (veri )
    dogru_ust =0 
    for d in veri :
        s =m .skorla ([o [2 ]for o in d ["secs"][0 ]])
        if int (np .argmax (s ))==int (np .argmax (d ["y"][0 ])):
            dogru_ust +=1 
    assert dogru_ust /len (veri )>0.7 ,f"yalniz {dogru_ust }/{len (veri )} dogru"


def test_siralayici_tek_secenekte_patlamaz ():
    import numpy as np 
    import p5v2_egit as PE 
    m =PE .Siralayici .__new__ (PE .Siralayici )
    s =PE .Siralayici .skorla (m ,np .zeros ((1 ,len (PS .OZ_AD ))))
    assert len (s )==1 
