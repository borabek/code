# -*- coding: utf-8 -*-
"""p5-v2 EGITIM + ORTAK SECIM -- gate'ten ONCE.

Plan (kullanici, 2026-08-09) + olculmus oncul
([[gate-before-poz-after-tavani-kirpiyor]]: ham pool kahini 0.3781 / before-gate 0.3155).

TASARIM
  * TUM ham candidates (gate YOK) -> p5v2_secenek.secenekler()
  * ETIKET: bire-a STRICT robot eslesmesi (greedy DEGIL) -- Macar with
  * SECIM: each adaya a secenek; AYNI fiziksel mouth EN FAZLA a adaya
    (bipartite kisit, `agiz_kimlik`)
  * EGITIM and INFERENCE AYNI secenekleri gorur (mevcut p5'te planar only
    inference'ta vardi -- that kusur here YOK)
  * MEVCUT (`v_o`) real fallback: skoru dusukse bile candidate SILINMEZ, MEVCUT'ta kalir
  * Gate SONRA, last kabul/kalibrasyon as (this betikte DEGIL)
TEZE SADIK: `v_o` always 0 numarali secenek; turetme/remesh/sinif count does not change.
"""
import argparse ,collections ,glob ,json ,os ,pickle ,sys 
import numpy as np 
import makbuz_hash 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,".")
import d6_record ,robot_cp ,wire_gate ,p5v2_secenek as PS 
from corpus_identity import step_kimlik as SK 
from scipy .optimize import linear_sum_assignment 
from sklearn .ensemble import RandomForestClassifier 

YANAL ,ACI ,EKSENEL =2.0 ,10.0 ,40.0 


def robot_uyar (p ,y ,g ,gd ):
    if p is None :
        return False 
    u =y /(np .linalg .norm (y )+1e-9 )
    v =g -p 
    ax =abs (float (v @u ))
    lat =float (np .linalg .norm (v -(v @u )*u ))
    gu =gd /(np .linalg .norm (gd )+1e-9 )
    ac =float (np .degrees (np .arccos (np .clip (float (u @gu ),-1 ,1 ))))
    return lat <=YANAL and ac <=ACI and ax <=EKSENEL 


def etiketle (secs ,G ,Gd ):
    """BIRE-BIR strict robot eslesmesi -> secenek basina 0/1.

    Greedy DEGIL: (candidate, GT) maliyet matrisi kurulur, Macar atar, only ATANAN
    ciftin O GT'yi saglayan secenekleri 1 becomes. Greedy olsaydi same GT birden
    very adayi POZITIF yapar and selector 'each candidate correct' ogrenirdi.
    """
    n ,m =len (secs ),len (G )
    y =[np .zeros (len (o ),int )for o in secs ]
    if not n or not m :
        return y 
    C =np .ones ((n ,m ))
    iyi ={}
    for i ,o in enumerate (secs ):
        for j in range (m ):
            for k ,(p ,yy ,_oz ,_a )in enumerate (o ):
                if robot_uyar (p ,yy ,G [j ],Gd [j ]):
                    C [i ,j ]=0.0 
                    iyi .setdefault ((i ,j ),[]).append (k )
                    break 
    ri ,ci =linear_sum_assignment (C )
    for a ,b in zip (ri ,ci ):
        if C [a ,b ]==0.0 :
            for k ,(p ,yy ,_oz ,_ag )in enumerate (secs [a ]):
                if robot_uyar (p ,yy ,G [b ],Gd [b ]):
                    y [a ][k ]=1 
    return y 


def sec (secs ,skor ,gate_skor =None ,gate_esik =None ):
    """ORTAK secim: each adaya a secenek, a AGIZ most extra a adaya.

    Macar with (candidate x mouth) atamasi; agzi olmayan (MEVCUT/NULL) secenekler
    kisit disi and atamadan SONRA degerlendirilir.
    """
    n =len (secs )
    agizlar =sorted ({a for o in secs for (_p ,_y ,_oz ,a )in o if a >=0 })
    P ,D =[None ]*n ,[None ]*n 
    if agizlar :
        idx ={a :k for k ,a in enumerate (agizlar )}
        C =np .ones ((n ,len (agizlar )))
        en ={}
        for i ,o in enumerate (secs ):
            for k ,(p ,y ,_oz ,a )in enumerate (o ):
                if a <0 :
                    continue 
                c =1.0 -float (skor [i ][k ])
                if c <C [i ,idx [a ]]:
                    C [i ,idx [a ]]=c 
                    en [(i ,idx [a ])]=k 
        ri ,ci =linear_sum_assignment (C )
        for i ,j in zip (ri ,ci ):
            k =en .get ((i ,j ))
            if k is None :
                continue 
                # mouth secenegi however MEVCUT'tan IYIYSE alinir
            if skor [i ][k ]>skor [i ][0 ]:
                P [i ],D [i ]=secs [i ][k ][0 ],secs [i ][k ][1 ]
    for i ,o in enumerate (secs ):
        if P [i ]is None :
        # NULL most yuksekse adayi at; degilse MEVCUT'ta kal (GERCEK fallback)
            k_null =len (o )-1 
            if skor [i ][k_null ]>skor [i ][0 ]:
                continue 
            P [i ],D [i ]=o [0 ][0 ],o [0 ][1 ]
            # GATE SON KABUL: ortak secimden SONRA, goreli esikle
    if gate_esik is not None and gate_skor is not None and len (gate_skor )==n :
        gs =np .asarray (gate_skor ,float )
        kes =max (gate_esik [1 ],gate_esik [0 ]*float (gs .max ())if len (gs )else 0.0 )
        for i in range (n ):
            if P [i ]is not None and gs [i ]<kes :
                P [i ]=None 
    tut =[i for i in range (n )if P [i ]is not None ]
    return (np .asarray ([P [i ]for i in tut ],float )if tut else np .zeros ((0 ,3 )),
    np .asarray ([D [i ]for i in tut ],float )if tut else np .zeros ((0 ,3 )))


def veri_kur (pidler ,rec_ ,ob_dir ,cyl ,acik ,S ,gate ):
    """Her part for (secenekler, labels, manufacturer). Gate SKOR ozniteligi
    for is used but ADAY ELEMEZ -- eleme p5-v2'nin isi."""
    from p1c_threshold import maske # noqa: F401  (kullanilmiyor; gate ELEMEZ)
    data_ =[]
    for pid in pidler :
        f =f"{ob_dir }/{pid }.npz"
        if not os .path .exists (f ):
            continue 
        r =rec_ [pid ]
        G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
        if not len (G ):
            continue 
        d =np .load (f )
        V =np .ascontiguousarray (d ["V"],np .float64 )
        F =np .ascontiguousarray (d ["F"],np .int64 )
        cps ,_o ,_c ,_p =robot_cp .derive_candidates (
        V ,F ,[np .asarray (q ,float )for q in d ["pbs"]],S .get (pid ))
        if not cps :
            continue 
        P =np .asarray ([c ["point"]for c in cps ],float )
        D =np .asarray ([c ["direction"]for c in cps ],float )
        vt =np .asarray ([c .get ("_votes",0 )for c in cps ],float )
        avg =np .asarray (d ["pbs"],float ).mean (0 )
        Xp =np .asarray (wire_gate .feats_for (V ,F ,avg ,cps ,robot_cp .CE ,
        robot_cp .CT ,step_path =S .get (pid )),float )
        gs =np .asarray (wire_gate .decision_score (gate ,Xp ),float )
        komsu =None 
        if len (D )>1 :
            B =D *np .sign (D @D [0 ])[:,None ]
            komsu =B .mean (0 );komsu /=(np .linalg .norm (komsu )+1e-12 )
        secs =PS .secenekler (P ,D ,cyl .get (pid ),acik .get (pid ),r ["diag"],
        gate_s =gs ,votes =vt ,komsu =komsu )
        data_ .append ({"pid":pid ,"mfg":r ["mfg"],"secs":secs ,"gate_skor":gs ,
        "y":etiketle (secs ,G ,Gd ),"G":G ,"Gd":Gd ,
        "diag":r ["diag"]})
    return data_ 


class Siralayici :
    """ADAY-ICI SIRALAMA (pairwise). Ikili siniflandirmanin instead of.

    WHY: p5-v2'nin real karari "this adayin secenekleri arasindan BIRINI sec".
    Ikili siniflandirici each secenegi BAGIMSIZ puanliyordu; ogrendigi sey
    ("this secenek correct mu") with kullanildigi sey ("hangisi EN IYI") same not.
    Pairwise: same adayin (correct, wrong) secenek ciftleri ten
    f(correct) > f(wrong) ogretilir -- feature FARKI ten ikili sinif.
    Cikarimda skor single secenek uzerinden hesaplanabilsin diye model FARKA
    egitilir and puanlama f(x) = P(x - 0 farki) instead of DOGRUDAN karar fonksiyonu
    as is used: double (x_i - x_j) -> 1 whereas x_i more iyi.
    """

    def __init__ (self ,n =300 ,leaf =5 ,seed =0 ,maks_cift =200000 ):
        self .clf =RandomForestClassifier (n_estimators =n ,min_samples_leaf =leaf ,
        n_jobs =-1 ,random_state =seed )
        self .maks_cift =maks_cift 

    def fit (self ,data_ ):
        rng =np .random .RandomState (0 )
        A ,B =[],[]
        for d in data_ :
            for i ,o in enumerate (d ["secs"]):
                y =d ["y"][i ]
                poz =np .where (y ==1 )[0 ]
                neg =np .where (y ==0 )[0 ]
                if not len (poz )or not len (neg ):
                    continue 
                F =np .asarray ([s [2 ]for s in o ],float )
                for pi in poz :
                # each pozitif for most extra 6 negatif (dengeli, patlamasin)
                    for ni in (neg if len (neg )<=6 else rng .choice (neg ,6 ,replace =False )):
                        A .append (F [pi ]-F [ni ]);B .append (1 )
                        A .append (F [ni ]-F [pi ]);B .append (0 )
        if not A :
            raise RuntimeError ("pairwise training cifti YOK")
        A =np .asarray (A ,float );B =np .asarray (B ,int )
        if len (A )>self .maks_cift :
            i =rng .choice (len (A ),self .maks_cift ,replace =False )
            A ,B =A [i ],B [i ]
        self .clf .fit (A ,B )
        self .n_cift =len (A )
        return self 

    def skorla (self ,F ):
        """Aday-ici skor: each secenegin DIGERLERINE karsi kazanma orani."""
        F =np .asarray (F ,float )
        n =len (F )
        if n ==1 :
            return np .array ([1.0 ])
        i ,j =np .triu_indices (n ,1 )
        P =self .clf .predict_proba (F [i ]-F [j ])[:,1 ]
        s =np .zeros (n )
        np .add .at (s ,i ,P );np .add .at (s ,j ,1.0 -P )
        return s /max (n -1 ,1 )


def egit (data_ ,kip =None ):
    kip =kip or os .environ .get ("P5V2_KIP","ikili")
    if kip =="ranking":
        m =Siralayici ().fit (data_ )
        return m ,(m .n_cift ,data_ [0 ]["secs"][0 ][0 ][2 ].__len__ ()),-1.0 
    X =np .vstack ([np .asarray ([s [2 ]for s in d ["secs"][i ]],float )
    for d in data_ for i in range (len (d ["secs"]))])
    y =np .concatenate ([d ["y"][i ]for d in data_ for i in range (len (d ["secs"]))])
    clf =RandomForestClassifier (n_estimators =300 ,min_samples_leaf =5 ,n_jobs =-1 ,
    random_state =0 ,class_weight ="balanced").fit (X ,y )
    return clf ,X .shape ,float (y .mean ())


def uygula (data_ ,clf ,gate_esik =None ):
    """gate_esik verilirse ORTAK SECIMDEN SONRA last kabul as uygulanir.

    Plandaki order: ham pool -> ortak secim -> GATE (last kabul/kalibrasyon).
    Gate'i ONCE uygulamak tavani kirpiyordu; SONRA uygulamak kesinligi toplar.
    """
    from sina_cluster import match_hungarian ,f1w 
    import canonical_d7 as KZ # MIKRO toplama -- headline olcegi
    from p1c_threshold import maske 
    T ,R =[],[]
    for d in data_ :
        if isinstance (clf ,Siralayici ):
            skor =[clf .skorla ([s [2 ]for s in o ])for o in d ["secs"]]
        else :
            skor =[clf .predict_proba (np .asarray ([s [2 ]for s in o ],float ))[:,1 ]
            for o in d ["secs"]]
        P ,D =sec (d ["secs"],skor ,gate_skor =d .get ("gate_skor"),
        gate_esik =gate_esik )
        G ,Gd =d ["G"],d ["Gd"]
        T .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,d ["diag"],0.0 ,180.0 ,True )[:3 ])
        R .append ((len (G ),)+match_hungarian (P ,D ,G ,Gd ,d ["diag"],YANAL ,ACI ,
        False ,signed =True )[:3 ])
        # MIKRO da dondur: headline MIKRO olcekte, f1w regime-agirlikli (two fold difference)
    return (f1w (T ),f1w (R )),(KZ .mikro (T ),KZ .mikro (R ))


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ob",default ="results/_p1_olasilik_g10")
    ap .add_argument ("--gate",default ="results/wire_gate_v7.pkl")
    a =ap .parse_args ()
    sv =d6_record .exam ();rec_ =d6_record .yukle (set (sv ["pidler"]))
    gate =pickle .load (open (a .gate ,"rb"))
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    cyl =pickle .load (open ("results/_d6_silindirler.pkl","rb"))
    acik =pickle .load (open ("results/_d6_acikliklar.pkl","rb"))
    pidler =sorted ({f [:-4 ]for f in os .listdir (a .ob )if f .endswith (".npz")}
    &set (rec_ ))
    print (f"part {len (pidler )} | veri kuruluyor...",flush =True )
    data_ =veri_kur (pidler ,rec_ ,a .ob ,cyl ,acik ,S ,gate )
    print (f"kullanilabilir part {len (data_ )}",flush =True )

    # MARKA-DISI (LOMO) -- havuzlanmis secim wrong objektif for selects
    mfgs =sorted ({d ["mfg"]for d in data_ })
    say =collections .Counter (d ["mfg"]for d in data_ )
    test_mf =[m for m ,c in say .most_common ()if c >=30 ]
    print (f"LOMO markalari: {test_mf }\n",flush =True )
    res_ ={}
    for m in test_mf :
        tr =[d for d in data_ if d ["mfg"]!=m ]
        te =[d for d in data_ if d ["mfg"]==m ]
        clf ,sh ,poz =egit (tr )
        (t ,r ),(tm ,rm )=uygula (te ,clf )
        # NESTED LOMO: gate esigi DIS test markasini HIC gormeden, EGITIM
        # markalarindan ayrilan a IC fold'da secilir. Sabit 0.40/0.30
        # kullanmak, esigi tum korpusta (therefore dis markaya da bakarak)
        # secilmis a degerle sabitlemek demekti.
        ic_mf =sorted ({d ["mfg"]for d in tr })
        ic_test =ic_mf [len (ic_mf )//2 ]# ic fold: a training markasi
        ic_tr =[d for d in tr if d ["mfg"]!=ic_test ]
        ic_te =[d for d in tr if d ["mfg"]==ic_test ]
        en_e ,en_r =(0.40 ,0.30 ),-1.0 
        if ic_te :
            ic_clf ,_sh ,_p =egit (ic_tr )
            for ratio in (0.30 ,0.40 ,0.50 ):
                for baseline in (0.20 ,0.30 ,0.40 ):
                    (_t ,_r ),_ =uygula (ic_te ,ic_clf ,gate_esik =(ratio ,baseline ))
                    if _r >en_r :
                        en_r ,en_e =_r ,(ratio ,baseline )
        (tg ,rg ),(tgm ,rgm )=uygula (te ,clf ,gate_esik =en_e )# gate SONRA
        # TABAN: secim YOK, hep MEVCUT (`v_o`) -- tezin own yolu
        from sina_cluster import match_hungarian ,f1w 
        import canonical_d7 as KZ 
        T0 ,R0 =[],[]
        for d in te :
            P0 =np .asarray ([o [0 ][0 ]for o in d ["secs"]],float )
            D0 =np .asarray ([o [0 ][1 ]for o in d ["secs"]],float )
            G ,Gd =d ["G"],d ["Gd"]
            T0 .append ((len (G ),)+match_hungarian (P0 ,D0 ,G ,Gd ,d ["diag"],0. ,180. ,True )[:3 ])
            R0 .append ((len (G ),)+match_hungarian (P0 ,D0 ,G ,Gd ,d ["diag"],YANAL ,ACI ,
            False ,signed =True )[:3 ])
        t0 ,r0 =f1w (T0 ),f1w (R0 )
        t0m ,r0m =KZ .mikro (T0 ),KZ .mikro (R0 )
        res_ [m ]={"n":len (te ),"ic_esik":list (en_e ),"tespit_taban":t0 ,"tespit":t ,
        "robot_taban":r0 ,"robot":r ,
        "tespit_gate":tg ,"robot_gate":rg ,
        "robot_gate_MIKRO":rgm ,"tespit_gate_MIKRO":tgm ,
        "robot_taban_MIKRO":r0m ,"robot_MIKRO":rm ,
        "tespit_fark":t -t0 ,"robot_fark":r -r0 }
        print (f"  {m :<6} n={len (te ):<4} robot: baseline {r0 :.4f} | p5v2 {r :.4f} "
        f"({r -r0 :+.4f}) | +GATE {rg :.4f} || MIKRO: baseline {r0m :.4f} -> "
        f"p5v2 {rm :.4f} -> +GATE {rgm :.4f}",
        flush =True )
    if res_ :
        rf =float (np .mean ([v ["robot_fark"]for v in res_ .values ()]))
        tf =float (np .mean ([v ["tespit_fark"]for v in res_ .values ()]))
        print (f"\nLOMO ORTALAMA: tespit {tf :+.4f} | robot {rf :+.4f}")
        json .dump ({"damga":makbuz_hash .damga (),"sonuc":res_ ,"robot_fark_ort":rf ,"tespit_fark_ort":tf ,
        "baseline":"secim YOK, hep MEVCUT (v_o)"},
        open ("results/p5v2_lomo.json","w"),indent =1 )
        print ("receipt -> results/p5v2_lomo.json")


if __name__ =="__main__":
    main ()
