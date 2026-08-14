# -*- coding: utf-8 -*-
"""Q2: RENGI GEOMETRIYE baglayabiliyor muyuz? (tikanikligin real testi)

Q1 KANITLADI: parcalarin %83.8'inde >=2 ayrik renk present and renkler metal/plastik ayrimini
tasiyor (gri-gumus metal, mavi/turuncu body). Yani channel BOS DEGIL -- kayitli "each parcada
gumusi" notu single parcaya dayaniyormus.

TIKANIKLIK: gmsh rengi ATIYOR; STEP metnindeki koordinatlar whereas parcanin YEREL cercevesinde
(axis isinde bunu angle kaybi as olcmustuk: medyan 21mm deviation). XCAF renk tablosunu
yukluyor but 0 lower-shape etiketi veriyor. Onceki denemede yuzler SIRAYA according to eslestirilmis and
null testte cakmisti (2/5).

YENI IMKAN: `brep_axes.cylinders` residual gmsh OCC'den KURESEL koordinatli silindirleri veriyor
(point, axis, radius). STEP metninden same silindirleri YEREL koordinatlariyla and RENKLERIYLE
ayristirabiliyoruz. Parcalar TEK KATI oldugundan (25/25 measured) ikisi arasindaki donusum TEK
a rijit harekettir: yaricapla eslestirip Kabsch with cozulebilir. Sira eslemesi DEGIL, GEOMETRI
eslemesi -- onceki denemeyi this separates.

KILL (onceden yazildi): parcalarin %60'inda <1.0mm residual with donusum kurtarilamazsa is KAPANIR.
NULL TEST zorunlu: same yordami RASTGELE eslemeyle kosuyoruz; that da basariyorsa test degersizdir.
"""
import os ,sys ,re ,json ,glob 
import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

# --- STEP metni ayristirma: yerel silindirler ---
ENT =re .compile (rb"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*?)\)\s*;",re .S )
REF =re .compile (rb"#(\d+)")


def parse (path ):
    """#id -> (type, ham argumanlar). Tek gecis, tum file."""
    with open (path ,"rb")as fh :
        raw =fh .read ()
    d ={}
    for m in ENT .finditer (raw ):
        d [int (m .group (1 ))]=(m .group (2 ).decode ("ascii","replace"),m .group (3 ))
    return d 


def yerel_silindirler (d ):
    """CYLINDRICAL_SURFACE -> (centre, axis, radius) YEREL cerceve."""
    out =[]
    for i ,(t ,a )in d .items ():
        if t !="CYLINDRICAL_SURFACE":
            continue 
        refs =[int (x )for x in REF .findall (a )]
        try :
            r =float (a .rsplit (b",",1 )[1 ])
        except ValueError :
            continue 
        if not refs :
            continue 
        ax =d .get (refs [0 ])
        if not ax or ax [0 ]!="AXIS2_PLACEMENT_3D":
            continue 
        p =[int (x )for x in REF .findall (ax [1 ])]
        if len (p )<2 :
            continue 
        try :
            c =[float (x )for x in d [p [0 ]][1 ].split (b"(")[1 ].split (b")")[0 ].split (b",")]
            n =[float (x )for x in d [p [1 ]][1 ].split (b"(")[1 ].split (b")")[0 ].split (b",")]
        except Exception :
            continue 
        if len (c )==3 and len (n )==3 :
            out .append ((i ,np .array (c ),np .array (n ),r ))
    return out 


def kabsch (A ,B ):
    """A -> B rijit donusum (R, t) and residual (RMS)."""
    ca ,cb =A .mean (0 ),B .mean (0 )
    H =(A -ca ).T @(B -cb )
    U ,_ ,Vt =np .linalg .svd (H )
    dsign =np .sign (np .linalg .det (Vt .T @U .T ))
    R =Vt .T @np .diag ([1 ,1 ,dsign ])@U .T 
    t =cb -R @ca 
    res =float (np .sqrt (((A @R .T +t -B )**2 ).sum (1 ).mean ()))
    return R ,t ,res 


def hizala (step ,rastgele =False ,rng =None ):
    """Yerel (metin) silindirleri kuresel (gmsh) silindirlere yaricapla eslestir + Kabsch."""
    import brep_axes as B 
    C ,A ,R_ =B .cylinders (step )
    if len (C )<4 :
        return None ,0 ,"gmsh<4 silindir"
    d =parse (step )
    loc =yerel_silindirler (d )
    if len (loc )<4 :
        return None ,0 ,"metin<4 silindir"
    lc =np .array ([x [1 ]for x in loc ]);lr =np .array ([x [3 ]for x in loc ])
    gc =np .asarray (C ,float );gr =np .asarray (R_ ,float )
    # YARICAP with eslestir: each yerel silindire, yaricapi 0.01mm inside which is kuresel silindir
    pa ,pb =[],[]
    kul =set ()
    order =rng .permutation (len (loc ))if rastgele else np .argsort (lr )
    for k in order :
        cand =[j for j in range (len (gc ))if j not in kul and abs (gr [j ]-lr [k ])<0.01 ]
        if not cand :
            continue 
        j =int (rng .choice (cand ))if rastgele else cand [0 ]
        kul .add (j );pa .append (lc [k ]);pb .append (gc [j ])
    if len (pa )<4 :
        return None ,len (pa ),"eslesme<4"
    Rm ,t ,res =kabsch (np .array (pa ),np .array (pb ))
    return (Rm ,t ),len (pa ),f"{res :.3f}mm"


def main ():
    n =int (sys .argv [1 ])if len (sys .argv )>1 else 20 
    files =sorted (glob .glob ("all_wscad_stp/*.stp"))
    rng =np .random .RandomState (0 )
    sel =[files [i ]for i in rng .choice (len (files ),min (n ,len (files )),replace =False )]
    print (f"{len (sel )} parcada donusum kurtarma testi\n",flush =True )
    print (f"{'part':<16}{'esl':>5}{'GERCEK artik':>15}{'NULL (rastgele)':>18}")
    iyi =nul_iyi =tot =0 
    rec_ =[]
    for f in sel :
        pid =os .path .basename (f ).split ("_")[1 ]
        try :
            _ ,k ,msg =hizala (f )
        except Exception as e :
            print (f"{pid :<16}{'-':>5}{'HATA '+type (e ).__name__ :>15}")
            continue 
        try :
            _ ,k2 ,msg2 =hizala (f ,rastgele =True ,rng =np .random .RandomState (7 ))
        except Exception :
            msg2 ="error"
        tot +=1 
        ok =msg .endswith ("mm")and float (msg [:-2 ])<1.0 
        nok =msg2 .endswith ("mm")and float (msg2 [:-2 ])<1.0 
        iyi +=ok ;nul_iyi +=nok 
        print (f"{pid :<16}{k :>5}{msg :>15}{msg2 :>18}{'  <- GECTI'if ok else ''}",flush =True )
        rec_ .append ({"pid":pid ,"eslesme":k ,"artik":msg ,"null":msg2 ,"gecti":bool (ok )})
    o =iyi /max (tot ,1 );no =nul_iyi /max (tot ,1 )
    print (f"\nGERCEK esleme: {iyi }/{tot } ({o :.3f}) <1.0mm")
    print (f"NULL  esleme : {nul_iyi }/{tot } ({no :.3f}) <1.0mm   "
    f"<- bu da yuksekse test DEGERSIZ")
    print (f"\nKILL: gercek ratio <0.60 ya da null ratio gercege yakinsa is KAPANIR -> "
    f"{'AC'if (o >=0.60 and no <o -0.3 )else 'KAPAT'}")
    json .dump ({"n":tot ,"gercek_oran":o ,"null_oran":no ,"kayit":rec_ ,
    "karar":"AC"if (o >=0.60 and no <o -0.3 )else "KAPAT"},
    open ("results/q2_colour_alignment.json","w"),indent =1 )
    print ("receipt -> results/q2_colour_alignment.json")


if __name__ =="__main__":
    main ()
