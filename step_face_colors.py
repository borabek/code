# -*- coding: utf-8 -*-
"""STEP yuzeylerini RENGIYLE BIRLIKTE oku (OpenCASCADE XCAF) -- metal/plastik ayrimi.

WHY GEREKLI:
  * Renk bilgisi STEP'te VAR (OVER_RIDING_STYLED_ITEM zinciri; 108/108 surface).
  * Ama gmsh onu DUSURUYOR: gmsh.model.getColor() butun yuzeylere TEK renk returns.
  * Metni elle ayristirip yuzeyleri SIRAYA according to eslestirme was tried and DOGRULANAMADI
    (5 parcanin 2'sinde tutarli, 3'unde not). Sira varsayimi guvenilir not.
  * STEPCAFControl_Reader rengi and geometriyi BIRLIKTE reads -> eslestirme sorunu ORTADAN KALKAR.

WHY IMPORTANT: [[low-cp-information-gap]] sizintili upper sinirla kanitladi ki kalan error a BILGI
sorunu -- data or kapasite not. "Kanalin dibinde METAL present mi" full da no sondanin sahip
olmadigi turden a sinyal.
"""
import numpy as np 

METAL_RGB =(0.824 ,0.824 ,0.784 )# each parcada gorulen gumusi ton = kontak metali
METAL_TOL =0.06 


def read (step_path ):
    """STEP -> [{'type','area','com','bbox_lo','bbox_hi','rgb','is_metal'}] (surface basina)."""
    from OCP .STEPCAFControl import STEPCAFControl_Reader 
    from OCP .TDocStd import TDocStd_Document 
    from OCP .XCAFDoc import XCAFDoc_DocumentTool 
    from OCP .TCollection import TCollection_ExtendedString 
    from OCP .TopExp import TopExp_Explorer 
    from OCP .TopAbs import TopAbs_FACE 
    from OCP .TopoDS import TopoDS 
    from OCP .BRepGProp import BRepGProp 
    from OCP .GProp import GProp_GProps 
    from OCP .BRepAdaptor import BRepAdaptor_Surface 
    from OCP .Bnd import Bnd_Box 
    from OCP .BRepBndLib import BRepBndLib 
    from OCP .Quantity import Quantity_Color 
    from OCP .XCAFDoc import XCAFDoc_ColorSurf ,XCAFDoc_ColorGen 

    doc =TDocStd_Document (TCollection_ExtendedString ("d"))
    rd =STEPCAFControl_Reader ()
    rd .SetColorMode (True )
    if rd .ReadFile (step_path )!=1 :
        raise RuntimeError (f"STEP okunamadi: {step_path }")
    rd .Transfer (doc )
    ctool =XCAFDoc_DocumentTool .ColorTool_s (doc .Main ())
    stool =XCAFDoc_DocumentTool .ShapeTool_s (doc .Main ())

    from OCP .TDF import TDF_LabelSequence 
    seq =TDF_LabelSequence ()
    stool .GetFreeShapes (seq )
    out =[]
    _KIND ={0 :"Plane",1 :"Cylinder",2 :"Cone",3 :"Sphere",4 :"Torus"}
    for i in range (1 ,seq .Length ()+1 ):
        shape =stool .GetShape_s (seq .Value (i ))
        exp =TopExp_Explorer (shape ,TopAbs_FACE )
        while exp .More ():
            f =TopoDS .Face_s (exp .Current ())
            col =Quantity_Color ()
            got =ctool .GetColor (f ,XCAFDoc_ColorSurf ,col )or ctool .GetColor (f ,XCAFDoc_ColorGen ,col )
            rgb =(round (col .Red (),3 ),round (col .Green (),3 ),round (col .Blue (),3 ))if got else None 
            g =GProp_GProps ();BRepGProp .SurfaceProperties_s (f ,g )
            com =g .CentreOfMass ();area =float (g .Mass ())
            bb =Bnd_Box ();BRepBndLib .Add_s (f ,bb )
            lo =np .array (bb .CornerMin ().Coord ());hi =np .array (bb .CornerMax ().Coord ())
            try :
                kind =_KIND .get (int (BRepAdaptor_Surface (f ).GetType ()),"Other")
            except Exception :
                kind ="Other"
            is_m =(rgb is not None and 
            all (abs (rgb [k ]-METAL_RGB [k ])<=METAL_TOL for k in range (3 )))
            out .append ({"type":kind ,"area":area ,
            "com":np .array ([com .X (),com .Y (),com .Z ()]),
            "bbox_lo":lo ,"bbox_hi":hi ,"rgb":rgb ,"is_metal":bool (is_m )})
            exp .Next ()
    return out 


def read_by_order (step_path ,dogrula =True ):
    """SILINDIRIK yuzleri RENK + KURESEL geometri with birlikte dondur.

    Doner: (cyls, ok) ; cyls = [{'radius','com','axis','rgb','is_metal'}], ok = bool

    WHY SIRA ESLEMESI: XCAF renk tablosunu yukluyor but lower-shape etiketi vermiyor
    (GetColor tum yuzler for False); gmsh rengi tamamen dusuruyor (OCCImportLabels=1 with bile).

    WHY SILINDIR ALT-DIZISI, TUM yuzler not: mutlak face order a parcada BIR KAYIYOR
    (measured: OCC indeksleri metin indekslerinin full +1'i). Ama SILINDIRIK yuzlerin lower-dizisi
    ikisinde de same sirada: 13 parcada 12'sinde yaricaplar ELEMAN ELEMAN equal
    (456/456, 224/224, 47/47, 31/31, 30/30, 29/29, 23/23, 19/19, 18/18, 13/13, 12/12).

    Sira hipotezi more before DOLAYLI a vekille ("metal yuzler more derinde must be") sinanip
    5 parcanin 2'sinde tuttugu for reddedilmisti. Dogrudan sinama present: each two taraf da
    radius veriyor. `dogrula=True` iken this kontrol PARCA BASINA is done; tutmayan part
    ok=False returns and cagiran onu ATLAR -- varsayim corpus genelinde kabul edilmez.
    """
    import re as _re 
    import numpy as _np 
    from OCP .STEPControl import STEPControl_Reader 
    from OCP .TopExp import TopExp_Explorer 
    from OCP .TopAbs import TopAbs_FACE 
    from OCP .TopoDS import TopoDS 
    from OCP .BRepAdaptor import BRepAdaptor_Surface 
    import q7_renk_degeri as _Q 

    _REF =_re .compile (rb"#(\d+)")
    d =_Q .parse (step_path )
    renk =_Q ._renk_haritasi (d )
    metin =[]# (radius, rgb) -- varlik-id during
    for i in sorted (k for k ,(t ,a )in d .items ()if t =="ADVANCED_FACE"):
        cyl =[r for r in (int (x )for x in _REF .findall (d [i ][1 ]))
        if r in d and d [r ][0 ]=="CYLINDRICAL_SURFACE"]
        if not cyl :
            continue 
        try :
            metin .append ((round (float (d [cyl [0 ]][1 ].rsplit (b",",1 )[1 ]),3 ),renk .get (i )))
        except ValueError :
            metin .append ((None ,renk .get (i )))

    rd =STEPControl_Reader ()
    rd .ReadFile (step_path )
    rd .TransferRoots ()
    exp =TopExp_Explorer (rd .OneShape (),TopAbs_FACE )
    occ =[]
    while exp .More ():
        a =BRepAdaptor_Surface (TopoDS .Face_s (exp .Current ()))
        if int (a .GetType ())==1 :
            try :
                cy =a .Cylinder ()
                ad =cy .Axis ().Direction ();ap =cy .Axis ().Location ()
                occ .append ((round (float (cy .Radius ()),3 ),
                _np .array ([ap .X (),ap .Y (),ap .Z ()],float ),
                _np .array ([ad .X (),ad .Y (),ad .Z ()],float )))
            except Exception :
                pass 
        exp .Next ()

        # A-VAKASI KURTARMA: some dosyalarda STYLED_ITEM yuze not KATIYA bagli, that zaman
        # face-basina renk haritasi BOS gelir. Olculdu (40 part): boyle 12 parcanin 9'u TEK KATI +
        # TEK RENK (orada kontrast fiziken dogamaz, data siniri) but 3'unde 11-15 KATI and 4-5 AYRIK
        # renk present -- onlar kurtarilabilir. Kati order face sirasindan very more kisa oldugu for
        # eslesme de more safe; yine de SAYI kontrolu is done, tutmazsa vazgecilir.
    if not renk :
        kati_renk =_kati_renkleri (d )
        if len (kati_renk )>1 :
            from OCP .TopAbs import TopAbs_SOLID 
            se =TopExp_Explorer (rd .OneShape (),TopAbs_SOLID )
            kati_yuz =[]
            while se .More ():
                fe =TopExp_Explorer (se .Current (),TopAbs_FACE )
                n_ =0 
                while fe .More ():
                    n_ +=1 ;fe .Next ()
                kati_yuz .append (n_ );se .Next ()
            if len (kati_yuz )==len (kati_renk )and sum (kati_yuz )==len (occ ):
                k =0 
                for n_ ,col in zip (kati_yuz ,kati_renk ):
                    for _ in range (n_ ):
                        if k <len (occ ):
                            occ [k ]["rgb"]=col 
                        k +=1 
                for o in occ :
                    rgb =o .get ("rgb")
                    o ["is_metal"]=bool (rgb is not None and 
                    all (abs (rgb [q ]-METAL_RGB [q ])<=METAL_TOL 
                    for q in range (3 )))
                return occ ,True 

    ok =len (occ )==len (metin )and len (occ )>0 
    if ok and dogrula :
        ok =all (m [0 ]is not None and abs (o [0 ]-m [0 ])<0.01 for o ,m in zip (occ ,metin ))
    out =[]
    for k ,(r ,c ,ax )in enumerate (occ ):
        rgb =metin [k ][1 ]if k <len (metin )else None 
        out .append ({"radius":r ,"com":c ,"axis":ax ,"rgb":rgb ,
        "is_metal":bool (rgb is not None and 
        all (abs (rgb [q ]-METAL_RGB [q ])<=METAL_TOL 
        for q in range (3 )))})
    return out ,bool (ok )


def metal_at (faces ,point ,radius =3.0 ):
    """Verilen noktanin `radius` inside METAL surface present mi + most yakin metale uzaklik."""
    p =np .asarray (point ,float )
    best =1e9 ;near =0 
    for f in faces :
        if not f ["is_metal"]:
            continue 
        d =float (np .linalg .norm (np .maximum (np .maximum (f ["bbox_lo"]-p ,p -f ["bbox_hi"]),0.0 )))
        best =min (best ,d )
        if d <=radius :
            near +=1 
    return near ,(best if best <1e9 else 99.0 )


def read_all_by_order (step_path ,max_shift =4 ):
    """TUM yuzleri (duzlem dahil) RENK + KURESEL geometri with dondur.

    Doner: (faces, ok); faces = [{'type','radius','com','axis','rgb','is_metal'}]

    `read_by_order`'dan IKI FARKI, ikisi de olculmus a eksigi kapatiyor:

    1) SILINDIR DEGIL, TUM YUZLER. Tel kelepcesi silindirik a surface DEGIL, duz metal a
       yuzeydir; only silindirlere bakan a "metale distance" ozelligi asil metali kacirir.
       (q8'de olculen metal_mesafe AUC 0.639 silindir-metaliyle alinmisti.)

    2) SABIT KAYMA (offset) COZULUR, part ATILMAZ. Mutlak face order a parcada full a
       eleman kayiyordu (measured: OCC indeksi = metin indeksi + 1). `read_by_order` bunu
       "dogrulanmadi" sayip parcayi atiyordu; here offset [-max_shift, max_shift] araliginda
       ARANIR and silindir yaricaplariyla DOGRULANIR. Kayma bulunamazsa ok=False.

    Dogrulama yine PARCA BASINA: each two taraf da silindir yaricapi verdigi for eslesme
    varsayimi each parcada yeniden kanitlanir.
    """
    import re as _re 
    import numpy as _np 
    from OCP .STEPControl import STEPControl_Reader 
    from OCP .TopExp import TopExp_Explorer 
    from OCP .TopAbs import TopAbs_FACE 
    from OCP .TopoDS import TopoDS 
    from OCP .BRepAdaptor import BRepAdaptor_Surface 
    from OCP .BRepGProp import BRepGProp 
    from OCP .GProp import GProp_GProps 
    import q7_renk_degeri as _Q 

    _REF =_re .compile (rb"#(\d+)")
    d =_Q .parse (step_path )
    renk =_Q ._renk_haritasi (d )
    metin =[]# (radius|None, rgb) -- varlik-id during
    for i in sorted (k for k ,(t ,a )in d .items ()if t =="ADVANCED_FACE"):
        cyl =[r for r in (int (x )for x in _REF .findall (d [i ][1 ]))
        if r in d and d [r ][0 ]=="CYLINDRICAL_SURFACE"]
        rad =None 
        if cyl :
            try :
                rad =round (float (d [cyl [0 ]][1 ].rsplit (b",",1 )[1 ]),3 )
            except ValueError :
                rad =None 
        metin .append ((rad ,renk .get (i )))

    rd =STEPControl_Reader ()
    rd .ReadFile (step_path )
    rd .TransferRoots ()
    exp =TopExp_Explorer (rd .OneShape (),TopAbs_FACE )
    occ =[]
    while exp .More ():
        f =TopoDS .Face_s (exp .Current ())
        a =BRepAdaptor_Surface (f )
        st =int (a .GetType ())
        rad =None ;ax =None ;com =None 
        if st ==1 :
            try :
                cy =a .Cylinder ()
                rad =round (float (cy .Radius ()),3 )
                ad =cy .Axis ().Direction ();ap =cy .Axis ().Location ()
                ax =_np .array ([ad .X (),ad .Y (),ad .Z ()],float )
                com =_np .array ([ap .X (),ap .Y (),ap .Z ()],float )
            except Exception :
                pass 
        g =GProp_GProps ();BRepGProp .SurfaceProperties_s (f ,g );c =g .CentreOfMass ()
        cen =_np .array ([c .X (),c .Y (),c .Z ()],float )
        occ .append ({"type":st ,"radius":rad ,"com":com if com is not None else cen ,
        "centroid":cen ,"axis":ax })
        exp .Next ()

    def _uyum (s ):
        """offset s for: metin[i] <-> occ[i+s]; silindir yaricaplari ne up to tutuyor."""
        tot =hit =0 
        for i ,(r ,_ )in enumerate (metin ):
            j =i +s 
            if r is None or not (0 <=j <len (occ ))or occ [j ]["radius"]is None :
                continue 
            tot +=1 
            hit +=abs (occ [j ]["radius"]-r )<0.01 
        return hit ,tot 

    best =(0 ,0 ,None )
    for s in range (-max_shift ,max_shift +1 ):
        h ,t =_uyum (s )
        if t >=3 and h >best [0 ]:
            best =(h ,t ,s )
    ok =best [2 ]is not None and best [0 ]==best [1 ]and best [1 ]>=3 
    s =best [2 ]if best [2 ]is not None else 0 
    for i ,(_ ,rgb )in enumerate (metin ):
        j =i +s 
        if 0 <=j <len (occ ):
            occ [j ]["rgb"]=rgb 
    for o in occ :
        rgb =o .get ("rgb")
        o ["rgb"]=rgb 
        o ["is_metal"]=bool (rgb is not None and 
        all (abs (rgb [k ]-METAL_RGB [k ])<=METAL_TOL for k in range (3 )))
    return occ ,bool (ok )


def kontrast_yuzler (faces ):
    """Parcanin BASKIN renginden different renkteki yuzler (alan instead of sayiya according to baskinlik).

    WHY RENK-BAGIMSIZ: `METAL_RGB` single a gumus tonuna sabitti and measured (40 part) ki
    parcalarin **%22'sinde renk VAR but that tona uyan face YOK** -- ureticiler different palet
    kullaniyor. Sabiti genisletmek de wrong olurdu: "gri = metal" demek gri PLASTIK govdeyi
    metal saymak becomes, klemenslerde gri body very yaygin.

    Renk-bagimsiz criterion: a parcada yuzlerin cogunlugu GOVDE'dir; govdeden FARKLI renkteki
    yuzler different malzemedir (kontak, yay, vida). Hangi tonun metal oldugunu BILMEK gerekmez.

    Doner: (kontrast yuzlerin centroid dizisi, baskin renk).

    MEASURED VE KULLANILMIYOR (q12, 80 part / 919 candidate):
        criterion                     AUC     TP ort   FP ort   |AUC-0.5|   kapsama
        metal-tonu (mevcut)      0.287    33.3mm   63.2mm     0.2129      38
        KONTRAST (this fonksiyon)  0.301    19.6mm   44.1mm     0.1986      52
    Kapsama %37 ARTIYOR but AYIRT EDICILIK DUSUYOR (-0.0143). Sebep: "baskin renkten different"
    olcutu metali not HER different rengi yakaliyor -- label alani, sign, ikinci type plastik.
    Yani more very parcada calisiyor but more gurultulu.

    LESSON: kapsama ayirt edicilik DEGILDIR. Ikisini ayri olcmeden birini otekinin instead of koyma.
    Yeniden denenecekse: kontrasti single basina not, metal-tonuyla BIRLIKTE (two ayri column)
    ver and gate karar versin.
    """
    import collections as _c 
    import numpy as _np 
    say =_c .Counter (f ["rgb"]for f in faces if f .get ("rgb")is not None )
    if not say :
        return _np .zeros ((0 ,3 )),None 
    baskin =say .most_common (1 )[0 ][0 ]
    P =[f .get ("centroid",f .get ("com"))for f in faces 
    if f .get ("rgb")is not None and f ["rgb"]!=baskin ]
    return (_np .array (P ,float )if P else _np .zeros ((0 ,3 ))),baskin 


def _kati_renkleri (d ):
    """STYLED_ITEM -> MANIFOLD_SOLID_BREP zincirinden KATI BASINA renk (varlik-id during)."""
    import re as _re 
    _REF =_re .compile (rb"#(\d+)")

    def rgb (sid ,dep =0 ):
        if dep >8 or sid not in d :
            return None 
        t ,a =d [sid ]
        if t =="COLOUR_RGB":
            try :
                return tuple (round (float (x ),3 )for x in a .split (b",")[1 :4 ])
            except Exception :
                return None 
        for r in _REF .findall (a ):
            v =rgb (int (r ),dep +1 )
            if v :
                return v 
        return None 

    KATI =("MANIFOLD_SOLID_BREP","BREP_WITH_VOIDS","SHELL_BASED_SURFACE_MODEL")
    esle ={}
    for i ,(t ,a )in d .items ():
        if t not in ("STYLED_ITEM","OVER_RIDING_STYLED_ITEM"):
            continue 
        refs =[int (x )for x in _REF .findall (a )]
        hed =[r for r in refs if r in d and d [r ][0 ]in KATI ]
        if not hed :
            continue 
        col =None 
        for r in refs :
            if r in hed :
                continue 
            col =rgb (r )
            if col :
                break 
        if col :
            for h in hed :
                esle .setdefault (h ,col )
    return [esle [k ]for k in sorted (esle )]
