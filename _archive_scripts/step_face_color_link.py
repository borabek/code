# -*- coding: utf-8 -*-
"""K7.1 -- YUZ<->RENK BAGI: STEP yuzeyini VARLIK NUMARASI uzerinden rengiyle eslestir.

SORUN (more before two times cakildi):
  * `gmsh` rengi tamamen dusuruyor.
  * `STEPCAFControl_Reader` (XCAF) renk tablosunu yukluyor but this dosyalarda surface basina
    renk DONDURMUYOR -- measured: 180/180 and 108/108 yuzeyde `rgb=None`.
  * Yuzeyleri SIRAYA according to eslestirme was tried, null testinde 2/5 with cakildi.

COZUM (prediction absent, open bag):
  1. STEP metni surface->renk bagini ACIKCA veriyor:
     `OVER_RIDING_STYLED_ITEM('',(#stil),#YUZEY,...)` -- ucuncu argument YUZEYIN KENDISI.
     Dogrulandi: 108 stil hedefinin 108'i de `ADVANCED_FACE` (kesisim full).
  2. OCC tarafinda `XSControl_TransferReader.EntityFromShapeResult()` a TopoDS_Face'ten
     onu ureten STEP varligini gives; `model.Number(entity)` de varlik numarasini.
  -> Iki tarafta AYNI anahtar (varlik numarasi) oldugu for order varsayimi devre disi.

WHY BU SINYAL IMPORTANT: [[low-cp-information-gap]] sizintili upper sinirla kanitladi ki remaining error
BILGI sorunu. Tel girisi a KONTAKTA biter, alet agzi a arm/yayda -- kanalin dibinde metal
gorunmesi full da no geometrik sondanin veremedigi fonksiyonel sinyal.
"""
import io 
import re 
import numpy as np 

METAL_RGB =(0.824 ,0.824 ,0.784 )# tarihsel: first gorulen gumusi ton
METAL_TOL =0.08 
import os as _os 
_WIDE_METAL =_os .environ .get ("CP_METAL_WIDE","0")not in ("0","false","False")


def is_metal (rgb ):
    """Metal mi? FIZIKSEL rule -- single a RGB'ye sabitlemek kapsamayi %78'de biraktiriyordu.

    Olculdu 2026-07-30: dosyalarda EN AZ IKI metalik gri present -- (0.82,0.82,0.78) and
    (0.56,0.59,0.60) -- artiKontak metali as makul a bakir tonu (0.69,0.55,0.53).
    Tek RGB sabiti bunlarin only birini yakaliyordu.

    MEASURED 2026-07-30 -- GENIS RULE ZARAR VERDI:
      dar rule  (only gumus, sat<0.10)  -> kapsama %36, CP-F1 katkisi **+0.0078**
      genis rule (sat<0.15 VEYA bakir)    -> kapsama %58, CP-F1 katkisi **-0.0044**
    Kapsamayi artirmak kazanci DUSURDU: added %22 nicelik as kapsama but nitelik as
    GURULTU (kontak olmayan gri yuzeyler metal sayiliyor). Bu yuzden DAR rule default.
    Genis rule CP_METAL_WIDE=1 with acilabilir (measured, tavsiye EDILMEZ).
    """
    if rgb is None :
        return False 
    r ,g ,b =float (rgb [0 ]),float (rgb [1 ]),float (rgb [2 ])
    mx ,mn =max (r ,g ,b ),min (r ,g ,b )
    sat =(mx -mn )/max (mx ,1e-9 )
    if _WIDE_METAL :
        if sat <0.15 and mx >0.40 :
            return True 
        if 0.15 <=sat <=0.45 and mx >0.50 and r >=g >=b :
            return True 
        return False 
    return sat <0.10 and mx >0.55 


def metal_colors_of (step_path ,cmap =None ):
    """Bu DOSYADA metal which is renk kumesi -- DOSYA-ICI GORELI rule.

    WHY GORELI: mutlak threshold (sat<0.10) dosyalarin %16'sinda metali kaciriyordu; esigi
    gevsetmek whereas kaliteyi bozdu (measured: kapsama %36->%58 but CP-F1 katkisi +0.0078 -> -0.0044,
    because kontak olmayan gri yuzeyler de metal sayildi).
    Bir terminalde plastikler DOYGUN, metal at least doygun olandir -- this, esigi gevsetmek not
    DOGRU OLCUTU kullanmaktir. Dogrulandi 2026-07-30: two kuralin da metal buldugu 19 parcada
    AYNI rengi seciyorlar (0 anlasmazlik), ustune 10 part kazandiriyor.
    """
    if cmap is None :
        cmap =face_colors_from_text (step_path )
    if not cmap :
        return set ()
    out ={tuple (np .round (v ,3 ))for v in cmap .values ()if is_metal (v )}
    if out :
        return out 
    def _sat (c ):
        mx ,mn =max (c ),min (c )
        return (mx -mn )/max (mx ,1e-9 ),mx 
    cand =sorted ({tuple (np .round (v ,3 ))for v in cmap .values ()},key =lambda c :_sat (c )[0 ])
    return {cand [0 ]}if cand and _sat (cand [0 ])[1 ]>0.35 else set ()


    # ---------------------------------------------------------------------------- metin tarafi
_NUM =r"[-+0-9.E]+"


def _entities (txt ):
    """#id -> (TIP, ham_argumanlar) sozlugu."""
    out ={}
    for m in re .finditer (r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\(",txt ):
        eid ,typ =m .group (1 ),m .group (2 )
        i =m .end ();depth =1 
        while i <len (txt )and depth :
            if txt [i ]=="(":
                depth +=1 
            elif txt [i ]==")":
                depth -=1 
            i +=1 
        out [eid ]=(typ ,txt [m .end ():i -1 ])
    return out 


def face_colors_from_text (step_path ):
    """{yuzey_varlik_no: (r,g,b)} -- STEP metninden, OPEN bag uzerinden.

    Zincir: OVER_RIDING_STYLED_ITEM/STYLED_ITEM -> PRESENTATION_STYLE_ASSIGNMENT -> ...
    -> COLOUR_RGB. Ara tipleri isimle not, ULASILABILIRLIKLE cozeriz: stilden baslayip
    referans grafinda COLOUR_RGB bulana up to yuruyoruz. Boylece CAD ureticisinin hangi ara
    tipi kullandigi onemli olmaz.
    """
    txt =io .open (step_path ,encoding ="latin-1",errors ="ignore").read ().replace ("\n","")
    ent =_entities (txt )

    def refs (eid ):
        return re .findall (r"#(\d+)",ent [eid ][1 ])if eid in ent else []

    def find_rgb (start ,budget =40 ):
        """Stil varligindan baslayip referanslari izleyerek COLOUR_RGB bul (BFS)."""
        seen ,q =set (),[start ]
        while q and budget >0 :
            cur =q .pop (0 );budget -=1 
            if cur in seen or cur not in ent :
                continue 
            seen .add (cur )
            typ ,args =ent [cur ]
            if typ =="COLOUR_RGB":
                v =re .findall (_NUM ,args )
                if len (v )>=3 :
                    try :
                        return tuple (float (x )for x in v [-3 :])
                    except ValueError :
                        return None 
            q .extend (refs (cur ))
        return None 

    out ={}
    for eid ,(typ ,args )in ent .items ():
        if typ not in ("OVER_RIDING_STYLED_ITEM","STYLED_ITEM"):
            continue 
        ids =re .findall (r"#(\d+)",args )
        if not ids :
            continue 
            # STYLED_ITEM(isim, (stiller), item) -> item, ADVANCED_FACE which is referans
        target =next ((i for i in reversed (ids )if ent .get (i ,("",""))[0 ]=="ADVANCED_FACE"),None )
        if target is None :
            continue 
        for s in ids :
            if s ==target :
                continue 
            rgb =find_rgb (s )
            if rgb is not None :
                out [target ]=rgb 
                break 
    return out 


    # ---------------------------------------------------------------------------- OCC tarafi
def faces_with_entity_ids (step_path ):
    """[(TopoDS_Face, varlik_no)] -- OCC yuzeyleri, STEP varlik numaralariyla.

    `EntityFromShapeResult(shape, 1)` ters yonde (sonuc->varlik) works; mode 1 = 'shape
    sonucunu ureten varlik'. Bulunamazsa that surface None with isaretlenir (SESSIZCE 0 yazilmaz).
    """
    from OCP .STEPControl import STEPControl_Reader 
    from OCP .TopExp import TopExp_Explorer 
    from OCP .TopAbs import TopAbs_FACE 
    from OCP .TopoDS import TopoDS 

    rd =STEPControl_Reader ()
    if rd .ReadFile (step_path )!=1 :
        raise RuntimeError (f"STEP okunamadi: {step_path }")
    rd .TransferRoots ()
    shape =rd .OneShape ()
    tr =rd .WS ().TransferReader ()
    model =rd .StepModel ()

    out =[]
    exp =TopExp_Explorer (shape ,TopAbs_FACE )
    while exp .More ():
        f =TopoDS .Face_s (exp .Current ())
        eid =None 
        try :
            e =tr .EntityFromShapeResult (f ,1 )
            if e is not None :
                n =model .Number (e )
                eid =str (n )if n else None 
        except Exception :
            eid =None 
        out .append ((f ,eid ))
        exp .Next ()
    return out 


def face_geometry (f ):
    """Yuzeyin tipi, alani, weight merkezi, silindir ekseni/yaricapi."""
    from OCP .BRepGProp import BRepGProp 
    from OCP .GProp import GProp_GProps 
    from OCP .BRepAdaptor import BRepAdaptor_Surface 
    KIND ={0 :"Plane",1 :"Cylinder",2 :"Cone",3 :"Sphere",4 :"Torus"}
    p =GProp_GProps ()
    BRepGProp .SurfaceProperties_s (f ,p )
    c =p .CentreOfMass ()
    a =BRepAdaptor_Surface (f )
    k =KIND .get (int (a .GetType ()),"Other")
    rad =axis =None 
    if k =="Cylinder":
        cyl =a .Cylinder ()
        rad =float (cyl .Radius ())
        ax =cyl .Axis ().Direction ()
        axis =(ax .X (),ax .Y (),ax .Z ())
    return {"type":k ,"area":float (p .Mass ()),
    "com":(c .X (),c .Y (),c .Z ()),"radius":rad ,"axis":axis }


def read_colored_faces (step_path ):
    """[{type,area,com,radius,axis,rgb,is_metal,entity}] -- renk BAGLI yuzeyler.

    rgb None whereas that yuzeyin rengi BULUNAMADI demektir; 0 ya da 'plastik' varsayilmaz
    (silent default, this depoda three times fake sinyal uretti).
    """
    cmap =face_colors_from_text (step_path )
    out =[]
    for f ,eid in faces_with_entity_ids (step_path ):
        g =face_geometry (f )
        rgb =cmap .get (eid )if eid else None 
        g ["entity"]=eid 
        g ["rgb"]=rgb 
        g ["is_metal"]=is_metal (rgb )
        out .append (g )
    return out 


    # ---------------------------------------------------------------------------- geometri: METINDEN
def metal_points_from_text (step_path ):
    """METAL yuzeylerin temsili 3B noktalari -- only STEP METNINDEN, OCC YOK.

    WHY OCC YOK: OCC'nin transfer kayitlari this dosyalarda bag vermiyor -- measured:
      * `XCAF` surface basina renk dondurmuyor (108/108 rgb=None)
      * `EntityFromShapeResult(f,-1)` varligi KOPYA as donduruyor, `model.Number()` 0
      * ters direction `ShapeResult(entity)` hepsinde NULL
      * `model.StringLabel()` "(#0..)" i.e. doldurulmamis
    Renk and konumu AYNI kaynaktan (metin) and AYNI anahtarla (varlik no) okumak, birlestirme
    sorununu tamamen ortadan kaldirir.

    Zincir: ADVANCED_FACE -> surface -> AXIS2_PLACEMENT_3D -> CARTESIAN_POINT.
    Doner: [{'entity','type','pt','axis','radius','rgb'}]
    """
    txt =io .open (step_path ,encoding ="latin-1",errors ="ignore").read ().replace (chr (10 ),"")
    ent =_entities (txt )
    cmap =face_colors_from_text (step_path )

    def nums (eid ):
        return [float (x )for x in re .findall (_NUM ,ent [eid ][1 ])]if eid in ent else []

    def refs (eid ):
        return re .findall (r"#(\d+)",ent [eid ][1 ])if eid in ent else []

    def placement (eid ,budget =6 ):
        """Yuzeyden AXIS2_PLACEMENT_3D'ye in, konum and axis dondur."""
        q ,seen =[eid ],set ()
        while q and budget >0 :
            cur =q .pop (0 );budget -=1 
            if cur in seen or cur not in ent :
                continue 
            seen .add (cur )
            if ent [cur ][0 ]=="AXIS2_PLACEMENT_3D":
                rr =refs (cur )
                loc =next ((r for r in rr if ent .get (r ,("",))[0 ]=="CARTESIAN_POINT"),None )
                axs =[r for r in rr if ent .get (r ,("",))[0 ]=="DIRECTION"]
                p =nums (loc )[:3 ]if loc else None 
                a =nums (axs [0 ])[:3 ]if axs else None 
                if p and len (p )==3 :
                    return p ,a 
            q .extend (refs (cur ))
        return None ,None 

    out =[]
    for eid ,rgb in cmap .items ():
        _met =is_metal (rgb )
        rr =refs (eid )
        surf =next ((r for r in rr if ent .get (r ,("",))[0 ].endswith ("SURFACE")
        or ent .get (r ,("",))[0 ]in ("PLANE","CYLINDRICAL_SURFACE",
        "CONICAL_SURFACE","TOROIDAL_SURFACE")),None )
        if surf is None :
            continue 
        typ =ent [surf ][0 ]
        pt ,ax =placement (surf )
        if pt is None :
            continue 
        rad =None 
        if typ =="CYLINDRICAL_SURFACE":
            v =nums (surf )
            rad =v [-1 ]if v else None 
        out .append ({"entity":eid ,"type":typ ,"pt":pt ,"axis":ax ,
        "radius":rad ,"rgb":rgb ,"is_metal":_met })
    return out 


def face_vertices_from_text (step_path ):
    """Renkli yuzeylerin GERCEK boundary kose noktalari -- only STEP metninden.

    WHY AXIS2_PLACEMENT DEGIL (measured, cerceve testi 0/108 with cakildi):
    a DUZLEMIN yerel orijini yuzeyin UZERINDE not; parcanin tamamen outside may be.
    O points X'te 38mm yayilirken mesh only 7mm idi.

    Dogru temsilci: yuzeyin SINIR KOSELERI (VERTEX_POINT). Onlar tanim geregi surface ten.
    Zincir: ADVANCED_FACE -> FACE_BOUND -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE
            -> VERTEX_POINT -> CARTESIAN_POINT
    Doner: [{'entity','rgb','is_metal','pts'(N,3),'center'}]
    """
    txt =io .open (step_path ,encoding ="latin-1",errors ="ignore").read ().replace (chr (10 ),"")
    ent =_entities (txt )
    cmap =face_colors_from_text (step_path )

    def refs (eid ):
        return re .findall (r"#(\d+)",ent [eid ][1 ])if eid in ent else []

    def vertices (face_eid ,budget =400 ):
        """Yuzeyden ulasilabilen VERTEX_POINT'lerin koordinatlari."""
        pts ,seen ,q =[],set (),[face_eid ]
        while q and budget >0 :
            cur =q .pop (0 );budget -=1 
            if cur in seen or cur not in ent :
                continue 
            seen .add (cur )
            typ =ent [cur ][0 ]
            if typ =="VERTEX_POINT":
                for r in refs (cur ):
                    if ent .get (r ,("",))[0 ]=="CARTESIAN_POINT":
                        v =[float (x )for x in re .findall (_NUM ,ent [r ][1 ])]
                        if len (v )>=3 :
                            pts .append (v [:3 ])
                continue 
                # yuzeyin KENDI placement'ina inmeyi engelle: only topoloji zincirini izle
            if typ in ("PLANE","CYLINDRICAL_SURFACE","CONICAL_SURFACE","TOROIDAL_SURFACE",
            "AXIS2_PLACEMENT_3D","CARTESIAN_POINT","DIRECTION"):
                continue 
            q .extend (refs (cur ))
        return np .array (pts ,float )if pts else np .zeros ((0 ,3 ))

    mcols =metal_colors_of (step_path ,cmap )
    out =[]
    for eid ,rgb in cmap .items ():
        pts =vertices (eid )
        if not len (pts ):
            continue 
        out .append ({"entity":eid ,"rgb":rgb ,
        "is_metal":tuple (np .round (rgb ,3 ))in mcols ,
        "pts":pts ,"center":pts .mean (0 )})
    return out 


def all_vertex_points (step_path ):
    """Dosyadaki TUM VERTEX_POINT koordinatlari -- hizalama for katinin TAMAMI.

    WHY: hizalamayi only RENKLI yuzeylerin koseleriyle yapmak, renkli yuzeyler katinin
    a KISMINI kapladiginda cakiyor -- measured 2026-07-30: 8 parcada bbox farki 4.9-32.4mm
    and residual >1mm. Kati'nin tamamiyla hizalayip donusumu renkli yuzeylere uygulamak correct.
    """
    txt =io .open (step_path ,encoding ="latin-1",errors ="ignore").read ().replace (chr (10 ),"")
    ent =_entities (txt )
    pts =[]
    for eid ,(typ ,args )in ent .items ():
        if typ !="VERTEX_POINT":
            continue 
        for r in re .findall (r"#(\d+)",args ):
            if ent .get (r ,("",))[0 ]=="CARTESIAN_POINT":
                v =[float (x )for x in re .findall (_NUM ,ent [r ][1 ])]
                if len (v )>=3 :
                    pts .append (v [:3 ])
    return np .array (pts ,float )if pts else np .zeros ((0 ,3 ))


if __name__ =="__main__":
    import sys 
    for p in sys .argv [1 :]:
        fs =read_colored_faces (p )
        got =sum (1 for f in fs if f ["rgb"]is not None )
        met =sum (1 for f in fs if f ["is_metal"])
        uniq ={tuple (np .round (f ["rgb"],3 ))for f in fs if f ["rgb"]}
        print (f"{p .split (chr (92 ))[-1 ][:44 ]:<46} yuz {len (fs ):>4} | rengi found {got :>4} "
        f"| metal {met :>3} | essiz renk {len (uniq )}")
