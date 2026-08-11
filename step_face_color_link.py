# -*- coding: utf-8 -*-
"""K7.1 -- YUZ<->RENK BAGI: STEP yuzeyini VARLIK NUMARASI uzerinden rengiyle eslestir.

SORUN (daha once iki kez cakildi):
  * `gmsh` rengi tamamen dusuruyor.
  * `STEPCAFControl_Reader` (XCAF) renk tablosunu yukluyor ama bu dosyalarda yuzey basina
    renk DONDURMUYOR -- olculdu: 180/180 ve 108/108 yuzeyde `rgb=None`.
  * Yuzeyleri SIRAYA gore eslestirme denendi, null testinde 2/5 ile cakildi.

COZUM (tahmin yok, acik bag):
  1. STEP metni yuzey->renk bagini ACIKCA veriyor:
     `OVER_RIDING_STYLED_ITEM('',(#stil),#YUZEY,...)` -- ucuncu argument YUZEYIN KENDISI.
     Dogrulandi: 108 stil hedefinin 108'i de `ADVANCED_FACE` (kesisim tam).
  2. OCC tarafinda `XSControl_TransferReader.EntityFromShapeResult()` bir TopoDS_Face'ten
     onu ureten STEP varligini verir; `model.Number(entity)` de varlik numarasini.
  -> Iki tarafta AYNI anahtar (varlik numarasi) oldugu icin sira varsayimi devre disi.

NEDEN BU SINYAL ONEMLI: [[low-cp-information-gap]] sizintili ust sinirla kanitladi ki kalan hata
BILGI sorunu. Tel girisi bir KONTAKTA biter, alet agzi bir kol/yayda -- kanalin dibinde metal
gorunmesi tam da hicbir geometrik sondanin veremedigi fonksiyonel sinyal.
"""
import io
import re
import numpy as np

METAL_RGB = (0.824, 0.824, 0.784)      # tarihsel: ilk gorulen gumusi ton
METAL_TOL = 0.08
import os as _os
_WIDE_METAL = _os.environ.get("CP_METAL_WIDE", "0") not in ("0", "false", "False")


def is_metal(rgb):
    """Metal mi? FIZIKSEL kural -- tek bir RGB'ye sabitlemek kapsamayi %78'de biraktiriyordu.

    Olculdu 2026-07-30: dosyalarda EN AZ IKI metalik gri var -- (0.82,0.82,0.78) ve
    (0.56,0.59,0.60) -- artiKontak metali olarak makul bir bakir tonu (0.69,0.55,0.53).
    Tek RGB sabiti bunlarin yalnizca birini yakaliyordu.

    OLCULDU 2026-07-30 -- GENIS KURAL ZARAR VERDI:
      dar kural  (yalniz gumus, sat<0.10)  -> kapsama %36, CP-F1 katkisi **+0.0078**
      genis kural (sat<0.15 VEYA bakir)    -> kapsama %58, CP-F1 katkisi **-0.0044**
    Kapsamayi artirmak kazanci DUSURDU: eklenen %22 nicelik olarak kapsama ama nitelik olarak
    GURULTU (kontak olmayan gri yuzeyler metal sayiliyor). Bu yuzden DAR kural varsayilan.
    Genis kural CP_METAL_WIDE=1 ile acilabilir (olculdu, tavsiye EDILMEZ).
    """
    if rgb is None:
        return False
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    mx, mn = max(r, g, b), min(r, g, b)
    sat = (mx - mn) / max(mx, 1e-9)
    if _WIDE_METAL:
        if sat < 0.15 and mx > 0.40:
            return True
        if 0.15 <= sat <= 0.45 and mx > 0.50 and r >= g >= b:
            return True
        return False
    return sat < 0.10 and mx > 0.55


def metal_colors_of(step_path, cmap=None):
    """Bu DOSYADA metal olan renk kumesi -- DOSYA-ICI GORELI kural.

    NEDEN GORELI: mutlak esik (sat<0.10) dosyalarin %16'sinda metali kaciriyordu; esigi
    gevsetmek ise kaliteyi bozdu (olculdu: kapsama %36->%58 ama CP-F1 katkisi +0.0078 -> -0.0044,
    cunku kontak olmayan gri yuzeyler de metal sayildi).
    Bir terminalde plastikler DOYGUN, metal en az doygun olandir -- bu, esigi gevsetmek degil
    DOGRU OLCUTU kullanmaktir. Dogrulandi 2026-07-30: iki kuralin da metal buldugu 19 parcada
    AYNI rengi seciyorlar (0 anlasmazlik), ustune 10 parca kazandiriyor.
    """
    if cmap is None:
        cmap = face_colors_from_text(step_path)
    if not cmap:
        return set()
    out = {tuple(np.round(v, 3)) for v in cmap.values() if is_metal(v)}
    if out:
        return out
    def _sat(c):
        mx, mn = max(c), min(c)
        return (mx - mn) / max(mx, 1e-9), mx
    cand = sorted({tuple(np.round(v, 3)) for v in cmap.values()}, key=lambda c: _sat(c)[0])
    return {cand[0]} if cand and _sat(cand[0])[1] > 0.35 else set()


# ---------------------------------------------------------------------------- metin tarafi
_NUM = r"[-+0-9.E]+"


def _entities(txt):
    """#id -> (TIP, ham_argumanlar) sozlugu."""
    out = {}
    for m in re.finditer(r"#(\d+)\s*=\s*([A-Z_0-9]+)\s*\(", txt):
        eid, typ = m.group(1), m.group(2)
        i = m.end(); depth = 1
        while i < len(txt) and depth:
            if txt[i] == "(":
                depth += 1
            elif txt[i] == ")":
                depth -= 1
            i += 1
        out[eid] = (typ, txt[m.end():i - 1])
    return out


def face_colors_from_text(step_path):
    """{yuzey_varlik_no: (r,g,b)} -- STEP metninden, ACIK bag uzerinden.

    Zincir: OVER_RIDING_STYLED_ITEM/STYLED_ITEM -> PRESENTATION_STYLE_ASSIGNMENT -> ...
    -> COLOUR_RGB. Ara tipleri isimle degil, ULASILABILIRLIKLE cozeriz: stilden baslayip
    referans grafinda COLOUR_RGB bulana kadar yuruyoruz. Boylece CAD ureticisinin hangi ara
    tipi kullandigi onemli olmaz.
    """
    txt = io.open(step_path, encoding="latin-1", errors="ignore").read().replace("\n", "")
    ent = _entities(txt)

    def refs(eid):
        return re.findall(r"#(\d+)", ent[eid][1]) if eid in ent else []

    def find_rgb(start, budget=40):
        """Stil varligindan baslayip referanslari izleyerek COLOUR_RGB bul (BFS)."""
        seen, q = set(), [start]
        while q and budget > 0:
            cur = q.pop(0); budget -= 1
            if cur in seen or cur not in ent:
                continue
            seen.add(cur)
            typ, args = ent[cur]
            if typ == "COLOUR_RGB":
                v = re.findall(_NUM, args)
                if len(v) >= 3:
                    try:
                        return tuple(float(x) for x in v[-3:])
                    except ValueError:
                        return None
            q.extend(refs(cur))
        return None

    out = {}
    for eid, (typ, args) in ent.items():
        if typ not in ("OVER_RIDING_STYLED_ITEM", "STYLED_ITEM"):
            continue
        ids = re.findall(r"#(\d+)", args)
        if not ids:
            continue
        # STYLED_ITEM(isim, (stiller), item) -> item, ADVANCED_FACE olan referans
        target = next((i for i in reversed(ids) if ent.get(i, ("", ""))[0] == "ADVANCED_FACE"), None)
        if target is None:
            continue
        for s in ids:
            if s == target:
                continue
            rgb = find_rgb(s)
            if rgb is not None:
                out[target] = rgb
                break
    return out


# ---------------------------------------------------------------------------- OCC tarafi
def faces_with_entity_ids(step_path):
    """[(TopoDS_Face, varlik_no)] -- OCC yuzeyleri, STEP varlik numaralariyla.

    `EntityFromShapeResult(shape, 1)` ters yonde (sonuc->varlik) calisir; mode 1 = 'shape
    sonucunu ureten varlik'. Bulunamazsa o yuzey None ile isaretlenir (SESSIZCE 0 yazilmaz).
    """
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS

    rd = STEPControl_Reader()
    if rd.ReadFile(step_path) != 1:
        raise RuntimeError(f"STEP okunamadi: {step_path}")
    rd.TransferRoots()
    shape = rd.OneShape()
    tr = rd.WS().TransferReader()
    model = rd.StepModel()

    out = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        eid = None
        try:
            e = tr.EntityFromShapeResult(f, 1)
            if e is not None:
                n = model.Number(e)
                eid = str(n) if n else None
        except Exception:
            eid = None
        out.append((f, eid))
        exp.Next()
    return out


def face_geometry(f):
    """Yuzeyin tipi, alani, agirlik merkezi, silindir ekseni/yaricapi."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    KIND = {0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere", 4: "Torus"}
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(f, p)
    c = p.CentreOfMass()
    a = BRepAdaptor_Surface(f)
    k = KIND.get(int(a.GetType()), "Other")
    rad = axis = None
    if k == "Cylinder":
        cyl = a.Cylinder()
        rad = float(cyl.Radius())
        ax = cyl.Axis().Direction()
        axis = (ax.X(), ax.Y(), ax.Z())
    return {"type": k, "area": float(p.Mass()),
            "com": (c.X(), c.Y(), c.Z()), "radius": rad, "axis": axis}


def read_colored_faces(step_path):
    """[{type,area,com,radius,axis,rgb,is_metal,entity}] -- renk BAGLI yuzeyler.

    rgb None ise o yuzeyin rengi BULUNAMADI demektir; 0 ya da 'plastik' varsayilmaz
    (sessiz varsayilan, bu depoda uc kez sahte sinyal uretti).
    """
    cmap = face_colors_from_text(step_path)
    out = []
    for f, eid in faces_with_entity_ids(step_path):
        g = face_geometry(f)
        rgb = cmap.get(eid) if eid else None
        g["entity"] = eid
        g["rgb"] = rgb
        g["is_metal"] = is_metal(rgb)
        out.append(g)
    return out


# ---------------------------------------------------------------------------- geometri: METINDEN
def metal_points_from_text(step_path):
    """METAL yuzeylerin temsili 3B noktalari -- yalnizca STEP METNINDEN, OCC YOK.

    NEDEN OCC YOK: OCC'nin transfer kayitlari bu dosyalarda bag vermiyor -- olculdu:
      * `XCAF` yuzey basina renk dondurmuyor (108/108 rgb=None)
      * `EntityFromShapeResult(f,-1)` varligi KOPYA olarak donduruyor, `model.Number()` 0
      * ters yon `ShapeResult(entity)` hepsinde NULL
      * `model.StringLabel()` "(#0..)" yani doldurulmamis
    Renk ve konumu AYNI kaynaktan (metin) ve AYNI anahtarla (varlik no) okumak, birlestirme
    sorununu tamamen ortadan kaldirir.

    Zincir: ADVANCED_FACE -> yuzey -> AXIS2_PLACEMENT_3D -> CARTESIAN_POINT.
    Doner: [{'entity','type','pt','axis','radius','rgb'}]
    """
    txt = io.open(step_path, encoding="latin-1", errors="ignore").read().replace(chr(10), "")
    ent = _entities(txt)
    cmap = face_colors_from_text(step_path)

    def nums(eid):
        return [float(x) for x in re.findall(_NUM, ent[eid][1])] if eid in ent else []

    def refs(eid):
        return re.findall(r"#(\d+)", ent[eid][1]) if eid in ent else []

    def placement(eid, budget=6):
        """Yuzeyden AXIS2_PLACEMENT_3D'ye in, konum ve eksen dondur."""
        q, seen = [eid], set()
        while q and budget > 0:
            cur = q.pop(0); budget -= 1
            if cur in seen or cur not in ent:
                continue
            seen.add(cur)
            if ent[cur][0] == "AXIS2_PLACEMENT_3D":
                rr = refs(cur)
                loc = next((r for r in rr if ent.get(r, ("",))[0] == "CARTESIAN_POINT"), None)
                axs = [r for r in rr if ent.get(r, ("",))[0] == "DIRECTION"]
                p = nums(loc)[:3] if loc else None
                a = nums(axs[0])[:3] if axs else None
                if p and len(p) == 3:
                    return p, a
            q.extend(refs(cur))
        return None, None

    out = []
    for eid, rgb in cmap.items():
        _met = is_metal(rgb)
        rr = refs(eid)
        surf = next((r for r in rr if ent.get(r, ("",))[0].endswith("SURFACE")
                     or ent.get(r, ("",))[0] in ("PLANE", "CYLINDRICAL_SURFACE",
                                                 "CONICAL_SURFACE", "TOROIDAL_SURFACE")), None)
        if surf is None:
            continue
        typ = ent[surf][0]
        pt, ax = placement(surf)
        if pt is None:
            continue
        rad = None
        if typ == "CYLINDRICAL_SURFACE":
            v = nums(surf)
            rad = v[-1] if v else None
        out.append({"entity": eid, "type": typ, "pt": pt, "axis": ax,
                    "radius": rad, "rgb": rgb, "is_metal": _met})
    return out


def face_vertices_from_text(step_path):
    """Renkli yuzeylerin GERCEK sinir kose noktalari -- yalnizca STEP metninden.

    NEDEN AXIS2_PLACEMENT DEGIL (olculdu, cerceve testi 0/108 ile cakildi):
    bir DUZLEMIN yerel orijini yuzeyin UZERINDE degil; parcanin tamamen disinda olabilir.
    O noktalar X'te 38mm yayilirken mesh yalniz 7mm idi.

    Dogru temsilci: yuzeyin SINIR KOSELERI (VERTEX_POINT). Onlar tanim geregi yuzey uzerinde.
    Zincir: ADVANCED_FACE -> FACE_BOUND -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE
            -> VERTEX_POINT -> CARTESIAN_POINT
    Doner: [{'entity','rgb','is_metal','pts'(N,3),'center'}]
    """
    txt = io.open(step_path, encoding="latin-1", errors="ignore").read().replace(chr(10), "")
    ent = _entities(txt)
    cmap = face_colors_from_text(step_path)

    def refs(eid):
        return re.findall(r"#(\d+)", ent[eid][1]) if eid in ent else []

    def vertices(face_eid, budget=400):
        """Yuzeyden ulasilabilen VERTEX_POINT'lerin koordinatlari."""
        pts, seen, q = [], set(), [face_eid]
        while q and budget > 0:
            cur = q.pop(0); budget -= 1
            if cur in seen or cur not in ent:
                continue
            seen.add(cur)
            typ = ent[cur][0]
            if typ == "VERTEX_POINT":
                for r in refs(cur):
                    if ent.get(r, ("",))[0] == "CARTESIAN_POINT":
                        v = [float(x) for x in re.findall(_NUM, ent[r][1])]
                        if len(v) >= 3:
                            pts.append(v[:3])
                continue
            # yuzeyin KENDI placement'ina inmeyi engelle: yalniz topoloji zincirini izle
            if typ in ("PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "TOROIDAL_SURFACE",
                       "AXIS2_PLACEMENT_3D", "CARTESIAN_POINT", "DIRECTION"):
                continue
            q.extend(refs(cur))
        return np.array(pts, float) if pts else np.zeros((0, 3))

    mcols = metal_colors_of(step_path, cmap)
    out = []
    for eid, rgb in cmap.items():
        pts = vertices(eid)
        if not len(pts):
            continue
        out.append({"entity": eid, "rgb": rgb,
                    "is_metal": tuple(np.round(rgb, 3)) in mcols,
                    "pts": pts, "center": pts.mean(0)})
    return out


def all_vertex_points(step_path):
    """Dosyadaki TUM VERTEX_POINT koordinatlari -- hizalama icin katinin TAMAMI.

    NEDEN: hizalamayi yalnizca RENKLI yuzeylerin koseleriyle yapmak, renkli yuzeyler katinin
    bir KISMINI kapladiginda cakiyor -- olculdu 2026-07-30: 8 parcada bbox farki 4.9-32.4mm
    ve residual >1mm. Kati'nin tamamiyla hizalayip donusumu renkli yuzeylere uygulamak dogru.
    """
    txt = io.open(step_path, encoding="latin-1", errors="ignore").read().replace(chr(10), "")
    ent = _entities(txt)
    pts = []
    for eid, (typ, args) in ent.items():
        if typ != "VERTEX_POINT":
            continue
        for r in re.findall(r"#(\d+)", args):
            if ent.get(r, ("",))[0] == "CARTESIAN_POINT":
                v = [float(x) for x in re.findall(_NUM, ent[r][1])]
                if len(v) >= 3:
                    pts.append(v[:3])
    return np.array(pts, float) if pts else np.zeros((0, 3))


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        fs = read_colored_faces(p)
        got = sum(1 for f in fs if f["rgb"] is not None)
        met = sum(1 for f in fs if f["is_metal"])
        uniq = {tuple(np.round(f["rgb"], 3)) for f in fs if f["rgb"]}
        print(f"{p.split(chr(92))[-1][:44]:<46} yuz {len(fs):>4} | rengi bulunan {got:>4} "
              f"| metal {met:>3} | essiz renk {len(uniq)}")
