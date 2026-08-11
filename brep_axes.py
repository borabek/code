# -*- coding: utf-8 -*-
"""STEP'in ANALITIK silindir eksenleri -- CP yonunu tahmin etmek yerine OKUMAK icin.

NEDEN: eksen yonu bugun olculdu ve robot-hazir F1'in 0.109'unu tek basina o tutuyor
(yanal <=2mm 0.6043 -> eksen sarti eklenince 0.4955). Mesh'ten eksen kestirmenin iki tavani var:
ray-probe delik koni acisindan (atan(r/L) ~ 11 derece) daha keskin olamaz, normal-kovaryans ise
yalnizca silindirik kanallarda tanimli (300 aday "silindirik degil" diye reddedildi).
Oysa STEP dosyasi silindir yuzeyleri ZATEN analitik tasiyor -- tahmin etmeye gerek yok.

NEDEN gmsh, neden regex DEGIL: ilk deneme STEP metnini regex ile okudu ve CP'den silindire
mesafe medyan 21mm / %75'i 111mm cikti. Sebep MONTAJ DONUSUMU: AXIS2_PLACEMENT_3D koordinatlari
parcanin YEREL cercevesinde yazili. gmsh'in OCC cekirdegi donusumleri uygular; dogrulandi:
3211485 icin global bbox [0,0,0]-[8.2,74.1,42.2], meshin bbox'i ile BIREBIR ayni.

Eksen, yuzeyin ORNEKLENEN NORMALLERINDEN cikarilir (silindirde normaller eksene diktir), boylece
gmsh surumune ozgu bir "eksen dondur" API'sine bagimli kalinmaz. Olculdu: silindiriklik
(ev0/ev1) tam silindirlerde 0.0000.
"""
import os, json, hashlib
import numpy as np

CACHE_DIR = "results/brep_cache"
import collections as _collections
REJECT = _collections.Counter()      # tani: hangi kapi kac kez None dondurdu


def _oku(yol):
    """JSON'u KAPATARAK oku. `json.load(open(x))` deseni dosya tanitici SIZDIRIR: bu fonksiyon
    aday basina cagriliyor ve 200 parcalik bir olcumde surec tanitici tukenmesinden SESSIZCE
    oldu (2026-07-31, t8 kosusu; log binlerce ResourceWarning ile doluydu)."""
    with open(yol, "r", encoding="utf-8") as f:
        return json.load(f)


def _yaz(yol, veri):
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(veri, f)


CYL_VER = 2      # v2: yaricap/merkez CEMBER OTURTMA ile (v1 centroid kullaniyordu, YAY'larda yanlis)


def _key(step_path, ver=""):
    st = os.stat(step_path)
    h = hashlib.md5(f"{os.path.abspath(step_path)}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, h + ver + ".json")


def _fit_circle(P, ax):
    """Eksene DIK duzlemde en kucuk kareler cember oturtma -> (eksen uzeri nokta, yaricap).

    NEDEN GEREKLI: STEP delikleri neredeyse her zaman dikis cizgisinden BOLUNMUS silindirik
    yuzler olarak yazilir -- olculdu: bir parcadaki silindirik yuzlerin %0'i tam tur, u-araligi
    ceyrek (1.57) ve yarim (3.14) turlarda yigiliyor. Yay uzerindeki noktalarin ORTALAMASI eksenin
    UZERINDE DEGILDIR; centroid'e olan medyan uzaklik da gercek yaricap degildir. Olculen sonuc:
    metin 0.25mm iken centroid tahmini 0.07mm (3.5 kat kucuk).

    Bu, uc yeri birden bozuyordu: robota giden `size_mm`, `axis_at`'in r_range filtresi (yaricap
    kucuk kestirilince mesru silindirler eleniyordu) ve `max_off_mm` mesafe testi (nokta eksende
    degildi). Cember oturtma yaya karsi bagisiktir: bir yay bile merkezi ve yaricapi belirler.
    """
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(ax @ e1) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ ax) * ax; e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(ax, e1)
    o = P.mean(0)
    rel = P - o
    x = rel @ e1; y = rel @ e2
    # Kasa cebirsel cember: x^2+y^2 + D x + E y + F = 0
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    try:
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return o, float(np.median(np.linalg.norm(rel - (rel @ ax)[:, None] * ax, axis=1)))
    cx, cy = -D / 2.0, -E / 2.0
    r2 = cx * cx + cy * cy - F
    if not np.isfinite(r2) or r2 <= 0:
        return o, float(np.median(np.linalg.norm(rel - (rel @ ax)[:, None] * ax, axis=1)))
    ctr = o + cx * e1 + cy * e2 + ((rel @ ax).mean()) * ax
    return ctr, float(np.sqrt(r2))


def cylinders(step_path, use_cache=True):
    """Doner: (N,3) eksen-uzeri nokta, (N,3) birim eksen, (N,) yaricap -- GLOBAL koordinatlarda.

    Sonuc diske onbelleklenir: gmsh ile STEP okumak parca basina saniyeler suruyor ve ayni
    parca olcum boyunca defalarca geliyor.
    """
    ck = _key(step_path, f"_c{CYL_VER}")
    if use_cache and os.path.exists(ck):
        try:
            d = _oku(ck)
            return (np.array(d["c"], float).reshape(-1, 3),
                    np.array(d["a"], float).reshape(-1, 3),
                    np.array(d["r"], float))
        except Exception:
            pass
    import gmsh
    C, A, R = [], [], []
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        for dim, tag in gmsh.model.getEntities(2):
            try:
                if gmsh.model.getType(dim, tag) != "Cylinder":
                    continue
                b = gmsh.model.getParametrizationBounds(dim, tag)
                us = np.linspace(b[0][0], b[1][0], 7)[1:-1]
                vs = np.linspace(b[0][1], b[1][1], 5)[1:-1]
                pars = np.array([[u, v] for u in us for v in vs], float).ravel()
                if not len(pars):
                    continue
                N = np.asarray(gmsh.model.getNormal(tag, pars), float).reshape(-1, 3)
                P = np.asarray(gmsh.model.getValue(dim, tag, pars), float).reshape(-1, 3)
                if len(N) < 4:
                    continue
                M = N.T @ N / len(N)
                ev, evec = np.linalg.eigh(M)
                ax = evec[:, 0] / (np.linalg.norm(evec[:, 0]) + 1e-12)
                ctr, rad = _fit_circle(P, ax)
                if not np.isfinite(rad) or rad <= 1e-6:
                    continue
                C.append(ctr); A.append(ax); R.append(rad)
            except Exception:
                continue
    finally:
        # finally SART: tek meshlenemeyen STEP'ten sonra gmsh durumu bozulup TUM parcalari
        # surunduruyordu (gmsh-finalize-leak, 2026-07-28).
        try:
            gmsh.finalize()
        except Exception:
            pass
    C = np.asarray(C, float).reshape(-1, 3)
    A = np.asarray(A, float).reshape(-1, 3)
    R = np.asarray(R, float)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            _yaz(ck, {"c": C.tolist(), "a": A.tolist(), "r": R.tolist()})
        except Exception:
            pass
    return C, A, R


def axis_at(point, seed_dir, cyl, max_off_mm=3.0, max_turn_deg=40.0, r_range=(0.5, 12.0),
            want_radius=False):
    """CP'nin uzerinde durdugu silindiri bul ve onun ANALITIK eksenini dondur (yoksa None).

    Eslestirme: CP, silindirin EKSEN CIZGISINE yakin olmali (delik agzinda CP eksendedir).
    Birden fazla aday varsa tohum yone en yakin eksen secilir -- yani B-rep karari verir,
    tohum yalnizca esitlik bozar.

    max_turn_deg: B-rep ekseni tohumdan bu kadar cok sapiyorsa muhtemelen BASKA bir silindire
    (komsu delige, vidaya) eslesmisizdir -> None. Kor duzeltme yapilmaz.
    """
    C, A, R = cyl
    if not len(C):
        return (None, None) if want_radius else None
    p = np.asarray(point, float)
    s = np.asarray(seed_dir, float)
    ns = np.linalg.norm(s)
    if ns < 1e-12:
        return (None, None) if want_radius else None
    s = s / ns
    rel = p - C
    al = (rel * A).sum(1)
    off = np.linalg.norm(rel - al[:, None] * A, axis=1)        # eksen cizgisine dik mesafe
    ok = (off <= max_off_mm) & (R >= r_range[0]) & (R <= r_range[1])
    if not ok.any():
        return (None, None) if want_radius else None
    cand = np.where(ok)[0]
    cos = np.abs(A[cand] @ s)
    j = cand[int(np.argmax(cos))]
    ang = np.degrees(np.arccos(min(1.0, float(np.abs(A[j] @ s)))))
    if ang > max_turn_deg:
        return (None, None) if want_radius else None
    d = A[j]
    d = d if float(np.dot(d, s)) >= 0 else -d
    # Yaricap da ANALITIK: agiz genisligini isinla olcmek yerine silindirin kendi yaricapini
    # kullanmak daha kesin. Isinla olcum, nokta kanal merkezinde degilse EN KUCUK mesafeyi
    # aliyor ve 0.05mm gibi fiziksel olmayan degerler uretebiliyordu (1070018'de 4/13 CP).
    return (d, float(R[j])) if want_radius else d


def planes(step_path, use_cache=True):
    """STEP'in DUZLEM yuzeyleri: (merkez, normal, yaricap) -- GLOBAL koordinatlarda.

    NEDEN: `cylinders()` yalnizca silindirleri okuyor, ama acikliklarin bir kismi YUVA/KELEPCE
    girisi -- silindir degil, DUZLEM duvarlardan olusan bir kanal. Olculdu: normal-kovaryans
    yontemi 300 adayi "silindirik degil" diye reddetti ve o adaylarda eksen hatasi yuksek kaldi.
    Bir yuva kanalinin ekseni de duvar normallerinin HEPSINE diktir; yani ayni fizik, farkli
    yuzey tipi. gmsh 3211485'te 217 yuzeyin 171'ini Plane olarak veriyor.

    'yaricap' = yuzey ornek noktalarinin merkezden en buyuk uzakligi; CP'nin SONSUZ duzleme
    degil GERCEK yuze yakin olup olmadigini elemek icin gerekli.
    """
    ck = _key(step_path).replace(".json", "_pl.json")
    if use_cache and os.path.exists(ck):
        try:
            d = _oku(ck)
            return (np.array(d["c"], float).reshape(-1, 3),
                    np.array(d["n"], float).reshape(-1, 3),
                    np.array(d["r"], float))
        except Exception:
            pass
    import gmsh
    C, N, R = [], [], []
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        for dim, tag in gmsh.model.getEntities(2):
            try:
                if gmsh.model.getType(dim, tag) != "Plane":
                    continue
                b = gmsh.model.getParametrizationBounds(dim, tag)
                us = np.linspace(b[0][0], b[1][0], 5)[1:-1]
                vs = np.linspace(b[0][1], b[1][1], 5)[1:-1]
                pars = np.array([[u, v] for u in us for v in vs], float).ravel()
                if not len(pars):
                    continue
                P = np.asarray(gmsh.model.getValue(dim, tag, pars), float).reshape(-1, 3)
                nn = np.asarray(gmsh.model.getNormal(tag, pars), float).reshape(-1, 3)
                if len(P) < 3:
                    continue
                nrm = nn.mean(0)
                ln = float(np.linalg.norm(nrm))
                if ln < 1e-9:
                    continue
                ctr = P.mean(0)
                C.append(ctr); N.append(nrm / ln)
                R.append(float(np.linalg.norm(P - ctr, axis=1).max()))
            except Exception:
                continue
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    C = np.asarray(C, float).reshape(-1, 3)
    N = np.asarray(N, float).reshape(-1, 3)
    R = np.asarray(R, float)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            _yaz(ck, {"c": C.tolist(), "n": N.tolist(), "r": R.tolist()})
        except Exception:
            pass
    return C, N, R


def axis_from_planes(point, seed_dir, pl, max_dist_mm=6.0, min_faces=3, flat_ratio=0.35,
                     max_turn_deg=60.0):
    """Yuva kanalinin eksenini yakin DUZLEM duvarlarindan cikar (yoksa None).

    Kanal ekseni tum duvar normallerine DIKTIR -> normal kovaryansinin en kucuk ozvektoru.
    Terminalin DUZ ON YUZU de yakindir ve normali eksene PARALEL oldugu icin kovaryansi bozar;
    bu yuzden tohum yone ~paralel normalli yuzler ATILIR (silindir kolunda ayni tuzak yasandi:
    yanlis yarim-uzay secimi olcumu bozup reddedilmesine yol aciyordu).
    """
    C, N, R = pl
    if not len(C):
        return None
    p = np.asarray(point, float)
    s = np.asarray(seed_dir, float)
    ns = float(np.linalg.norm(s))
    if ns < 1e-12:
        return None
    s = s / ns
    d = np.linalg.norm(C - p, axis=1)
    near = d <= np.maximum(max_dist_mm, R)          # gercek yuze yakin mi (sonsuz duzleme degil)
    # ON YUZ ELEMESI: normali eksene ~paralel olan duzlem kanal duvari DEGIL, agzin kapagidir
    near &= np.abs(N @ s) < 0.80
    if int(near.sum()) < min_faces:
        REJECT["duzlem_yetersiz"] += 1
        return None
    n = N[near]
    M = n.T @ n / len(n)
    ev, evec = np.linalg.eigh(M)
    if ev[1] <= 1e-9 or ev[0] / max(ev[1], 1e-12) > flat_ratio:
        REJECT["duzlem_kanal_degil"] += 1
        return None
    ax = evec[:, 0] / (np.linalg.norm(evec[:, 0]) + 1e-12)
    if float(np.dot(ax, s)) < 0:
        ax = -ax
    if np.degrees(np.arccos(min(1.0, abs(float(np.dot(ax, s)))))) > max_turn_deg:
        REJECT["duzlem_tohumdan_uzak"] += 1
        return None
    REJECT["duzlem_kabul"] += 1
    return ax


def axis_point(point, seed_dir, cyl, max_off_mm=3.0, r_range=(0.5, 12.0), max_turn_deg=60.0):
    """CP'yi eslesen silindirin EKSEN CIZGISI uzerine izdusur (eksenel konumu KORUYARAK).

    `axis_at` YONU dondurur; bu NOKTAYI tasir. Ikisi ayri: uretici ConnectionPoint'i acikligin
    ekseni UZERINDEDIR, ama bizim noktamiz mesh kumesinin agirlik merkezidir ve aciklik
    asimetrik ortuldugunde eksenden kayar.

    Secim kurali `axis_at` ile AYNI olmali -- baska bir silindire izdusurmek noktayi
    tamamen baska bir delige tasirdi.
    """
    C, A, R = cyl
    if not len(C):
        return None
    p = np.asarray(point, float)
    s = np.asarray(seed_dir, float)
    ns = np.linalg.norm(s)
    if ns < 1e-12:
        return None
    s = s / ns
    rel = p - C
    al = (rel * A).sum(1)
    off = np.linalg.norm(rel - al[:, None] * A, axis=1)
    ok = (off <= max_off_mm) & (R >= r_range[0]) & (R <= r_range[1])
    if not ok.any():
        return None
    cand = np.where(ok)[0]
    j = cand[int(np.argmax(np.abs(A[cand] @ s)))]
    if np.degrees(np.arccos(min(1.0, float(np.abs(A[j] @ s))))) > max_turn_deg:
        return None
    return C[j] + float((p - C[j]) @ A[j]) * A[j]
