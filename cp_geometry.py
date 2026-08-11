# -*- coding: utf-8 -*-
"""CP GEOMETRI YARDIMCILARI -- tum olcum/gorsel araclarinin ORTAK kullandigi donusumler.

ANA GERCEK: URETICININ ConnectionPoint'i KONTAK YUVASINDADIR, govdenin ICINDE. Delik/agiz oradan
INSERT YONU boyunca birkac mm oteded, yuzeydedir. Model AGZI tahmin eder. GT'yi yuvada varsayan her
olcum yanlis olur (isaretci kati icinde kalir, teshis yanlis noktaya bakar, tek yonlu eksen
penceresi ters taraftaki agzi kacirir).

2026-07-28 KOK HATA VE COZUMU -- kopyalamadan once oku:
  trimesh'in contains() / ray.intersects_location() fonksiyonlarinin IKISI DE `rtree` paketine
  baglidir ve bu ortamda rtree KURULU DEGIL -> her cagri ModuleNotFoundError atiyordu. Eski kodda
  `except Exception: continue` bunu SESSIZCE yutuyor, fonksiyonlar hicbir sey yapmadan girdiyi geri
  donduruyordu. Sonuc: yuva->agiz tasimasi hic olmadi, disari-yon secimi hic calismadi, ve iki
  fonksiyon da "basarili" gorundu. Ayrica remesh'lenmis govdeler WATERTIGHT DEGIL -- contains()
  dogru arac olmazdi zaten.
  Bu yuzden burada DIS BAGIMLILIK YOK: Moller-Trumbore isin-ucgen kesisimi saf numpy ile yazildi.
  Watertight olmayan meshlerde de calisir, sessizce basarisiz OLMAZ.
"""
import numpy as np


def _mesh_arrays(mesh):
    """trimesh nesnesi ya da (V, F) ciftini kabul et."""
    if hasattr(mesh, "vertices"):
        return np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    V, F = mesh
    return np.asarray(V, float), np.asarray(F, np.int64)


def ray_hits(mesh, origin, direction, max_mm=1e9, eps=1e-6):
    """Isin-ucgen kesisim mesafeleri (artan sirali). Saf numpy, bagimlilik yok.

    Moller-Trumbore. Doner: origin'den itibaren pozitif t mesafeleri (mm), <= max_mm olanlar.
    """
    V, F = _mesh_arrays(mesh)
    o = np.asarray(origin, float); d = np.asarray(direction, float)
    n = float(np.linalg.norm(d))
    if n < 1e-12 or not len(F):
        return np.zeros(0)
    d = d / n
    tri = V[F]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-12
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tv = o - tri[:, 0]
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = np.einsum("j,ij->i", d, qv) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    m = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > eps) & (t <= max_mm)
    t = np.sort(t[m])
    if len(t) < 2:
        return t
    # ORTAK KENAR/KOSE TEKILLESTIRME: isin iki ucgenin paylastigi kenardan gecerse AYNI kesisim iki
    # kez sayilir (kutu yuzeyinde olculdu: [15., 15.]). Bu, parite testini bozar -- icerideki nokta
    # 'disarida' gorunur. Ayni t degerleri tek kesisime indirilir; 1e-7 mm'den yakin iki AYRI yuzey
    # gercek geometride yoktur.
    return t[np.concatenate(([True], np.diff(t) > 1e-7))]


def _reach(mesh, point):
    """Noktadan meshi TAMAMEN gecmeye yetecek isin menzili.

    Sabit bir 'span' YETMEZ: govdeden uzaktaki bir nokta icin isin, cikis yuzeyine varmadan
    kirpilir, kesisim sayisi TEK gorunur ve parite testi 'iceride' der (testte yakalandi).
    Menzil, noktanin uzakligini da icermek zorunda.
    """
    V, _ = _mesh_arrays(mesh)
    c = (V.max(0) + V.min(0)) / 2.0
    span = float(np.linalg.norm(V.max(0) - V.min(0)))
    return float(np.linalg.norm(np.asarray(point, float) - c)) + span * 1.5


def is_inside(mesh, point, probe=None):
    """Parite testi: rastgele bir isin tek sayida yuzey kesiyorsa nokta ICERIDEDIR.

    Watertight olmayan meshlerde tek isin yanilabilir -> 5 isinla oy cokluguna bakilir.
    """
    far = _reach(mesh, point)
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(5, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    odd = sum(len(ray_hits(mesh, point, dd, far)) % 2 == 1 for dd in dirs)
    return odd >= 3


def seat_to_mouth(mesh, seat, direction, max_mm=45.0, **_):
    """Uretici CP'si GOVDENIN ICINDEYSE onu ACIKLIGIN AGZINA tasi: eksen boyunca ILK YUZEY KESISIMI.

    Iki yone de isin at (insert yonunun isareti parcadan parcaya degisir), en yakin kesisimi sec.
    Doner: (agiz_noktasi, isaretli_offset_mm). Kesisim yoksa seat'i aynen dondurur (offset 0).

    ONEMLI (2026-07-28 olcum): CP'nin ICERIDE olmasi PARCAYA GORE DEGISIR -- 3270115'te 16/16 iceride
    ama 3273112'de 0/19, zaten yuzeyde. Bu yuzden once ICERIDE MI diye bakilir; disaridaki nokta
    OYNATILMAZ. (Yoksa bos alandaki bir CP icin isin karsi duvara carpar ve nokta 36mm oteye
    firlatilirdi -- olculdu.)
    """
    d = np.asarray(direction, float); n = float(np.linalg.norm(d))
    seat = np.asarray(seat, float)
    if n < 1e-9:
        return seat, 0.0
    d = d / n
    if not is_inside(mesh, seat):
        return seat, 0.0
    best_t, best_s = None, 1.0
    for sgn in (+1.0, -1.0):
        h = ray_hits(mesh, seat, sgn * d, max_mm)
        if len(h) and (best_t is None or h[0] < best_t):
            best_t, best_s = float(h[0]), sgn
    if best_t is None:
        return seat, 0.0
    return seat + best_s * d * best_t, best_s * best_t


_DIR_CACHE = {}


def _candidate_dirs(n_dirs=None):
    """Aday eksen yonleri (isaretsiz -> yarim kure yeter).

    n_dirs None/3 ise ESKI davranis: yalnizca X, Y, Z. Daha buyukse Fibonacci kuresiyle
    esit dagilimli yarim kure ornegi uretilir; koordinat eksenleri her zaman iceride tutulur
    ki eksene hizali durumlarda eski sonuc birebir korunsun.
    """
    n = 3 if n_dirs is None else int(n_dirs)
    if n <= 3:
        return [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    key = n
    if key in _DIR_CACHE:
        return _DIR_CACHE[key]
    m = 2 * n                                   # tam kure uret, yarisini al
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    theta = np.pi * (1 + 5 ** 0.5) * i
    D = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    D = D[D[:, 2] >= 0]                         # yarim kure (isaretsiz eksen)
    D = np.vstack([np.eye(3), D])               # koordinat eksenleri MUTLAKA denensin
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    out = [d for d in D]
    _DIR_CACHE[key] = out
    return out


def _refine_dirs(best, k=12, spread_deg=12.0):
    """Kazanan yonun cevresinde ince tarama -- kaba izgara cozunurlugunu asmak icin."""
    b = np.asarray(best, float); b /= np.linalg.norm(b) + 1e-12
    tmp = np.array([1.0, 0, 0]) if abs(b[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(b, tmp); u /= np.linalg.norm(u) + 1e-12
    v = np.cross(b, u)
    out = []
    for r in (0.5, 1.0):
        a = np.radians(spread_deg) * r
        for j in range(k):
            t = 2 * np.pi * j / k
            d = np.cos(a) * b + np.sin(a) * (np.cos(t) * u + np.sin(t) * v)
            out.append(d / np.linalg.norm(d))
    return out


def channel_axis(mesh, point, fallback=None, min_gain=1.6, n_dirs=None):
    """Acikligin GERCEK eksenini GEOMETRIDEN bul (isaretsiz eksen dondurur).

    NEDEN: yonu 'ham tahminin en buyuk bileseni' ile en yakin eksene yuvarlamak, ham tahmin
    kararsizken TAMAMEN yanlis eksene yuvarliyordu -- olculdu (2026-07-29): parcalarin yarisinda
    aci hatasi 0.0 derece, diger yarisinda 42-48 derece. Iki kutuplu imza = yanlis eksen secimi.

    FIZIKSEL OLCUT: bir tel kanali, kendi ekseni boyunca IKI TARAFTA DA acik olan tek eksendir
    (bir taraf disari, diger taraf kanalin dibi). Dik eksenler delik yaricapi kadar gidip duvara
    carpar. Bu yuzden her eksen icin min(ileri_serbest, geri_serbest) hesaplanir; kanal ekseni
    bunu maksimize eder.

    Kazanan, ikinciyi min_gain katindan fazla gecemezse karar GUVENSIZ sayilir ve fallback
    dondurulur (kor bir tahmin yerine mevcut degeri korumak daha guvenli).

    n_dirs: denenecek aday yon sayisi. VARSAYILAN 3 = yalnizca X/Y/Z (eski davranis).
    KOK NEDEN BURADAYDI (2026-07-30): bu fonksiyon yalnizca 3 koordinat eksenini deniyordu, yani
    EGIK bir eksen dondurmesi matematiksel olarak imkansizdi. Oysa uretici ConnectionPoint
    yonlerinin %19.1'i eksene 10 dereceden fazla egik (8534 CP'de olculdu, en fazla 43.1 derece;
    PXC %16.9, WEI %21.7) -- klemenste tel acili bir huniden girer. Bu yuzden eslesen CP'lerin
    %18.2'sinde eksenimiz 15-90 derece sapiyordu. n_dirs > 3 verildiginde kure uzerinde gercek
    arama yapilir (kaba tarama + kazananin cevresinde ince tarama).
    """
    p = np.asarray(point, float)
    reach = _reach(mesh, p)
    scores = []
    for e in _candidate_dirs(n_dirs):
        hp = ray_hits(mesh, p, +e, reach)
        hm = ray_hits(mesh, p, -e, reach)
        dp = float(hp[0]) if len(hp) else reach
        dm = float(hm[0]) if len(hm) else reach
        # ACIK TARAF SARTI: gercek bir tel kanali EN AZ BIR TARAFTAN acik havaya cikar (o taraf
        # hicbir yuzeyi kesmez). Yalnizca min(ileri, geri) bakmak bunu istemiyordu ve dar bir
        # oyugu "kanal" sanabiliyordu; sonuc, okun kanalin karsi duvarina 7-13mm mesafede
        # bakmasiydi (olculdu 2026-07-29, PXC.2770943'te 3 isaretci).
        open_side = int(len(hp) == 0) + int(len(hm) == 0)
        scores.append((open_side, min(dp, dm), e))
    # once ACIK TARAFI olan eksenler, sonra daha DERIN kanal
    scores.sort(key=lambda x: (-x[0], -x[1]))
    if scores[0][0] == 0:                          # hicbir eksen disari acilmiyor -> karar yok
        return None if fallback is None else np.asarray(fallback, float)
    best, second = scores[0][1], scores[1][1]
    if best <= 1e-6 or best >= reach * 0.999:      # ya hicbir sey ya her sey acik -> karar yok
        return None if fallback is None else np.asarray(fallback, float)
    if scores[0][0] == scores[1][0] and second > 1e-9 and best < min_gain * second:
        return None if fallback is None else np.asarray(fallback, float)   # net kazanan yok
    win = scores[0]
    if n_dirs is not None and int(n_dirs) > 3:      # kazananin cevresinde INCE tarama
        for e in _refine_dirs(win[2]):
            hp = ray_hits(mesh, p, +e, reach); hm = ray_hits(mesh, p, -e, reach)
            dp = float(hp[0]) if len(hp) else reach
            dm = float(hm[0]) if len(hm) else reach
            cand = (int(len(hp) == 0) + int(len(hm) == 0), min(dp, dm), e)
            if (cand[0], cand[1]) > (win[0], win[1]):
                win = cand
    return win[2]


def outward_along_axis(mesh, point, direction, max_mm=None, **_):
    """Eksen boyunca +d / -d'den hangisi DISARI (bos tarafa) bakiyorsa onu dondur (birim vektor).

    Karari GERCEK GOVDE veriyor: her yonde onundeki yuzey kesisim SAYISINA bakilir. Az kesisim =
    o tarafta daha az malzeme = disari. Beraberlikte ilk yuzeyi daha UZAKTA olan (yani daha genis
    bos alani olan) yon kazanir.

    NEDEN BOUNDING-BOX DEGIL: kutu testi, noktanin parcanin neresinde durduguna gore isareti ters
    cevirir; ayni yuzeydeki CP'ler zit yonlere savrulur (2026-07-28 gorsel hatasi).
    """
    d = np.asarray(direction, float); n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    d = d / n
    if max_mm is None:
        max_mm = _reach(mesh, point)
    hp = ray_hits(mesh, point, +d, max_mm)
    hm = ray_hits(mesh, point, -d, max_mm)
    fp = float(hp[0]) if len(hp) else np.inf
    fm = float(hm[0]) if len(hm) else np.inf

    # IKI REJIM AYRI: nokta malzemenin ICINDEYSE 'disari' = KISA yoldan cikilan taraf. Serbest
    # alandaysa 'disari' = onunde daha AZ malzeme / daha COK bosluk olan taraf. Tek kural ikisine
    # birden uymuyor: icerideki nokta icin 'daha cok bosluk' olcutu ters cevap veriyordu (test).
    if (len(hp) % 2 == 1) or (len(hm) % 2 == 1):
        return +d if fp <= fm else -d
    key_p = (len(hp), -fp)
    key_m = (len(hm), -fm)
    if key_p != key_m:
        return +d if key_p < key_m else -d

    # BERABERLIK -- ve bu NADIR DEGIL: ince terminalde bir agizdan bakinca iki yon de bos kalir
    # (ikisinde de 0 kesisim, ikisinde de fp=fm=inf). Eski kural burada '<=' ile HER ZAMAN +d
    # donuyordu; model yonu ise ISARETSIZ bir eksen oldugu icin isaret KEYFI kaliyor ve ayni
    # yuzeydeki oklar rastgele ters donuyordu (2026-07-29 gorsel hatasi, olculdu).
    # Karari MALZEME DAGILIMI verir: eksen etrafindaki dar silindirde hangi tarafta daha AZ
    # vertex varsa disarisi odur.
    V = np.asarray(mesh.vertices, float)
    p0 = np.asarray(point, float)
    rel = V - p0
    al = rel @ d
    perp = np.linalg.norm(rel - al[:, None] * d[None, :], axis=1)
    span = float(np.linalg.norm(V.max(0) - V.min(0)))
    r_probe = max(2.0, 0.05 * span)
    l_probe = max(4.0, 0.25 * span)
    near = perp <= r_probe
    n_p = int((near & (al > 0) & (al <= l_probe)).sum())
    n_m = int((near & (al < 0) & (al >= -l_probe)).sum())
    if n_p != n_m:
        return +d if n_p < n_m else -d

    # Hala beraberse (gercek delik-boyu gecen kanal): deterministik son kural -- govde
    # merkezinden UZAGA. Bu durumda iki yon de fiziksel olarak gecerlidir.
    c = 0.5 * (V.max(0) + V.min(0))
    return +d if float((p0 - c) @ d) >= 0 else -d


def exit_length(mesh, point, direction, margin=6.0, min_len=0.0, max_len=None):
    """Noktadan `direction` boyunca TUM malzemeyi gecip disari cikmak icin gereken UZUNLUK.

    Isin uzerindeki EN UZAK yuzey kesisimi + margin. Boylece isaretci igne, tam gerektigi kadar
    uzar -- ne govde icinde kaybolur ne de parcadan uzun olur.
    (Onceki cozum boyu tahmin edip 1.6x'lik dongulerle buyutuyordu; 100mm'lik parcada 121mm igne
    uretti. Bu fonksiyon dogru boyu TEK SEFERDE olcer.)
    """
    h = ray_hits(mesh, point, direction, _reach(mesh, point))   # sabit span DEGIL: uzak nokta kirpilirdi
    L = (float(h[-1]) if len(h) else 0.0) + margin
    L = max(L, min_len)
    return min(L, max_len) if max_len else L


def mouths_for(mesh, seats, directions, **kw):
    """Cok sayida GT icin toplu yuva->agiz. Doner: (agizlar Nx3, offsetler N)."""
    if not len(seats):
        return np.zeros((0, 3)), np.zeros(0)
    ms, offs = [], []
    for s, d in zip(np.asarray(seats, float), np.asarray(directions, float)):
        m, o = seat_to_mouth(mesh, s, d, **kw)
        ms.append(m); offs.append(o)
    return np.array(ms), np.array(offs)


def mouth_width(mesh, point, axis, n_dirs=24, max_mm=None):
    """Agizdaki BOSLUGUN gercek genisligi -- kanal eksenine DIK olculur.

    NEDEN: robot_cp eskiden size_mm'i segmentlenmis BOLGENIN alanindan turetiyordu
    (2*sqrt(alan/pi)). Contact sinifi tum kontak yuzeyini kapsadigi icin bu, 4mm'lik
    klemenslerde ~25mm "delik capi" veriyordu -- fiziksel olarak imkansiz. Robotun ihtiyaci
    olan sey telin gecegi acikligin genisligi, o yuzden OLCULUR: agiz noktasindan eksene dik
    yonlerde isin atilir ve ilk yuzeye mesafeye bakilir.

    Doner: (ic_cap_mm, ort_cap_mm) = 2*min ve 2*ortalama mesafe.
    Yuvarlak delikte ikisi esittir; yassi yuvada ic_cap DAR kenari verir (telin sigmasi
    gereken olcu), ort_cap ise yuvanin genel acikligini.
    Hicbir yonde yuzey bulunmazsa (nokta govdenin disindaysa) (0.0, 0.0) doner -- cagiran
    taraf eski tahminine geri donebilsin diye, sessizce uydurma bir sayi URETILMEZ.
    """
    V, _F = _mesh_arrays(mesh)
    if max_mm is None:
        # Menzil ACIKLIKTAN buyuk olmali. Kosegenin %25'i ile denendi ve 15mm'lik bir delik
        # 0.00 dondurdu (isin karsi duvara ULASAMADI) -- yani "olcemedim" gibi gorunen sessiz
        # bir kirpma. %50 gercek terminallerdeki en genis agiz icin fazlasiyla yeterli.
        max_mm = 0.50 * float(np.linalg.norm(V.max(0) - V.min(0)))
    a = np.asarray(axis, float)
    na = float(np.linalg.norm(a))
    if na < 1e-12:
        return 0.0, 0.0
    a = a / na
    # eksene dik ortonormal taban
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, tmp); u /= np.linalg.norm(u) + 1e-12
    v = np.cross(a, u)
    d = []
    for k in range(n_dirs):
        th = 2.0 * np.pi * k / n_dirs
        r = np.cos(th) * u + np.sin(th) * v
        h = ray_hits(mesh, point, r, max_mm)
        if len(h):
            d.append(float(h[0]))
    if not d:
        return 0.0, 0.0
    d = np.asarray(d)
    return round(2.0 * float(d.min()), 2), round(2.0 * float(d.mean()), 2)


import collections as _collections
REJECT = _collections.Counter()      # tani: hangi kapi kac kez None dondurdu


def channel_axis_normals(mesh, point, seed_dir, radius=6.0, min_faces=8, max_turn_deg=60.0,
                         flat_ratio=0.35):
    """Kanal eksenini DUVAR NORMALLERINDEN cikar -- ray-probe'dan cok daha keskin.

    NEDEN BU YONTEM: channel_axis() ray atarak min(ileri,geri) serbest mesafeyi maksimize eder.
    Bu olcut, delik kendi koni acisindan (atan(r/L)) daha iyi cozemez: r=2.5mm L=15mm bir yuvada
    eksenin +-9.5 derecelik konisi icindeki TUM yonler dibe ayni mesafede carpar, yani skor DUZDUR.
    Sentetikte olculdu: egik kanalda kalan hata 9.2 derece, hangi cozunurlukte taransa taransin.
    Gercek terminalde r~2mm L~10mm -> ~11 derece taban sinir. Yetmez.

    SILINDIRIK KANALDA eksen, duvar normallerine DIKTIR. Yani normal kovaryans matrisinin
    EN KUCUK ozdegerine ait ozvektoru eksendir. Sentetik dogrulama: 0-43 derece egimde sapma
    0.00 derece; 0.05mm mesh gurultusuyle 0.03-0.07 derece.

    seed_dir: mevcut (kaba) yon. Yalnizca YUZ SECIMI icin kullanilir -- terminalin DUZ ON YUZU
    de CP'nin yakinindadir ve normalleri eksene PARALEL oldugu icin kovaryansi bozar. Bu yuzden
    yalnizca agiz duzleminin GERISINDEKI (kanal icindeki) yuzler alinir. Sonuc seed'den
    max_turn_deg'den fazla saparsa olcum guvenilmez sayilir ve None doner (kor tahmin yok).
    """
    V, F = _mesh_arrays(mesh)
    p = np.asarray(point, float)
    s = np.asarray(seed_dir, float)
    ns = float(np.linalg.norm(s))
    if ns < 1e-12 or not len(F):
        REJECT["gecersiz_girdi"] += 1
        return None
    s = s / ns
    tri = V[F]
    c = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(nrm, axis=1)                     # 2 x alan
    rel = c - p
    inside = rel @ s <= 0.5                              # agiz duzleminin gerisi = kanal ici
    keep = (a2 > 1e-12) & (np.linalg.norm(rel, axis=1) <= radius) & inside
    if int(keep.sum()) < min_faces:
        REJECT["yuz_yetersiz"] += 1
        return None
    n = nrm[keep] / a2[keep, None]
    w = a2[keep]
    C = (n * w[:, None]).T @ n / w.sum()
    ev, evec = np.linalg.eigh(C)
    if ev[1] <= 1e-9 or ev[0] / max(ev[1], 1e-12) > flat_ratio:
        REJECT["silindirik_degil"] += 1
        return None            # normaller bir duzleme yayilmiyor -> silindirik kanal degil
    d = evec[:, 0]
    d = d / (np.linalg.norm(d) + 1e-12)
    if float(np.dot(d, s)) < 0:
        d = -d
    if np.degrees(np.arccos(min(1.0, abs(float(np.dot(d, s)))))) > max_turn_deg:
        REJECT["tohumdan_uzak"] += 1
        return None            # seed'den kopuk -> guvenilmez
    REJECT["kabul"] += 1
    return d


def channel_axis_robust(mesh, point, seed_dir, radius=8.0, min_faces=8, flat_ratio=0.35,
                        max_turn_deg=90.0):
    """Ekseni COK TOHUMLA olc ve kazanani GEOMETRI sectirsin -- tek tohuma guvenme.

    COZULEN TUZAK (olculdu 2026-07-30): channel_axis_normals yuz secimini TOHUM yone gore
    yapiyor ("agzin gerisi = kanal ici"). Tohum 90 derece yanlissa yanlis yarim-uzay secilir,
    terminalin DUZ ON YUZU orneklemeye girer, normaller uc boyuta yayilir ve fonksiyon
    "silindirik degil" deyip REDDEDER. Red ise yanlis tohumu oldugu gibi birakir.
    Kendi kendini besleyen bir dongu: yanlis tohum -> red -> yanlis tohum. Cok-CP parcalarda
    eslesen CP'lerin %22.3'u 45 dereceden fazla sapiyordu ve bu oran yaricaptan BAGIMSIZDI
    (yaricap taramasi: %22.1-22.3 arasi hic kipirdamadi) -- yani sebep komsu delik kirliligi
    degil, tam olarak bu dongu.

    YONTEM: aday tohumlar = verilen tohum + alti koordinat yonu. Her biri icin yuzler secilir
    ve normal kovaryansi cikarilir. Kazanan, EN SILINDIRIK olan (ev0/ev1 orani en kucuk) --
    yani karari tohum degil geometrinin kendisi verir. Hicbir aday esigi gecemezse None doner
    (kor tahmin uretmez).

    Dikdortgen yuva/kelepçe girisi de gecerlidir: iki paralel duvar + iki uc normalleri yine
    eksene DIKTIR, yani kovaryansin en kucuk ozvektoru yine eksendir. Reddedilen sey silindir
    olmamasi degil, ON YUZUN orneklemeye karismasiydi.
    """
    V, F = _mesh_arrays(mesh)
    p = np.asarray(point, float)
    s0 = np.asarray(seed_dir, float)
    n0 = float(np.linalg.norm(s0))
    if n0 < 1e-12 or not len(F):
        return None
    s0 = s0 / n0
    tri = V[F]
    c = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(nrm, axis=1)
    rel = c - p
    near = (a2 > 1e-12) & (np.linalg.norm(rel, axis=1) <= radius)
    if int(near.sum()) < min_faces:
        REJECT["yuz_yetersiz"] += 1
        return None

    seeds = [s0]
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        seeds += [e, -e]
    best = None
    for sd in seeds:
        keep = near & (rel @ sd <= 0.5)
        if int(keep.sum()) < min_faces:
            continue
        n = nrm[keep] / a2[keep, None]
        w = a2[keep]
        C = (n * w[:, None]).T @ n / w.sum()
        ev, evec = np.linalg.eigh(C)
        if ev[1] <= 1e-9:
            continue
        ratio = ev[0] / ev[1]
        if best is None or ratio < best[0]:
            best = (ratio, evec[:, 0], sd)
    if best is None or best[0] > flat_ratio:
        REJECT["silindirik_degil"] += 1
        return None
    d = best[1] / (np.linalg.norm(best[1]) + 1e-12)
    if float(np.dot(d, s0)) < 0:
        d = -d
    if np.degrees(np.arccos(min(1.0, abs(float(np.dot(d, s0)))))) > max_turn_deg:
        REJECT["tohumdan_uzak"] += 1
        return None
    REJECT["kabul"] += 1
    return d
