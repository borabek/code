# -*- coding: utf-8 -*-
"""Single source of truth for CP (connection-point) derivation ten Scheffler-segmented meshes.

DEFINITION (cp-v3-thesis-connection-classes, 2026-07-20, arbiter-selected): a CP is
  * every **CableEntry** connected component (round OR square/clamp wire entry), PLUS
  * every **Contact** component with thesis insertion depth |v_o - v_s| >= ct_depth_min
    (the depth gate drops flat screw heads / pads) that is not within dedupe_mm of an
    already-accepted CP (CableEntry first; nearby connection features = one terminal).

WHY (evidence, 2026-07-20 -- the "CP = round hole only" fallacy correction):
  * Thesis line 1606 derives the v_o opening midpoint for "Kabelein fuehrung ODER Kontaktierung"
    -- BOTH classes are connection openings; TERMINAL_TYPES was always {CONTACT, CABLE_ENTRY}.
  * The cp-v2 (CableEntry-only) "manufacturer validation" rested ten 2/102 parts; ten the other 2
    known-count parts (0271017 mfr=8, 3209635 mfr=3) it was completely blind.
  * ARBITER (results/cp_def_eval.json, 9 in-scope PXC terminals vs Desktop\\JSON manufacturer
    ConnectionPoints): cp-v2 F1 = 0.000 (finds NONE of the 18 CPs -- PXC clamp entries are
    Contact-labelled); winner = this definition (F1 0.388, recall 0.722). n=18 is small but is
    the only part-matched manufacturer GT; 4 simple candidates raced, no over-fitting.

Point = ConnectionPoint.entry_point (v_o opening midpoint, thesis Abb.44/5.3.6); direction =
axis-snapped outward approach vector. cp-v2 behaviour stays reachable via classes=(CABLE_ENTRY,).

Both viz and eval import `connection_points()` so they never diverge.
"""
import copy 
import numpy as np 
import connector3d 
from cp_geometry import channel_axis ,channel_axis_normals ,outward_along_axis 

import os as _os 
USE_CHANNEL_AXIS =_os .environ .get ("CP_CHANNEL_AXIS","1")not in ("0","false","False")
# D1.1: geometrik yonlendirme. Olculdu but corpus-agirlikli net -0.0036 -> VARSAYILAN KAPALI.
# P9 (2026-07-30): VARSAYILAN OPEN. Dagitilan gate GEOMETRIK yonlendirmeyle cikarilmis
# adaylarla egitildi; closed calistirmak gate'i uyumsuz birakiyordu.
# Olculdu (same 100 part, sizintisiz): 0.7865 -> 0.8016 (+0.0151), two regime de yukseldi.
USE_GEO_ORIENT =_os .environ .get ("CP_GEO_ORIENT","1")not in ("0","false","False")
# Eksen ince ayari (normal-kovaryans). MEASURED VE URUNE ALINDI 2026-07-30 -> varsayilan OPEN.
# Ayni 100 part, sizintisiz, single degisken: robot-hazir F1 (lateral <=2mm VE axis <=10 derece)
# 0.4677 -> 0.4904 (+0.0227); <=5 derece kolunda +0.0210. Tespit F1 +0.0008 (zarar absent).
# Eksen angle hatasi >15 derece which is CP orani %29.2 -> %24.9.
USE_AXIS_NORMALS =_os .environ .get ("CP_AXIS_NORMALS","1")not in ("0","false","False")
# B-rep analitik axis (gmsh/OCC). Olculdu and URUNE ALINDI 2026-07-30 -> varsayilan OPEN.
USE_BREP_AXIS =_os .environ .get ("CP_BREP_AXIS","1")not in ("0","false","False")
# B-rep silindir eslestirme kapilari. Cember-oturtma duzeltmesi yaricaplari 3.5 fold buyuttu,
# that is why ikisi de yeniden ayarlanabilir must be (see. asagidaki cagri yerindeki not).
BREP_MAX_OFF =float (_os .environ .get ("CP_BREP_MAX_OFF","5.0"))
# NOKTAYI analitik silindir eksenine izdusur (YON already oradan aliniyordu, NOKTA alinmiyordu).
# Varsayilan KAPALI -- uctan uca olculmeden acilmaz.
# MEASURED VE KAYBETTI 2026-08-01 (results/r6_izdusum_uctan_uca.json, 200 part, sizintisiz):
#   tespit 0.7439 -> 0.6984 (-0.0454) | robot-hazir 0.4500 -> 0.4054 (-0.0445)
# Aday duzeyinde +2.1 score gorunuyordu. Sebep: noktayi oynatmak gate'in ozelliklerini
# (nn_dist, n_close, clustering) degistiriyor and dagitilan gate IZDUSUMSUZ adaylarla egitildi.
# Acmak for full corpus yeniden uretilip gate yeniden egitilmeli; candidate duzeyi kazanci (+2 score)
# 80 dakikalik yeniden uretimi HAK ETMIYOR. VARSAYILAN KAPALI.
USE_AXIS_PROJECT =_os .environ .get ("CP_AXIS_PROJECT","0")not in ("0","false","False")
# ASAMA IZLEME: a list atanirsa each CP for YON ASAMALARI buraya yazilir (teshis amacli).
# Varsayilan None -> urun yolunda no maliyet absent.
_ASAMA_IZ =None 
BREP_R_MAX =float (_os .environ .get ("CP_BREP_R_MAX","12.0"))

_BODY_CACHE ={}


def _body_of (V ,F ):
    """Ayni mesh for trimesh nesnesini BIR KEZ kur (CP basina yeniden kurmak pahali)."""
    key =(id (V ),id (F ),len (V ),len (F ))
    b =_BODY_CACHE .get (key )
    if b is None :
        import trimesh 
        b =trimesh .Trimesh (np .asarray (V ,float ),np .asarray (F ),process =False )
        _BODY_CACHE .clear ()# single girdilik: parts arasi bellek sizmasin
        _BODY_CACHE [key ]=b 
    return b 

CABLE_ENTRY =int (connector3d .CABLE_ENTRY )
CONTACT =int (connector3d .CONTACT )
CP_CLASSES =(CABLE_ENTRY ,CONTACT )# the two "connection" classes (for the conf mask)
DEFAULT_CLASSES =CP_CLASSES # cp-v3: both thesis connection classes
CT_DEPTH_MIN_MM =1.0 # cp-v3: min insertion depth for a Contact CP
DEDUPE_MM_DEFAULT =10.0 # cp-v3: nearby connection features = one terminal


def _cluster_terminals (kept ,cluster_mm ):
    """PREDICTION-side per-terminal merge: a single manufacturer terminal has several connection
    features (clamp, screw, wire slot) each segmented separately -> single-linkage-merge CPs within
    cluster_mm into ONE CP (the robot inserts before per terminal). Point = cluster centroid; the
    representative (CableEntry-preferred, else largest) keeps direction; source_label = CableEntry if
    any member is CableEntry else Contact. Raised arbiter F1 0.39 -> ~0.65 (precision lever; recall is
    segmentation-bound). NEVER applied to GT (GT stays per-region-component)."""
    if cluster_mm <=0 or len (kept )<=1 :
        return kept 
    pts =np .array ([k ["point"]for k in kept ],float )
    parent =list (range (len (kept )))
    def find (x ):
        while parent [x ]!=x :parent [x ]=parent [parent [x ]];x =parent [x ]
        return x 
    d =np .linalg .norm (pts [:,None ,:]-pts [None ,:,:],axis =2 )
    for i in range (len (kept )):
        for j in range (i +1 ,len (kept )):
            if d [i ,j ]<cluster_mm :parent [find (i )]=find (j )
    groups ={}
    for i in range (len (kept )):
        groups .setdefault (find (i ),[]).append (kept [i ])
    out =[]
    for members in groups .values ():
        rep =next ((m for m in members if m ["source_label"]==CABLE_ENTRY ),
        max (members ,key =lambda m :m ["n_verts"]))
        out .append ({"point":np .mean ([m ["point"]for m in members ],0 ),"direction":rep ["direction"],
        "source_label":CABLE_ENTRY if any (m ["source_label"]==CABLE_ENTRY for m in members )else CONTACT ,
        "n_verts":int (sum (m ["n_verts"]for m in members )),
        "area":float (sum (m ["area"]for m in members )),
        "confidence":float (np .mean ([m ["confidence"]for m in members ])),
        # carry the thesis insertion depth through the merge -- it is what places the CP at
        # the CONTACT (v_s) rather than the mouth (v_o), which is where the MANUFACTURER
        # defines its ConnectionPoint (measured 2026-07-22: same direction, same other two
        # axes, deeper by the insertion depth). Dropping it here silently returned 0.
        "insertion_depth_mm":float (max (m .get ("insertion_depth_mm",0.0 )for m in members )),
        # Yaricap da tasinmali: clustering new dictionary kuruyor and tasinmayan
        # each alan SESSIZCE kayboluyor -- insertion_depth_mm bunu a times yasadi.
        "brep_radius_mm":next ((m .get ("brep_radius_mm")for m in members 
        if m .get ("brep_radius_mm")),None ),
        "n_merged":len (members )})
    return out 


    # Yuvarlama esigi: olculen axis a koordinat eksenine this aciDAN uzaksa egim GERCEK sayilir
    # and korunur. VARSAYILAN 90 = always yuvarla (old davranis) -- because 10 derece with MEASURED
    # and KAYBETTI (asagi bak). Knob duruyor: channel_axis egimi gorebilir hale gelirse single satirla acilir.
SNAP_MAX_DEG =float (_os .environ .get ("CP_SNAP_MAX_DEG","90"))


def _snap_axis (d ,outward_ref ,snap_max_deg =None ):
    """Yonu disari bakacak sekilde yonlendir; SADECE small a duzeltmeyse eksene yuvarla.

    ESKI RATIONALE YANLISTI. Bu fonksiyonun docstring'i "Manufacturer InsertDirections are 100%
    axis-aligned" diyordu and each yonu most yakin koordinat eksenine yuvarliyordu. Korpusta measured
    (8534 manufacturer CP): only %75.6'si eksene hizali, **%19.1'i 10 dereceden extra EGIK**
    (most extra 43.1 derece), and this single ureticiye ozgu not (PXC %16.9, WEI %21.7). Klemenslerde
    tel girisi most zaman acili a huniden gecer -- egim gercektir.

    Sonucu measured: eslesen CP'lerin %18.2'sinde eksenimiz manufacturer yonunden 15-90 derece
    sapiyordu (most ~90), because 45 dereceye yaklasan real a egim YANLIS eksene yuvarlaniyor.
    Hata part basina ya hep ya never gorunuyordu (93 parcanin 19'u %100 wrong) -- single single hard
    opening not, sistematik a yuvarlama hatasi oldugunun isareti.

    MEASURED VE KAYBETTI (2026-07-30, same 100 part, sizintisiz): esigi 10 dereceye cekmek
    robot-hazir F1'i 0.4677 -> 0.4429 (-0.0248) DUSURDU, tespit F1'ini de -0.0044.
    Sebep tanida ortaya output: yuvarlama kalkinca axis angle hatasi neredeyse HIC degismedi
    (>15 derece sapan double %29.2 -> %28.9). Yani olculen ham eksenler de already eksene hizali --
    **channel_axis egimi GOREMIYOR**. Suclu yuvarlama not, a fold more derinde: axis olcumu.
    Bu yuzden varsayilan 90 (always yuvarla) as REVERTED; knob duruyor ki axis olcumu
    duzeldiginde single degiskenle acilabilsin.
    """
    d =np .asarray (d ,float )
    n =float (np .linalg .norm (d ))
    if n <1e-12 :
        return np .asarray (outward_ref ,float )/(np .linalg .norm (outward_ref )+1e-12 )
    d =d /n 
    thr =SNAP_MAX_DEG if snap_max_deg is None else float (snap_max_deg )
    ax =int (np .argmax (np .abs (d )))
    e =np .zeros (3 );e [ax ]=1.0 
    off_deg =np .degrees (np .arccos (min (1.0 ,float (np .abs (d [ax ])))))
    if off_deg >thr :# real egim -- KORU, only isareti disari cevir
        return d if float (np .dot (d ,outward_ref ))>=0 else -d 
    if np .dot (e ,outward_ref )<0 :
        e =-e 
    return e 



def _split_elongated (V ,idx ,split_ratio ):
    """Split ONE fragment only if it really contains SEVERAL openings -- detected by a WAIST.

    WHY NOT "elongated -> slice into k" (the first version): measured against the manufacturer
    arbiter ten 182 Weidmueller parts it recovered 9 real CPs (TP 113 -> 122) but invented 56 false
    ones (FP 141 -> 197), F1 .229 -> .232 = nothing. It slices every elongated fragment, including
    single long openings that must stay whole.

    What the annotator actually pointed at is a DOUBLE-LOBED region: two openings joined by a thin
    neck, which the segmentation paints as one blob and which then yields ONE CP sitting in the gap
    between them. That has a signature: projected onto its principal axis the vertex density is
    BIMODAL -- two humps with a real valley between. So: split at the valley, and only when the
    valley is deep enough to be a genuine neck rather than sampling noise.

    split_ratio is now the VALLEY DEPTH threshold: the dip must fall to <= split_ratio * the smaller
    surrounding peak (0.6 = the neck is at most 60% as dense as the thinner lobe). Returns one array
    per lobe, or a single array when there is no convincing waist.
    """
    P =V [idx ]
    c =P .mean (0 )
    X =P -c 
    try :
        _ ,_ ,Vt =np .linalg .svd (X ,full_matrices =False )
    except np .linalg .LinAlgError :
        return [idx ]
    t =X @Vt [0 ]
    span =float (t .max ()-t .min ())
    if span <=1e-6 or len (idx )<60 :
        return [idx ]
    nb =max (8 ,min (24 ,len (idx )//12 ))
    hist ,edges =np .histogram (t ,bins =nb )
    if hist .max ()<=0 :
        return [idx ]
    h =hist .astype (float )/hist .max ()
    # find the deepest interior valley that has a real peak ten BOTH sides
    best =None 
    for i in range (2 ,nb -2 ):
        left =h [:i ].max ();right =h [i +1 :].max ()
        if left <0.35 or right <0.35 :# need a genuine lobe ten each side
            continue 
        depth =h [i ]/max (min (left ,right ),1e-9 )
        if best is None or depth <best [0 ]:
            best =(depth ,i )
    if best is None or best [0 ]>split_ratio :
        return [idx ]# no convincing neck -> keep whole
    cut =0.5 *(edges [best [1 ]]+edges [best [1 ]+1 ])
    a =idx [t <=cut ];b =idx [t >cut ]
    if len (a )<25 or len (b )<25 :
        return [idx ]
    return [a ,b ]


def connection_points (V ,F ,labels ,min_v =1 ,dedupe_mm =DEDUPE_MM_DEFAULT ,probs =None ,
min_conf =0.0 ,min_area =0.0 ,vertex_conf =0.0 ,conn_promote =0.0 ,
step_path =None ,
classes =DEFAULT_CLASSES ,
axis_snap =True ,ct_depth_min_mm =CT_DEPTH_MIN_MM ,cluster_mm =0.0 ,
outward_min =0.0 ,split_ratio =0.0 ,point_mode ="v_o"):
    """Return a list of CPs: {point(3,), direction(3,), source_label, n_verts, area, confidence}.

    classes      : which label classes count as a CP. Default (CABLE_ENTRY, CONTACT) = cp-v3
                   (arbiter-selected, see module docstring). Pass classes=(CABLE_ENTRY,) for the
                   superseded cp-v2 (CableEntry-only; arbiter F1 0.000 ten the PXC manufacturer set).
    min_v        : min vertices per component. For GT use the definition's gt_min_vertices (1);
                   for PREDICTION use the prediction post-proc value. Kept SEPARATE by the caller.
    conn_promote : PREDICTION only, opt-in (0.0 = off). Promote a vertex whose SUMMED connection
                   probability (CE+CT) reaches this threshold, even when Housing won the argmax.
                   Without it, candidate formation can only ever shrink the argmax mask -- see CC-B.
    vertex_conf  : per-vertex confidence mask (PREDICTION only). Demote connection vertices with
                   class-prob < vertex_conf to Housing before connected-components -> erodes the
                   low-conf bridge the model paints between two openings so they split (recall)
                   and removes low-conf FP blobs (precision). Needs probs. Never applied to GT.
    dedupe_mm    : cross-class merge distance (only meaningful when classes has both). Two
                   SAME-class openings are never merged. 0 disables.
    ct_depth_min_mm : min thesis insertion depth |v_o - v_s| for a CONTACT component to count as a
                   CP (drops flat screw-head/pad pseudo-contacts; cp-v3 arbiter winner). Only
                   applies when CONTACT is in `classes`. 0/None disables the gate.
    cluster_mm   : PREDICTION-side per-terminal merge radius. >0 single-linkage-merges the kept CPs
                   into one-per-terminal centroids (a terminal has several connection features each
                   segmented separately). Raised arbiter F1 0.39 -> ~0.65 (precision). 0 = off (GT
                   default -- never cluster GT). See _cluster_terminals.
    outward_min  : PREDICTION-side precision gate. Drop a CP whose entry point sits in the INNER part
                   of the body along its own outward ray -- outward = ((p-bc).u) / max_v((V-bc).u)
                   with u = (p-bc)/|p-bc|; 1.0 = ten the hull, <0.5 = inner half. MEASURED (product
                   selftrain_120 ten the 9-part PXC arbiter): real manufacturer CPs sit at outward
                   ~0.77, spurious FPs at ~0.54; a round prior in [0.50,0.70] keeps ALL 13 TP
                   (recall 0.722 untouched) and drops 7-11 of 19 FP -> F1 0.520 -> 0.60-0.67. CAVEAT:
                   measured ten n=18 CPs; NOT yet confirmed held-out (batch-2 labels) -> default 0.0
                   (OFF). A recessed real CP (deep gland) could be gated -> keep the threshold low.
                   0 = off (GT default -- never gate GT). Physically: wires enter the OUTER housing.
    split_ratio  : PREDICTION-side splitting of a fragment that merged TWO adjacent openings, now
                   gated ten a WAIST (bimodal vertex density along the principal axis) rather than ten
                   elongation. The value is the valley-depth threshold: split only if the neck falls
                   to <= split_ratio x the thinner lobe (try 0.5-0.7). 0 = off (GT default -- never
                   split GT). Elongation-based slicing was measured and rejected: +9 TP but +56 FP ten
                   the 182-part Weidmueller arbiter. See _split_elongated.
    """
    # B-rep silindirleri (varsa) BIR KEZ yuklenir; diske onbelleklenir.
    _brep_cyl =None 
    _brep_pl =None 
    _brep_axis_point =None 
    if USE_BREP_AXIS and step_path :
        try :
            import brep_axes as _bx 
            _brep_cyl =_bx .cylinders (step_path )
            _brep_pl =_bx .planes (step_path )
            _brep_axis_pl =_bx .axis_from_planes 
            _brep_axis_at =_bx .axis_at 
            _brep_axis_point =getattr (_bx ,"axis_point",None )
            if not len (_brep_cyl [0 ]):
                _brep_cyl =None 
        except Exception :
            _brep_cyl =None 
            _brep_pl =None 

    classes =tuple (int (c )for c in classes )
    V =np .asarray (V ,float );F =np .asarray (F ,int );labels =np .asarray (labels )
    bc =V .mean (0 )
    probs =None if probs is None else np .asarray (probs ,float )
    if probs is not None and vertex_conf >0.0 :
        pc =probs [np .arange (len (labels )),labels ]
        labels =labels .copy ()
        labels [np .isin (labels ,classes )&(pc <vertex_conf )]=int (connector3d .HOUSING )
    if probs is not None and conn_promote >0.0 :
    # YUKSELTME (2026-07-28, CC-B'den turedi): candidate olusumu argmax'tan basliyordu and vertex_conf
    # SADECE dusuruyordu. Sonuc: baglanti olasiligi kayda value but Housing'in under kalan
    # vertex ASLA candidate olamiyordu -- esigi ne up to dusursen dusur. CC-B olcumu: very-CP
    # parcalarinda KACIRILAN GT'lerin %76'sinda yakinda CE+CT olasiligi >= 0.30 present, but
    # argmax orani 0.00. Yani sinyal VARDI, gate kapaliydi.
    # Bu step TOPLAM baglanti olasiligi esigi asan vertexleri baskin baglanti sinifina carries.
    # OPT-IN: varsayilan 0.0 -> mevcut urun davranisi BIREBIR korunur.
        cls =list (classes )
        pconn =probs [:,cls ].sum (1 )
        promote =(~np .isin (labels ,classes ))&(pconn >=conn_promote )
        if promote .any ():
            labels =labels .copy ()
            labels [promote ]=np .asarray (cls )[probs [np .ix_ (np .where (promote )[0 ],cls )].argmax (1 )]
    frags =connector3d .build_fragments (V ,F ,labels ,min_vertices =min_v )
    # CableEntry first so it wins ties ten any cross-class dedupe, then Contact
    frags =sorted ((f for f in frags if int (f .label )in classes ),
    key =lambda f :0 if int (f .label )==CABLE_ENTRY else 1 )
    if split_ratio >0.0 :# PREDICTION only: break a merged row of openings back apart
        split =[]
        for f in frags :
            parts =_split_elongated (V ,np .asarray (f .idx ,int ),split_ratio )
            if len (parts )==1 :
                split .append (f )
            else :
                for pidx in parts :
                    g =copy .copy (f );g .idx =pidx ;split .append (g )
        frags =split 
    kept =[]
    for f in frags :
        idx =np .asarray (f .idx ,int )
        conf =1.0 if probs is None else float (probs [idx ,int (f .label )].mean ())
        area =float (getattr (f ,"area",0.0 ))
        if conf <min_conf or area <min_area :
            continue 
        cp =connector3d .ConnectionPoint ([f ])
        try :
            cp .compute_direction (V ,F ,smooth_subdiv =0 ,body_center =bc )
            d =np .asarray (cp .approach_vector ,float )
        except Exception :
            d =np .asarray (cp .entry_point ,float )-bc 
            # cp-v3: gate CONTACT components by thesis insertion depth (drops flat screw-head/pad
            # pseudo-contacts that are not real connection openings; arbiter-selected, F1 0.388 vs
            # cp-v2's 0.000). Never gates CableEntry.
        if int (f .label )==CONTACT and ct_depth_min_mm and cp .insertion_depth_mm <ct_depth_min_mm :
            continue 
            # THESIS Abb.44/5.3.6: three centroids. Default v_o (opening mouth, where the wire enters).
            # point_mode="v_v" uses the Volumenschwerpunkt (convex-hull centroid) which the thesis says
            # is less mesh-biased than the cluster mean -- test whether it matches the manufacturer CP better.
        if point_mode =="v_v"and getattr (cp ,"v_v",None )is not None :
            p =np .asarray (cp .v_v ,float )
        else :
            p =np .asarray (cp .entry_point ,float )
            # cross-class only: a CableEntry and a co-located Contact are one terminal; two same-class
            # openings are always distinct terminals -> never merged (0444048 3 Contacts ~8-9mm apart).
        if dedupe_mm >0.0 and any (
        np .linalg .norm (p -k ["point"])<dedupe_mm and k ["source_label"]!=int (f .label )
        for k in kept ):
            continue 
        n =np .linalg .norm (d )
        d =d /n if n >1e-6 else (p -bc )/(np .linalg .norm (p -bc )+1e-9 )
        _brep_r =None # axis_snap kapaliyken de tanimli must be
        # Yon a array asamadan geciyor: tez normali -> channel_axis -> yuvarlama ->
        # normal-kovaryans -> B-rep. Hangisinin BOZDUGUNU bilmeden duzeltmek korlemesine becomes.
        _iz ={"tez_normali":d .copy ()}if _ASAMA_IZ is not None else None 
        if axis_snap :
        # ONCE GEOMETRI: acikligin ekseni kanalin kendisinden olculur. Ham tahminin most large
        # bilesenine according to yuvarlamak, prediction kararsizken wrong eksene dusuyordu (measured
        # 2026-07-29: parcalarin yarisinda 0.0 deg, digerlerinde 42-48 deg = wrong axis).
        # channel_axis kararsizsa None returns and old davranisa dusulur.
        # CP_CHANNEL_AXIS=0 with kapatilabilir -- A/B olcumu for (urun degisikligi olculmeden
        # kabul edilmez; this direction duzeltmesi outward_min kapisini da etkiliyor).
            _bd =_body_of (V ,F )
            ax =channel_axis (_bd ,p ,fallback =None )if USE_CHANNEL_AXIS else None 
            if _iz is not None :
                _iz ["channel_axis"]=(ax .copy ()if ax is not None else None )
            d =_snap_axis (ax if ax is not None else d ,p -bc )
            if _iz is not None :
                _iz ["yuvarlanmis"]=d .copy ()
                # EKSEN INCE AYARI (normal-kovaryans). channel_axis + _snap_axis birlikte EGIK
                # kanallari goremiyor: channel_axis only X/Y/Z deniyordu and ray-probe olcutu
                # already hole koni acisindan (atan(r/L) ~ 11 derece) more keskin olamaz; _snap_axis
                # de kalani most yakin eksene yuvarliyordu. Oysa manufacturer yonlerinin %19.1'i eksene
                # 10 dereceden extra egik (8534 CP; PXC %16.9, WEI %21.7) -- tel acili huniden girer.
                # Silindirik kanalda wall normalleri eksene DIKTIR; kovaryansin most small ozvektoru
                # ekseni gives (sentetik: 0-43 derece egimde 0.00 deviation, gurultuyle 0.03-0.07).
                # Yuvarlanmis direction SADECE face secimi for seed; measurement guvensizse None returns and
                # yuvarlanmis direction aynen kalir.
            if USE_AXIS_NORMALS :
            # Yaricap TARANDI (onbellekten, inference absent). x3/max12 -> x2/max8 kismak:
            # very-CP F1 0.5416 -> 0.5463, low-CP axis hatasi (>15d) %20.9 -> %18.9.
            # Belirleyici which is UST SINIR (12 -> 8mm): genis radius dense bloklarda
            # KOMSU acikligin duvarini da topluyordu. Buyutmek (max 20mm) belirgin KOTU.
                _r =float (np .sqrt (max (float (area ),1e-6 )/np .pi ))*2.0 
                _rad =max (3.0 ,min (_r ,8.0 ))
                # IKI TUR: rafine edilen direction new seed becomes. Tohum duzeldikce channel-ici face
                # secimi de duzelir, measurement keskinlesir. Olculdu: 1 kind 0.4904 -> 2 kind 0.4955,
                # 3. and 4. turda DEGISMIYOR (doyuyor) -> 2 kind yeterli, fazlasi empty maliyet.
                for _ in range (2 ):
                    _fine =channel_axis_normals (_bd ,p ,d ,radius =_rad )
                    if _fine is None :
                        break 
                    d =_fine 
                if _iz is not None :
                    _iz ["normal_kovaryans"]=d .copy ()
                    # B-REP EKSENI: STEP silindir yuzeyleri ekseni ANALITIK tasiyor -- prediction etmek
                    # instead of OKU. gmsh'in OCC cekirdegi montaj donusumlerini uyguladigi for
                    # koordinatlar mesh with AYNI cercevede (dogrulandi: 3211485 bbox birebir).
                    # Olculdu (same 100 part, sizintisiz): robot-hazir 0.4904 -> 0.5152 (+0.0248),
                    # very-CP F1 0.5416 -> 0.5558, axis hatasi >15d %24.9 -> %16.0, >45d %19.4 -> %14.2.
                    # Duz metin regex with DENENDI VE OLDU (montaj donusumu absent: CP-silindir mesafesi
                    # medyan 21mm). gmsh sart.
            if USE_BREP_AXIS and _brep_cyl is not None :
            # BU IKI SAYI CARPIK OLCUMLERE GORE AYARLANMISTI: radius yay-centroid'inden
            # kestirilirken ~3.5 fold KUCUK cikiyordu, i.e. r_range upper siniri 12mm gercekte
            # ~42mm demekti (fiilen filtresiz) and axis NOKTASI eksenden ~r up to kaymisti,
            # max_off_mm=5.0 that kaymayi tolere ediyordu. Cember oturtma duzeltmesinden
            # (metinle 327/327 dogrulandi) after ikisi de yeniden ayarlanmali -> P2 taramasi.
            # P2 TARAMASI YAPILDI 2026-08-01 (results/r5_brep_kapi.json), RESULT: FARK YOK.
            # Uctan uca max_off 5.0 / 3.0 / 2.0 -> robot-hazir 0.4500 / 0.4495 / 0.4492,
            # angle<=10 %75.8 / %76.0 / %76.1. Kapi degeri belirleyici DEGIL; 5.0 kaliyor.
            # (Aday duzeyinde this tarama devasa difference gosteriyordu -- that deney NOKTAYI da
            #  izdusuruyordu, urun whereas only YONU aliyor. Aday duzeyi yine yaniltti.)
                _ba ,_brep_r =_brep_axis_at (p ,d ,_brep_cyl ,max_off_mm =BREP_MAX_OFF ,
                max_turn_deg =60.0 ,r_range =(0.5 ,BREP_R_MAX ),
                want_radius =True )
                if _iz is not None :
                    _iz ["brep_eksen"]=(_ba .copy ()if _ba is not None else None )
                if _ba is not None :
                    d =_ba 
                    # NOKTAYI DA EKSENE TASI (2026-08-01). Simdiye up to B-rep'ten only YON
                    # aliniyordu, point mesh kumesinin weight merkezinde kaliyordu. Oysa manufacturer
                    # ConnectionPoint'i acikligin EKSENI UZERINDEDIR and robot-hazir olcutu full
                    # as noktanin GT axis DOGRUSUNA dik mesafesini olcer. Centroid, opening
                    # asimetrik ortuldugunde eksenden kayar (measured: 2mm barini gecemeyen
                    # noktalarin medyani 3.16mm, yarisi 2-3mm bandinda).
                    # EKSENEL konum korunur (mouth derinligi does not change); only eksene DIK deviation
                    # sifirlanir. Silindir secimi direction with AYNI kurala baglidir -- baska a
                    # silindire izdusurmek noktayi bambaska a delige tasirdi.
                    if USE_AXIS_PROJECT and _brep_axis_point is not None :
                        _pp =_brep_axis_point (p ,d ,_brep_cyl ,max_off_mm =BREP_MAX_OFF ,
                        r_range =(0.5 ,BREP_R_MAX ))
                        if _pp is not None :
                            p =_pp 
                elif _brep_pl is not None :
                # I: silindir SUSTUGUNDA yuva kolu. Aciklarin a kismi silindir not,
                # DUZLEM duvarli channel (yuva/kelepce girisi); ekseni yine tum wall
                # normallerine diktir. Olculdu (same 100 part): robot-hazir 0.5272 ->
                # 0.5415, low-CP 0.5236 -> 0.5393, axis >15d %15.8 -> %15.2.
                # SECICI olmak sart: more gevsek ayarlar more COK candidate kullanip more KOTU
                # sonuc veriyor (134/154 kullanim -> 0.5136). Silindir kolunun dersi same.
                    _bp =_brep_axis_pl (p ,d ,_brep_pl ,max_dist_mm =6.0 ,min_faces =4 ,
                    flat_ratio =0.20 ,max_turn_deg =45.0 )
                    if _bp is not None :
                        d =_bp 
                        # YONLENDIRME: _snap_axis isareti BBOX MERKEZINE according to veriyor. Ayni depodaki
                        # cp_geometry.outward_along_axis'in docstring'i this yontemi CURUTUYOR ("kutu testi,
                        # noktanin parcanin neresinde durduguna according to isareti ters cevirir; same yuzeydeki
                        # CP'ler zit yonlere savrulur"). Olculdu 2026-07-30: CP'lerin **%31'inde** (153/496)
                        # bbox isareti geometrik correct yonun TERSI. Bu, gate'in most guclu geometrik ozelligi
                        # which is `outward`i isareti ters besliyordu. Karar residual GERCEK GOVDEYE veriliyor.
                        # DURUM (2026-07-30, measured): bbox isareti CP'lerin %31'inde geometrik correct
                        # yonun TERSI. Duzeltmek very-CP F1'ini 0.6154 -> 0.7767 does (+0.161) AMA
                        # low-CP'yi -0.023 dusuruyor; corpus agirligi (%89.5/%10.5) with net -0.0036,
                        # i.e. kill kriterinin (+0.02) ALTINDA. Dagitilan gate de ESKI yonlendirmeyle
                        # egitildi -- acmak gate'i uyumsuz birakir.
                        # Bu yuzden VARSAYILAN KAPALI. Acmak for: CP_GEO_ORIENT=1 (and gate yeniden uydurulmali,
                        # eslesen model: results/wire_gate_regrow_rt2.pkl).
            if _iz is not None :
                _iz ["son"]=d .copy ();_iz ["nokta"]=np .asarray (p ,float ).copy ()
                _ASAMA_IZ .append (_iz )
            if USE_GEO_ORIENT :
                _od =outward_along_axis (_bd ,p ,d )
                if float (np .dot (d ,_od ))<0 :
                    d =-d 
        kept .append ({"point":p ,"direction":d ,"source_label":int (f .label ),
        "n_verts":int (len (idx )),"area":area ,"confidence":conf ,
        "insertion_depth_mm":float (cp .insertion_depth_mm ),
        "brep_radius_mm":(float (_brep_r )if _brep_r else None )})
    if cluster_mm >0.0 :
        kept =_cluster_terminals (kept ,cluster_mm )# PREDICTION-side one-CP-per-terminal merge
    if outward_min >0.0 and kept :
        gated =[]
        for k in kept :
            u =k ["point"]-bc ;nu =np .linalg .norm (u )
            if nu <1e-9 :
                gated .append (k );continue 
            u =u /nu 
            reach =float (((V -bc )@u ).max ())
            outward =float ((k ["point"]-bc )@u )/(reach +1e-9 )
            if outward >=outward_min :
                gated .append (k )
        kept =gated 
    return kept 
