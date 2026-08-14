# -*- coding: utf-8 -*-
"""BAGIMSIZ CIZIM DENETCISI -- uretilmis GLB'yi DISKTEN GERI OKUR and 9 maddeyi sinar.

WHY SEPARATE DOSYA: 2026-07-28'de gorsellestirme three times "duzeltildi" and ucunde de own kendini
"gecti" ilan etti; because dogrulama cizen kodun ICINDEYDI and same bozuk cagriyi kullaniyordu
(rtree yoklugunda trimesh.contains each seferinde patliyor, except yutuyordu). Bu denetci cizim
sirasindaki HICBIR ara degiskene erisemez: single girdisi .glb + .receipt.json.

Kullanim:
    .venv/Scripts/python.exe glb_audit.py                 # results/robot_glb icindeki HEPSI
    .venv/Scripts/python.exe glb_audit.py a.glb b.glb     # secili dosyalar
    .venv/Scripts/python.exe glb_audit.py --json rapor.json

Cikis kodu: 0 = all of them gecti, 1 = at least a ihlal. Sessiz gecis YOKTUR: denetleyemedigi seyi
"gecti" saymaz, DENETLENEMEDI der and this da ihlaldir.
"""
import os ,sys ,json ,glob 
import numpy as np 
import trimesh 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
from viz_contract import (VIZ_COLORS ,GT_ROLES ,PRED_ROLES ,classify_colors ,receipt_path ,
MODES ,ROBOT_ROLE ,COMPONENTS_PER_MARKER as CPM ,
MAX_MARKER_OFFSET_MM )
from cp_geometry import ray_hits ,is_inside 

TOL_BASE_MM =0.05 # M4: baseline konumu toleransi
TOL_F1 =0.005 # M8: file adi F1 toleransi
TOL_CP_MM =MAX_MARKER_OFFSET_MM # M9: sozlesmeden okunur -- cizicinin sinirinin AYNISI


def _components_by_role (mesh ,roles ):
    """Verilen rollere ait yuzeyleri ayir and BAGLI BILESEN sayisini dondur (+ lower-mesh)."""
    vr =classify_colors (mesh .visual .vertex_colors )
    keep_v =np .isin (vr .astype (str ),list (roles ))
    if not keep_v .any ():
        return 0 ,None 
    fmask =keep_v [mesh .faces ].all (axis =1 )
    if not fmask .any ():
        return 0 ,None 
    sub =mesh .submesh ([np .where (fmask )[0 ]],append =True ,repair =False )
    try :
        comps =sub .split (only_watertight =False )
    except Exception as e :
        raise RuntimeError (f"bilesen ayrimi yapilamadi: {type (e ).__name__ }: {e }")
    return len (comps ),sub 


def audit_one (glb_path ):
    """Tek GLB'yi nine maddeye according to sina. Doner: (gecti_mi, ihlaller, bilgi)."""
    v =[]# ihlaller
    rp =receipt_path (glb_path )
    if not os .path .exists (rp ):
        return False ,[f"DENETLENEMEDI: receipt none ({os .path .basename (rp )})"],{}
    rec =json .load (open (rp ,encoding ="utf-8"))

    scene =trimesh .load (glb_path ,process =False )
    mesh =scene .to_mesh ()if hasattr (scene ,"to_mesh")else scene 
    if mesh is None or not hasattr (mesh ,"vertices"):
        return False ,["DENETLENEMEDI: GLB'den mesh cikarilamadi"],{}
    cols =getattr (mesh .visual ,"vertex_colors",None )
    if cols is None or len (cols )!=len (mesh .vertices ):
        return False ,["DENETLENEMEDI: GLB'de vertex rengi none -- roller ayirt edilemez"],{}

    roles =classify_colors (cols )
    unknown =int ((roles ==None ).sum ())# noqa: E711
    if unknown :
        v .append (f"M6: {unknown } vertex TANIMSIZ renkte (sozlesme disi renk kullanilmis)")

        # ---- govdeyi ayir: denetci own ray testlerini BUNUN on runs ----
    body_v =(roles =="body")
    fmask =body_v [mesh .faces ].all (axis =1 )
    if not fmask .any ():
        return False ,["DENETLENEMEDI: GLB'de body (gri) yuzeyi none"],{}
    body =mesh .submesh ([np .where (fmask )[0 ]],append =True ,repair =False )
    if rec .get ("body_faces")and abs (len (body .faces )-rec ["body_faces"])>0 :
        v .append (f"body yuzey sayisi tutmuyor: GLB {len (body .faces )} vs receipt {rec ['body_faces']}")

        # MOD: receipt hangi gorunumun cizildigini soyler. robot_only'de manufacturer isaretcisi and mavi
        # line BULUNMAMALIDIR -- denetci bunu gevsetmez, TERSINE cevirir (fazlasi da ihlaldir).
    mode =rec .get ("mode","compare")
    if mode not in MODES :
        return False ,[f"DENETLENEMEDI: makbuzda unknown mod {mode !r }"],{}

        # ---- M1/M2: isaretci sayilari ----
        # Okli isaretci = kure + body + koni (CPM bilesen). Yonu belirlenemeyen isaretci OK
        # CIZILMEDEN only kure as cizilir (1 bilesen) and makbuzda arrow=false writes.
        # Denetci bunu bilir, because aksi halde durust a beyani "missing isaretci" sayardi.
    def _comp_budget (ms ):
        a =sum (1 for m in ms if m .get ("arrow",True ))
        return CPM *a +(len (ms )-a )
    ms_gt =[m for m in rec ["markers"]if m ["role"]in GT_ROLES ]
    ms_pr =[m for m in rec ["markers"]if m ["role"]in PRED_ROLES ]
    n_gt_c ,_ =_components_by_role (mesh ,GT_ROLES )
    n_pr_c ,_ =_components_by_role (mesh ,PRED_ROLES )
    if n_gt_c !=_comp_budget (ms_gt ):
        v .append (f"M1: manufacturer isaretcisi bilesen {n_gt_c }, beklenen {_comp_budget (ms_gt )}"
        +(" (robot-only gorunumde HIC olmamali)"if mode =="robot_only"else ""))
    if n_pr_c !=_comp_budget (ms_pr ):
        v .append (f"M2: robot isaretcisi bilesen {n_pr_c }, beklenen {_comp_budget (ms_pr )}")

        # ---- M6: renk = mod sozlesmesi ----
    if mode =="robot_only":
        exp ={"gt_hit":0 ,"gt_miss":0 ,ROBOT_ROLE :rec ["n_pred"],
        **({}if ROBOT_ROLE =="pred_fp"else {"pred_fp":0 })}
    else :
        exp ={"gt_hit":rec ["tp"],"gt_miss":rec ["fn"],"pred_ok":rec ["tp"],"pred_fp":rec ["fp"]}
    for role ,n_exp in exp .items ():
        n_c ,_ =_components_by_role (mesh ,(role ,))
        want =_comp_budget ([m for m in rec ["markers"]if m ["role"]==role ])
        if n_c !=want :
            v .append (f"M6: '{role }' bilesen {n_c }, beklenen {want } ({n_exp } isaretci, mod={mode })")

            # ---- M7: mavi line count = tp (robot_only'de 0) ----
    n_link ,_ =_components_by_role (mesh ,("link",))
    if n_link !=rec ["n_links"]:
        v .append (f"M7: mavi cizgi {n_link }, beklenen {rec ['n_links']} (mod={mode })")

        # ---- M3/M4/M5/M9: each isaretci for geometri (DENETCININ KENDI isin testi) ----
    n_in =n_badbase =n_baddir =n_far =0 
    for m in rec ["markers"]:
        tip =np .asarray (m ["tip"],float );base =np .asarray (m ["base"],float )
        d =np .asarray (m ["dir"],float )
        # M9: isaretci KENDI CP'sinin next to durmali. 2026-07-29'da oklar butun blogu delip
        # karsi yuzeye firladi (306 isaretcinin 148'i, max 88.9mm) and audit bunu GORMEDI --
        # because sayim, renk and direction dogruydu, only YER yanlisti. Artik olculuyor.
        if "cp"in m :
            if float (np .linalg .norm (base -np .asarray (m ["cp"],float )))>TOL_CP_MM :
                n_far +=1 
        if not m .get ("arrow",True ):# oksuz (yonu beyan edilmemis) isaretci: M3/M4/M5 absent
            continue 
        if is_inside (body ,tip ):# M3
            n_in +=1 
        if np .linalg .norm ((base +d *m ["len"])-tip )>TOL_BASE_MM :# M4
            n_badbase +=1 
            # M5: igne DISARI bakmali. Olcut "uctan ileride never malzeme absent" not, "uctan ileride
            # YAKIN malzeme absent" -- because U/L biciminde a parcada isin bosluktan gecip parcanin
            # UZAK kolunu kesebilir; this a direction hatasi not, body seklinin sonucudur (measured
            # 2026-07-29: 3 isaretci, three DISARIDA, first malzeme 1 ok boyundan uzakta).
            # Gercek ihlal, ucun a cebe GOMULMESIDIR: malzeme igne boyundan more yakinda.
        h_fwd =ray_hits (body ,tip ,d ,float (np .linalg .norm (body .extents ))*1.05 )
        if len (h_fwd )and float (h_fwd [0 ])<float (m ["len"]):
            n_baddir +=1 
    if n_in :v .append (f"M3: {n_in }/{len (rec ['markers'])} igne ucu GOVDE ICINDE")
    if n_badbase :v .append (f"M4: {n_badbase } isaretcinin tabani CP noktasinda not (>{TOL_BASE_MM }mm)")
    if n_baddir :v .append (f"M5: {n_baddir } igne ucundan ILERIDE hala malzeme present (direction disari not)")
    if n_far :v .append (f"M9: {n_far }/{len (rec ['markers'])} isaretci KENDI CP'sinden {TOL_CP_MM }mm den uzakta (ok parcadan kopuk)")

    # ---- M8: file adi = receipt (compare'de F1, robot_only'de CP count) ----
    base_name =os .path .basename (glb_path )
    if mode =="robot_only":
        try :
            n_name =int (base_name .rsplit ("_CP",1 )[1 ].rsplit (".glb",1 )[0 ])
            if n_name !=rec ["n_pred"]:
                v .append (f"M8: file adi CP{n_name } vs receipt {rec ['n_pred']}")
        except Exception :
            v .append (f"M8: file adindan CP sayisi okunamadi ({base_name })")
    else :
        try :
            f1_name =float (base_name .rsplit ("_F1_",1 )[1 ].rsplit (".glb",1 )[0 ])
            if abs (f1_name -rec ["f1"])>TOL_F1 :
                v .append (f"M8: file adi F1 {f1_name } vs receipt {rec ['f1']:.3f}")
        except Exception :
            v .append (f"M8: file adindan F1 okunamadi ({base_name })")

    info ={"part":rec .get ("part"),"mfg":rec .get ("mfg"),"mode":mode ,"n_gt":rec ["n_gt"],
    "n_pred":rec ["n_pred"],"f1":rec ["f1"],"markers":len (rec ["markers"])}
    return (len (v )==0 ),v ,info 


def main (argv ):
    out_json =None 
    if "--json"in argv :
        i =argv .index ("--json");out_json =argv [i +1 ];argv =argv [:i ]+argv [i +2 :]
    files =argv or sorted (glob .glob ("results/robot_glb/*.glb"))
    if not files :
        print ("no GLB to audit");return 1 

    rows ,n_ok =[],0 
    print (f"{'file':<36} {'mode':>10} {'CP':>4} {'F1':>5}  result")
    for f in files :
        ok ,viol ,info =audit_one (f )
        n_ok +=int (ok )
        rows .append ({"glb":os .path .basename (f ),"ok":ok ,"violations":viol ,**info })
        tag ="PASS"if ok else "VIOLATION"
        f1v =info .get ("f1")
        print (f"{os .path .basename (f ):<36} {info .get ('mode','-'):>10} {info .get ('n_pred','-'):>4} "
        f"{(f'{f1v :.2f}'if isinstance (f1v ,(int ,float ))else '-'):>5}  {tag }")
        for x in viol :
            print (f"      ! {x }")
    print (f"\n{n_ok }/{len (files )} GLB files satisfy the contract")
    if out_json :
        json .dump ({"n_files":len (files ),"n_ok":n_ok ,"rows":rows },
        open (out_json ,"w",encoding ="utf-8"),indent =1 )
        print (f"report -> {out_json }")
    return 0 if n_ok ==len (files )else 1 


if __name__ =="__main__":
    sys .exit (main (sys .argv [1 :]))
