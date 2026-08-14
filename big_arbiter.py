# -*- coding: utf-8 -*-
"""THE BIG ARBITER: score the product against MANUFACTURER ConnectionPoints ten ~379 parts / 1133 CPs.

WHY THIS EXISTS: every product decision in this project has been made ten a 9-part / 18-CP manufacturer
arbiter whose noise is about +-0.10 -- big enough that it killed the 9k-resolution model (-0.156) and
contradicted the 82-CP held-out set ten pos_weight, with no way to tell which was right. DataSet 1
(1101 JSONs, Downloads) changes that: 400 of its parts have BOTH a STEP file in all_wscad_stp AND
manufacturer ConnectionPoints, and 379 of those are terminals that appear in NEITHER our training set
NOR the batch-4 held-out set. That is a clean, 63x larger arbiter (noise ~+-0.015).

Frame alignment VALIDATED ten a 25-part sample: 25/25 aligned, median align residual 0.167, median
bounding-box difference 0.00mm ten ~82mm parts (one outlier at 6.4mm).

Scores exactly what the PRODUCT emits: 6000-vertex remesh -> seed ensemble -> cp-v3.1 derivation with
the 2026-07-22 post-proc (cluster_mm 5, min_v 45, vertex_conf 0.7, depth gate 1.0).

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe big_arbiter.py \
         --ckpts results/seg_extra/human77c_s0.pt ... [--limit N] [--tag name]
"""
import os ,sys ,glob ,json ,time ,argparse 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D 
import connector3d ,cp_openings ,thesis_remesh 
from cad_eval import align_frames ,icp_refine 
from infer_step_cp import step_to_mesh ,load_any 

CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
OP ="results/step_infer/ops";DS ="_ds1/DataSet"


def greedy (P ,G ,tol ,Gdir =None ,axis_tol =None ):
    """Match predictions to manufacturer CPs.

    AXIS-AWARE MODE (Gdir given): match ten the distance PERPENDICULAR to the manufacturer's own
    InsertDirection, i.e. "is this the same opening?", and allow any depth along that axis (up to
    axis_tol). WHY: measured ten 61 manufacturer CPs, our prediction sits median 1.4mm off laterally
    but +14.8mm OUTWARD along the insertion axis -- the thesis defines the CP as v_o, the opening
    MOUTH (Abb.44), while the manufacturer reports the CONTACT deeper inside. Same opening, different
    convention. Scoring the euclidean distance therefore measures the convention gap, not the model;
    scoring the perpendicular distance measures what the thesis-faithful product actually claims.
    """
    if not len (P )or not len (G ):
        return 0 ,len (P ),len (G )
    if Gdir is not None :
        diff =P [:,None ,:]-G [None ,:,:]# (p, g, 3)
        along =(diff *Gdir [None ,:,:]).sum (-1 )# depth along each CP's own axis
        dm =np .linalg .norm (diff -along [...,None ]*Gdir [None ,:,:],axis =-1 )# perpendicular
        if axis_tol is not None :
            dm =np .where (np .abs (along )<=axis_tol ,dm ,np .inf )# reject absurd depths
    else :
        dm =np .linalg .norm (diff :=(P [:,None ,:]-G [None ,:,:]),axis =2 )
    order =sorted ((dm [i ,j ],i ,j )for i in range (len (P ))for j in range (len (G )))
    up ,ug ,tp =set (),set (),0 
    for d ,i ,j in order :
        if d >tol :break 
        if i in up or j in ug :continue 
        up .add (i );ug .add (j );tp +=1 
    return tp ,len (P )-tp ,len (G )-tp 


def gt_gecerli (jf ):
    """GT PUANLANABILIR MI? Degilse part korpusa GIRMEZ (2026-08-04'te eklendi).

    WHY IT EXISTS -- AL VAKASI: gorulmemis manufacturer sinavinda AL ureticisinde canli urun de,
    TABAN da, v3 de TAM 0.000 aldi. Uc hipotez sirayla curutuldu:
      1. "align_frames bozuk"      -> HAYIR, AL residual 0.13mm (ELMEX 0.11 with same)
      2. "direction vektoru sifir"       -> filtrelenmis kumede oyle gorunmedi (AL already elenmisti)
      3. HAM JSON'a bakildi        -> `InsertDirection {'X':0.0,'Y':0.0,'Z':0.0}` HARFIYEN SIFIR
                                      and 19 CP'nin HEPSI same noktada (bbox min == max)
    Yani GT'nin kendisi kullanilamaz durumda; urun wrong bulmuyor, OLCULEMIYOR.

    TUM KORPUSTA MEASURED: 4432 parcanin **27'sinde** direction sifir (26 AL + 1 A-B),
    **26'sinda** tum CP'ler same noktada. Korpusun %0.6'si -- but AL small a manufacturer
    oldugu for dengeli ornekleme onu exam kumesinin %8.4'une tasiyor and sonucu
    SAHTE as agirlastiriyordu.

    WHY BURADA: `eligible()` egitimin de, olcumun de, exam kumesinin de TEK bogaz
    noktasi. Filtreyi kullanan yerlere single single eklemek instead of kaynakta kapatiliyor --
    otherwise a sonraki betik yine dejenere GT'yi puanlar and "model kotu" der.
    """
    # NOTE: here `except Exception` KULLANMA. Ilk surumde oyleydi and dosyayi
    # `io.open` with aciyordum -- but `io` this modulde import EDILMEMIS. Dogan NameError'i
    # genis except YUTTU, fonksiyon each parcada False dondu and corpus 4432'den **0'a**
    # dustu; no error mesaji cikmadi, only "27 cikarildi" instead of "4720 cikarildi"
    # yazdi. Genis except, own yazim hatani data bulgusu like gosterir.
    # Cozum two katmanli: (1) `open()` yeterli, `io` gerekmiyor -- `np` already row 21'de
    # modul duzeyinde; (2) only GERCEK okuma/form hatalari yakalanir.
    try :
        with open (jf ,encoding ="utf-8-sig")as fh :
            j =json .load (fh )
    except (OSError ,ValueError ,UnicodeDecodeError ):
        return False 
    cps =j .get ("ConnectionPoints")or []
    if not cps :
        return False 
    try :
        G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in cps ],float )
        D =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in cps ],float )
    except (KeyError ,TypeError ,ValueError ):
        return False 
    if not np .isfinite (G ).all ()or not np .isfinite (D ).all ():
        return False 
    if (np .linalg .norm (D ,axis =1 )<1e-6 ).all ():
        return False # direction absent -> axial tolerans anlamsiz
    if len (G )>1 and float (np .abs (G .max (0 )-G .min (0 )).max ())<1e-6 :
        return False # butun CP'ler same noktada -> tekil esleme imkansiz
    return True 


def eligible ():
    """Parts with STEP + manufacturer CPs that are NOT in training and NOT in the batch-4 held-out."""
    gecersiz =[]
    # KIMLIK AYRISTIRMASI (2026-08-04 duzeltmesi, H1): eskiden each two tarafta da
    # `basename.split("_")[<n>]` kullaniliyordu. Bu, ALT CIZGI iceren manufacturer kodlarinda
    # (`A-B_N.1492-H4_...` -> mfg "A-B", pid BOS) and lower cizgili part numaralarinda
    # (`ELMEX.KUT16_GY_...` -> "KUT16") kiriliyordu: 481 file wrong, 291'i BOS kimlik
    # (i.e. STEP'i gelse bile korpusa ASLA giremezdi), 35 kimlik cakisiyordu.
    # ETKI MEASURED and NOTR: eslesme 1926 -> 1926, measurement kumesi 194/194 korundu, only
    # ELMEX.KUDD4D1_GY'nin kimligi "KUDD4D1" -> "KUDD4D1_GY" became (same file).
    # Bugun kazanc sifir because lower cizgili parcalarin STEP'i absent; value ILERIDE.
    from korpus_kimlik import kimlik as _kimlik ,step_kimlik as _skimlik 
    step ={_skimlik (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    seen =set ()
    for d in ("_label_targets","_label_targets_2","_label_targets_3","_label_targets_4"):
        seen |={os .path .basename (os .path .normpath (p ))for p in glob .glob (d +"/*/")}
    import scheffler_dataset as ds 
    for sp in ("train","val"):
        seen |={s ["part_id"]for s in ds .load_split ("wscad_corpus_scheffler_exact",sp ,verify_hashes =False )}
        # LEAKAGE GUARD: manufacturer parts whose labels were fed into training (labels_from_mfg.py writes
        # a deterministic per-part train/test split) must never be scored here.
    rc_file ="_label_targets_recall/trained_parts.json"
    if os .path .exists (rc_file ):
        rc =set (json .load (open (rc_file )).get ("parts",[]))
        seen |=rc 
        print (f"  [leakage guard] {len (rc )} recall-trained WEI parts excluded from scoring",flush =True )
    sp_file ="_mfg_labels/split.json"
    if os .path .exists (sp_file ):
        mfg_split =json .load (open (sp_file ))
        trained ={k for k ,v in mfg_split .items ()if v .get ("split")=="train"}
        seen |=trained 
        print (f"  [leakage guard] {len (trained )} manufacturer-label TRAIN parts excluded from scoring",flush =True )
        # URETICI TAKMA ADLARI (2026-08-04, H2): DataSet5 with A-B and A-B_N same parcayi IKI
        # KEZ getirdi -- measured: 288 part, HEPSI same STEP dosyasini paylasiyor and CP sayilari
        # BIREBIR same. Yani same fiziksel parcanin two NORM kaydi (IEC/NEMA like).
        # UC ZARARI VARDI:
        #   1. gate egitiminde CIFT AGIRLIK (A-B'ye correct deviation)
        #   2. split sizintisi (a kopya egitimde, digeri testte) -- geometri grubu bunu
        #      yakaliyordu but tesadufen
        #   3. EN SINSISI: "A-B'yi disarida birak" testinde A-B_N ICERIDE kalir; i.e. manufacturer
        #      HIC disarida birakilmamis becomes and producer-out satirlari SESSIZCE SISER.
        # COZUM: takma name birlestirilir + same (pid, STEP) ciftinden TEK kayit tutulur.
    TAKMA ={"A-B_N":"A-B"}
    out ,gorulen =[],set ()
    for f in sorted (glob .glob (os .path .join (DS ,"*ElectricalTerminal*.json"))):
        mfg ,pid =_kimlik (f )
        if pid not in step :
            continue 
        if pid in seen and not os .environ .get ("BA_ALLOW_SEEN"):
            continue 
        mfg =TAKMA .get (mfg ,mfg )
        if (pid ,step [pid ])in gorulen :# same fiziksel part, ikinci norm kaydi
            continue 
        if not gt_gecerli (f ):
            gecersiz .append ((mfg ,pid ))
            continue 
        gorulen .add ((pid ,step [pid ]))
        out .append ((mfg ,pid ,f ,step [pid ]))
    if gecersiz :
        import collections as _c 
        print (f"  [GT gecerlilik] {len (gecersiz )} part CIKARILDI (puanlanamaz GT): "
        f"{dict (_c .Counter (m for m ,_ in gecersiz ))}",flush =True )
    return out 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--ckpts",nargs ="+",required =True )
    ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--only-parts",nargs ="+",default =[],help ="restrict to these part ids (pipeline cross-check)")
    ap .add_argument ("--only-mfg",default ="",help ="restrict to one manufacturer (PXC/WEI/...)")
    ap .add_argument ("--shuffle",action ="store_true",
    help ="sample RANDOMLY instead of taking the alphabetically-first parts. "
    "LEARNED THE HARD WAY 2026-07-22: the first 40 entries are all one product "
    "family (04xxxxx) and gave a completely misleading picture (they need a 13mm "
    "inward shift to match, while the 9-part arbiter's 3xxxxxx parts match at 0mm). "
    "Alphabetical == one family == a biased sample.")
    ap .add_argument ("--cluster-mm",type =float ,default =5.0 )
    ap .add_argument ("--min-v",type =int ,default =45 )
    ap .add_argument ("--vertex-conf",type =float ,default =0.7 )
    ap .add_argument ("--remesh-target",type =int ,default =6000 )
    ap .add_argument ("--icp",action ="store_true",help ="FAZ 1: ICP-refined hizalama ile de eslestir; coarse vs ICP F1 karsilastir")
    ap .add_argument ("--skip-parts",nargs ="+",default =[],help ="hang eden patolojik parcalari atla (step_to_mesh/remesh takilan)")
    ap .add_argument ("--verbose-parts",action ="store_true",help ="each parcayi islemeden before pid head (hang teshisi)")
    ap .add_argument ("--inward-mm",type =float ,default =0.0 ,
    help ="shift each predicted CP INWARD along its own insertion axis before matching. "
    "MEASURED 2026-07-22: the manufacturer defines its ConnectionPoint at the "
    "CONTACT/seat inside the terminal, while our v_o is the opening MOUTH -- same "
    "direction, same other two axes, differing only in depth (7-18mm on the parts "
    "checked). This is a DEFINITION difference, not a model error.")
    ap .add_argument ("--use-depth",action ="store_true",
    help ="place the match point at each CP's OWN seat (v_o - insertion_depth*dir) "
    "instead of a fixed --inward-mm. Physically right: the manufacturer's CP is "
    "the contact inside the terminal, and the depth varies per opening "
    "(measured 0.6-15.2mm on one part), so a single constant cannot fit all.")
    ap .add_argument ("--to-center",action ="store_true",
    help ="slide each predicted CP along its own insertion axis until it reaches the "
    "part's CENTRE plane. MEASURED: the manufacturer's ConnectionPoints sit at "
    "0.40-0.60 of the part's extent along that axis (i.e. essentially the middle, "
    "where the clamp grips) while our v_o is the outer mouth. Unlike a fixed "
    "offset this adapts to part size.")
    ap .add_argument ("--axis-aware",action ="store_true",
    help ="score 'did we find the SAME OPENING' (perpendicular distance to the "
    "manufacturer's insertion axis) instead of raw euclidean distance, which is "
    "dominated by the v_o-vs-contact convention offset. See greedy().")
    ap .add_argument ("--axis-tol",type =float ,default =40.0 ,help ="max allowed depth difference along the axis")
    ap .add_argument ("--split-ratio",type =float ,default =0.0 ,
    help ="split an elongated fragment that spans a ROW of adjacent openings. Rejected "
    "earlier against the HUMAN-derived held-out (F1 .733 -> .508) -- but that GT "
    "MERGES double openings exactly like the model does, so it could only punish "
    "splitting. The manufacturer lists every CP separately, so this arbiter is "
    "the only ruler that can reward it. (Annotator spotted the merge by eye.)")
    ap .add_argument ("--point-mode",default ="v_o",choices =["v_o","v_v"])
    ap .add_argument ("--tag",default ="product")
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    a =ap .parse_args ()

    models =[load_any (c ,dev =a .device )for c in a .ckpts ]
    parts =eligible ()
    if a .only_mfg :
        parts =[p for p in parts if p [0 ]==a .only_mfg ]
    if a .only_parts :
        keep =set (a .only_parts );parts =[p for p in parts if p [1 ]in keep ]
    if a .shuffle :
        import random ;random .Random (0 ).shuffle (parts )
    if a .limit :parts =parts [:a .limit ]
    print (f"{len (parts )} uygun part | {len (models )} model | remesh {a .remesh_target } "
    f"| cluster {a .cluster_mm } min_v {a .min_v }",flush =True )

    T =Fp =Fn =0 ;Ti =[0 ,0 ,0 ];rows =[];t0 =time .time ();skipped =0 
    SKIP =set (a .skip_parts )
    for k ,(mfg ,pid ,jf ,stp )in enumerate (parts ,1 ):
        if pid in SKIP :
            skipped +=1 ;continue 
        if a .verbose_parts :print (f"  [{k }/{len (parts )}] {pid } ({mfg })...",flush =True )
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[p ["X"],p ["Y"],p ["Z"]]for p in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"]["X"],c ["Point"]["Y"],c ["Point"]["Z"]]
            for c in j .get ("ConnectionPoints",[])],float )
            Gd =np .array ([[c ["InsertDirection"]["X"],c ["InsertDirection"]["Y"],c ["InsertDirection"]["Z"]]
            for c in j .get ("ConnectionPoints",[])],float )
            if len (Gd ):Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            if not len (G ):continue 
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =a .remesh_target )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            acc =None 
            for model ,meta ,_ in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =a .device ,op_cache_dir =OP ,return_probs =True )
                pb =np .asarray (pb ,float );acc =pb if acc is None else acc +pb 
            probs =acc /len (models );lab =probs .argmax (-1 )
            cps =cp_openings .connection_points (V ,F ,lab ,min_v =a .min_v ,classes =(CE ,CT ),dedupe_mm =10.0 ,
            probs =probs ,vertex_conf =a .vertex_conf ,ct_depth_min_mm =1.0 ,
            cluster_mm =a .cluster_mm ,split_ratio =a .split_ratio )
            R ,t ,ares =align_frames (Vr ,Vj )# ares = hizalama residual (mm); large = STEP!=JSON geometri
            if cps :
                P0 =np .array ([np .asarray (c ["point"])for c in cps ],float )
                if a .to_center :
                    Dv =np .array ([np .asarray (c ["direction"],float )for c in cps ],float )
                    Dv =Dv /(np .linalg .norm (Dv ,axis =1 ,keepdims =True )+1e-9 )
                    ctr =0.5 *(V .min (0 )+V .max (0 ))
                    P0 =P0 -(((P0 -ctr )*Dv ).sum (1 ,keepdims =True ))*Dv 
                elif a .use_depth :
                    Dv =np .array ([np .asarray (c ["direction"],float )for c in cps ],float )
                    Dv =Dv /(np .linalg .norm (Dv ,axis =1 ,keepdims =True )+1e-9 )
                    dep =np .array ([float (c .get ("insertion_depth_mm",0.0 ))for c in cps ])[:,None ]
                    P0 =P0 -dep *Dv 
                elif a .inward_mm :
                    Dv =np .array ([np .asarray (c ["direction"],float )for c in cps ],float )
                    Dv =Dv /(np .linalg .norm (Dv ,axis =1 ,keepdims =True )+1e-9 )
                    P0 =P0 -a .inward_mm *Dv # mouth -> toward the seat
            else :
                P0 =np .zeros ((0 ,3 ))
            tol =max (3.0 ,0.06 *float (np .linalg .norm (Vj .max (0 )-Vj .min (0 ))))
            P =P0 @R .T +t if len (P0 )else P0 
            tp ,fp ,fn =(greedy (P ,G ,tol ,Gd ,a .axis_tol )if a .axis_aware else greedy (P ,G ,tol ))
            T +=tp ;Fp +=fp ;Fn +=fn 
            row ={"mfg":mfg ,"part_id":pid ,"mfg_cps":len (G ),"pred":len (P0 ),
            "tp":tp ,"fp":fp ,"fn":fn ,"align_res":float (ares )}
            if a .icp :# FAZ 1: ICP-refined hizalama with de eslestir
                Ri ,ti ,ires =icp_refine (Vr ,Vj ,R ,t )
                Pi =P0 @Ri .T +ti if len (P0 )else P0 
                tpi ,fpi ,fni =(greedy (Pi ,G ,tol ,Gd ,a .axis_tol )if a .axis_aware else greedy (Pi ,G ,tol ))
                Ti [0 ]+=tpi ;Ti [1 ]+=fpi ;Ti [2 ]+=fni 
                row .update ({"tp_icp":tpi ,"fp_icp":fpi ,"fn_icp":fni ,"align_res_icp":float (ires )})
            rows .append (row )
        except Exception as e :
            skipped +=1 
            continue 
        if k %10 ==0 :
            pr =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 )
            print (f"  {k }/{len (parts )}  F1={2 *pr *rc /max (pr +rc ,1e-9 ):.3f}  "
            f"(TP{T } FP{Fp } FN{Fn })  {time .time ()-t0 :.0f}s",flush =True )

    pr =T /max (T +Fp ,1 );rc =T /max (T +Fn ,1 );f1 =2 *pr *rc /max (pr +rc ,1e-9 )
    print (f"\n=== BIG ARBITER ({len (rows )} part, {T +Fn } manufacturer CP, {skipped } atlandi) ===")
    print (f"  {a .tag }: F1={f1 :.3f}  P={pr :.3f}  R={rc :.3f}  (TP{T } FP{Fp } FN{Fn })")
    for m in sorted ({r ["mfg"]for r in rows }):
        s =[r for r in rows if r ["mfg"]==m ]
        t_ ,f_ ,n_ =sum (r ["tp"]for r in s ),sum (r ["fp"]for r in s ),sum (r ["fn"]for r in s )
        p_ =t_ /max (t_ +f_ ,1 );r_ =t_ /max (t_ +n_ ,1 )
        print (f"    {m }: {len (s ):4d} part  F1={2 *p_ *r_ /max (p_ +r_ ,1e-9 ):.3f}  P={p_ :.3f} R={r_ :.3f}")

        # HIZALAMA-STRATIFIED (objektif, modelden bagimsiz): align_res large = STEP!=JSON geometri -> GT yerlestirilemez.
        # Bu parcalarda modelin correct tahmini bile FN sayilir. Kriter align residual, MODEL PERFORMANSI DEGIL.
    def f1_sub (sub ):
        t_ =sum (r ["tp"]for r in sub );f_ =sum (r ["fp"]for r in sub );n_ =sum (r ["fn"]for r in sub )
        p_ =t_ /max (t_ +f_ ,1 );r_ =t_ /max (t_ +n_ ,1 );return 2 *p_ *r_ /max (p_ +r_ ,1e-9 ),p_ ,r_ 
    ares_all =sorted (r .get ("align_res",0.0 )for r in rows )
    print (f"\n  --- HIZALAMA denetimi (align_res mm; kucuk=iyi, buyuk=STEP!=JSON) ---")
    print (f"    residual dagilimi: med {ares_all [len (ares_all )//2 ]:.1f}  p75 {ares_all [int (len (ares_all )*0.75 )]:.1f}  p90 {ares_all [int (len (ares_all )*0.9 )]:.1f}  max {ares_all [-1 ]:.1f}")
    for thr in (10.0 ,7.0 ,5.0 ,3.0 ):
        keep =[r for r in rows if r .get ("align_res",0.0 )<=thr ]
        nx =len (rows )-len (keep )
        f1k ,pk ,rk =f1_sub (keep )
        line =f"    residual<={thr :>4.1f}mm: {len (keep ):4d} part ({nx :3d} haric)  F1={f1k :.3f} P={pk :.3f} R={rk :.3f}"
        for m in sorted ({r ["mfg"]for r in rows }):
            sm =[r for r in keep if r ["mfg"]==m ]
            if sm :line +=f"  |{m } F1={f1_sub (sm )[0 ]:.3f}"
        print (line )
    print (f"    NOT: kriter objektif (align residual = STEP/JSON geometri uyumu), model performansi DEGIL. Hem tam hem filtreli gosteriliyor.")

    if a .icp and any ("tp_icp"in r for r in rows ):# FAZ 1: ICP-refined hizalama karsilastirmasi
        ti ,fi ,ni =Ti 
        pit =ti /max (ti +fi ,1 );rit =ti /max (ti +ni ,1 );f1i =2 *pit *rit /max (pit +rit ,1e-9 )
        ir =sorted (r .get ("align_res_icp",0.0 )for r in rows if "align_res_icp"in r )
        cr =sorted (r .get ("align_res",0.0 )for r in rows if "align_res_icp"in r )
        print (f"\n  === FAZ 1: ICP-REFINED HIZALAMA (recovery) ===")
        print (f"    residual median: coarse {cr [len (cr )//2 ]:.1f}mm -> ICP {ir [len (ir )//2 ]:.1f}mm  (p90 {cr [int (len (cr )*0.9 )]:.1f} -> {ir [int (len (ir )*0.9 )]:.1f})")
        print (f"    F1: coarse {f1 :.3f} -> ICP {f1i :.3f}  (P {pit :.3f} R {rit :.3f})")
        for m in sorted ({r ["mfg"]for r in rows if "align_res_icp"in r }):
            sm =[r for r in rows if r ["mfg"]==m and "tp_icp"in r ]
            tt =sum (r ["tp_icp"]for r in sm );ff =sum (r ["fp_icp"]for r in sm );nn =sum (r ["fn_icp"]for r in sm )
            pp =tt /max (tt +ff ,1 );rr =tt /max (tt +nn ,1 )
            print (f"      {m }: F1={2 *pp *rr /max (pp +rr ,1e-9 ):.3f}")
            # ICP + residual-filtre (gercekten different parcalari da cikar)
        for thr in (7.0 ,5.0 ):
            keep =[r for r in rows if r .get ("align_res_icp",99 )<=thr and "tp_icp"in r ]
            tt =sum (r ["tp_icp"]for r in keep );ff =sum (r ["fp_icp"]for r in keep );nn =sum (r ["fn_icp"]for r in keep )
            pp =tt /max (tt +ff ,1 );rr =tt /max (tt +nn ,1 )
            print (f"    ICP + residual<={thr }mm: {len (keep )} part ({len (rows )-len (keep )} STEP!=JSON haric)  F1={2 *pp *rr /max (pp +rr ,1e-9 ):.3f}")
    os .makedirs ("results",exist_ok =True )
    out =f"results/big_arbiter_{a .tag }.json"
    json .dump ({"tag":a .tag ,"ckpts":a .ckpts ,"n_parts":len (rows ),"n_cps":T +Fn ,
    "precision":round (pr ,4 ),"recall":round (rc ,4 ),"f1":round (f1 ,4 ),
    "cluster_mm":a .cluster_mm ,"min_v":a .min_v ,"remesh_target":a .remesh_target ,
    "per_part":rows },open (out ,"w"),indent =1 )
    print (f"  -> {out }")


if __name__ =="__main__":
    main ()
