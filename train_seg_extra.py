# -*- coding: utf-8 -*-
"""Retrain the 5-class DiffusionNet segmentation ten the ENLARGED corpus: our 71 train + the 132
downloaded EEC parts (`_eec_extra/`) = 203 train, validate ten our 20 val, and NEVER touch the
locked-11 test. Goal: close the generalisation gap with 2.3x real human-labelled data.

Self-contained (does not need the corpus manifest): loads our splits via scheffler_dataset and the
extra parts by parsing OBJ+labels. Operators stay ten CPU and move per-part (a 4GB GPU cannot hold
all -> WDDM spill). Weighted NLL with inverse-frequency class weights. Saves best-by-val-mIoU to
results/seg_extra/best.pt.

Usage: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe train_seg_extra.py [--limit N --epochs E]
"""
import argparse ,os ,glob 
import numpy as np 
import torch 
import diffusionnet as D 
import scheffler_dataset as dataset 

# OPERATOR ONBELLEGI k_eig'e GORE AYRILIR (2026-07-29): single a mesh-anahtarli directory, 64 and 96
# eigen with calisan kosular between surekli "overwriting cache -- not enough eigenvalues" +
# WinError 32 file kilidi dongusune giriyordu (training ilerlemiyordu). robot_cp same sorunu
# ops_k{64,96} ayrimiyla cozmustu; here da same ayrim yapiliyor.
OPS_BASE ="results/seg_extra/ops";NCLS =5 ;NCLS_CE =3 # CableEntry class index


def _rand_rot (dev ,max_ang =1.05 ):# moderate random rotation (<= ~60deg) for augmentation
    v =torch .randn (3 ,device =dev );v =v /(v .norm ()+1e-9 )
    ang =(torch .rand (1 ,device =dev )*2 -1 )*max_ang 
    K =torch .zeros (3 ,3 ,device =dev )
    K [0 ,1 ],K [0 ,2 ],K [1 ,0 ],K [1 ,2 ],K [2 ,0 ],K [2 ,1 ]=-v [2 ],v [1 ],v [2 ],-v [0 ],-v [1 ],v [0 ]
    return torch .eye (3 ,device =dev )+torch .sin (ang )*K +(1 -torch .cos (ang ))*(K @K )


def load_obj (p ):
    V ,F =[],[]
    for ln in open (p ):
        if ln .startswith ("v "):
            V .append ([float (x )for x in ln .split ()[1 :4 ]])
        elif ln .startswith ("f "):
            F .append ([int (t .split ("/")[0 ])-1 for t in ln .split ()[1 :4 ]])
    return np .asarray (V ,float ),np .asarray (F ,int )


def load_extra (root ="_eec_extra"):
    out =[]
    for d in sorted (glob .glob (f"{root }/*/")):
    # NOTE: os.path.normpath, NOT d.rstrip("/") -- ten Windows glob returns a trailing BACKSLASH,
    # rstrip("/") leaves it, basename() then returns "" and every part is silently skipped
    # (this bug made a whole "+132 EEC" experiment train ten 71 parts while reporting success).
        pid =os .path .basename (os .path .normpath (d ))
        try :
            V ,F =load_obj (os .path .join (d ,f"{pid }.obj"))
            L =np .array ([int (x )for x in open (os .path .join (d ,f"{pid }.labels.txt")).read ().split ()])
        except Exception :
            continue 
        if len (V )and len (F )and len (V )==len (L )and F .max ()<len (V )and set (np .unique (L ))<=set (range (5 )):
            rec ={"part_id":pid ,"verts":V ,"faces":F ,"labels":L }
            # FB-2: TEL/ALET yardimci etiketi (varsa). -1 maskeli / 0 tel / 1 alet.
            # Tezin 5-sinif etiketini DEGISTIRMEZ: ayri file, ayri kafa, ayri loss.
            _ax =os .path .join (d ,pid +".aux.txt")
            if os .path .exists (_ax ):
                try :
                    A =np .array ([int (x )for x in open (_ax ).read ().split ()],np .int64 )
                    if len (A )==len (V ):
                        rec ["aux"]=A 
                except Exception :
                    pass 
                    # ADJUDICATION: vertices the annotator marked "unsure" are neither positive nor negative.
                    # Without this they would be supervised as NEGATIVE by the masked BCE (which pushes p_pos
                    # down ten every unmarked vertex) -- i.e. a guess would become a hard training signal.
            ig =os .path .join (d ,f"{pid }.ignore.txt")
            if os .path .exists (ig ):
                try :
                    idx =np .array ([int (x )for x in open (ig ).read ().split ()],np .int64 )
                    m =np .zeros (len (V ),bool )
                    m [idx [(idx >=0 )&(idx <len (V ))]]=True 
                    rec ["ignore"]=m 
                except Exception :
                    pass 
            out .append (rec )
    return out 


def prep (samples ,k_eig ):
    data =[]
    for s in samples :
        V =np .ascontiguousarray (s ["verts"],np .float64 );F =np .ascontiguousarray (s ["faces"],np .int64 )
        try :
            ops =D .precompute_operators (V ,F ,k_eig ,op_cache_dir =f"{OPS_BASE }_k{int (k_eig )}")
        except Exception :
            continue 
        rec ={"ops":ops ,"lab":torch .tensor (np .asarray (s ["labels"]),dtype =torch .long ),
        "partial_ce":bool (s .get ("partial_ce",False ))}
        if s .get ("aux")is not None :
            rec ["aux"]=torch .tensor (np .asarray (s ["aux"]),dtype =torch .long )
        if s .get ("ignore")is not None :
            rec ["ignore"]=torch .tensor (np .asarray (s ["ignore"]),dtype =torch .bool )
        data .append (rec )
    return data 


def to_dev (ops ,dev ):
    return {k :(v .to (dev )if hasattr (v ,"to")else v )for k ,v in ops .items ()}


NCLS_CT =1 # Contact class index (cp-v3: a CP can be CableEntry OR a depth-gated Contact)


def miou (model ,meta ,data ,dev ):
    """Dogrulama metrigi. KISMI etiketli orneklerde MASKELI olcer.

    BULUNAN HATA (2026-08-07): dogrulama kumesi kismi etiketli parcalara cevrilince
    Conn-IoU 0.5878 -> 0.0995'e COKTU. Sebep: kismi etikette YALNIZ CableEntry signed,
    geri kalan 0 (Housing). Maskesiz IoU, modelin DOGRU prediction ettigi Contact bolgelerini
    YANLIS POZITIF sayiyordu. Bu, EGITIM HEDEFIYLE CELISIR -- egitimde loss maskeli
    (`partial_ce` dallarinda only CE-vs-not denetleniyor).
    Boyle a sinyalle secim yapmak, Contact'i DAHA AZ prediction eden modeli "iyi" sanmak
    demekti; i.e. correction diye konan sey YENI a yanlilik yaratirdi.

    Cozum: kismi ornekte only ISARETLI bolgenin CIVARINI puanla -- signed vertices
    and modelin baglanti dedigi vertices. Isaretsiz body bolgeleri metrikten CIKARILIR.
    Tam etiketli orneklerde davranis BIREBIR ESKISI GIBI kalir.
    """
    model .eval ();inter =np .zeros (NCLS );union =np .zeros (NCLS );acc =0 ;tot =0 
    kismi_kayip =[]# kismi etiketli orneklerde EGITIM HEDEFI (maskeli BCE)
    ci =cu =0 # CableEntry+Contact merged = the cp-v3 "Connection" class (the real CP driver)
    with torch .no_grad ():
        for d in data :
            ops =to_dev (d ["ops"],dev );x =D ._model_input (ops ,meta )
            pr =D ._forward (model ,ops ,x ).argmax (-1 ).cpu ().numpy ();gt =d ["lab"].numpy ()
            kismi =bool (d .get ("partial_ce"))
            if kismi :
            # SECIM SINYALI = EGITIM HEDEFININ AYNISI (2026-08-07, ikinci correction).
            # Ilk duzeltmem yetmedi: Conn-IoU'nun BIRLESIMI already (pc|gc) oldugu for
            # maske orada no sey degistirmiyordu and acc %6'ya dusuyordu -- because
            # kismi etikette sign CE, training hedefi whereas CE+CT; SINIF-BAZLI esitlik
            # modelin DOGRU Contact tahminini wrong sayiyor.
            # Metrik ICAT ETMEK instead of training kaybinin KENDISINI kullaniyoruz:
            # maskeli BCE, ayrik veride. Tanim geregi training hedefiyle BIREBIR same.
                import torch .nn .functional as _Fn 
                lo =D ._forward (model ,ops ,x )
                sm2 =torch .softmax (lo ,-1 )
                pp =(sm2 [:,NCLS_CE ]+sm2 [:,NCLS_CT ]).clamp (1e-6 ,1 -1e-6 )
                yy =(d ["lab"].to (dev )==NCLS_CE ).float ()
                w =torch .where (yy >0 ,torch .full_like (yy ,20.0 ),torch .ones_like (yy ))
                loss =(_Fn .binary_cross_entropy (pp ,yy ,weight =w )).item ()
                kismi_kayip .append (loss )
            pc =np .isin (pr ,(NCLS_CE ,NCLS_CT ));gc =np .isin (gt ,(NCLS_CE ,NCLS_CT ))
            if kismi :
            # MASKE: signed vertices + modelin baglanti dedigi vertices.
            # Isaretsiz body metrige GIRMEZ (orada GT bilgisi YOK).
                m =gc |pc 
                if not m .any ():
                    continue 
                ci +=(pc &gc &m ).sum ();cu +=((pc |gc )&m ).sum ()
                acc +=((pr ==gt )&m ).sum ();tot +=int (m .sum ())
                for c in (NCLS_CE ,NCLS_CT ):
                    inter [c ]+=((pr ==c )&(gt ==c )&m ).sum ()
                    union [c ]+=(((pr ==c )|(gt ==c ))&m ).sum ()
                continue 
            acc +=(pr ==gt ).sum ();tot +=len (gt )
            for c in range (NCLS ):
                inter [c ]+=((pr ==c )&(gt ==c )).sum ();union [c ]+=((pr ==c )|(gt ==c )).sum ()
            ci +=(pc &gc ).sum ();cu +=(pc |gc ).sum ()
    iou =inter /np .maximum (union ,1 )
    # KISMI val varsa SECIM SINYALI maskeli BCE'nin NEGATIFI becomes (large = iyi),
    # so cagiran taraf "connection_iou'yu ENBUYUKLE" mantigini degistirmeden works.
    if kismi_kayip :
        return (float (np .mean (iou )),acc /max (tot ,1 ),float (iou [NCLS_CE ]),
        float (-np .mean (kismi_kayip )))
    return float (np .mean (iou )),acc /max (tot ,1 ),float (iou [NCLS_CE ]),float (ci /max (cu ,1 ))


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--epochs",type =int ,default =200 );ap .add_argument ("--limit",type =int ,default =0 )
    ap .add_argument ("--train-limit",type =int ,default =0 ,help ="learning-curve: train on a seed-fixed subset of N train parts, val kept FULL")
    ap .add_argument ("--k-eig",type =int ,default =64 );ap .add_argument ("--lr",type =float ,default =1e-3 )
    # GIRDI OZNITELIGI (2026-08-13). `cfg` bunu SABIT "xyz" yaziyordu, i.e.
    # modele YALNIZCA HAM KOORDINAT veriliyordu and `hks` secenegi never
    # denenemiyordu. Fark mekanik as onemli: `xyz` DISSALDIR (model
    # mutlak konuma baglanir), `hks` ICSELDIR (donme/otelemeye duyarsiz) --
    # gorulmemis brand kosulunda istenen full as budur.
    ap .add_argument ("--input-features",choices =("xyz","hks"),default ="xyz",
    help ="xyz = ham koordinat (dissal) | hks = isi cekirdegi "
    "imzasi (icsel, donme/oteleme duyarsiz)")
    ap .add_argument ("--sinir-weight",type =float ,default =0.0 ,
    help ="Y13 sinir-farkindali loss agirligi. CP fiziksel "
    "olarak bir SINIRDIR (mouth cemberi) ama mevcut loss "
    "BOLGEYI hedefler. 0 = kapali (davranis degismez).")
    ap .add_argument ("--tversky-gamma",type =float ,default =1.0 ,
    help ="Y14 Focal-Tversky ussu. 1.0 = klasik Tversky "
    "(varsayilan, davranis degismez). 0.75/1.33 tipik.")
    ap .add_argument ("--tversky",type =float ,default =0.25 )# thesis overlap term for rare classes
    ap .add_argument ("--lr-decay-every",type =int ,default =100 );ap .add_argument ("--lr-decay-rate",type =float ,default =0.75 )
    ap .add_argument ("--checkpoint-out",required =True ,help ="where to save best-by-val-mIoU (no default -> cannot clobber the product model)")
    ap .add_argument ("--seed",type =int ,default =0 )
    # FEW-SHOT (K6.5-b): k part 200 parcalik korpusun inside KAYBOLUR. Iki kip:
    #  --only-kismi : YALNIZ k parcayla adapte ol (agresif; unutma riski VAR)
    #  --kismi-tekrar : korpusu koru but k parcayi N times tekrarla (gerceksi recete)
    ap .add_argument ("--yalniz-kismi",action ="store_true",
    help ="FEW-SHOT: training kumesi YALNIZ --partial-dir olsun")
    ap .add_argument ("--kismi-tekrar",type =int ,default =1 ,
    help ="FEW-SHOT: kismi ornekleri N kez tekrarla (agirliklandirma)")
    ap .add_argument ("--init-from",default ="",
    help ="FEW-SHOT/FINE-TUNE: bu ckpt'ten baslat (sifirdan degil). "
    "Mimari birebir ayni olmali; strict=True ile yuklenir.")
    ap .add_argument ("--train-dir",default ="",help ="RESOLUTION EXPERIMENT: load the corpus TRAIN split from a load_extra-style dir (e.g. _corpus12k_train) instead of scheffler_dataset -- lets us retrain at a different remesh resolution without touching the frozen corpus")
    ap .add_argument ("--val-dir",default ="",help ="same for VAL (must match the train resolution)")
    ap .add_argument ("--val-partial",action ="store_true",
    help ="VAL dizini KISMI etiketli (yalniz CableEntry signed). Dogrulama "
    "metrigi MASKELI olcer -- unsigned body metrige girmez. Bu bayrak "
    "olmadan Conn-IoU yapay olarak COKER (0.5878 -> 0.0995 measured) ve "
    "secim, Contact'i AZ tahmin eden modeli odullendirir.")
    ap .add_argument ("--no-extra",action ="store_true",help ="ablation: train on our 71 only (skip the 132 EEC extra)")
    ap .add_argument ("--pseudo-dir",default ="",help ="self-training: also load MODEL-PREDICTED pseudo-labels from this dir (e.g. _pseudo_extra). NOT human labels -- only adopt if val Connection IoU improves")
    ap .add_argument ("--partial-dir",nargs ="+",default =[],help ="human PARTIAL label dir(s) (CableEntry marked, else 0). Accepts multiple dirs (e.g. _label_targets _label_targets_2). Trained with a MASKED loss that only supervises CableEntry-vs-not, so the unmarked classes are NOT taught as Housing")
    ap .add_argument ("--aux-wire",action ="store_true",
    help ="FB-2 TEL/ALET yardimci supervizyonu. Paylasilan govdeye IKINCI "
    "bir kafa eklenir; ana 5-sinif kafasi ve kaybi BIT DUZEYINDE "
    "degismez (tez ihlali YOK). Gerekce: tezin Contact sinifi "
    "tasarimi geregi Kontaktierung bzw. Werkzeugeinschub -- tel ve "
    "alet TEK sinif, yani backbone ayrimi SILMEK uzere egitildi.")
    ap .add_argument ("--aux-w",type =float ,default =0.5 ,help ="yardimci kaybin agirligi")
    ap .add_argument ("--aux-pos-weight",type =float ,default =2.0 ,
    help ="ALET pozitif agirligi (measured: alet/tel tepe orani ~0.46)")
    ap .add_argument ("--label-smooth",type =float ,default =0.0 ,
    help ="Y6: kismi BCE hedefini yumusatir (0.05 tipik). "
    "Kismi etiketler opening kenarinda belirsiz; kesin "
    "1.0 hedefi asiri confidence uretir.")
    ap .add_argument ("--partial-pos-weight",type =float ,default =20.0 ,help ="positive weight for the masked CableEntry BCE (CableEntry is ~1.5% of vertices)")
    ap .add_argument ("--partial-target",choices =["connection","cableentry"],default ="connection",
    help ="which channel the human partial marks supervise. connection = CableEntry+Contact (default; the annotator marks wire openings, and MEASURED the corpus calls ~half of them Contact -- supervising CableEntry alone fights the corpus). cableentry = the literal channel (kept to reproduce the 0.586/0.584 runs)")
    ap .add_argument ("--terminal-only",action ="store_true",help ="ablation: add only the EEC parts whose category_guess is terminal_block")
    ap .add_argument ("--augment",action ="store_true",help ="recipe: random rotation+jitter of xyz input (operators intrinsic) -> orientation/mesh robustness")
    ap .add_argument ("--augment-maxang",type =float ,default =1.05 ,help ="max augment rotation angle (rad); small (0.2-0.4) = mild, 1.05 = aggressive (the heuristic-tuned one that hurt quality)")
    ap .add_argument ("--ce-weight-mult",type =float ,default =1.0 ,help ="recipe: multiply the CableEntry class weight (target the missed-CableEntry failure mode)")
    ap .add_argument ("--thesis-weights",action ="store_true",help ="use refit91's exact thesis class weights (vs inv-freq) -> test if its edge is reproducible held-out")
    ap .add_argument ("--select-metric",choices =["miou","cableentry_iou","connection_iou"],default ="connection_iou",
    help ="save best-by this HUMAN-GT val metric. connection_iou (CableEntry+Contact merged) = the cp-v3 CP-placement selector (default); cableentry_iou = cp-v2-era; miou = old mean")
    a =ap .parse_args ()
    # CIKTI DIZINI GARANTI (2026-08-08): `--checkpoint-out results/seg_b9/...`
    # verildiginde directory otherwise torch.save ILK KAYITTA patliyor -- i.e. training
    # 10 epoch bosuna kosuyor and olduktan after difference ediliyor (40 dk loss).
    _cd =os .path .dirname (a .checkpoint_out )
    if _cd :
        os .makedirs (_cd ,exist_ok =True )
        # FINE-TUNE MIMARI UYUMU (K6.5-b, 2026-08-07): source ckpt own cfg'sini carries.
        # g10 `n_eig=96` with egitilmis, this betigin varsayilani 64. Uyusmazsa model
        # EGITILDIGINDEN BASKA a oztabanla runs; error VERMEZ, only kotu sonuc
        # gives and "few-shot does not work" diye YANLIS a arm kapatilirdi.
    _ilk_cfg =None 
    if a .init_from :
        import torch as _t 
        _ck =_t .load (a .init_from ,map_location ="cpu",weights_only =False )
        _ilk_cfg =(_ck .get ("cfg")if isinstance (_ck ,dict )else None )or {}
        _ne =_ilk_cfg .get ("n_eig")
        if _ne and int (_ne )!=int (a .k_eig ):
            print (f"  [fine-tune] k_eig {a .k_eig } -> {_ne } (kaynak ckpt'ten alindi)",
            flush =True )
            a .k_eig =int (_ne )
        del _ck 
    CKPT =a .checkpoint_out 
    np .random .seed (a .seed );torch .manual_seed (a .seed )
    os .makedirs ("results/seg_extra",exist_ok =True )
    os .makedirs (f"{OPS_BASE }_k{int (a .k_eig )}",exist_ok =True )
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    tr =load_extra (a .train_dir )if a .train_dir else dataset .load_split ("wscad_corpus_scheffler_exact","train",verify_hashes =False )
    va =load_extra (a .val_dir )if a .val_dir else dataset .load_split ("wscad_corpus_scheffler_exact","val",verify_hashes =False )
    if a .train_dir :
        print (f"  [resolution] corpus from {a .train_dir } ({len (tr )} train) / {a .val_dir } ({len (va )} val), "
        f"median {sorted (len (s0 ['verts'])for s0 in tr )[len (tr )//2 ]if tr else 0 } verts",flush =True )
    extra =[]if a .no_extra else load_extra ()
    if a .pseudo_dir :# SELF-TRAINING: model-predicted pseudo-labels (NOT human). Only kept if it
        p =load_extra (a .pseudo_dir )# beats the human-GT val baseline -- see make_pseudo_labels.py
        print (f"  [self-training] + {len (p )} PSEUDO-labelled parts from {a .pseudo_dir }",flush =True )
        extra =extra +p 
    partial =[]
    if a .partial_dir :# HUMAN partial labels: only CableEntry is marked, the rest is left 0. A normal
        for pdir in a .partial_dir :# 5-class loss would teach "Contact/SnapPoint = Housing"
            partial +=load_extra (pdir )# -> masked CableEntry-vs-not loss instead
        for s in partial :s ["partial_ce"]=True 
    if a .val_partial :
    # VAL de kismi etiketli -> metrik MASKELI olsun (otherwise maske never devreye girmez:
    # bayrak only EGITIM listesine konuyordu, val'e not).
        for s in va :s ["partial_ce"]=True 
        # GUARD: an all-zero partial file is NOT neutral -- under the masked BCE every vertex becomes
        # "not CableEntry", actively teaching the model to suppress CableEntry ten a part that does
        # have entries (i.e. an un-annotated part, not a genuinely entry-less one). Skip those.
        empty =[s ["part_id"]for s in partial if not (np .asarray (s ["labels"])==NCLS_CE ).any ()]
        if empty :
            print (f"  [partial-human] SKIPPING {len (empty )} un-annotated part(s) with 0 CableEntry "
            f"(would teach 'no entries here'): {empty }",flush =True )
            partial =[s for s in partial if s ["part_id"]not in set (empty )]
        print (f"  [partial-human] + {len (partial )} CableEntry-only labelled parts from {a .partial_dir } "
        f"(masked loss; unmarked classes NOT supervised)",flush =True )
    if a .kismi_tekrar >1 and partial :
        partial =partial *a .kismi_tekrar 
        print (f"  [few-shot] kismi ornekler {a .kismi_tekrar }x tekrarlandi -> {len (partial )}",
        flush =True )
    if a .yalniz_kismi :
        if not partial :
            raise SystemExit ("--yalniz-kismi verildi ama --partial-dir bos")
        print (f"  [few-shot] YALNIZ KISMI KIP: corpus ({len (tr )}) + extra ({len (extra )}) "
        f"BIRAKILDI, training {len (partial )} ornek",flush =True )
        tr ,extra =[],[]
    if a .terminal_only and extra :
        import json as _j 
        term =set (r ["part_id"]for r in _j .load (open ("eec_extra_manifest.json"))["parts"]if r ["category_guess"]=="terminal_block")
        extra =[s for s in extra if s ["part_id"]in term ]
    print (f"our train {len (tr )} + extra {len (extra )} = {len (tr )+len (extra )} train | val {len (va )} | test LOCKED (untouched)",flush =True )
    train_samples =tr +extra 
    if a .train_limit and a .train_limit <len (train_samples ):# learning-curve: subset TRAIN only, val fixed
        idx =np .random .RandomState (a .seed ).permutation (len (train_samples ))[:a .train_limit ]
        train_samples =[train_samples [i ]for i in idx ]
        print (f"  [learning-curve] train subset -> {len (train_samples )} parts (val kept full {len (va )})",flush =True )
    if a .limit :
        train_samples =train_samples [:a .limit ];va =va [:max (3 ,a .limit //4 )]
        # inverse-frequency class weights from the training labels
    cnt =np .zeros (NCLS )
    for s in train_samples :
        for c in range (NCLS ):
            cnt [c ]+=(np .asarray (s ["labels"])==c ).sum ()
    if a .thesis_weights :# refit91's exact class weights -> is its edge reproducible held-out (vs inv-freq)?
        w =np .array ([46.4605 ,221.5453 ,1156.4556 ,546.8523 ,514.3766 ])
    else :
        w =cnt .sum ()/(NCLS *np .maximum (cnt ,1 ))
    w [NCLS_CE ]*=a .ce_weight_mult # boost CableEntry (the class the model misses ten unseen parts)
    w =torch .tensor (w ,dtype =torch .float32 ,device =dev )
    print (f"class weights (inv-freq, CE x{a .ce_weight_mult }): {w .cpu ().numpy ().round (3 )} | augment={a .augment }",flush =True )

    print ("prep operators (cached) ...",flush =True )
    tr_d =prep (train_samples ,a .k_eig )+prep (partial ,a .k_eig );va_d =prep (va ,a .k_eig )
    npart =sum (1 for d in tr_d if d .get ("partial_ce"))
    print (f"prepared train {len (tr_d )} ({npart } partial-human) val {len (va_d )}",flush =True )
    cfg ={"input_features":a .input_features ,"loss":"nll","n_diffusion_blocks":3 ,"width":64 ,"n_eig":a .k_eig ,
    "dropout":0.3 ,"tversky_weight":a .tversky ,"tversky_alpha":0.3 ,"tversky_beta":0.7 ,
    "tversky_gamma":a .tversky_gamma ,
    "sinir_weight":a .sinir_weight }
    model ,meta =D .build_diffusionnet (cfg ,n_classes =NCLS );model =model .to (dev )
    # FEW-SHOT / FINE-TUNE: sifirdan not, verilen ckpt'ten basla (K6.5-b).
    # Mimari AYNI olmak zorunda (k_eig dahil) -- strict=True bilerek, silent kismi
    # yukleme "fine-tune ettim" sanip aslinda rastgele agirlikla egitmeye path acar.
    if a .init_from :
    # weights_only=False BILEREK: PyTorch 2.6 varsayilani True and KENDI
    # ckpt'lerimizdeki numpy skalerlerini reddediyor. Dosya bizim training
    # betigimizin ciktisi (guvenilir source); disaridan gelen ckpt yuklenmez.
        _sd =torch .load (a .init_from ,map_location =dev ,weights_only =False )
        if isinstance (_sd ,dict ):
        # bizim ckpt formatimiz: {"state": ..., "cfg": ..., "meta": ...}
            for _k in ("state","model","state_dict"):
                if _k in _sd :
                    _sd =_sd [_k ]
                    break 
        model .load_state_dict (_sd ,strict =True )
        print (f"BASLANGIC AGIRLIGI: {a .init_from } (fine-tune kipi)",flush =True )
        # FB-2: YARDIMCI KAFA. Govde paylasilir (last_lin ONCESI 128-d hidden temsil),
        # cikis ayridir. Ana 5-sinif kafasi and kaybi HIC DEGISMEZ.
    aux_lin =None ;_gizli ={}
    if a .aux_wire :
    # GIZLI BOYUT MODELDEN OKUNUR: cfg["width"]=64 but last_lin'in girdisi 128.
    # (cfg'ye guvenmek "mat1 and mat2 shapes cannot be multiplied" with patliyordu.)
        _w =int (model .last_lin .in_features )
        aux_lin =torch .nn .Linear (_w ,1 ).to (dev )
        def _kanca (mod ,giren ,cikan ):
            _gizli ["z"]=giren [0 ]
        model .last_lin .register_forward_hook (_kanca )
        n_aux =sum (1 for _d in tr_d if _d .get ("aux")is not None )
        print (f"  [aux-wire] {n_aux }/{len (tr_d )} parcada TEL/ALET etiketi var "
        f"| agirlik {a .aux_w } | pos_w {a .aux_pos_weight }",flush =True )
    _par =list (model .parameters ())+(list (aux_lin .parameters ())if aux_lin else [])
    opt =torch .optim .Adam (_par ,lr =a .lr )
    sched =torch .optim .lr_scheduler .StepLR (opt ,step_size =a .lr_decay_every ,gamma =a .lr_decay_rate )
    best =-1 
    for ep in range (1 ,a .epochs +1 ):
        model .train ();tot =0.0 
        for i in np .random .permutation (len (tr_d )):
            d =tr_d [i ];opt .zero_grad ()
            ops =to_dev (d ["ops"],dev );lab =d ["lab"].to (dev )
            if a .augment :# rotate xyz input (operators are intrinsic -> unchanged) + small jitter
                x =ops ["verts"]@_rand_rot (dev ,a .augment_maxang ).T 
                x =x +torch .randn_like (x )*(0.005 *float (x .abs ().max ()))
            else :
                x =D ._model_input (ops ,meta )
            out =D ._forward (model ,ops ,x )
            if d .get ("partial_ce"):
            # HUMAN PARTIAL labels: only the wire openings were marked; every other vertex was
            # left 0 but actually contains Contact/SnapPoint/LabelSurface. Supervising those as
            # Housing would corrupt the model -> masked BCE that supervises only "is this vertex
            # part of a connection opening", leaving every other class free.
            #
            # WHICH CHANNEL: --partial-target connection (default) sums CableEntry+Contact.
            # MEASURED 2026-07-20: ten the human-marked vertices the corpus-trained model says
            # Contact 49.8% / Housing 32.0% / LabelSurface 13.6% / CableEntry only 4.5% -- the
            # annotator marked square/clamp entries that the corpus calls CONTACT. Supervising
            # the CableEntry channel alone therefore fights the corpus ten ~half the marks. The
            # union is also the right target: cp-v3 defines a CP as CableEntry OR Contact, and
            # both product metrics (Connection IoU, arbiter CP F1) score the union.
                sm =torch .softmax (out ,-1 )
                p_pos =(sm [:,NCLS_CE ]+sm [:,NCLS_CT ]if a .partial_target =="connection"
                else sm [:,NCLS_CE ]).clamp (1e-6 ,1 -1e-6 )
                y =(lab ==NCLS_CE ).float ()
                # Y6 LABEL SMOOTHING (2026-08-13). Kismi labels ELLE
                # isaretlendi and opening KENARINDA belirsizdir; conclusive 1.0/0.0
                # hedefi modeli asiri kendine safe yapar. eps with hedef
                # (1-eps)/eps'e cekilir. Augmentasyon (bugunun kazanani)
                # with same aileden a DUZENLEYICIDIR.
                if a .label_smooth >0 :
                    y =y *(1.0 -a .label_smooth )+(1.0 -y )*a .label_smooth 
                pw =a .partial_pos_weight 
                per_v =-(pw *y *torch .log (p_pos )+(1 -y )*torch .log (1 -p_pos ))
                ig =d .get ("ignore")
                if ig is not None :# adjudicated "unsure" -> no signal either way
                    keep =~ig .to (per_v .device )
                    loss =per_v [keep ].mean ()if keep .any ()else per_v .sum ()*0.0 
                else :
                    loss =per_v .mean ()
            else :
            # Y13: `faces` verilirse boundary-farkindali terim devreye
            # girer (weight `sinir_weight`, varsayilan 0 = does not change).
                loss =D ._compute_loss (out ,lab ,w ,meta ,cfg ,
                faces =ops .get ("faces"))
                # FB-2 YARDIMCI KAYIP: ana loss above hesaplandi and DEGISMEDI;
                # here only USTUNE ekleniyor. Maskeli: -1 which is vertices sinyal vermez.
            if aux_lin is not None and d .get ("aux")is not None and "z"in _gizli :
                ya =d ["aux"].to (dev )
                mk =ya >=0 
                if bool (mk .any ()):
                # hidden temsil (1, V, 128) geliyor -> duzlestir, otherwise maske
                # [V] with tensor [1,V] uyusmuyor.
                    lg =aux_lin (_gizli ["z"]).reshape (-1 )
                    pw =torch .tensor (float (a .aux_pos_weight ),device =dev )
                    la =torch .nn .functional .binary_cross_entropy_with_logits (
                    lg [mk ],ya [mk ].float (),pos_weight =pw )
                    loss =loss +float (a .aux_w )*la 
            loss .backward ();opt .step ();tot +=float (loss )
        sched .step ()
        print (f"ep{ep :3d} loss{tot /max (len (tr_d ),1 ):.4f} lr{opt .param_groups [0 ]['lr']:.2e}",flush =True )
        if ep %10 ==0 or ep ==a .epochs :
            mi ,ac ,ce_iou ,conn_iou =miou (model ,meta ,va_d ,dev )
            sel ={"cableentry_iou":ce_iou ,"connection_iou":conn_iou }.get (a .select_metric ,mi )
            print (f"  -> val mIoU={mi :.4f} acc={ac :.4f} CE_IoU={ce_iou :.4f} Conn_IoU={conn_iou :.4f} "
            f"(select on {a .select_metric }={sel :.4f})",flush =True )
            if sel >best :
                best =sel ;torch .save ({"state":model .state_dict (),"cfg":cfg ,"meta":meta ,
                "aux_state":(aux_lin .state_dict ()if aux_lin is not None else None ),
                "val_miou":mi ,"val_acc":ac ,"val_cableentry_iou":ce_iou ,
                "val_connection_iou":conn_iou ,
                "select_metric":a .select_metric },CKPT )
                # #8 FIX: corpus-val WEI-underfit epoch secebilir. Last-epoch'u da kaydet -> ikisini WEI arbiter'da
                # (real hedef) karsilastirip most iyisi secilir (post-hoc WEI selection). NcLS/masked-WEI-val instead of this.
    last_ckpt =CKPT .replace (".pt","_last.pt")
    torch .save ({"state":model .state_dict (),"cfg":cfg ,"meta":meta ,"select_metric":"last_epoch"},last_ckpt )
    print (f"DONE best val {a .select_metric }={best :.4f} -> {CKPT }",flush =True )
    print (f"DONE last epoch -> {last_ckpt }",flush =True )


if __name__ =="__main__":
    main ()
