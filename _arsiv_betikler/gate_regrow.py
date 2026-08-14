# -*- coding: utf-8 -*-
"""GATE'I BUYUMUS KORPUSLA YENIDEN EGIT -- olculmemis single canli kaldirac.

DURUM: dagitilan wire-gate 1041 parcadan cikarilmis adaylarla egitildi (25.07). Korpus that
gunden beri 1906 uygun parcaya output: gate'in HIC gormedigi 865 part present.

WHY UMUT VAR (S5 bulgusu, measured): candidate arzini degiten a mudahale KUCUK korpusta notr
okunur; corpus two katina cikinca same double +0.037 verdi. Gate de candidate dagilimini ogrenen
a bilesen -- same mantik gecerli.

SAFETY: urun gate'i (results/wire_gate.pkl) BU BETIKTE ASLA UZERINE YAZILMAZ. Yeni model
ayri dosyaya gider; kabul karari OOF olcumune bakilarak AYRICA verilir.

KILL (olcumden before yazildi):
  * OOF CP-F1 katkisi < +0.02  -> reddedilir
  * VEYA manufacturer-disi sondada loss > 0.01 -> reddedilir (ExtraTrees dersi: easy bolunmede
    kazanan, distribution kayinca kaybedebilir; holdout harcamadan ONCE sinanir)
"""
import os ,sys ,json ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

TAG =os .environ .get ("GATE_TAG","")# output dosyalarini ayirmak for (orn. "_minv4")
NPZ_NEW =f"results/gate_regrow_data{TAG }.npz"
PKL_NEW =f"results/wire_gate_regrow{TAG }.pkl"
NPZ_OLD ="results/f1_sweep_data.npz"
PARTIAL =f"results/gate_regrow_partial{TAG }.npz"# ara kayit -- cokme/oldurme sonrasi devam
INFLIGHT ="results/gate_regrow_inflight.txt"# this an islenen part (zehirli-part korumasi)
SKIPFILE ="results/gate_regrow_skip.txt"# kalici atlama listesi


def _load_skip ():
    """Daha before bizi kilitleyen parcalari atla.

    2026-07-29: single a WEI parcasi spektral ayristirmada 47 DAKIKA %100 CPU'da dondu and
    no output uretmedi; 1700 parcalik inference bellekteydi and oldurulunce ucup gitti.
    Artik each part ISLENMEDEN ONCE adi INFLIGHT'a yazilir. Kosu a more baslatildiginda
    orada duran part "bizi olduren part"dir and kalici atlama listesine alinir.
    """
    skip =set ()
    if os .path .exists (SKIPFILE ):
        skip |={x .strip ()for x in open (SKIPFILE )if x .strip ()}
    if os .path .exists (INFLIGHT ):
        raw =open (INFLIGHT ).read ().strip ().split ()
        stuck =raw [0 ]if raw else ""
        attempt =int (raw [1 ])if len (raw )>1 else 1 
        if stuck :
        # ILK KESINTIDE KARA LISTEYE ALMA. Mekanizma "part asildi" with "process disaridan
        # olduruldu"yu ayirt edemiyor -- 2026-07-29 gece elektrik kesildi and masum a part
        # (2466530000) zehirli sayilacakti. Gercekten asilan part IKINCI denemede de asilir;
        # that yuzden kalici atlama however 2. kesintiden after is done.
            if attempt >=2 :
                skip .add (stuck )
                with open (SKIPFILE ,"a")as fh :
                    fh .write (stuck +"\n")
                print (f"  [zehirli part] {stuck } IKI kez kosuyu kilitledi -> kalici atlama",
                flush =True )
            else :
                _RETRY [stuck ]=attempt +1 
                print (f"  [yeniden dene] {stuck } bir kez yarim kaldi (elektrik/oldurme olabilir) "
                f"-> {attempt +1 }. deneme, kara listeye ALINMADI",flush =True )
        os .remove (INFLIGHT )
    return skip 


_RETRY ={}# a times half kalan parts -> sonraki deneme numarasi


def resume_partial (parts ,partial_path ):
    """PARTIAL'i oku and satirlari GUNCEL parts listesine yeniden esle.

    Devam PARCA NUMARASINA according to is done, DONGU INDEKSINE according to DEGIL. eligible() diske new STEP
    dustukce buyur; this each enumerate indeksini kaydirir. Eski indekslerle devam etmek YANLIS
    parcalari skips and ngt'yi (part basina GT count) kaydirir -> regime ayrimi and recall paydasi
    SESSIZCE bozulur, i.e. metrigin kendisi. tests/test_gate_regrow_resume.py bunu korur.

    Doner: (Xs, votes, tp, grp, mfgs, fams, pts, dirs, pids, ngt, done, atilan_aday_sayisi)
    """
    d =np .load (partial_path ,allow_pickle =True )
    if "pids"not in d :
        raise SystemExit (
        f"{partial_path } part numarasi icermiyor (eski bicim) -> corpus buyuduyse guvenle "
        f"devam edilemez. Ya dosyayi silip bastan kosun, ya eski korpusla bitirin.")
    newk ={pid :k for k ,(_m ,pid ,_j ,_s )in enumerate (parts ,1 )}
    old_pid =[str (q )for q in d ["pids"].tolist ()]
    g2p ={int (g ):q for g ,q in zip (d ["groups"].tolist (),old_pid )}
    keep =np .array ([q in newk for q in old_pid ],bool )
    pids =[q for q in old_pid if q in newk ]
    grp =[newk [q ]for q in pids ]
    ngt ={newk [g2p [int (g )]]:int (n )for g ,n in zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ())
    if g2p .get (int (g ))in newk }
    return ([d ["X"][keep ]],[d ["votes"][keep ]],[d ["y"][keep ]],grp ,
    d ["mfg"][keep ].tolist (),d ["fams"][keep ].tolist (),
    [d ["pts"][keep ]],[d ["dirs"][keep ]],pids ,ngt ,set (grp ),int ((~keep ).sum ()))


def extract ():
    import torch 
    import diffusionnet as D ,cp_openings ,thesis_remesh ,wire_gate 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from big_arbiter import eligible ,CE ,CT 
    from json_dataset import family_key 
    import f1_sweep 

    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cfg =json .load (open ("cp_config.json"))
    cks =cfg ["robot_vote2_checkpoints"]
    models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    pp =cfg ["prediction_postproc"]

    parts =list (eligible ())
    skip =_load_skip ()

    Xs ,votes ,tp ,grp ,mfgs ,fams ,ngt =[],[],[],[],[],[],{}
    pts ,dirs ,pids =[],[],[]
    done =set ()
    if os .path .exists (PARTIAL ):# ONCEKI KOSUDAN DEVAM ET
        (Xs ,votes ,tp ,grp ,mfgs ,fams ,pts ,dirs ,pids ,ngt ,done ,
        gone )=resume_partial (parts ,PARTIAL )
        print (f"  [devam] {PARTIAL } icinden {len (ngt )} part yuklendi, indeksler guncel korpusa "
        f"yeniden eslendi"+(f" ({gone } candidate artik korpusta olmayan parcalardan atildi)"
        if gone else ""),flush =True )

    print (f"{len (parts )} parts | ensemble {len (cks )} | {len (done )} done | {len (skip )} skipped",
    flush =True )

    def _checkpoint ():
        if not Xs :
            return 
        gid =np .array (sorted (ngt ));ng =np .array ([ngt [g ]for g in gid ])
        kw ={}
        if pts :
            kw =dict (pts =np .vstack (pts ),dirs =np .vstack (dirs ),pids =np .array (pids ))
        np .savez (PARTIAL ,X =np .vstack (Xs ),votes =np .concatenate (votes ),y =np .concatenate (tp ),
        groups =np .array (grp ),mfg =np .array (mfgs ),fams =np .array (fams ),
        grp_ids =gid ,ngt =ng ,**kw )

    t0 =time .time ()
    for k ,(mfg ,pid ,jf ,stp )in enumerate (parts ,1 ):
        if k in done or pid in skip :
            continue 
        with open (INFLIGHT ,"w")as fh :# ISLENMEDEN ONCE isaretle (pid + deneme no)
            fh .write (f"{pid } {_RETRY .get (pid ,1 )}")
        try :
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j .get ("ConnectionPoints",[])],float )
            if not len (G ):
                continue 
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )

            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            R ,t ,_ =align_frames (Vr ,Vj )
            Gm =(G -t )@R ;Gdm =Gd @R 

            # CALISMA MANTIGINI BIREBIR TAKLIT ET (D1.3 bulgusu, measured 2026-07-30):
            # eskiden here conn_promote HIC gecilmiyordu -> gate, calisma aninda very-CP
            # adaylarinin %73'unu (256/350) EGITIMDE HIC GORMEDIGI turden candidate as
            # skorluyordu. robot_cp before promote'SUZ produces, router very-CP derse
            # promote'LU yeniden produces. Ayni sirayi here da uyguluyoruz.
            import robot_cp as _RC 
            pbs =[]
            acc =None 
            for model ,meta in models :
                _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
                op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
                return_probs =True )
                pb =np .asarray (pb ,float )
                acc =pb if acc is None else acc +pb 
                pbs .append (pb )

                # PARITE (2026-08-01 denetimi): candidate uretimi residual URUNUN own fonksiyonundan.
                # Eskiden here AYRI a path vardi and IKI places sapiyordu:
                #   * step_path GECILMIYORDU -> B-rep analitik ekseni/fiziksel ozellikler egitimde YOK
                #   * f1_sweep.union_all kullaniliyordu, urun whereas _vote2 (BENZERSIZ model sayimi)
                #     -> egitimde votes 12'ye cikiyordu, uründe ceiling 4. `votes` gate'in durust
                #        bolmede GENELLESEN TEK ozelligi oldugu for this ozellikle agir a sapmaydi.
            if os .environ .get ("GATE_AVG_MEMBER","0")not in ("0","false","False"):
                raise SystemExit ("GATE_AVG_MEMBER PARITEYI BOZAR (urun 4 uye kullanir). "
                "Acmak icin once urun tarafina 5. uye eklenmeli.")
            cps ,probs ,_is_hi ,_ =_RC .derive_candidates (V ,F ,pbs ,stp ,cfg =cfg )
            if not cps :
                continue 
                # step_path SART: WG_FIZ_FEATS=1 iken B-rep fiziksel ozellikleri buradan geliyor.
                # Verilmezse that 5 column SESSIZCE sifir becomes and gate onlari "bilgisiz" ogrenir --
                # E maddesinde full as this yasandi (17 sutunlu gate 4 new sutunu sifir aliyordu).
            X =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =stp )

            # ETIKET: big_arbiter konvansiyonu (eksene dik distance + +-40mm axial pencere)
            P =np .array ([c ["point"]for c in cps ],float )
            tol =max (3.0 ,0.06 *float (np .linalg .norm (V .max (0 )-V .min (0 ))))
            diff =P [:,None ,:]-Gm [None ,:,:]
            al =(diff *Gdm [None ,:,:]).sum (-1 )
            pe =np .linalg .norm (diff -al [...,None ]*Gdm [None ,:,:],axis =-1 )
            pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
            yy =np .zeros (len (P ),int );up ,ug =set (),set ()
            for dd ,a ,b in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (Gm ))):
                if dd >tol or a in up or b in ug :
                    continue 
                up .add (a );ug .add (b );yy [a ]=1 

            Xs .append (X );votes .append (np .array ([c ["_votes"]for c in cps ]));tp .append (yy )
            # K7 for: candidate konum/yonu + part no. Renk ozelligi GPU gerektirmez, offline eklenir.
            pts .append (P );dirs .append (np .array ([c .get ("direction",(0 ,0 ,1 ))for c in cps ],float ))
            pids +=[pid ]*len (cps )
            grp +=[k ]*len (cps );mfgs +=[1 if mfg =="WEI"else 0 ]*len (cps )
            fams +=[family_key (pid )]*len (cps );ngt [k ]=len (Gm )
        except Exception :
            continue 
        if k %50 ==0 :
            el =time .time ()-t0 
            rem =sum (1 for kk in range (k +1 ,len (parts )+1 )if kk not in done )
            print (f"  {k }/{len (parts )}  {el :.0f}s  ({len (ngt )} parts held)",flush =True )
            _checkpoint ()# ARA KAYIT: buradan sonrasi cokse bile emek kaybolmaz

    if os .path .exists (INFLIGHT ):
        os .remove (INFLIGHT )
    X =np .vstack (Xs );v =np .concatenate (votes );y =np .concatenate (tp )
    gid =np .array (sorted (ngt ));ng =np .array ([ngt [g ]for g in gid ])
    kw =dict (pts =np .vstack (pts ),dirs =np .vstack (dirs ),pids =np .array (pids ))if pts else {}
    np .savez (NPZ_NEW ,X =X ,votes =v ,y =y ,groups =np .array (grp ),mfg =np .array (mfgs ),
    fams =np .array (fams ),grp_ids =gid ,ngt =ng ,**kw )
    print (f"  -> {NPZ_NEW }  ({len (y )} candidates, {len (gid )} parts)",flush =True )


def evaluate ():
    """Leakage-free OOF: old gate verisi vs buyumus data, AYNI protokolle."""
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    import pickle 

    def oof_f1 (d ,group_key ,thr =0.35 ):
        X ,y =d ["X"],d ["y"]
        g =d [group_key ]
        oof =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,g ):
            clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (X [tr ],y [tr ])
            oof [te ]=clf .predict_proba (X [te ])[:,1 ]
        keep =oof >=thr 
        tp =int ((y [keep ]==1 ).sum ());nk =int (keep .sum ());npos =int ((y ==1 ).sum ())
        p =tp /max (nk ,1 );r =tp /max (npos ,1 )
        return 2 *p *r /max (p +r ,1e-9 ),p ,r ,oof 

    new =np .load (NPZ_NEW ,allow_pickle =True )
    old =np .load (NPZ_OLD ,allow_pickle =True )
    print (f"{'set':<14}{'parts':>7}{'cands':>8}{'P':>8}{'R':>8}{'F1':>8}")
    f_old ,p_o ,r_o ,_ =oof_f1 (old ,"groups")
    print (f"{'deployed':<14}{len (old ['grp_ids']):>7}{len (old ['y']):>8}{p_o :>8.3f}{r_o :>8.3f}{f_old :>8.4f}")
    f_new ,p_n ,r_n ,_ =oof_f1 (new ,"groups")
    print (f"{'regrown':<14}{len (new ['grp_ids']):>7}{len (new ['y']):>8}{p_n :>8.3f}{r_n :>8.3f}{f_new :>8.4f}")
    print (f"\nGATE-LEVEL GAIN: {f_new -f_old :+.4f}")

    # ZOR BOLUNME: manufacturer-disi (ExtraTrees dersi -- holdout harcamadan before sinanir)
    X ,y ,m =new ["X"],new ["y"],new ["mfg"]
    hard =[]
    for held in (0 ,1 ):
        tr =m !=held ;te =m ==held 
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (X [tr ],y [tr ])
        s =clf .predict_proba (X [te ])[:,1 ]>=0.35 
        tp =int ((y [te ][s ]==1 ).sum ())
        p =tp /max (int (s .sum ()),1 );r =tp /max (int ((y [te ]==1 ).sum ()),1 )
        hard .append (2 *p *r /max (p +r ,1e-9 ))
    print (f"manufacturer-out (regrown): PXC-held {hard [0 ]:.4f} | WEI-held {hard [1 ]:.4f}")

    clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (X ,y )
    import wire_gate as WG 
    pickle .dump ({"clf":clf ,"feat_names":WG .FEAT_NAMES },open (PKL_NEW ,"wb"))
    rec ={"deployed_oof_f1":f_old ,"regrown_oof_f1":f_new ,"gain":f_new -f_old ,
    "manufacturer_out":{"pxc_held":hard [0 ],"wei_held":hard [1 ]},
    "n_parts_deployed":int (len (old ["grp_ids"])),"n_parts_regrown":int (len (new ["grp_ids"])),
    "kill_passed":bool ((f_new -f_old )>=0.02 )}
    json .dump (rec ,open ("results/gate_regrow.json","w"),indent =1 )
    print (f"\nKILL (>= +0.02): {'PASSED'if rec ['kill_passed']else 'DEAD'}")
    print (f"model -> {PKL_NEW }  (product gate NOT overwritten)")
    print ("receipt -> results/gate_regrow.json")


if __name__ =="__main__":
    if "--eval-only"in sys .argv :
        evaluate ()
    else :
        extract ();evaluate ()
