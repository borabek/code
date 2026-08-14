# -*- coding: utf-8 -*-
"""INSTANCE-OYLAMA BASLIGI (tezin Ausblick'inin data-verimli hali, Masterarbeit row 2099).

TEZ NE DIYOR: 2D YOLOv6 Kontaktierung/Kabeleinfuehrung for "nicht empfehlenswert" (ortulme; more
extra etiketle cozulemez) -> YERINE "ein 3D-basiertes YOLO ... auf Punktwolken ... 3D-Bounding-Box
Labels ausgehend von den Scheitelpunkt-Labels". Sifirdan 3D detektor bizim 853 instance'imiz for AZ
data; that is why same fikri MEVCUT DiffusionNet uzerine a OYLAMA BASLIGI as kuruyoruz:

  each baglanti-verteksi -> ait oldugu INSTANCE MERKEZINE offset (3 ek output kanali)
  inference: oy = verteks + offset -> oylari kumele -> each cluster BIR CP

WHY: bugun CP'leri kaybettigimiz yer post-proc. min_v 30 small fragmentleri atiyor, cluster_mm 5
komsu aciklikari birlestiriyor/boluyor -- FBI teshisi model-FN'lerinin %63'unde modelin ZATEN
atesledigini showed. Oylama, elle-ayarli esiklerin instead of KUMELEMEYI OGRENIR.

Cikti: 8 channel (0-4 sinif logit, 5-7 offset). Sinif kaybi first 5'te (mevcut recete aynen), offset
kaybi only baglanti-vertekslerinde (smooth-L1). Egitim verisi certain-source (corpus + insan kismi).

Kullanim: PYTHONPATH=_diffusion_net_repo/src .venv/Scripts/python.exe train_vote.py \
    --partial-dir _label_targets _label_targets_2 _label_targets_3 _label_targets_recall _label_targets_recall_hard \
    --seed 2 --k-eig 96 --checkpoint-out results/seg_extra/vote_s2.pt
"""
import os ,sys ,argparse 
import numpy as np ,torch 
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import diffusionnet as D ,connector3d 
import scheffler_dataset as dataset 
import train_seg_extra as T 
from derive_3d_boxes import components 

NCLS =5 
N_OFF =3 
CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )


def offset_targets (V ,F ,L ):
    """each baglanti-verteksi for: ait oldugu instance'in merkezine offset. (N,3) hedef + (N,) maske."""
    tgt =np .zeros ((len (V ),3 ),np .float32 )
    valid =np .zeros (len (V ),bool )
    mask =np .isin (L ,(CE ,CT ))
    if not mask .any ():
        return tgt ,valid 
    for comp in components (V ,F ,mask ):
        if len (comp )<4 :
            continue 
        c =V [comp ].mean (0 )
        tgt [comp ]=(c [None ,:]-V [comp ]).astype (np .float32 )
        valid [comp ]=True 
    return tgt ,valid 


def prep_vote (samples ,k_eig ):
    """train_seg_extra.prep + offset hedefleri."""
    out =T .prep (samples ,k_eig )
    for d ,s in zip (out ,samples ):
        V =np .asarray (s ["verts"],float );F =np .asarray (s ["faces"],np .int64 )
        L =np .asarray (s ["labels"],np .int64 )
        tgt ,valid =offset_targets (V ,F ,L )
        d ["off_tgt"]=torch .from_numpy (tgt )
        d ["off_valid"]=torch .from_numpy (valid )
    return out 


def evaluate (model ,meta ,data ,dev ):
    """sinif metrikleri (first 5 channel) + offset hatasi (mm)."""
    model .eval ()
    inter =np .zeros (NCLS );union =np .zeros (NCLS );corr =0 ;tot =0 ;off_err =[]
    with torch .no_grad ():
        for d in data :
            ops =T .to_dev (d ["ops"],dev );lab =d ["lab"].to (dev )
            out =D ._forward (model ,ops ,D ._model_input (ops ,meta ))
            pred =out [:,:NCLS ].argmax (-1 )
            corr +=int ((pred ==lab ).sum ());tot +=len (lab )
            for c in range (NCLS ):
                p =pred ==c ;t =lab ==c 
                inter [c ]+=float ((p &t ).sum ());union [c ]+=float ((p |t ).sum ())
            v =d ["off_valid"].to (dev )
            if v .any ():
                e =(out [:,NCLS :][v ]-d ["off_tgt"].to (dev )[v ]).norm (dim =-1 )
                off_err .append (float (e .mean ()))
    iou =inter /np .maximum (union ,1 )
    conn =(inter [CE ]+inter [CT ])/max (union [CE ]+union [CT ],1 )
    return iou .mean (),corr /max (tot ,1 ),iou [CE ],conn ,(float (np .mean (off_err ))if off_err else -1.0 )


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--partial-dir",nargs ="+",default =[])
    ap .add_argument ("--partial-pos-weight",type =float ,default =20.0 )
    ap .add_argument ("--epochs",type =int ,default =200 )
    ap .add_argument ("--k-eig",type =int ,default =96 )
    ap .add_argument ("--lr",type =float ,default =1e-3 )
    ap .add_argument ("--lr-decay-every",type =int ,default =50 )
    ap .add_argument ("--lr-decay-rate",type =float ,default =0.75 )
    ap .add_argument ("--seed",type =int ,default =0 )
    ap .add_argument ("--off-weight",type =float ,default =1.0 ,help ="offset kaybinin agirligi")
    ap .add_argument ("--checkpoint-out",required =True )
    ap .add_argument ("--device",default ="cuda"if torch .cuda .is_available ()else "cpu")
    a =ap .parse_args ()
    torch .manual_seed (a .seed );np .random .seed (a .seed )
    dev =a .device 

    tr =dataset .load_split ("wscad_corpus_scheffler_exact","train",verify_hashes =False )
    va =dataset .load_split ("wscad_corpus_scheffler_exact","val",verify_hashes =False )
    partial =[]
    for pdir in a .partial_dir :
        partial +=T .load_extra (pdir )
    print (f"train {len (tr )} corpus + {len (partial )} insan kismi | val {len (va )} | k_eig {a .k_eig }",flush =True )

    # sinif agirliklari (inv-freq) -- mevcut receteyle same
    cnt =np .zeros (NCLS )
    for s in tr :
        u ,c =np .unique (np .asarray (s ["labels"],np .int64 ),return_counts =True )
        cnt [u ]+=c 
    w =torch .tensor ((cnt .sum ()/np .maximum (cnt ,1 ))/NCLS ,dtype =torch .float32 ,device =dev )
    print (f"class weights: {w .cpu ().numpy ().round (3 )}",flush =True )

    print ("prep operators + offset hedefleri ...",flush =True )
    tr_d =prep_vote (tr ,a .k_eig )+prep_vote (partial ,a .k_eig )
    va_d =prep_vote (va ,a .k_eig )
    nv =sum (int (d ["off_valid"].any ())for d in tr_d )
    print (f"hazir train {len (tr_d )} ({nv } parcada instance hedefi) val {len (va_d )}",flush =True )

    # loss="ce" is REQUIRED here, not a style choice: with loss="nll" build_diffusionnet puts a
    # log_softmax ten the LAST layer across ALL C_out channels -- which would squash the 3 offset
    # channels into log-probabilities and silently destroy the regression. "ce" -> last_activation
    # None -> raw logits for the 5 class channels and free real values for the 3 offset channels.
    cfg ={"input_features":"xyz","loss":"ce","n_diffusion_blocks":3 ,"width":64 ,
    "n_eig":a .k_eig ,"dropout":0.3 ,"tversky_weight":0.0 ,
    "vote_head":True ,"n_seg_classes":NCLS ,"n_offset":N_OFF }
    model ,meta =D .build_diffusionnet (cfg ,n_classes =NCLS +N_OFF );model =model .to (dev )
    opt =torch .optim .Adam (model .parameters (),lr =a .lr )
    sched =torch .optim .lr_scheduler .StepLR (opt ,step_size =a .lr_decay_every ,gamma =a .lr_decay_rate )

    best =-1 
    for ep in range (1 ,a .epochs +1 ):
        model .train ();tot =0.0 
        for i in np .random .permutation (len (tr_d )):
            d =tr_d [i ];opt .zero_grad ()
            ops =T .to_dev (d ["ops"],dev );lab =d ["lab"].to (dev )
            out =D ._forward (model ,ops ,D ._model_input (ops ,meta ))
            logits =out [:,:NCLS ];off =out [:,NCLS :]
            if d .get ("partial_ce"):
                sm =torch .softmax (logits ,-1 )
                p_pos =(sm [:,CE ]+sm [:,CT ]).clamp (1e-6 ,1 -1e-6 )
                y =(lab ==CE ).float ()
                per_v =-(a .partial_pos_weight *y *torch .log (p_pos )+(1 -y )*torch .log (1 -p_pos ))
                ig =d .get ("ignore")
                cls_loss =per_v [~ig .to (dev )].mean ()if ig is not None else per_v .mean ()
            else :# logits are RAW (cfg loss="ce" -> no last activation)
                cls_loss =torch .nn .functional .cross_entropy (logits ,lab ,weight =w )
            v =d ["off_valid"].to (dev )
            if v .any ():
                off_loss =torch .nn .functional .smooth_l1_loss (off [v ],d ["off_tgt"].to (dev )[v ])
            else :
                off_loss =off .sum ()*0.0 
            loss =cls_loss +a .off_weight *off_loss 
            loss .backward ();opt .step ();tot +=float (loss )
        sched .step ()
        print (f"ep{ep :3d} loss{tot /max (len (tr_d ),1 ):.4f} lr{opt .param_groups [0 ]['lr']:.2e}",flush =True )
        if ep %10 ==0 or ep ==a .epochs :
            mi ,ac ,ce_iou ,conn ,oerr =evaluate (model ,meta ,va_d ,dev )
            print (f"  -> val mIoU={mi :.4f} acc={ac :.4f} CE_IoU={ce_iou :.4f} Conn_IoU={conn :.4f} "
            f"offset_err={oerr :.2f}mm",flush =True )
            if conn >best :
                best =conn 
                torch .save ({"state":model .state_dict (),"cfg":cfg ,"meta":meta ,
                "val_connection_iou":conn ,"val_offset_err_mm":oerr ,
                "n_seg_classes":NCLS ,"n_offset":N_OFF },a .checkpoint_out )
    print (f"DONE best val connection_iou={best :.4f} -> {a .checkpoint_out }",flush =True )


if __name__ =="__main__":
    main ()
