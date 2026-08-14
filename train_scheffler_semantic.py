"""Train five-class DiffusionNet ten the exact human-labelled WSCAD subset.

The command can see only the published 71-part train and 20-part validation
splits.  The dedicated final-test evaluator is a separate executable.
"""

from __future__ import annotations 

import argparse 
import hashlib 
import json 
import logging 
from pathlib import Path 
import platform 
import random 
import sys 

import numpy as np 


HERE =Path (__file__ ).resolve ().parent 
VENDORED_DIFFUSION_NET =HERE /"_diffusion_net_repo"/"src"
if VENDORED_DIFFUSION_NET .is_dir ():
    sys .path .insert (0 ,str (VENDORED_DIFFUSION_NET ))

import diffusionnet # noqa: E402
from connector_constants import CLASS_NAMES # noqa: E402
import scheffler_dataset as dataset # noqa: E402


def _sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b""):
            digest .update (block )
    return digest .hexdigest ()


def _seed_everything (seed :int )->None :
    random .seed (seed )
    np .random .seed (seed )
    import torch 

    torch .manual_seed (seed )
    if torch .cuda .is_available ():
        torch .cuda .manual_seed_all (seed )


def _choose_device (requested :str )->str :
    import torch 

    if requested =="auto":
        return "cuda"if torch .cuda .is_available ()else "cpu"
    if requested =="cuda"and not torch .cuda .is_available ():
        raise RuntimeError ("--device cuda requested but torch.cuda.is_available() is false")
    return requested 


def _inverse_frequency_weights (samples :list [dict ])->list [float ]:
    counts_by_name =dataset .class_counts (samples )
    counts =np .asarray ([counts_by_name [name ]for name in CLASS_NAMES ],dtype =float )
    if np .any (counts <=0 ):
        raise ValueError (f"cannot calculate inverse-frequency weights: {counts_by_name }")
    weights =counts .sum ()/(len (counts )*counts )
    return weights .tolist ()


def _write_json (path :Path ,payload :dict )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    temporary =path .with_suffix (path .suffix +".tmp")
    temporary .write_text (json .dumps (payload ,indent =2 ,ensure_ascii =False )+"\n",encoding ="utf-8")
    temporary .replace (path )


def parse_args ()->argparse .Namespace :
    parser =argparse .ArgumentParser (
    description =__doc__ ,formatter_class =argparse .ArgumentDefaultsHelpFormatter 
    )
    parser .add_argument ("--corpus",type =Path ,default =HERE /"wscad_corpus_scheffler_exact")
    parser .add_argument ("--checkpoint",type =Path ,default =HERE /"results"/"scheffler_semantic"/"best.pt")
    parser .add_argument ("--run-report",type =Path ,default =HERE /"results"/"scheffler_semantic"/"train_report.json")
    parser .add_argument ("--history-log",type =Path ,default =HERE /"results"/"scheffler_semantic"/"train_history.jsonl")
    parser .add_argument ("--op-cache",type =Path ,default =HERE /"results"/"scheffler_semantic"/"operators")
    parser .add_argument ("--epochs",type =int ,default =200 )
    parser .add_argument ("--eval-every",type =int ,default =5 )
    parser .add_argument ("--seed",type =int ,default =20260716 )
    parser .add_argument ("--device",choices =("auto","cpu","cuda"),default ="auto")
    parser .add_argument ("--input-features",choices =("xyz","hks"),default ="xyz")
    parser .add_argument ("--learning-rate",type =float ,default =1e-3 )
    parser .add_argument ("--lr-decay-every",type =int ,default =100 )
    parser .add_argument ("--lr-decay-rate",type =float ,default =0.75 )
    parser .add_argument ("--loss",choices =("nll","ce"),default ="nll")
    parser .add_argument ("--blocks",type =int ,default =3 )
    parser .add_argument ("--width",type =int ,default =64 )
    parser .add_argument ("--n-eig",type =int ,default =64 )
    parser .add_argument ("--dropout",type =float ,default =0.3 )
    parser .add_argument ("--tversky-weight",type =float ,default =0.0 )
    parser .add_argument ("--accum-steps",type =int ,default =1 )
    parser .add_argument (
    "--class-weights",
    choices =("thesis","uniform","inverse_frequency"),
    default ="thesis",
    help ="thesis uses the existing published five-class weight vector",
    )
    parser .add_argument ("--preflight-only",action ="store_true")
    parser .add_argument ("--force",action ="store_true",help ="replace an existing checkpoint/report")
    return parser .parse_args ()


def main ()->int :
    args =parse_args ()
    if args .epochs <1 or args .eval_every <1 or args .n_eig <2 :
        raise ValueError ("epochs/eval-every must be positive and n-eig must be >= 2")
    corpus =args .corpus .resolve ()

    # These are the only label loaders in this executable. There is intentionally
    # no allow_locked flag or test split argument here.
    train_samples =dataset .load_split (corpus ,"train",verify_hashes =True )
    val_samples =dataset .load_split (corpus ,"val",verify_hashes =True )
    if len (train_samples )!=71 or len (val_samples )!=20 :
        raise dataset .CorpusContractError (
        f"refusing split drift: train={len (train_samples )}, val={len (val_samples )}"
        )

    config ={
    "input_features":args .input_features ,
    "learning_rate":args .learning_rate ,
    "lr_decay_every":args .lr_decay_every ,
    "lr_decay_rate":args .lr_decay_rate ,
    "loss":args .loss ,
    "n_diffusion_blocks":args .blocks ,
    "c_width":args .width ,
    "n_eig":args .n_eig ,
    "dropout":args .dropout ,
    "tversky_weight":args .tversky_weight ,
    "accum_steps":args .accum_steps ,
    "strict_precompute":True ,
    }
    train_counts =dataset .class_counts (train_samples )
    val_counts =dataset .class_counts (val_samples )
    preflight ={
    "status":"preflight_verified",
    "corpus_manifest_sha256":dataset .manifest_sha256 (corpus ),
    "train":{"parts":len (train_samples ),"class_counts":train_counts },
    "val":{"parts":len (val_samples ),"class_counts":val_counts },
    "test_labels_opened_by_training":False ,
    "test_labels_used_for_training_or_model_selection":False ,
    "config":config ,
    }
    if args .preflight_only :
        print (json .dumps (preflight ,indent =2 ,ensure_ascii =False ))
        return 0 

    checkpoint =args .checkpoint .resolve ()
    report_path =args .run_report .resolve ()
    history_path =args .history_log .resolve ()
    if not args .force :
        existing =[str (path )for path in (checkpoint ,report_path ,history_path )if path .exists ()]
        if existing :
            raise FileExistsError (
            "refusing to overwrite frozen training artifacts; pass --force explicitly: "
            +", ".join (existing )
            )
    checkpoint .parent .mkdir (parents =True ,exist_ok =True )
    history_path .parent .mkdir (parents =True ,exist_ok =True )
    history_path .write_text ("",encoding ="utf-8")
    args .op_cache .resolve ().mkdir (parents =True ,exist_ok =True )

    if args .class_weights =="thesis":
        weights =None 
        recorded_weights =diffusionnet .LOSS_WEIGHTS_BY_ID 
    elif args .class_weights =="uniform":
        weights =[1.0 ]*len (CLASS_NAMES )
        recorded_weights =weights 
    else :
        weights =_inverse_frequency_weights (train_samples )
        recorded_weights =weights 

    _seed_everything (args .seed )
    device =_choose_device (args .device )
    history :list [dict ]=[]

    def report (metrics :dict )->None :
        history .append (metrics )
        with history_path .open ("a",encoding ="utf-8")as handle :
            handle .write (json .dumps (metrics ,ensure_ascii =False )+"\n")
            handle .flush ()
        print (
        f"epoch={metrics ['epoch']:03d} "
        f"val_accuracy={metrics ['accuracy']:.4f} "
        f"val_macro_f1={metrics ['mean_dice']:.4f} "
        f"val_mean_iou={metrics ['mean_iou']:.4f}",
        flush =True ,
        )

    logging .basicConfig (level =logging .INFO ,format ="%(asctime)s %(levelname)s %(message)s")
    best =diffusionnet .train_diffusionnet (
    config ,
    train_samples ,
    val_samples ,
    epochs =args .epochs ,
    device =device ,
    weights =weights ,
    report =report ,
    eval_every =args .eval_every ,
    checkpoint_path =str (checkpoint ),
    op_cache_dir =str (args .op_cache .resolve ()),
    )
    if not checkpoint .is_file ():
        raise RuntimeError ("training returned without writing the best checkpoint")

    import torch 

    receipt ={
    **preflight ,
    "status":"training_complete_model_selected_on_validation_only",
    "seed":args .seed ,
    "device":device ,
    "epochs":args .epochs ,
    "eval_every":args .eval_every ,
    "class_weight_mode":args .class_weights ,
    "class_weights_by_id":recorded_weights ,
    "best_validation":best ,
    "history":history ,
    "history_log":str (history_path ),
    "history_log_sha256":_sha256 (history_path ),
    "checkpoint":str (checkpoint ),
    "checkpoint_sha256":_sha256 (checkpoint ),
    "splits":{
    "train":dataset .split_receipt (corpus ,"train"),
    "val":dataset .split_receipt (corpus ,"val"),
    },
    "code_sha256":{
    name :_sha256 (HERE /name )
    for name in ("diffusionnet.py","scheffler_dataset.py",Path (__file__ ).name )
    },
    "environment":{
    "python":platform .python_version (),
    "torch":torch .__version__ ,
    "cuda_available":torch .cuda .is_available (),
    "cuda_device":torch .cuda .get_device_name (0 )if torch .cuda .is_available ()else None ,
    },
    "test_labels_opened_by_training":False ,
    "test_labels_used_for_training_or_model_selection":False ,
    }
    _write_json (report_path ,receipt )
    print (f"checkpoint={checkpoint }")
    print (f"train_report={report_path }")
    return 0 


if __name__ =="__main__":
    raise SystemExit (main ())
