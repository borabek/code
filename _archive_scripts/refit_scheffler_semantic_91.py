"""Refit a frozen semantic configuration ten all 91 development parts.

This stage is allowed only after ``train_scheffler_semantic.py`` selected the
configuration and epoch ten 71 train / 20 validation.  It performs no validation
and opens no locked-test label; the last fixed epoch is the final checkpoint.
"""

from __future__ import annotations 

import argparse 
import hashlib 
import json 
import logging 
from pathlib import Path 
import platform 
import sys 


HERE =Path (__file__ ).resolve ().parent 
VENDORED_DIFFUSION_NET =HERE /"_diffusion_net_repo"/"src"
if VENDORED_DIFFUSION_NET .is_dir ():
    sys .path .insert (0 ,str (VENDORED_DIFFUSION_NET ))

import diffusionnet # noqa: E402
import scheffler_dataset as dataset # noqa: E402
from train_scheffler_semantic import _choose_device ,_seed_everything # noqa: E402


def _sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b""):
            digest .update (block )
    return digest .hexdigest ()


def _write_json (path :Path ,payload :dict )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    temporary =path .with_suffix (path .suffix +".tmp")
    temporary .write_text (json .dumps (payload ,indent =2 ,ensure_ascii =False )+"\n",encoding ="utf-8")
    temporary .replace (path )


def parse_args ()->argparse .Namespace :
    base =HERE /"results"/"scheffler_semantic"
    parser =argparse .ArgumentParser (
    description =__doc__ ,formatter_class =argparse .ArgumentDefaultsHelpFormatter 
    )
    parser .add_argument ("--corpus",type =Path ,default =HERE /"wscad_corpus_scheffler_exact")
    parser .add_argument ("--selection-report",type =Path ,default =base /"train_report.json")
    parser .add_argument ("--checkpoint",type =Path ,default =base /"refit91.pt")
    parser .add_argument ("--refit-report",type =Path ,default =base /"refit91_report.json")
    parser .add_argument ("--op-cache",type =Path ,default =base /"operators")
    parser .add_argument ("--device",choices =("auto","cpu","cuda"),default ="auto")
    parser .add_argument ("--preflight-only",action ="store_true")
    parser .add_argument ("--force",action ="store_true")
    return parser .parse_args ()


def main ()->int :
    args =parse_args ()
    corpus =args .corpus .resolve ()
    selection_path =args .selection_report .resolve ()
    if not selection_path .is_file ():
        raise FileNotFoundError (f"missing validation-selection receipt: {selection_path }")
    selection =json .loads (selection_path .read_text (encoding ="utf-8"))
    if selection .get ("status")!="training_complete_model_selected_on_validation_only":
        raise RuntimeError ("selection receipt is not a completed 71/20 validation run")
    if selection .get ("test_labels_used_for_training_or_model_selection")is not False :
        raise RuntimeError ("selection receipt does not prove test isolation")
    if selection .get ("corpus_manifest_sha256")!=dataset .manifest_sha256 (corpus ):
        raise RuntimeError ("corpus manifest changed after validation selection")

    selection_checkpoint =Path (selection ["checkpoint"]).resolve ()
    if not selection_checkpoint .is_file ():
        raise FileNotFoundError (f"missing selection checkpoint: {selection_checkpoint }")
    if _sha256 (selection_checkpoint )!=selection .get ("checkpoint_sha256"):
        raise RuntimeError ("selection checkpoint SHA-256 drift")
    for name in ("diffusionnet.py","scheffler_dataset.py"):
        expected =selection .get ("code_sha256",{}).get (name )
        if expected !=_sha256 (HERE /name ):
            raise RuntimeError (
            f"{name } changed after validation selection; rerun selection before final refit"
            )

    best =selection .get ("best_validation",{})
    if "epoch"not in best :
        raise RuntimeError ("selection receipt has no best validation epoch")
    fixed_epochs =int (best ["epoch"])+1 
    if fixed_epochs <1 :
        raise RuntimeError (f"invalid frozen epoch count: {fixed_epochs }")
    config =dict (selection ["config"])
    config ["strict_precompute"]=True 

    train_samples =dataset .load_split (corpus ,"train",verify_hashes =True )
    val_samples =dataset .load_split (corpus ,"val",verify_hashes =True )
    development =train_samples +val_samples 
    if len (development )!=91 :
        raise dataset .CorpusContractError (f"final refit requires exactly 91 parts, got {len (development )}")
    ids =[sample ["part_id"]for sample in development ]
    if len (ids )!=len (set (ids )):
        raise dataset .CorpusContractError ("duplicate part IDs in 91-part final refit")

    preflight ={
    "status":"ready_for_fixed_91_refit",
    "corpus_manifest_sha256":dataset .manifest_sha256 (corpus ),
    "selection_report":str (selection_path ),
    "selection_report_sha256":_sha256 (selection_path ),
    "selection_checkpoint_sha256":selection ["checkpoint_sha256"],
    "frozen_best_epoch_zero_based":int (best ["epoch"]),
    "fixed_refit_epochs":fixed_epochs ,
    "development_parts":len (development ),
    "config":config ,
    "test_labels_opened_by_refit":False ,
    "test_labels_used_for_training_or_model_selection":False ,
    }
    if args .preflight_only :
        print (json .dumps (preflight ,indent =2 ,ensure_ascii =False ))
        return 0 

    checkpoint =args .checkpoint .resolve ()
    report_path =args .refit_report .resolve ()
    if not args .force :
        existing =[str (path )for path in (checkpoint ,report_path )if path .exists ()]
        if existing :
            raise FileExistsError ("refusing to overwrite final-refit artifacts: "+", ".join (existing ))
    checkpoint .parent .mkdir (parents =True ,exist_ok =True )
    args .op_cache .resolve ().mkdir (parents =True ,exist_ok =True )

    weight_mode =selection ["class_weight_mode"]
    weights =None if weight_mode =="thesis"else selection ["class_weights_by_id"]
    seed =int (selection ["seed"])
    _seed_everything (seed )
    device =_choose_device (args .device )
    history =[]

    def report (row :dict )->None :
        history .append (row )
        print (
        f"fixed_refit_epoch={row ['epoch']+1 :03d}/{fixed_epochs :03d} "
        f"train_loss={row ['train_loss']:.6f}",
        flush =True ,
        )

    logging .basicConfig (level =logging .INFO ,format ="%(asctime)s %(levelname)s %(message)s")
    refit_result =diffusionnet .refit_diffusionnet (
    config ,
    development ,
    epochs =fixed_epochs ,
    device =device ,
    weights =weights ,
    report =report ,
    checkpoint_path =str (checkpoint ),
    op_cache_dir =str (args .op_cache .resolve ()),
    )
    if not checkpoint .is_file ():
        raise RuntimeError ("fixed refit returned without a checkpoint")

    import torch 

    receipt ={
    **preflight ,
    "status":"final_refit_91_fixed_epoch_no_model_selection",
    "seed":seed ,
    "device":device ,
    "class_weight_mode":weight_mode ,
    "class_weights_by_id":selection ["class_weights_by_id"],
    "refit_result":refit_result ,
    "train_loss_history_not_used_for_model_selection":history ,
    "checkpoint":str (checkpoint ),
    "checkpoint_sha256":_sha256 (checkpoint ),
    "splits_combined_only_after_freeze":{
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
    },
    "test_labels_opened_by_refit":False ,
    "test_labels_used_for_training_or_model_selection":False ,
    }
    _write_json (report_path ,receipt )
    print (f"refit91_checkpoint={checkpoint }")
    print (f"refit91_report={report_path }")
    return 0 


if __name__ =="__main__":
    raise SystemExit (main ())

