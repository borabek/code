"""One-shot semantic evaluation ten the locked 11-part, three-rater benchmark."""

from __future__ import annotations 

import argparse 
from datetime import datetime ,timezone 
import hashlib 
import json 
from pathlib import Path 
import platform 
import sys 

import numpy as np 


HERE =Path (__file__ ).resolve ().parent 
VENDORED_DIFFUSION_NET =HERE /"_diffusion_net_repo"/"src"
if VENDORED_DIFFUSION_NET .is_dir ():
    sys .path .insert (0 ,str (VENDORED_DIFFUSION_NET ))

import diffusionnet # noqa: E402
import metrics # noqa: E402
import scheffler_dataset as dataset # noqa: E402


CONFIRMATION ="OPEN_LOCKED_11_ONCE"


def _sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b""):
            digest .update (block )
    return digest .hexdigest ()


def _write_json_exclusive (path :Path ,payload :dict )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    with path .open ("x",encoding ="utf-8")as handle :
        json .dump (payload ,handle ,indent =2 ,ensure_ascii =False )
        handle .write ("\n")


def _write_json_replace (path :Path ,payload :dict )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    temporary =path .with_suffix (path .suffix +".tmp")
    temporary .write_text (json .dumps (payload ,indent =2 ,ensure_ascii =False )+"\n",encoding ="utf-8")
    temporary .replace (path )


def _macro_part_summary (reports :list [dict ])->dict :
    scalar_keys =("accuracy","mean_dice","mean_iou","weighted_iou")
    summary ={key :float (np .mean ([report [key ]for report in reports ]))for key in scalar_keys }
    for metric_key in ("dice_per_class","iou_per_class"):
        names =reports [0 ][metric_key ]
        summary [metric_key ]={}
        for name in names :
            values =[report [metric_key ][name ]for report in reports if report [metric_key ][name ]is not None ]
            summary [metric_key ][name ]=float (np .mean (values ))if values else None 
    return summary 


def _minimal_report (report :dict )->dict :
    return {
    "accuracy":report ["accuracy"],
    "mean_dice":report ["mean_dice"],
    "mean_iou":report ["mean_iou"],
    "weighted_iou":report ["weighted_iou"],
    "dice_per_class":report ["dice_per_class"],
    "iou_per_class":report ["iou_per_class"],
    "confusion":report ["confusion"],
    }


def _choose_device (requested :str )->str :
    import torch 

    if requested =="auto":
        return "cuda"if torch .cuda .is_available ()else "cpu"
    if requested =="cuda"and not torch .cuda .is_available ():
        raise RuntimeError ("--device cuda requested but CUDA is unavailable")
    return requested 


def parse_args ()->argparse .Namespace :
    base =HERE /"results"/"scheffler_semantic"
    parser =argparse .ArgumentParser (
    description =__doc__ ,formatter_class =argparse .ArgumentDefaultsHelpFormatter 
    )
    parser .add_argument ("--corpus",type =Path ,default =HERE /"wscad_corpus_scheffler_exact")
    parser .add_argument ("--checkpoint",type =Path ,default =base /"refit91.pt")
    parser .add_argument (
    "--model-receipt","--train-report",dest ="train_report",type =Path ,
    default =base /"refit91_report.json",
    help ="frozen validation-selection or fixed-91-refit receipt",
    )
    parser .add_argument ("--output",type =Path ,default =base /"locked_test_result.json")
    parser .add_argument ("--opened-marker",type =Path ,default =base /"LOCKED_TEST_OPENED.json")
    parser .add_argument ("--op-cache",type =Path ,default =base /"test_operators")
    parser .add_argument ("--device",choices =("auto","cpu","cuda"),default ="auto")
    parser .add_argument (
    "--confirm-final-test",
    metavar =CONFIRMATION ,
    help =f"required exact phrase: {CONFIRMATION }",
    )
    parser .add_argument (
    "--preflight-only",
    action ="store_true",
    help ="verify the frozen checkpoint/receipt without opening test labels",
    )
    return parser .parse_args ()


def main ()->int :
    args =parse_args ()
    corpus =args .corpus .resolve ()
    checkpoint =args .checkpoint .resolve ()
    train_report_path =args .train_report .resolve ()
    output =args .output .resolve ()
    marker =args .opened_marker .resolve ()

    contract =dataset .validate_manifest_contract (corpus )
    if not checkpoint .is_file ()or not train_report_path .is_file ():
        raise FileNotFoundError ("frozen checkpoint and train_report.json are both required")
    train_report =json .loads (train_report_path .read_text (encoding ="utf-8"))
    checkpoint_hash =_sha256 (checkpoint )
    receipt_status =train_report .get ("status")
    allowed_statuses ={
    "training_complete_model_selected_on_validation_only",
    "final_refit_91_fixed_epoch_no_model_selection",
    }
    if receipt_status not in allowed_statuses :
        raise RuntimeError ("model receipt does not prove a frozen leakage-safe model")
    if train_report .get ("checkpoint_sha256")!=checkpoint_hash :
        raise RuntimeError ("checkpoint SHA-256 does not match the frozen training receipt")
    if train_report .get ("corpus_manifest_sha256")!=dataset .manifest_sha256 (corpus ):
        raise RuntimeError ("corpus manifest changed after training")
    if receipt_status =="training_complete_model_selected_on_validation_only":
        if train_report .get ("test_labels_opened_by_training")is not False :
            raise RuntimeError ("selection receipt does not prove that training kept test labels closed")
    else :
        if train_report .get ("test_labels_opened_by_refit")is not False :
            raise RuntimeError ("refit receipt does not prove that refit kept test labels closed")
        if train_report .get ("development_parts")!=91 :
            raise RuntimeError ("final-refit receipt does not prove use of all 91 development parts")
        if train_report .get ("refit_result",{}).get ("validation_or_test_metrics_computed")is not False :
            raise RuntimeError ("final-refit receipt indicates forbidden model selection metrics")
    if train_report .get ("test_labels_used_for_training_or_model_selection")is not False :
        raise RuntimeError ("training receipt does not prove a leakage-free model selection")

    preflight ={
    "status":"ready_but_locked",
    "checkpoint":str (checkpoint ),
    "checkpoint_sha256":checkpoint_hash ,
    "model_receipt":str (train_report_path ),
    "model_receipt_status":receipt_status ,
    "corpus_manifest_sha256":dataset .manifest_sha256 (corpus ),
    "locked_test_parts":contract ["counts"]["test_locked"],
    "test_labels_opened_by_evaluator":False ,
    "test_labels_used_for_training_or_model_selection":False ,
    }
    if args .preflight_only :
        print (json .dumps (preflight ,indent =2 ,ensure_ascii =False ))
        return 0 

    if args .confirm_final_test !=CONFIRMATION :
        raise PermissionError (
        f"locked test not opened; pass --confirm-final-test {CONFIRMATION } "
        "only after config, preprocessing, and epoch are frozen"
        )
    if marker .exists ()or output .exists ():
        raise FileExistsError (
        "the locked benchmark has already been opened or evaluated; "
        f"marker={marker .exists ()}, output={output .exists ()}"
        )

    opened_at =datetime .now (timezone .utc ).isoformat ()
    _write_json_exclusive (
    marker ,
    {
    **preflight ,
    "status":"locked_test_open_attempt_started",
    "opened_at_utc":opened_at ,
    "confirmation":CONFIRMATION ,
    "test_labels_opened_by_evaluator":True ,
    },
    )

    # No test mesh or label is loaded before the irreversible marker above.
    samples =dataset .load_split (
    corpus ,"test_locked",allow_locked =True ,verify_hashes =True 
    )
    if len (samples )!=11 :
        raise dataset .CorpusContractError (f"locked test must contain 11 parts, got {len (samples )}")

    device =_choose_device (args .device )
    model ,meta ,checkpoint_config =diffusionnet .load_checkpoint (str (checkpoint ),device =device )
    args .op_cache .resolve ().mkdir (parents =True ,exist_ok =True )
    all_pred :list [np .ndarray ]=[]
    all_true :list [np .ndarray ]=[]
    per_part :list [dict ]=[]
    rater_pooled_pred =[[],[],[]]
    rater_pooled_true =[[],[],[]]

    for index ,sample in enumerate (samples ,1 ):
        pred =diffusionnet .predict (
        model ,
        meta ,
        sample ["verts"],
        sample ["faces"],
        device =device ,
        op_cache_dir =str (args .op_cache .resolve ()),
        )
        truth =sample ["labels"]
        if len (pred )!=len (truth ):
            raise RuntimeError (f"{sample ['part_id']}: prediction/label length mismatch")
        majority_report =metrics .segmentation_report (pred ,truth ,n_classes =5 )
        rater_reports =[]
        for rater_index ,rater_truth in enumerate (
        dataset .load_rater_labels (corpus ,sample ["part_id"],allow_locked =True )
        ):
            rater_report =metrics .segmentation_report (pred ,rater_truth ,n_classes =5 )
            rater_reports .append (_minimal_report (rater_report ))
            rater_pooled_pred [rater_index ].append (pred )
            rater_pooled_true [rater_index ].append (rater_truth )
        per_part .append (
        {
        "part_id":sample ["part_id"],
        "human_region_cp_status":sample ["human_region_cp_status"],
        "majority_vote":_minimal_report (majority_report ),
        "individual_raters":rater_reports ,
        }
        )
        all_pred .append (pred )
        all_true .append (truth )
        print (f"evaluated {index }/11: {sample ['part_id']}",flush =True )

    pooled =metrics .segmentation_report (
    np .concatenate (all_pred ),np .concatenate (all_true ),n_classes =5 
    )
    per_part_majority =[entry ["majority_vote"]for entry in per_part ]
    rater_pooled =[
    _minimal_report (
    metrics .segmentation_report (
    np .concatenate (rater_pooled_pred [index ]),
    np .concatenate (rater_pooled_true [index ]),
    n_classes =5 ,
    )
    )
    for index in range (3 )
    ]
    import torch 

    result ={
    "status":"locked_final_test_complete",
    "opened_at_utc":opened_at ,
    "completed_at_utc":datetime .now (timezone .utc ).isoformat (),
    "checkpoint_sha256":checkpoint_hash ,
    "corpus_manifest_sha256":dataset .manifest_sha256 (corpus ),
    "checkpoint_config":checkpoint_config ,
    "benchmark":{
    "parts":11 ,
    "primary_ground_truth":"published_three_rater_majority_vote",
    "semantic_region_metrics_only":True ,
    "pooled_vertices":_minimal_report (pooled ),
    "macro_over_parts":_macro_part_summary (per_part_majority ),
    "prediction_vs_each_rater_pooled":rater_pooled ,
    "per_part":per_part ,
    },
    "cp_bridge_scope":{
    "label":"human-region-derived, not human-clicked CP",
    "cable_entry_candidate_parts":9 ,
    "review_required_parts":2 ,
    "automatic_candidate_coverage":9 /11 ,
    "cp_f1_not_reported_by_this_semantic_evaluator":True ,
    },
    "environment":{
    "python":platform .python_version (),
    "torch":torch .__version__ ,
    "device":device ,
    },
    }
    _write_json_replace (output ,result )
    _write_json_replace (
    marker ,
    {
    **preflight ,
    "status":"locked_final_test_complete",
    "opened_at_utc":opened_at ,
    "completed_at_utc":result ["completed_at_utc"],
    "result":str (output ),
    "result_sha256":_sha256 (output ),
    "test_labels_opened_by_evaluator":True ,
    },
    )
    print ("LOCKED FINAL TEST COMPLETE")
    metrics .print_report (pooled )
    print (f"result={output }")
    return 0 


if __name__ =="__main__":
    raise SystemExit (main ())
