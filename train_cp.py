# Training entry point for connection-point detection ten ABB JSON data (Option B).
#
# Ties together:
#   json_dataset  – stream parts from the corpus (1 GB single-array file OR a
#                   directory of per-part files)
#   cp_targets    – encode terminal-block locations -> per-vertex targets;
#                   decode predictions -> discrete connection points
#   cp_regressor  – the per-vertex 7-channel head + combined loss + train loop
#   metrics_cp    – localisation (mm) / angular (deg) / precision / recall
#
# Four backbones:
#   --backbone mlp           : light per-vertex MLP ten xyz (only needs torch).
#                              Memorises geometry -> good for overfit smoke-tests,
#                              but has no receptive field so it does NOT generalise.
#   --backbone diffusionnet  : (needs the diffusion_net pkg
#                              + robust_laplacian/potpourri3d -> Linux/WSL).
#   --backbone knngraph      : torch+scipy EdgeConv ten a kNN graph. Has a real
#                              geometric receptive field (generalises) but NO
#                              native geometry deps -> runs ten native Windows.
#   --backbone hierpoint     : hierarchical point U-Net (hierpoint.py) -- same
#                              torch+scipy-only footprint as knngraph, but with
#                              FPS downsampling levels + feature propagation:
#                              whole-patch receptive field at a fraction of the
#                              EdgeConv memory (T1200-sized; see hierpoint.py).
#
# Memory / 1 GB scale: parts are streamed; with --max-parts you cap how many are
# materialised. The eigenbasis for the diffusionnet backbone is the expensive
# step and should be cached per part (see diffusionnet.precompute_operators).

import os 
import sys 
import json 
import logging 
import argparse 

# Must be set BEFORE anything imports torch: proactively garbage-collect the
# CUDA caching allocator at 80% of its capped limit (the cap itself is set in
# cp_regressor._init_training) instead of growing the pool. On 4 GB WDDM cards
# pool growth silently spills into SYSTEM memory and the run crawls 10-100x.
os .environ .setdefault ("PYTORCH_CUDA_ALLOC_CONF","garbage_collection_threshold:0.8")

import numpy as np 

import json_dataset as jd 
import cp_targets as ct 
import cp_regressor as cpr 
import metrics as mcp 

logger =logging .getLogger (__name__ )


# ---------------------------------------------------------------------------
# split a corpus by PartNr (deterministic hash -> stable across runs)
# ---------------------------------------------------------------------------

def _bucket (key ,seed ):
    """Stable hash of a split key -> a float in [0,1)."""
    import hashlib 
    h =int (hashlib .sha256 ((str (seed )+":"+str (key )).encode ()).hexdigest (),16 )
    return (h %1000 )/1000.0 


def split_part_ids (ids ,val_frac =0.2 ,seed =0 ,group_keys =None ):
    """Deterministic train/val split (hash-based, stable).

    group_keys: optional parallel list of GROUP keys (see json_dataset.build_group_keys).
    When given, the bucket is decided per GROUP, so all parts sharing a key (a part
    family / near-duplicate geometry) land ten the SAME side -- this is what stops
    sibling variants leaking across train/val. Default (None) hashes each PartNr,
    the original per-part behaviour.
    """
    keys =group_keys if group_keys is not None else ids 
    val ,train =set (),[]
    for pid ,key in zip (ids ,keys ):
        if _bucket (key ,seed )<val_frac :
            val .add (pid )
        else :
            train .append (pid )
    return set (train ),val 


def three_way_split (ids ,val_frac =0.2 ,test_frac =0.0 ,seed =0 ,group_keys =None ,
balance_search =30 ,balance_tol =0.35 ,
family_keys =None ,family_balance_tol =0.5 ):
    """Train / val / TEST split from one stable hash space, group-aware.

    The held-out test set is carved from the same hash so it is frozen and never
    overlaps train/val: bucket in [0,val_frac) -> val, [val_frac,val_frac+test_frac)
    -> test, else train. test_frac=0 reproduces split_part_ids (test empty). The
    test split is for a FINAL, unbiased report only -- it must never feed training
    or best-checkpoint/decode selection.

    balance_search: grouping (--split-group prefix/geometry) can produce a few big
    atomic groups (measured: one geometry group is 7.9% of the real 479-part corpus
    -- many PartNrs that share literally identical mesh geometry, correctly kept
    together to prevent leakage). Because a whole group moves to one side together,
    a single hash seed can swing the ACHIEVED val/test fraction far from what was
    requested with no error at all -- measured 0.109-0.292 actual val for a
    requested 0.2 across seeds 0-4 ten the real corpus, making F1 across different
    seeds/runs not apples-to-apples. If `seed`'s achieved split deviates from the
    target by more than `balance_tol` (relative), deterministically try
    seed+1..seed+balance_search-1 and keep whichever is closest to the target --
    every candidate still keeps groups fully intact (this searches for a better
    BALANCED assignment of whole groups, it does not weaken the leakage guarantee).
    Sticks with `seed` whenever already close enough, so ungrouped (group_keys=None)
    or naturally fine-grained splits are unaffected. Fully deterministic given the
    same inputs, so a --resume reconstructs the identical split. Pass
    balance_search=0 to disable and use `seed` exactly (old behaviour).

    family_keys: optional parallel list (e.g. json_dataset.prefix_family per id) used
    ONLY to check per-family train-fraction skew -- NOT for the split assignment
    itself (that stays governed by group_keys). The AGGREGATE balance_search above
    can pick a seed where the WHOLE-CORPUS val/test fraction looks fine while a
    SPECIFIC small family is starved: measured ten cp_knn_v17 (real corpus, geometry
    grouping), the aggregate-balanced seed still left SIE with only 48% of its parts
    in train, PXC 53%, A-B 50%, RIT 25% (vs a nominal ~65% train fraction) -- ABB and
    wscaduniverse were fine (79%/62%) so the aggregate check never flagged it. That
    starvation, compounding a separate patch/non-patch gradient-share bug, correlated
    almost exactly with which families the resulting model never learned to detect
    at all. If any family's train fraction deviates from the nominal (1-val_frac-
    test_frac) by more than family_balance_tol (relative), this joins the same
    seed search above (a seed must satisfy BOTH the aggregate AND the per-family
    bound to be accepted) -- still never splits a group, only searches for a
    better-balanced whole-group assignment. Families with very few parts (<4) are
    exempt (too few to have a meaningful "fraction" -- a single coin-flip group can't
    be fixed by search without literally moving individual parts, which would break
    grouping).
    """
    def _split_for (sd ):
        keys =group_keys if group_keys is not None else ids 
        train ,val ,test =[],set (),set ()
        for pid ,key in zip (ids ,keys ):
            b =_bucket (key ,sd )
            if b <val_frac :
                val .add (pid )
            elif b <val_frac +test_frac :
                test .add (pid )
            else :
                train .append (pid )
        return set (train ),val ,test 

    fam_of =dict (zip (ids ,family_keys ))if family_keys is not None else None 
    fam_counts ={}
    if fam_of is not None :
        for fam in fam_of .values ():
            fam_counts [fam ]=fam_counts .get (fam ,0 )+1 
    nominal_train_frac =max (1e-6 ,1.0 -val_frac -test_frac )

    def _family_skew (train_ids_ ):
        """Worst-family relative deviation of train-fraction from nominal (0 if no
        family_keys, or every eligible family is within tolerance)."""
        if fam_of is None :
            return 0.0 
        fam_train ={}
        for pid in train_ids_ :
            fam =fam_of .get (pid )
            if fam is not None :
                fam_train [fam ]=fam_train .get (fam ,0 )+1 
        worst =0.0 
        for fam ,cnt in fam_counts .items ():
            if cnt <4 :# too few parts for "fraction" to mean anything
                continue 
            achieved =fam_train .get (fam ,0 )/cnt 
            worst =max (worst ,abs (achieved -nominal_train_frac )/nominal_train_frac )
        return worst 

    train_ids ,val_ids ,test_ids =_split_for (seed )
    if group_keys is not None and balance_search and val_frac >0 :
        n =max (1 ,len (ids ))

        def _skew_ratios (tr ,v ,t ):
            """(dv, dt, df) each as a FRACTION OF ITS OWN TOLERANCE (>1 = violates
            that specific bound). Using each component's own tolerance -- instead of
            a single max(balance_tol, family_balance_tol) scalar -- matters: since
            family_balance_tol (0.5) is looser than balance_tol (0.35) by design (a
            single-family outlier is tolerated a bit more than the aggregate), a
            naive combined threshold of max(balance_tol, family_balance_tol)=0.5
            would silently ALSO loosen the AGGREGATE check to 0.5 -- verified this
            regression directly: with the naive combined check, a seed achieving
            val=14.7% test=8.4% (aggregate test skew 0.44, clearly over the intended
            0.35 aggregate tolerance) was NOT corrected, reproducing exactly the
            '--val-frac 0.2 request -> ~11-15% actual' failure mode the aggregate-
            only balance search (an earlier fix) was built to prevent."""
            dv =(abs (len (v )/n -val_frac )/val_frac )/balance_tol 
            dt =((abs (len (t )/n -test_frac )/test_frac )/balance_tol 
            if test_frac >0 else 0.0 )
            df =_family_skew (tr )/family_balance_tol 
            return dv ,dt ,df 

        def _violates (ratios ):
            return max (ratios )>1.0 

        cur_ratios =_skew_ratios (train_ids ,val_ids ,test_ids )
        best_seed ,best_ratios =seed ,cur_ratios 
        if _violates (cur_ratios ):
            best =(train_ids ,val_ids ,test_ids )
            for k in range (1 ,balance_search ):
                cand =_split_for (seed +k )
                c_train ,c_val ,c_test =cand 
                if not c_val or (test_frac >0 and not c_test ):
                    continue 
                r =_skew_ratios (c_train ,c_val ,c_test )
                if max (r )<max (best_ratios ):
                    best_ratios ,best_seed ,best =r ,seed +k ,cand 
                    if not _violates (best_ratios ):
                        break 
            if best_seed !=seed :
                train_ids ,val_ids ,test_ids =best 
                n_ =max (1 ,len (train_ids )+len (val_ids )+len (test_ids ))
                logger .warning (
                "three_way_split: seed=%d gave a skewed grouped split (val "
                "target %.2f, or a family train-fraction outlier) -- auto-picked "
                "seed=%d instead (val=%.3f test=%.3f vs requested val=%.2f "
                "test=%.2f, worst-family skew %.2f); pass balance_search=0 to "
                "disable this and use seed=%d exactly",
                seed ,val_frac ,best_seed ,len (val_ids )/n_ ,
                len (test_ids )/n_ ,val_frac ,test_frac ,
                _family_skew (train_ids )*family_balance_tol ,seed )
    return train_ids ,val_ids ,test_ids 


    # ---------------------------------------------------------------------------
    # evaluate a trained model (any backbone) ten prepared samples
    # ---------------------------------------------------------------------------

def _nms_radius (s ,clearance_mm =5.0 ):
    """Decode merge radius CEILING = the fixed tool clearance, DECOUPLED from sigma.

    Previously this was max(2*sigma, clearance). sigma is ~2% of the bbox diagonal,
    which ten an elongated part is tens of mm, so 2*sigma ballooned the merge radius
    ABOVE the spacing between distinct connection points -- decode then collapsed two
    real CPs into one (a guaranteed recall cap, independent of model quality, and
    worse at inference where sigma is the uncapped bbox value). The votes for a
    single CP already cluster tightly because each voting vertex predicts the offset
    TO that CP, so the merge radius only needs to cover offset error + tool
    clearance: a fixed clearance (tunable via --nms-clearance-mm / sweep_decode) is
    the correct, GT-free, train==inference ceiling. `s` is kept for signature compat.

    This value is a CEILING, not the radius actually used: cp_targets.decode_predictions
    further caps it to a fraction of each part's own bbox diagonal (nms_scale_frac),
    so a small part with genuinely close-together CPs isn't forced to merge them just
    because the fixed clearance constant happens to exceed their real spacing.
    """
    return float (clearance_mm )


def _print_part_metrics (rep ):
    """One line of per-part keypoint metrics: accuracy / precision / recall / F1
    plus the TP/FP/FN counts and mean loc/ang error. loc/ang show '-' when the
    part had no matched detection (their mean is NaN)."""
    loc =rep ["mean_loc_err_mm"];ang =rep ["mean_ang_err_deg"]
    loc_s ="   -   "if (isinstance (loc ,float )and np .isnan (loc ))else "%5.2fmm"%loc 
    ang_s ="   -   "if (isinstance (ang ,float )and np .isnan (ang ))else "%5.1fdeg"%ang 
    print ("  %-26s acc=%.2f  prec=%.2f  rec=%.2f  F1=%.2f   "
    "TP=%d FP=%d FN=%d   loc=%s ang=%s"
    %(rep ["part_nr"],rep ["accuracy"],rep ["precision"],rep ["recall"],
    rep ["f1"],rep ["tp"],rep ["fp"],rep ["fn"],loc_s ,ang_s ))


def evaluate (samples ,parts_meta ,infer_arr ,dist_thresh_mm =5.0 ,
heatmap_thresh =0.3 ,min_votes =1 ,nms_clearance_mm =5.0 ,progress =True ,
print_points =False ):
    """Decode predictions for each sample and aggregate keypoint metrics.

    infer_arr(sample, part_meta) -> (N,7) numpy array of per-vertex predictions
    (heatmap already sigmoid'd, direction unit). This abstracts the backbone so
    the same decode+metrics path serves both the MLP and DiffusionNet models.
    min_votes: a decoded cluster needs at least this many voting vertices (#6:
    >1 suppresses lone-vertex false positives -> higher precision).

    progress: log a heartbeat every ~20% of parts (only when >=20 parts) so a
    slow eval -- a single 470k-vertex part is ~30s of CPU inference -- is visibly
    making progress instead of looking frozen.
    """
    import time 
    reports =[]
    n =len (samples )
    hb =max (1 ,n //5 )
    t0 =time .time ()
    for j ,(s ,pm )in enumerate (zip (samples ,parts_meta ),1 ):
        arr =infer_arr (s ,pm )
        preds =ct .decode_predictions (pm ["vertices"],arr ,
        heatmap_thresh =heatmap_thresh ,
        nms_radius_mm =_nms_radius (s ,nms_clearance_mm ),
        min_votes =min_votes )
        rep =mcp .keypoint_report (preds ,s ["gt_points"],s ["gt_directions"],
        dist_thresh_mm =dist_thresh_mm )
        rep ["part_nr"]=pm ["part_nr"]
        reports .append (rep )
        if print_points :
            _print_part_metrics (rep )
        if progress and n >=20 and (j %hb ==0 or j ==n ):
            logger .info ("    eval %d/%d parts (%.0fs)",j ,n ,time .time ()-t0 )
    return reports 


def aggregate (reports ):
    """Per-part means (macro) + pooled detection counts (micro) in one dict.

    macro keys (accuracy/precision/recall/f1/loc/ang) average each part's
    score; micro_* pool TP/FP/FN over all parts first, which weights every
    ground-truth point equally and is what 'overall corpus accuracy' means.
    """
    if not reports :
        return {}
    keys =["accuracy","precision","recall","f1",
    "mean_loc_err_mm","mean_ang_err_deg"]
    agg ={}
    for k in keys :
        vals =[r [k ]for r in reports if not (isinstance (r [k ],float )and np .isnan (r [k ]))]
        agg [k ]=float (np .mean (vals ))if vals else float ("nan")
    agg ["n_parts"]=len (reports )
    agg ["total_gt"]=sum (r ["n_gt"]for r in reports )
    agg ["total_tp"]=sum (r ["tp"]for r in reports )
    agg ["total_fp"]=sum (r ["fp"]for r in reports )
    agg ["total_fn"]=sum (r ["fn"]for r in reports )
    tp ,fp ,fn =agg ["total_tp"],agg ["total_fp"],agg ["total_fn"]
    agg ["micro_precision"]=tp /(tp +fp )if (tp +fp )else 0.0 
    agg ["micro_recall"]=tp /(tp +fn )if (tp +fn )else 0.0 
    pr =agg ["micro_precision"]+agg ["micro_recall"]
    agg ["micro_f1"]=(2 *agg ["micro_precision"]*agg ["micro_recall"]/pr 
    if pr else 0.0 )
    agg ["micro_accuracy"]=tp /(tp +fp +fn )if (tp +fp +fn )else 1.0 
    return agg 


def _val_metrics_fn (val_s ,val_m ,infer_builder ,dist_thresh_mm ,min_votes ,
heatmap_thresh =0.3 ,nms_clearance_mm =5.0 ):
    """Build metrics_fn(model, meta) for the training bookkeeper: decode and
    score the validation parts, returning the aggregate keypoint metrics
    (accuracy / precision / recall / f1 / loc / ang). None when no val set --
    the loop then skips val evaluation and best-by-F1 tracking.

    heatmap_thresh / nms_clearance_mm are the decode knobs that govern the
    precision/recall trade-off; they must match the values used for the final
    train/val evaluate() so the best-by-F1 checkpoint is selected under the same
    decoding the report uses."""
    if not val_s :
        return None 

    def metrics_fn (m ,mt ):
        return aggregate (evaluate (val_s ,val_m ,infer_builder (m ,mt ),
        dist_thresh_mm =dist_thresh_mm ,
        heatmap_thresh =heatmap_thresh ,
        min_votes =min_votes ,
        nms_clearance_mm =nms_clearance_mm ))
    return metrics_fn 


    # ---------------------------------------------------------------------------
    # main train+eval
    # ---------------------------------------------------------------------------

def train_and_eval (source ,backbone ="mlp",epochs =300 ,val_frac =0.2 ,
extra_sources =None ,max_bbox_mm =None ,keep_prefixes =None ,
max_parts =None ,device ="cpu",dist_thresh_mm =5.0 ,seed =0 ,
op_cache_dir =None ,max_gpu_verts =60000 ,
best_path =None ,last_path =None ,history_path =None ,
save_every =1 ,resume_from =None ,
min_votes =1 ,heat_loss ="centernet",
focal_gamma =2.0 ,centernet_alpha =2.0 ,lr_decay_every =0 ,lr_decay_rate =0.5 ,
accum_steps =1 ,low_memory =False ,eval_every =10 ,patience =0 ,
heatmap_thresh =0.3 ,nms_clearance_mm =5.0 ,heat_pos_weight =50.0 ,
augment =0 ,aug_rotate =True ,aug_jitter_frac =0.0 ,aug_reflect =False ,
aug_dropout_frac =0.0 ,aug_cable_frac =0.0 ,aug_cube_rotate =False ,
split_group ="none",test_frac =0.0 ,cp_surface_tol_frac =None ,
neg_frac =0.0 ,max_neg =None ,
best_metric ="micro_f1",resume_strict =False ,snapshot_every =0 ,
w_heat =1.0 ,w_off =5.0 ,w_dir =1.0 ,weight_decay =0.0 ,
lr =1e-3 ,lr_schedule ="step",warmup_epochs =0 ,grad_clip =0.0 ,
offset_heat_weight =True ,part_weight_mode ="none",
knn_config =None ,dn_config =None ,hp_config =None ,
print_points =False ,amp =True ,
knn_cache_dir =None ,dir_sign_invariant =False ,init_from =None ,
hard_mining =False ,offset_loss ="l1",offset_huber_beta =0.01 ,
min_delta =0.0 ,collapse_gap =0.03 ,fwd_verts =0 ,
pin_host_memory =None ,sign_inv_prefixes ="",split_seed =None ,
family_boost ="",exclude_parts_file =None ):
    """Load parts, prepare targets, train, and evaluate.

    Returns (train_agg, val_agg, train_reports, val_reports, test_agg, test_reports).

    best_path / last_path : full-state .ckpt files (best-by-val-F1, rolling
    resumable snapshot). resume_from continues a previous run from its saved
    epoch toward `epochs` total -- including ten another machine after a git
    pull. history_path collects the per-eval val metrics as JSON.

    split_group : 'none' (split each PartNr independently), 'prefix' (group
    part-family variants by stripped PartNr) or 'geometry' (group near-identical
    meshes) -- grouping stops sibling variants leaking across train/val.
    test_frac : carve a frozen held-out TEST split (never trains, never tunes) for
    one unbiased final number. cp_surface_tol_frac : drop CPs farther than this
    fraction of the bbox diagonal from the mesh (off-surface mislabels).
    neg_frac / max_neg : include zero-CP parts (valid negatives) in the FIT set so
    the model learns to stay silent where there is nothing -- only those ten the
    train side are used, never val/test.

    augment : number of augmented clones to add per TRAINING part (0 = off).
    Each clone is a random rigid rotation (+ optional reflection / vertex jitter)
    of the part, with CP points/directions transformed consistently. The clones
    enter the FIT set only; the train/val reports use the original parts, so val
    metrics stay deployment-real. The main generalisation lever for the raw-xyz
    backbones (mlp/knngraph), which are otherwise pose-overfit.
    """
    parts =[]
    for part in jd .iter_parts (source ):
        parts .append (part )
        if max_parts and len (parts )>=max_parts :
            break 
    for extra in (extra_sources or []):
        for part in jd .iter_parts (extra ):
            parts .append (part )
            if max_parts and len (parts )>=max_parts :
                break 
    if not parts :
        raise SystemExit (f"no parts found in {source }")
    logger .info ("loaded %d parts (%s)",len (parts ),
    (source if not extra_sources else 
    f"{source } + {len (extra_sources )} extra source(s)"))

    # Scope filter: keep only parts whose PartNr starts with one of keep_prefixes.
    # Use "wscaduniverse,PXC" to train a TERMINAL-BLOCK specialist (the deliverable
    # target) and exclude contactors/drives/breakers, whose different CP semantics
    # (raised screw terminals vs recessed wire-entry holes) otherwise dilute the model.
    if keep_prefixes :
        prefs =tuple (s .strip ()for s in keep_prefixes if s .strip ())
        before =len (parts )
        parts =[p for p in parts if str (p .part_nr ).startswith (prefs )]
        logger .info ("scope filter --keep-prefixes %s: kept %d of %d part(s)",
        ",".join (prefs ),len (parts ),before )
        if not parts :
            raise SystemExit ("no parts left after --keep-prefixes filter")

            # Scope filter: drop specific PartNrs listed in a file (one per line, # for
            # comments). Finer-grained than keep_prefixes: the PXC human-GT set turned
            # out to mix real terminal blocks with out-of-scope device types (touch
            # panels, box PCs, power supplies, Inline I/O modules) under the same
            # prefix -- those are excluded from BOTH fit and eval so the metrics stay
            # scope-honest (terminal blocks only).
    if exclude_parts_file :
        excl ={l .split ("#",1 )[0 ].strip ()
        for l in open (exclude_parts_file ,encoding ="utf-8")}
        excl .discard ("")
        before =len (parts )
        parts =[p for p in parts if str (p .part_nr )not in excl ]
        logger .info ("scope filter --exclude-parts-file %s: dropped %d of %d "
        "part(s)",exclude_parts_file ,before -len (parts ),before )
        if not parts :
            raise SystemExit ("no parts left after --exclude-parts-file filter")

            # Scope filter: drop parts whose bbox diagonal exceeds max_bbox_mm. The corpus
            # mixes small CONNECTORS (the task; ~6 CPs, <250mm) with large industrial DRIVES
            # / cabinets (ACS/ACH/ACQ580 etc.; ~42 CPs, 300-1100mm) that are a different,
            # much harder object class. Drives are 88% of all CPs and the model fails ten them
            # (F1~30%), dragging the metric AND diluting capacity away from connectors. With
            # the filter the whole model focuses ten the connector task. None = keep all.
    if max_bbox_mm :
        kept =[]
        for p in parts :
            V =np .asarray (p .vertices ,dtype =np .float64 )
            diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))if len (V )else 0.0 
            if diag <=max_bbox_mm :
                kept .append (p )
        n_drop =len (parts )-len (kept )
        logger .info ("scope filter --max-bbox-mm %.0f: kept %d connector part(s), "
        "dropped %d large/out-of-scope part(s)",max_bbox_mm ,len (kept ),n_drop )
        parts =kept 
        if not parts :
            raise SystemExit ("no parts left after --max-bbox-mm filter")

    ids =[p .part_nr for p in parts ]
    group_keys =jd .build_group_keys (parts ,mode =split_group )
    if split_group !="none":
        n_groups =len (set (group_keys ))
        logger .info ("split-group=%s: %d parts -> %d groups",split_group ,
        len (parts ),n_groups )
        # over-aggressive grouping (e.g. 'prefix' ten uniformly-named parts) can
        # collapse the whole corpus into a couple of groups, which the hash split
        # then sends to ONE side -> empty val/test and no best-tracking. Warn early.
        if n_groups <5 and len (parts )>=10 :
            logger .warning ("split-group=%s collapsed %d parts into only %d group(s) "
            "-- the train/val/test split will likely be degenerate. "
            "Use --split-group geometry (groups by mesh, naming-"
            "independent) or none if val/test come out empty.",
            split_group ,len (parts ),n_groups )
            # family_keys: independent of group_keys/split_group -- used ONLY to guard
            # against a specific small part-family being starved of train-side data even
            # when the whole-corpus val/test fraction looks balanced (see three_way_split's
            # family_keys docstring; measured ten cp_knn_v17: SIE/PXC/A-B/RIT were left with
            # 25-53% of their parts in train vs a ~65% nominal, correlating with those exact
            # families never being learned at all).
    family_keys =[jd .prefix_family (p .part_nr )for p in parts ]
    # split_seed: decouples the train/val/test PARTITION from the run's model-init/
    # augmentation RNG. Multi-seed replicas (variance CI, seed ensembles) MUST share
    # one split -- if the partition follows the run seed, every replica gets a
    # different val set, the per-seed numbers aren't comparable, and a decode-level
    # ensemble of the replicas is ILLEGITIMATE ten any of their val sets (each model
    # trained ten parts that sit in another replica's val -> leakage). Default None
    # keeps the old behaviour (partition follows `seed`) for single-run workflows.
    eff_split_seed =seed if split_seed is None else int (split_seed )
    train_ids ,val_ids ,test_ids =three_way_split (
    ids ,val_frac =val_frac ,test_frac =test_frac ,seed =eff_split_seed ,
    group_keys =group_keys ,family_keys =family_keys )
    # auto-recover from a degenerate GROUPED split: --split-group prefix ten
    # uniformly-named parts (e.g. synthetic part_0001..) can collapse the corpus
    # into ~1 group, which the hash then sends to ONE side -> empty val/test and
    # no best-checkpoint tracking. Fall back to a per-part split so the run stays
    # valid regardless of the --split-group / data combination.
    if split_group !="none"and len (parts )>=10 and (
    not val_ids or (test_frac >0 and not test_ids )):
        logger .warning ("split-group=%s produced a degenerate split (val=%d, test=%d) "
        "-- falling back to a per-part split",split_group ,
        len (val_ids ),len (test_ids ))
        train_ids ,val_ids ,test_ids =three_way_split (
        ids ,val_frac =val_frac ,test_frac =test_frac ,seed =eff_split_seed ,
        group_keys =None )
        # guarantee at least one part trains even ten a tiny corpus
    if not train_ids :
        train_ids ={ids [0 ]}
        val_ids .discard (ids [0 ]);test_ids .discard (ids [0 ])

        # --- RESUME: the manifest of the run being resumed is AUTHORITATIVE ---
        # The split above is RE-DERIVED from the live corpus, and three_way_split's
        # balance search depends ten the corpus CONTENTS (and their order): add or drop
        # a few parts between the launch and the resume and it can legitimately pick a
        # different seed -- silently moving an already-trained part into val, which
        # makes the resumed best_f1/history (and every number reported from them)
        # measured ten a leaked val set. The checkpoint's config fingerprint cannot
        # catch this: seed/val_frac/split_group are all unchanged. So ten a resume we
        # replay the frozen split from the manifest instead, and say so out loud when
        # it disagrees with what a fresh derivation would have produced.
    if resume_from and best_path :
        _man_path =os .path .splitext (best_path )[0 ]+"_manifest.json"
        _man_split =None 
        if os .path .exists (_man_path ):
            try :
                with open (_man_path ,encoding ="utf-8")as fh :
                    _man_split ={str (e ["part_nr"]):e ["split"]
                    for e in json .load (fh ).get ("parts",[])}
            except Exception as exc :# noqa: BLE001
                logger .warning ("resume: split manifest %s unreadable (%s) -- falling "
                "back to the re-derived split",_man_path ,exc )
        if _man_split :
            live ={str (i )for i in ids }
            m_val ={i for i in ids if _man_split .get (str (i ))=="val"}
            m_test ={i for i in ids if _man_split .get (str (i ))=="test"}
            new_parts =[i for i in ids if str (i )not in _man_split ]
            gone =[k for k in _man_split if k not in live ]
            moved =sum (1 for i in ids if str (i )in _man_split 
            and ((i in val_ids )!=(i in m_val )
            or (i in test_ids )!=(i in m_test )))
            if moved or new_parts or gone :
                logger .warning (
                "RESUME SPLIT DRIFT: a fresh derivation disagrees with the "
                "manifest on %d part(s); %d part(s) are new since the launch, "
                "%d manifest part(s) are no longer in the corpus. REPLAYING THE "
                "MANIFEST split (new parts -> train) so val/test stay exactly "
                "what this run has been measured on -- re-deriving would leak "
                "already-trained parts into val.",moved ,len (new_parts ),len (gone ))
            else :
                logger .info ("resume: re-derived split matches the manifest exactly "
                "(%d parts) -- no drift",len (ids ))
            val_ids ,test_ids =m_val ,m_test 
            train_ids ={i for i in ids if i not in val_ids and i not in test_ids }
        elif not os .path .exists (_man_path ):
            logger .warning ("resume: no split manifest next to %s -- the split is being "
            "RE-DERIVED from the live corpus; if the corpus changed "
            "since the launch, val may now contain trained parts",
            best_path )

            # Immutable split manifest: exactly which PartNr landed in train/val/test, its
            # geometry group, family, and a content hash (sha256 of vertex bytes -- same
            # convention as _knn_graph_disk's cache key) -- so a later diagnostic/tuning
            # script can load THIS split verbatim instead of re-deriving it from --seed/
            # --split-group/--max-bbox-mm/etc (which, if any of those flags are passed
            # differently or the underlying corpus files change, silently reconstructs a
            # DIFFERENT split and reports meaningless numbers -- exactly the class of bug
            # already found and fixed before this session for several diagnostic scripts).
    if best_path :
        try :
            import hashlib as _hashlib 
            manifest ={
            "run_config":{"seed":seed ,"split_seed":eff_split_seed ,
            "split_group":split_group ,
            "val_frac":val_frac ,"test_frac":test_frac ,
            "source":str (source ),
            "extra_sources":[str (s )for s in (extra_sources or [])],
            "max_bbox_mm":max_bbox_mm ,"keep_prefixes":keep_prefixes ,
            "exclude_parts_file":exclude_parts_file },
            "parts":[
            {"part_nr":str (p .part_nr ),
            "split":("val"if p .part_nr in val_ids else 
            "test"if p .part_nr in test_ids else "train"),
            "geometry_group":str (group_keys [i ]),
            "family":family_keys [i ],
            "n_cps":int (p .n_cps ),
            "vertex_hash":_hashlib .sha256 (
            np .asarray (p .vertices ,dtype =np .float32 ).tobytes ()
            ).hexdigest ()[:16 ]}
            for i ,p in enumerate (parts )],
            }
            manifest_path =os .path .splitext (best_path )[0 ]+"_manifest.json"
            with open (manifest_path ,"w",encoding ="utf-8")as fh :
                json .dump (manifest ,fh ,indent =1 )
            logger .info ("wrote split manifest -> %s (%d parts)",manifest_path ,
            len (manifest ["parts"]))
        except Exception as exc :# noqa: BLE001
            logger .warning ("could not write split manifest (%s) -- continuing "
            "without it",exc )

            # train_s/train_m are the ORIGINAL train parts (used for the train report);
            # fit_s is what the trainer actually optimises ten -- originals plus `augment`
            # augmented clones per part (+ optional negatives). Keeping them separate means
            # augmentation never leaks into the reported metrics and val/test stay pristine.
    train_s ,train_m ,val_s ,val_m ,fit_s =[],[],[],[],[]
    test_s ,test_m =[],[]
    train_negs =[]# zero-CP parts ten the train side
    aug_rng =np .random .default_rng (seed +1 )
    n_aug =max (0 ,int (augment ))

    # Family-aware augment boost: a data-scarce family (e.g. RIT=4, A-B=16 real
    # parts) gets the SAME flat --augment N as a data-rich one (ABB=151), so it
    # remains starved in absolute training-example count even after fixing the
    # patch/non-patch graph-count imbalance (a separate, larger effect) and the
    # per-family split skew (three_way_split's family_keys). Give scarce families a
    # few EXTRA augmented clones -- sqrt-scaled (not linear) and capped, so a
    # 34x-rarer family (RIT vs ABB) gets a modest ~2x extra clones, not 34x (which
    # would just overfit the handful of real parts it has instead of helping).
    fam_train_counts ={}
    if n_aug >0 :
        for p in parts :
            if p .n_cps >0 and p .part_nr not in val_ids and p .part_nr not in test_ids :
                fam =jd .prefix_family (p .part_nr )
                fam_train_counts [fam ]=fam_train_counts .get (fam ,0 )+1 
    max_fam_count =max (fam_train_counts .values ())if fam_train_counts else 1 
    _FAMILY_AUG_CAP =8 # hard ceiling ten the EXTRA (beyond n_aug) clones/part

    # EXPLICIT per-family boost ("PXC:3,foo:2"): multiplies that family's clone
    # count ten top of the automatic sqrt scaling. The automatic boost equalizes
    # by COUNT only; a family can be count-adequate yet PERFORMANCE-starved --
    # measured: hp_v22's single PXC val part yields 0 matched detections and the
    # 3-part human-GT eval shows ML blind/near-miss ten PXC push-in geometry,
    # while wscaduniverse (CAD-labeled, 6.5x more parts) dominates the loss.
    # This is the P0 "balanced hp_v24" knob: oversample PXC without touching
    # the wscad pipeline.
    fam_boost ={}
    for tok in str (family_boost or "").split (","):
        tok =tok .strip ()
        if not tok :
            continue 
        name ,_ ,mult =tok .partition (":")
        try :
            fam_boost [name .strip ()]=max (1.0 ,float (mult or 1 ))
        except ValueError :
            logger .warning ("--family-boost: ignoring malformed token %r",tok )
    if fam_boost :
        logger .info ("family augment boost active: %s",fam_boost )

    def _n_aug_for (part_nr ):
        if n_aug <=0 or not fam_train_counts :
            return n_aug 
        fam =jd .prefix_family (part_nr )
        cnt =fam_train_counts .get (fam ,max_fam_count )
        extra =max (0 ,int (np .sqrt (max_fam_count /max (1 ,cnt )))-1 )
        base =n_aug +min (extra ,_FAMILY_AUG_CAP )
        return int (round (base *fam_boost .get (fam ,1.0 )))

    for p in parts :
        in_val =p .part_nr in val_ids 
        in_test =p .part_nr in test_ids 
        if p .n_cps ==0 and not in_val and not in_test :
        # train-side zero-CP parts feed the (sampled, capped) negatives pool below
            train_negs .append (p )
            continue 
            # val/test-side zero-CP parts are SCORED, not skipped. They used to hit the
            # `continue` above, so 8 of the 278 val parts never reached the evaluator (the
            # history's val_n_parts=270 is exactly 278 - 8). That silently deleted the one
            # question a deployed detector is judged ten -- "does it invent connection points
            # ten a part that has none?" -- because a zero-CP part can only ever produce FPs.
            # An empty prediction ten an empty part is a legitimate, and perfect, result.
        s =cpr .prepare_sample (p ,dedup =True ,cp_surface_tol_frac =cp_surface_tol_frac )
        meta ={"part_nr":p .part_nr ,"vertices":p .vertices }
        if in_val :
            val_s .append (s );val_m .append (meta )
        elif in_test :
            test_s .append (s );test_m .append (meta )
        else :
            train_s .append (s );train_m .append (meta )
            fit_s .append (s )
            for _ in range (_n_aug_for (p .part_nr )):
                ap =cpr .augment_part (p ,aug_rng ,rotate =aug_rotate ,
                jitter_frac =aug_jitter_frac ,reflect =aug_reflect ,
                dropout_frac =aug_dropout_frac ,
                cable_frac =aug_cable_frac ,cube_rotate =aug_cube_rotate )
                # quiet=True: a clone is a rigid copy, so its CP-surface/dedup/sigma
                # warnings are identical to the original's -- printing them per clone
                # just floods the log (and falsely implicates augmentation).
                fit_s .append (cpr .prepare_sample (ap ,dedup =True ,quiet =True ,
                cp_surface_tol_frac =cp_surface_tol_frac ))

                # negatives: a sampled fraction (capped) of the train-side zero-CP parts.
    n_neg =0 
    if neg_frac and neg_frac >0.0 and train_negs :
        k =int (round (neg_frac *len (train_negs )))
        if max_neg is not None :
            k =min (k ,int (max_neg ))
        k =max (0 ,min (k ,len (train_negs )))
        if k :
            pick =np .random .default_rng (seed +2 ).choice (len (train_negs ),k ,replace =False )
            for j in pick :
                fit_s .append (cpr .prepare_sample (train_negs [int (j )],dedup =True ,
                cp_surface_tol_frac =cp_surface_tol_frac ))
            n_neg =k 
    logger .info ("train parts=%d (+%d augmented +%d negatives = %d fit)  "
    "val parts=%d  test parts=%d  (avail zero-CP train-side=%d)",
    len (train_s ),len (fit_s )-len (train_s )-n_neg ,n_neg ,len (fit_s ),
    len (val_s ),len (test_s ),len (train_negs ))
    # an empty val set ten a real corpus means no best-by-F1 tracking and no honest
    # number -- almost always a degenerate split (over-aggressive --split-group, or
    # too-small --val-frac). Make it impossible to miss.
    if not val_s and len (parts )>=10 :
        logger .warning ("VALIDATION SET IS EMPTY -- best-checkpoint tracking is "
        "disabled and val/test metrics will be blank. Likely cause: "
        "--split-group=%s collapsed the corpus. Fix with "
        "--split-group geometry|none or a larger --val-frac.",
        split_group )

        # Make the big-part recall ceiling VISIBLE: count how many val/test GT points
        # become unrecoverable because the inference subsample drops their region.
    if backbone in ("knngraph","hierpoint")and max_gpu_verts :
        for tag ,ss in (("val",val_s ),("test",test_s )):
            n_sub =n_gt =n_risk =0 
            for s in ss :
                if len (s ["verts_norm"])>max_gpu_verts :
                    n_sub +=1 
                    g ,r =cpr .subsample_coverage (s ["verts_norm"],s ["gt_points"],
                    s ["center"],s ["scale"],max_gpu_verts )
                    n_gt +=g ;n_risk +=r 
            if n_sub :
                logger .info ("%s: %d part(s) subsampled (>%d verts); %d/%d GT points "
                "at risk of becoming unrecoverable FNs (%.1f%%)",
                tag ,n_sub ,max_gpu_verts ,n_risk ,n_gt ,
                100.0 *n_risk /max (1 ,n_gt ))

                # decode operating point bundled into every saved checkpoint, so a raw .ckpt
                # deploys with THIS run's knobs (not DEFAULT_DECODE).
    decode ={"heatmap_thresh":heatmap_thresh ,"min_votes":min_votes ,
    "nms_clearance_mm":nms_clearance_mm ,"dist_thresh_mm":dist_thresh_mm }
    # training fingerprint stored in the checkpoint so a --resume can detect a
    # mismatched continuation (incomparable best_f1 / history).
    train_config ={"val_frac":val_frac ,"seed":seed ,
    "split_seed":eff_split_seed ,"split_group":split_group ,
    "family_boost":family_boost ,
    "exclude_parts_file":exclude_parts_file ,
    "test_frac":test_frac ,"heat_pos_weight":heat_pos_weight ,
    "heat_loss":heat_loss ,"focal_gamma":focal_gamma ,
    "centernet_alpha":centernet_alpha ,
    "augment":augment ,"best_metric":best_metric ,
    "max_gpu_verts":max_gpu_verts ,"backbone":backbone ,
    "source":str (source ),"w_heat":w_heat ,"w_off":w_off ,
    "w_dir":w_dir ,"weight_decay":weight_decay ,"lr":lr ,
    "epochs":epochs ,"eval_every":eval_every ,
    "patience":patience ,"lr_schedule":lr_schedule ,
    "lr_decay_every":lr_decay_every ,
    "lr_decay_rate":lr_decay_rate ,
    "warmup_epochs":warmup_epochs ,
    "ckpt_every":save_every ,
    "snapshot_every":snapshot_every ,
    "min_delta":min_delta ,"collapse_gap":collapse_gap ,
    "offset_heat_weight":offset_heat_weight ,
    "part_weight_mode":part_weight_mode ,
    "offset_loss":offset_loss ,"offset_huber_beta":offset_huber_beta ,
    # scope/data-provenance -- NOT previously recorded, so a reloaded
    # checkpoint couldn't tell you what corpus/scope it was actually
    # trained ten without re-reading the shell history.
    "extra_sources":[str (s )for s in (extra_sources or [])],
    "max_bbox_mm":max_bbox_mm ,"keep_prefixes":keep_prefixes ,
    "neg_frac":neg_frac ,"max_neg":max_neg ,
    "hard_mining":hard_mining ,"dir_sign_invariant":dir_sign_invariant ,
    "sign_inv_prefixes":sign_inv_prefixes ,"fwd_verts":fwd_verts ,
    "aug_rotate":aug_rotate ,"aug_jitter_frac":aug_jitter_frac ,
    "aug_reflect":aug_reflect ,"aug_dropout_frac":aug_dropout_frac ,
    "aug_cable_frac":aug_cable_frac ,"aug_cube_rotate":aug_cube_rotate ,
    "init_from":init_from }

    if backbone =="mlp":
        import torch 

        def _infer_builder_mlp (m ,mt ):
            def f (s ,pm ):
                m .eval ()
                with torch .no_grad ():
                    x =torch .tensor (s ["verts_norm"],dtype =torch .float32 ,
                    device =device )
                    return cpr .pred_to_array (m (x ),offset_scale =s ["scale"])
            return f 
        _infer_builder =_infer_builder_mlp 

        metrics_fn =_val_metrics_fn (val_s ,val_m ,_infer_builder ,
        dist_thresh_mm ,min_votes ,
        heatmap_thresh =heatmap_thresh ,
        nms_clearance_mm =nms_clearance_mm )
        model =cpr .train_cpmlp (
        fit_s ,epochs =epochs ,device =device ,
        log_every =max (1 ,epochs //6 ),
        metrics_fn =metrics_fn ,eval_every =eval_every ,patience =patience ,
        best_path =best_path ,last_path =last_path ,save_every =save_every ,
        history_path =history_path ,resume_from =resume_from ,seed =seed ,
        decode =decode ,train_config =train_config ,best_metric =best_metric ,
        resume_strict =resume_strict ,snapshot_every =snapshot_every ,
        weight_decay =weight_decay ,grad_clip =grad_clip ,
        offset_heat_weight =offset_heat_weight ,part_weight_mode =part_weight_mode ,
        lr_schedule =lr_schedule ,warmup_epochs =warmup_epochs ,
        min_delta =min_delta ,collapse_gap =collapse_gap )
        infer_arr =_infer_builder (model ,None )
    elif backbone in ("knngraph","hierpoint"):
    # torch-only backbones (no native geometry deps) -> run ten native Windows.
    # hierpoint shares the knngraph trainer/inference harness end to end
    # (vertex selection, patching, vote merge, caches); only the network and
    # its graph structure differ (cp_regressor dispatches ten `backbone`).
    # cache val kNN graphs across evals (same arrays reused every metrics_fn).
        _knn_graph_cache ={}

        def _infer_builder_knn (m ,mt ):
            def f (s ,pm ):
            # match the per-part vertex-selection used in training: terminal blocks
            # (wscad+PXC) were spatial-patched, other parts uniform-subsampled.
                patch =cpr .is_patch_part (s .get ("part_nr",""))
                return cpr .infer_knngraph (m ,mt ,s ["verts_norm"],device =device ,
                max_gpu_verts =max_gpu_verts ,
                offset_scale =s ["scale"],
                graph_cache =_knn_graph_cache ,
                knn_cache_dir =knn_cache_dir ,patch =patch ,
                part_nr =s .get ("part_nr"))
            return f 
        _infer_builder =_infer_builder_knn 

        metrics_fn =_val_metrics_fn (val_s ,val_m ,_infer_builder ,
        dist_thresh_mm ,min_votes ,
        heatmap_thresh =heatmap_thresh ,
        nms_clearance_mm =nms_clearance_mm )
        model ,meta =cpr .train_knngraph_regressor (
        fit_s ,config =(hp_config if backbone =="hierpoint"else knn_config ),
        backbone =backbone ,epochs =epochs ,device =device ,
        log_every =1 ,max_gpu_verts =max_gpu_verts ,lr =lr ,
        metrics_fn =metrics_fn ,eval_every =eval_every ,patience =patience ,
        best_path =best_path ,last_path =last_path ,save_every =save_every ,
        history_path =history_path ,resume_from =resume_from ,seed =seed ,
        heat_pos_weight =heat_pos_weight ,w_heat =w_heat ,w_off =w_off ,w_dir =w_dir ,
        heat_loss =heat_loss ,focal_gamma =focal_gamma ,centernet_alpha =centernet_alpha ,
        lr_decay_every =lr_decay_every ,lr_decay_rate =lr_decay_rate ,
        accum_steps =accum_steps ,amp =amp ,
        decode =decode ,train_config =train_config ,best_metric =best_metric ,
        resume_strict =resume_strict ,snapshot_every =snapshot_every ,
        weight_decay =weight_decay ,lr_schedule =lr_schedule ,
        warmup_epochs =warmup_epochs ,grad_clip =grad_clip ,
        offset_heat_weight =offset_heat_weight ,part_weight_mode =part_weight_mode ,
        knn_cache_dir =knn_cache_dir ,dir_sign_invariant =dir_sign_invariant ,
        init_from =init_from ,hard_mining =hard_mining ,
        offset_loss =offset_loss ,offset_huber_beta =offset_huber_beta ,
        min_delta =min_delta ,collapse_gap =collapse_gap ,
        fwd_verts =fwd_verts or None ,pin_host_memory =pin_host_memory ,
        sign_inv_prefixes =sign_inv_prefixes )
        infer_arr =_infer_builder (model ,meta )
    else :# diffusionnet
        def _infer_builder_dn (m ,mt ):
            def f (s ,pm ):
                return cpr .infer_diffusionnet (
                m ,mt ,pm ["vertices"],s ["faces"],s ["verts_norm"],
                op_cache_dir =op_cache_dir ,device =device ,
                max_gpu_verts =max_gpu_verts ,offset_scale =s ["scale"])
            return f 
        _infer_builder =_infer_builder_dn 

        metrics_fn =_val_metrics_fn (val_s ,val_m ,_infer_builder ,
        dist_thresh_mm ,min_votes ,
        heatmap_thresh =heatmap_thresh ,
        nms_clearance_mm =nms_clearance_mm )
        model ,meta =cpr .train_diffusionnet_regressor (
        fit_s ,config =dn_config ,epochs =epochs ,device =device ,op_cache_dir =op_cache_dir ,
        log_every =max (1 ,epochs //10 ),max_gpu_verts =max_gpu_verts ,lr =lr ,
        metrics_fn =metrics_fn ,eval_every =eval_every ,patience =patience ,
        best_path =best_path ,last_path =last_path ,save_every =save_every ,
        history_path =history_path ,resume_from =resume_from ,seed =seed ,
        heat_pos_weight =heat_pos_weight ,w_heat =w_heat ,w_off =w_off ,w_dir =w_dir ,
        heat_loss =heat_loss ,focal_gamma =focal_gamma ,
        lr_decay_every =lr_decay_every ,lr_decay_rate =lr_decay_rate ,
        accum_steps =accum_steps ,low_memory =low_memory ,
        decode =decode ,train_config =train_config ,best_metric =best_metric ,
        resume_strict =resume_strict ,snapshot_every =snapshot_every ,
        weight_decay =weight_decay ,lr_schedule =lr_schedule ,
        warmup_epochs =warmup_epochs ,grad_clip =grad_clip ,
        offset_heat_weight =offset_heat_weight ,part_weight_mode =part_weight_mode ,
        min_delta =min_delta ,collapse_gap =collapse_gap )
        infer_arr =_infer_builder (model ,meta )

        # Reload the BEST checkpoint before computing the final reports below.
        # `model` up to this point is whatever was in memory when the training loop
        # exited -- that's only guaranteed to BE the best epoch if patience triggered
        # exactly at the best point; if the run instead reached --epochs (or was
        # interrupted) after training continued PAST its best validation epoch, the
        # in-memory model is a LATER, possibly worse state, and reporting from it
        # silently understates (or overstates) what the actually-saved/deployed best
        # checkpoint can do. Report ten the artifact that will actually be used.
    if best_path and os .path .exists (best_path ):
        try :
            model ,meta ,_bb =cpr .load_model (best_path ,device =device )
            infer_arr =_infer_builder (model ,meta )
            logger .info ("reloaded best checkpoint (%s) for the final train/val/"
            "test report below",best_path )
        except Exception as exc :# noqa: BLE001
            logger .warning ("could not reload best checkpoint %s for the final "
            "report (%s) -- reporting the in-memory end-of-"
            "training model instead",best_path ,exc )

            # CP_FREE_TRAIN_SOURCE=1 drops the train parts' full-res source arrays after the
            # trainer has copied what it needs into `prepared` (~30GB ten corpus v8 -- the
            # difference between finishing and dying at 88% commit during prep). Those parts
            # can no longer be re-evaluated, so report ten the ones that survive rather than
            # crashing HERE, after the whole run. val/test samples are never freed.
    live_train =[(s ,m )for s ,m in zip (train_s ,train_m )
    if s .get ("verts_norm")is not None ]
    if len (live_train )<len (train_s ):
        logger .info ("train report: %d/%d parts freed by CP_FREE_TRAIN_SOURCE "
        "(memory) -- reporting on the rest; val/test are unaffected",
        len (train_s )-len (live_train ),len (train_s ))
    train_reports =evaluate ([s for s ,_ in live_train ],[m for _ ,m in live_train ],
    infer_arr ,
    dist_thresh_mm =dist_thresh_mm ,
    heatmap_thresh =heatmap_thresh ,min_votes =min_votes ,
    nms_clearance_mm =nms_clearance_mm )if live_train else []
    if print_points and val_s :
        print ("\n=== VAL per-part metrics (accuracy / precision / recall / F1) ===")
    val_reports =(evaluate (val_s ,val_m ,infer_arr ,
    dist_thresh_mm =dist_thresh_mm ,
    heatmap_thresh =heatmap_thresh ,min_votes =min_votes ,
    nms_clearance_mm =nms_clearance_mm ,
    print_points =print_points )
    if val_s else [])
    # frozen held-out TEST split: one unbiased number, scored at the deploy
    # decode point. Never touched training or best-checkpoint/decode selection.
    test_reports =(evaluate (test_s ,test_m ,infer_arr ,
    dist_thresh_mm =dist_thresh_mm ,
    heatmap_thresh =heatmap_thresh ,min_votes =min_votes ,
    nms_clearance_mm =nms_clearance_mm )
    if test_s else [])
    return (aggregate (train_reports ),aggregate (val_reports ),train_reports ,
    val_reports ,aggregate (test_reports ),test_reports )


def _prefix (part_nr ):
    """Training-log per-family display prefix -- delegates to
    jd.prefix_family, the single source of truth also used by tune_thresholds.py
    / predict.py for per-family decode thresholds."""
    return jd .prefix_family (part_nr )


def _print_agg (title ,agg ,reports =None ):
    """Readable metric block: per-part means + pooled (micro) values.
    When `reports` is given, also print per-prefix breakdown automatically."""
    print (f"\n=== {title } ===")
    if not agg :
        print ("  (no parts)")
        return 
    print ("  parts=%d  GT=%d  TP=%d  FP=%d  FN=%d"
    %(agg ["n_parts"],agg ["total_gt"],agg ["total_tp"],
    agg ["total_fp"],agg ["total_fn"]))
    for k in ("accuracy","precision","recall","f1"):
        print ("  %-9s : %.2f%%   (pooled %.2f%%)"
        %(k ,agg [k ]*100 ,agg ["micro_"+k ]*100 ))
    print ("  loc error : %.3f mm    ang error : %.2f deg"
    %(agg ["mean_loc_err_mm"],agg ["mean_ang_err_deg"]))

    if reports :
    # Per-prefix breakdown (wscad / PXC / ABB / SYNTH / ...)
        by_pfx ={}
        for r in reports :
            pfx =_prefix (r .get ("part_nr","unknown"))
            by_pfx .setdefault (pfx ,[]).append (r )
        if len (by_pfx )>1 :
            print ("  --- per-prefix (micro) ---")
            for pfx in sorted (by_pfx ):
                sub =aggregate (by_pfx [pfx ])
                print ("  %-22s  parts=%3d  F1=%5.1f%%  P=%5.1f%%  R=%5.1f%%  "
                "TP=%4d  FP=%4d  FN=%4d"
                %(pfx ,sub ["n_parts"],
                100 *sub ["micro_f1"],
                100 *sub ["micro_precision"],
                100 *sub ["micro_recall"],
                sub ["total_tp"],sub ["total_fp"],sub ["total_fn"]))


def _load_yaml_config (path ):
    """Load a YAML/JSON config file and return a flat dict of CLI-style args.
    Keys use underscores (matching argparse dest names). Example config.yaml:
        backbone: knngraph
        knn_normals: true
        knn_curvature: true
        epochs: 120
        augment: 4
        run_name: cp_knn_v15
    """
    import json as _json 
    with open (path ,encoding ="utf-8")as fh :
        text =fh .read ()
    try :
        import yaml as _yaml # optional dependency
        data =_yaml .safe_load (text )
    except ImportError :
        data =_json .loads (text )# fall back to JSON (YAML superset for simple configs)
    return {str (k ):v for k ,v in (data or {}).items ()}


def main (argv =None ):
    ap =argparse .ArgumentParser (description ="Train connection-point detector on ABB JSON data",
    formatter_class =argparse .ArgumentDefaultsHelpFormatter )
    ap .add_argument ("--config",default =None ,metavar ="FILE",
    help ="YAML (or JSON) config file -- sets argument defaults; "
    "explicit CLI flags override. Keys = argparse dest names "
    "(underscores). Example: backbone: knngraph, epochs: 120")
    ap .add_argument ("source",help ="corpus: a directory, a single big JSON array, or one part file")
    ap .add_argument ("--extra-source",action ="append",default =[],dest ="extra_sources",
    metavar ="DIR",help ="additional corpus directory to merge into training "
    "(can be repeated). Parts from extra sources follow "
    "the same split logic as the primary source.")
    ap .add_argument ("--max-bbox-mm",type =float ,default =None ,dest ="max_bbox_mm",
    help ="scope filter: drop parts whose bbox diagonal exceeds this (mm). "
    "Use ~250 to keep CONNECTORS and exclude large industrial drives/"
    "cabinets (a different object class that dominates CPs and tanks "
    "the metric). Applies to all sources; default keeps everything.")
    ap .add_argument ("--keep-prefixes",default =None ,
    help ="scope filter: comma-separated PartNr prefixes to KEEP (e.g. "
    "'wscaduniverse,PXC' for a terminal-block-only model). Drops "
    "everything else. Applied before --max-bbox-mm.")
    ap .add_argument ("--backbone",
    choices =["mlp","diffusionnet","knngraph","hierpoint"],
    default ="mlp")
    ap .add_argument ("--epochs",type =int ,default =300 ,
    help ="TOTAL epochs to reach; with --resume, training continues "
    "from the checkpoint's epoch up to this target")
    ap .add_argument ("--val-frac",type =float ,default =0.2 )
    ap .add_argument ("--max-parts",type =int ,default =None )
    ap .add_argument ("--dist-thresh-mm",type =float ,default =5.0 )
    ap .add_argument ("--device",default ="cpu")
    ap .add_argument ("--seed",type =int ,default =0 )
    ap .add_argument ("--family-boost",default ="",dest ="family_boost",
    help ="explicit per-family augment multiplier, e.g. 'PXC:3' or "
    "'PXC:3,FES:2' (family = json_dataset.prefix_family). "
    "Multiplies that family's augmented-clone count ON TOP of "
    "the automatic sqrt count-balancing -- the P0 'balanced' "
    "knob for oversampling a performance-starved family "
    "(measured: PXC recall ~0 on hp_v22 val) without touching "
    "the rest of the pipeline")
    ap .add_argument ("--exclude-parts-file",default =None ,
    dest ="exclude_parts_file",
    help ="file of PartNrs (one per line, # comments) to DROP from "
    "both fit and eval. Finer-grained scope filter than "
    "--keep-prefixes: e.g. pxc_out_of_scope.txt removes the "
    "12 PXC-prefixed parts that are NOT terminal blocks "
    "(touch panels, box PCs, power supplies, Inline I/O -- "
    "verified against vendor catalog 2026-07-10)")
    ap .add_argument ("--split-seed",type =int ,default =None ,dest ="split_seed",
    help ="seed for the train/val/test PARTITION only (default: "
    "follow --seed). Multi-seed replicas for variance CI or a "
    "seed ensemble MUST pin this to one value (e.g. 0) so every "
    "replica trains/validates on the SAME parts -- otherwise "
    "per-seed numbers aren't comparable and ensembling the "
    "replicas leaks train parts into each other's val")
    ap .add_argument ("--op-cache-dir",default =None ,
    help ="cache dir for DiffusionNet mesh operators "
    "(strongly recommended for the 479-file corpus)")
    ap .add_argument ("--max-gpu-verts",type =int ,default =60000 ,
    help ="parts larger than this run on CPU even with --device cuda "
    "(avoids OOM on small GPUs); 0 forces GPU for all parts")
    ap .add_argument ("--ckpt-dir",default ="checkpoints",
    help ="directory for the .ckpt files (default: checkpoints/, "
    "tracked in git so other machines get them on clone/pull)")
    ap .add_argument ("--run-name",default =None ,
    help ="basename for the checkpoint files "
    "(default: cp_<backbone>)")
    ap .add_argument ("--resume",action ="store_true",
    help ="continue training from <ckpt-dir>/<run-name>_last.ckpt "
    "(falls back to _best.ckpt); restores model+optimizer+"
    "scheduler+epoch, so a run can hop between machines")
    ap .add_argument ("--resume-from",default =None ,
    help ="explicit checkpoint path to resume from (overrides --resume)")
    ap .add_argument ("--hard-mining",action ="store_true",dest ="hard_mining",
    help ="loss-proportional part sampling: after epoch 0, oversample "
    "parts with high recent loss (EMA). Focuses training on hard "
    "examples. Compatible with graph batching.")
    ap .add_argument ("--init-from",default =None ,dest ="init_from",
    help ="FINE-TUNE init: load ONLY the model weights from a "
    "(pretrained) checkpoint, then start a FRESH run (epoch 0, "
    "fresh optimizer/history). The pretrain->finetune recipe: "
    "pretrain on the synthetic corpus (synth_blocks.py), then "
    "--init-from that .ckpt and fine-tune on the real corpus. "
    "Must match backbone hyperparams (k/width/layers/normals). "
    "Ignored if --resume/--resume-from is also given.")
    ap .add_argument ("--ckpt",default =None ,
    help ="explicit path for the BEST checkpoint "
    "(default <ckpt-dir>/<run-name>_best.ckpt); reload with "
    "cp_regressor.load_model")
    ap .add_argument ("--ckpt-every",type =int ,default =1 ,
    help ="write the resumable last.ckpt every N epochs "
    "(default 1; 0 = only at the end)")
    ap .add_argument ("--print-points",action ="store_true",
    help ="after training, print per-part keypoint metrics for the "
    "validation parts: accuracy / precision / recall / F1 plus "
    "TP/FP/FN and mean loc/ang error, one line per part")
    ap .add_argument ("--out",default =None ,
    help ="write aggregate + per-part metrics (with timing) to this JSON file")
    ap .add_argument ("--export-model",default =None ,
    help ="after training, write a lean inference-only .pt from the "
    "best checkpoint (weights+meta+backbone+decode params, no "
    "optimizer/history). Load it with predict.py")
    ap .add_argument ("--min-votes",type =int ,default =1 ,
    help ="min voting vertices per decoded CP. Default 1: the GT-decode "
    "ceiling on the real corpus is R=0.94 at min_votes=1 vs only "
    "0.79 at 2 (diag_encoding.py), so >1 throws away recoverable "
    "CPs AND biases best-checkpoint selection. Raise it only via "
    "sweep_decode if precision needs it")
    ap .add_argument ("--heatmap-thresh",type =float ,default =0.3 ,
    help ="decode: min sigmoid heat for a vertex to vote for a CP. "
    "Higher -> fewer false positives, but too high fires "
    "nothing. This is just the training-time eval point; lock "
    "the deployment operating point with sweep_decode.py")
    ap .add_argument ("--nms-clearance-mm",type =float ,default =5.0 ,
    help ="decode: votes are merged within this mm radius as a CEILING "
    "(cp_targets.decode_predictions further shrinks it to a fraction "
    "of each part's own bbox diagonal, so small parts with tight CP "
    "spacing don't get merged just because this constant exceeds "
    "their real spacing); larger collapses scattered votes into "
    "fewer detections")
    ap .add_argument ("--heat-pos-weight",type =float ,default =50.0 ,
    help ="heatmap BCE positive weight (1 + w*h). The heat target is "
    "~99%% background, so this stays high (default 50, known-good) "
    "or the model collapses to ~0 heat everywhere (recall 0). "
    "Lower it (e.g. 10-15) only if the heatmap floods/over-fires "
    "at a FULL epoch budget (knngraph/diffusionnet only)")
    ap .add_argument ("--heat-loss",choices =["bce","focal","centernet"],
    default ="centernet",
    help ="heatmap loss (DEFAULT centernet -- bce/focal flood or "
    "collapse on this ~99%%-background heatmap). 'centernet' is "
    "the penalty-reduced focal "
    "loss normalised by #keypoints -- the right choice for this "
    "sparse heatmap; 'bce'/'focal' average over all vertices and "
    "either over-fire or collapse on the 99%% background")
    ap .add_argument ("--focal-gamma",type =float ,default =2.0 )
    ap .add_argument ("--centernet-alpha",type =float ,default =2.0 ,dest ="centernet_alpha",
    help ="CenterNet focal exponent (default 2.0). Lower values (1.0-1.5) "
    "penalise FN less harshly -> higher recall at cost of precision. "
    "Use 1.5 to shift precision/recall balance toward recall.")
    ap .add_argument ("--lr",type =float ,default =1e-3 ,
    help ="AdamW learning rate (also settable via config 'lr:'). "
    "Lower (e.g. 3e-4) for fine-tuning from --init-from.")
    ap .add_argument ("--lr-decay-every",type =int ,default =0 ,
    help ="StepLR step size in epochs (0 disables the schedule)")
    ap .add_argument ("--lr-decay-rate",type =float ,default =0.5 )
    ap .add_argument ("--accum-steps",type =int ,default =1 ,
    help ="gradient accumulation over N parts before opt.step")
    ap .add_argument ("--low-memory",action ="store_true",
    help ="reload eigenbasis from cache per step instead of holding "
    "all in RAM (for corpora larger than memory)")
    ap .add_argument ("--knn-k",type =int ,default =16 ,
    help ="knngraph: neighbours per vertex in the kNN graph (default 16). "
    "Larger k = wider receptive field per layer but slower/more "
    "memory; sweep 12-24")
    ap .add_argument ("--knn-width",type =int ,default =128 ,
    help ="knngraph: EdgeConv feature width (default 128)")
    ap .add_argument ("--knn-layers",type =int ,default =4 ,
    help ="knngraph: number of residual EdgeConv blocks (default 4)")
    ap .add_argument ("--knn-global",action ="store_true",
    help ="knngraph: add a PointNet-style global-context feature "
    "(global max-pool concatenated before the head). Helps the "
    "model judge a vertex relative to the whole part -- usually a "
    "precision/recall win for these sparse connection points")
    ap .add_argument ("--knn-normals",action ="store_true",
    help ="knngraph: add per-vertex PCA surface normals as input "
    "(3->6 dims). Gives the model the concavity/normal-discontinuity "
    "cue raw xyz lacks -- the recall lever for terminal openings the "
    "xyz-only model misses. Computed from the kNN graph (no faces "
    "needed), so train and inference stay matched.")
    ap .add_argument ("--knn-curvature",action ="store_true",
    help ="knngraph: add PCA curvature scalar (lam0/sum(lam)) as 7th input feature "
    "alongside --knn-normals (6->7 dims). Requires --knn-normals.")
    ap .add_argument ("--knn-concavity",action ="store_true",
    help ="knngraph: add concavity score (inward-facing normal dot product) "
    "as extra input feature. 1=fully concave (CP hole), 0=convex. "
    "Requires --knn-normals.")
    ap .add_argument ("--knn-edge-dist",action ="store_true",
    help ="knngraph: add approximate distance-to-boundary feature. "
    "Low = near mesh edge. Requires --knn-normals.")
    ap .add_argument ("--hp-widths",default ="64,128,256,256",metavar ="W0,W1,W2,W3",
    help ="hierpoint: feature widths at levels L0..L3 "
    "(default 64,128,256,256 -- wide only where points are few)")
    ap .add_argument ("--hp-ratios",default ="4,16,64",metavar ="R1,R2,R3",
    help ="hierpoint: level sizes as divisors of N (default 4,16,64 "
    "-> N/4, N/16, N/64 points)")
    ap .add_argument ("--hp-k-levels",default ="16,16,8",metavar ="K1,K2,K3",
    help ="hierpoint: within-level kNN at L1..L3 (default 16,16,8)")
    ap .add_argument ("--hp-layers",default ="1,1,2,1",metavar ="L0,L1,L2,L3",
    help ="hierpoint: LeanEdgeConv blocks per level (default 1,1,2,1)")
    ap .add_argument ("--hp-dropout",type =float ,default =0.1 ,
    help ="hierpoint: dropout before the head (default 0.1)")
    ap .add_argument ("--hp-no-features",action ="store_true",
    help ="hierpoint ABLATION: raw xyz input only (c_in=3) instead of "
    "the default xyz+normals+curvature+concavity+edge_dist "
    "(c_in=9)")
    ap .add_argument ("--hp-no-curvature",action ="store_true",
    help ="hierpoint ABLATION: keep normals but DROP the PCA curvature/"
    "concavity/edge_dist features (c_in=6). These are lambda-ratio "
    "features that read ~0 on the flat facets of a coarse JSON export "
    "mesh but non-zero on a fine STEP tessellation -- the dominant "
    "JSON->STEP feature mismatch. Normals (direction) transfer better.")
    ap .add_argument ("--prep-cache-dir","--knn-cache-dir",dest ="knn_cache_dir",
    default =None ,
    help ="directory for the persistent PREP cache (SHA256-keyed): "
    "kNN graphs AND -- for hierpoint -- the pooling "
    "hierarchies. The hierarchy prebuild is the single most "
    "expensive startup step (MULTIPLE HOURS on the 1749-part "
    "corpus) and without this flag it is recomputed from "
    "scratch on every launch AND every --resume. Set it once "
    "(e.g. --prep-cache-dir prep_cache/) and resumes start in "
    "seconds. Cache writes are skipped when the disk has <3GB "
    "free, so a full disk degrades to 'no cache', never a "
    "crash. (--knn-cache-dir is the old name for this flag.)")
    ap .add_argument ("--dn-eig",type =int ,default =128 ,
    help ="diffusionnet: eigenbasis size k_eig (default 128, capped at "
    "MAX_K_EIG=128). The literature default of 64 cannot represent "
    "the sharp single-vertex heat peak the CP target needs -- it "
    "collapsed in cp_dn_v1/v2 (max_heat stuck ~0.10). 128 is the "
    "documented fix. NOTE: changing this invalidates a 64-built "
    "op-cache, so point --op-cache-dir at a fresh directory")
    ap .add_argument ("--dn-width",type =int ,default =64 ,
    help ="diffusionnet: DiffusionNet feature width c_width (default 64)")
    ap .add_argument ("--dn-blocks",type =int ,default =3 ,
    help ="diffusionnet: number of diffusion blocks (default 3)")
    ap .add_argument ("--dn-dropout",type =float ,default =0.3 ,
    help ="diffusionnet: dropout rate (default 0.3)")
    ap .add_argument ("--augment",type =int ,default =0 ,
    help ="add N augmented clones per TRAINING part (0=off). Each is a "
    "random rigid rotation (+ optional jitter) with CP points/"
    "directions transformed consistently. The main generalisation "
    "lever for mlp/knngraph (raw xyz -> no pose invariance). Try 4-8. "
    "NOTE: with --backbone diffusionnet each clone is fresh geometry, "
    "so the eigenbasis cache grows ~Nx and precompute takes ~Nx longer")
    ap .add_argument ("--no-aug-rotate",dest ="aug_rotate",action ="store_false",
    help ="disable the random rotation in --augment (keep only jitter)")
    ap .add_argument ("--aug-jitter-frac",type =float ,default =0.0 ,
    help ="augmentation vertex jitter: Gaussian std as a fraction of the "
    "bbox diagonal, added to vertices only (GT CPs stay). ~0.005 "
    "simulates mesh/scan noise; 0 disables jitter")
    ap .add_argument ("--aug-reflect",action ="store_true",
    help ="augmentation: also mirror each clone across a random axis with "
    "p=0.5 (connector geometry is often mirror-symmetric). CP points/"
    "directions are mirrored too and face winding is flipped")
    ap .add_argument ("--aug-dropout-frac",type =float ,default =0.0 ,
    dest ="aug_dropout_frac",
    help ="augmentation: drop a spherical region of vertices whose radius = "
    "this fraction of bbox diagonal (e.g. 0.15 drops ~10-20%% of verts "
    "from a random patch). Simulates cable/housing occlusion or a "
    "partial real scan. Faces are invalidated; only safe for "
    "knngraph/mlp (point-cloud backbones). 0 = disabled")
    ap .add_argument ("--aug-cable-frac",type =float ,default =0.0 ,
    dest ="aug_cable_frac",
    help ="augmentation: for each augmented clone, occlude a random fraction "
    "of CP openings with a synthetic cable cylinder (dense ring of "
    "points). Labels unchanged -- model learns 'cable != CP absence'. "
    "E.g. 0.3 = 30%% of CPs occluded per clone. 0 = disabled")
    ap .add_argument ("--aug-cube-rotate",action ="store_true",dest ="aug_cube_rotate",
    help ="augmentation: rotate each clone by one of the 24 octahedral "
    "(90-degree, axis-PRESERVING) rotations instead of a continuous "
    "one. Teaches frame/axis-permutation invariance WITHOUT erasing "
    "the axis-aligned CP-direction prior -- for JSON->STEP transfer "
    "where the STEP arrives in a signed-axis-permuted frame")
    ap .set_defaults (aug_rotate =True ,aug_cube_rotate =False )
    ap .add_argument ("--split-group",choices =["none","prefix","geometry"],
    default ="none",
    help ="how to group parts for a LEAKAGE-SAFE train/val split: 'none' "
    "(per-PartNr, old default), 'prefix' (group part-family variants "
    "by stripped PartNr), 'geometry' (group near-identical meshes). "
    "Stops sibling variants inflating val F1 by leaking across splits")
    ap .add_argument ("--test-frac",type =float ,default =0.0 ,
    help ="carve a frozen held-out TEST split (fraction). It never trains "
    "and never tunes decode/checkpoints -- one unbiased final number "
    "reported alongside train/val. 0 = no test split")
    ap .add_argument ("--cp-surface-tol-frac",type =float ,default =None ,
    help ="drop any GT connection point farther than this fraction of the "
    "bbox diagonal from the nearest vertex (off-surface mislabels). "
    "Unset = keep all, only warn at 5%%")
    ap .add_argument ("--neg-frac",type =float ,default =0.0 ,
    help ="include this fraction of the train-side ZERO-CP parts as "
    "negatives in the fit set, so the model learns to stay silent "
    "where there is nothing (reduces false positives). 0 = drop them")
    ap .add_argument ("--max-neg",type =int ,default =None ,
    help ="cap on the number of negative (zero-CP) parts added by --neg-frac")
    ap .add_argument ("--best-metric",choices =["micro_f1","f1"],default ="micro_f1",
    help ="metric the best checkpoint is selected on. Default micro_f1 "
    "(pooled -- the metric sweeps/reports/deployment use); 'f1' is "
    "the per-part macro mean (the old behaviour)")
    ap .add_argument ("--no-strict-resume",dest ="resume_strict",action ="store_false",
    help ="on --resume, only WARN (don't refuse) if a critical "
    "hyperparameter (val_frac/seed/loss/augment/...) differs from "
    "the checkpoint. Default is strict (refuse) -- a resumed run "
    "with a silently different split/loss produces a best_f1/"
    "history that isn't comparable to the earlier epochs, which "
    "is easy to miss until much later. Pass this flag only when "
    "you deliberately changed a hyperparameter and accept that.")
    ap .set_defaults (resume_strict =True )
    ap .add_argument ("--snapshot-every",type =int ,default =0 ,
    help ="also write a never-overwritten <run>_ep<N>.ckpt every N epochs, "
    "so an earlier good model is recoverable if a later best overfits "
    "(0 = off; only best+last are kept)")
    # --- loss design ---
    ap .add_argument ("--w-heat",type =float ,default =1.0 ,
    help ="loss weight on the heatmap term (centernet). Default 1.0 -- "
    "DO NOT raise above 2.0 with --heat-loss centernet: centernet "
    "normalises by #keypoints so its gradient approaches 0 once "
    "the heatmap fits; a high w_heat then starves the direction "
    "head and can cause ang>90 deg collapse (observed at w_heat=3 "
    "in v7, epoch 25). Sweep w-heat/w-off/w-dir for "
    "precision/recall vs localisation trade-off")
    ap .add_argument ("--w-off",type =float ,default =5.0 ,
    help ="loss weight on the offset (localisation) term")
    ap .add_argument ("--offset-loss",choices =["l1","huber"],default ="l1",
    dest ="offset_loss",
    help ="offset error function (default l1, plain mean-abs-error). "
    "'huber' (smooth-L1) concentrates gradient on shrinking a "
    "near-correct offset's last few mm instead of applying the "
    "same constant gradient regardless of error size -- targets "
    "the 'near-miss FP' failure mode (measured: 66%% of a real "
    "trained model's FPs are within 10mm of a true CP but outside "
    "the 5mm match radius)")
    ap .add_argument ("--offset-huber-beta",type =float ,default =0.01 ,
    dest ="offset_huber_beta",
    help ="huber transition point in NORMALISED units (mm/scale); "
    "below this, error is penalised quadratically, above it "
    "linearly (only used when --offset-loss huber)")
    ap .add_argument ("--w-dir",type =float ,default =1.0 ,
    help ="loss weight on the direction cosine term")
    ap .add_argument ("--no-offset-heat-weight",dest ="offset_heat_weight",
    action ="store_false",
    help ="disable weighting the offset/dir loss by target heat (by "
    "default the vertices that actually vote get the most accurate "
    "offsets)")
    ap .set_defaults (offset_heat_weight =True )
    ap .add_argument ("--part-weight",choices =["none","keypoints"],default ="none",
    help ="weight each part's loss by its #connection-points ('keypoints') "
    "so multi-CP parts are not down-weighted to a 1-CP part (default "
    "none)")
    # --- regularisation / optimisation ---
    ap .add_argument ("--weight-decay",type =float ,default =0.0 ,
    help ="AdamW weight decay (L2). Try 1e-4..1e-2 on this small corpus to "
    "fight overfitting (0 = plain Adam)")
    ap .add_argument ("--knn-dropout",type =float ,default =0.0 ,
    help ="knngraph: dropout before the output head (0 = off; try 0.1-0.3)")
    ap .add_argument ("--grad-clip",type =float ,default =0.0 ,
    help ="clip gradient L2 norm before opt.step (0 = off; ~1-5 tames the "
    "occasional large EdgeConv gradient with batch=1)")
    ap .add_argument ("--lr-schedule",choices =["step","plateau","cosine"],
    default ="step",
    help ="LR schedule: step (StepLR via --lr-decay-every), plateau "
    "(ReduceLROnPlateau on val F1), or cosine. Add --warmup-epochs "
    "for a linear warmup")
    ap .add_argument ("--warmup-epochs",type =int ,default =0 ,
    help ="linear LR warmup over this many epochs before the main schedule")
    ap .add_argument ("--amp",action ="store_true",
    help ="knngraph+cuda: enable mixed precision (fp16 autocast + "
    "GradScaler). OFF by default and only worth it at a very high "
    "--max-gpu-verts (>=~16k): EdgeConv is bandwidth-bound, so "
    "below the memory cliff fp16 overhead makes it SLOWER. Prefer "
    "lowering --max-gpu-verts (8000) for speed instead")
    ap .add_argument ("--eval-every",type =int ,default =10 ,
    help ="run val eval + best-checkpoint every N epochs (diffusionnet)")
    ap .add_argument ("--patience",type =int ,default =6 ,
    help ="early stop after this many stale val evals (0 disables). "
    "Default 6: with --eval-every 5 this means 30 stale epochs "
    "before stopping, which prevents running past the best "
    "checkpoint into collapse territory (seen in v7 at epoch 25)")
    ap .add_argument ("--min-delta",type =float ,default =0.0 ,dest ="min_delta",
    help ="minimum absolute improvement in --best-metric required to "
    "reset patience and overwrite best.ckpt (e.g. 0.005)")
    ap .add_argument ("--collapse-gap",type =float ,default =0.03 ,dest ="collapse_gap",
    help ="warn at final summary when last validation metric is this "
    "far below the best metric (default 0.03)")
    ap .add_argument ("--dir-sign-invariant",dest ="dir_sign_invariant",
    action ="store_true",default =False ,
    help ="use sign-invariant direction loss: treats dir and -dir as "
    "equally correct. Use when insert-axis sign is ambiguous "
    "(e.g. step_openings labels for symmetric through-holes)")
    ap .add_argument ("--dir-sign-inv-prefixes",dest ="sign_inv_prefixes",
    default ="",metavar ="P1,P2",
    help ="comma-separated part_nr prefixes whose direction loss is "
    "sign-invariant (escape hatch for corpora with untrusted "
    "insert-axis signs, e.g. 'wscaduniverse' for the OLD "
    "wscad_corpus labels). Default: none -- signed 1-cos "
    "everywhere, matching the SIGNED eval angle metric")
    ap .add_argument ("--fwd-verts",dest ="fwd_verts",type =int ,default =0 ,
    help ="per-forward vertex chunk budget on the GPU (0 = "
    "--max-gpu-verts). Raise it for memory-lean backbones to "
    "fuse several patches per launch (the T1200 path is "
    "launch-latency-bound, not memory-bound)")
    ap .add_argument ("--pin-host-memory",dest ="pin_host_memory",
    action ="store_true",default =False ,
    help ="OPT-IN: pin the per-part host tensors (t/m) so H2D copies "
    "overlap compute. OFF by default -- on WDDM (Windows) "
    "pinning thousands of tensors can corrupt the CUDA context "
    "and crash the run. Enable only on a driver where it's safe")
    # --config: load YAML/JSON defaults BEFORE parsing CLI (CLI overrides config)
    pre ,_ =ap .parse_known_args (argv )
    if hasattr (pre ,"config")and pre .config :
        cfg =_load_yaml_config (pre .config )
        # --extra-source is action="append": set_defaults would seed the CLI
        # namespace with the config's list, and any CLI --extra-source occurrence
        # then APPENDS to it (argparse append actions extend the existing default,
        # they don't replace it) -- silently violating "explicit CLI overrides
        # config" for this one flag (a config-driven sweep expecting a CLI
        # --extra-source to REPLACE the config's sources would silently train ten
        # extra data instead). If the CLI passes --extra-source at all, drop the
        # config's value so only the CLI-given sources are used.
        cli_argv =argv if argv is not None else sys .argv [1 :]
        if "extra_sources"in cfg and any (
        a =="--extra-source"or a .startswith ("--extra-source=")
        for a in cli_argv ):
            cfg .pop ("extra_sources")
            # Inject config as if they were CLI defaults (so explicit CLI still wins)
        ap .set_defaults (**cfg )
    args =ap .parse_args (argv )
    logging .basicConfig (level =logging .INFO ,format ="%(message)s")

    # checkpoint file layout: best (by val F1), last (resume point), history
    run_name =args .run_name or f"cp_{args .backbone }"
    if args .ckpt :
        best_path =args .ckpt 
        stem =os .path .splitext (args .ckpt )[0 ]
        last_path =stem +"_last.ckpt"
        history_path =stem +"_history.json"
    else :
        best_path =os .path .join (args .ckpt_dir ,run_name +"_best.ckpt")
        last_path =os .path .join (args .ckpt_dir ,run_name +"_last.ckpt")
        history_path =os .path .join (args .ckpt_dir ,run_name +"_history.json")

    resume_from =args .resume_from 
    if resume_from is None and args .resume :
        if os .path .exists (last_path ):
            resume_from =last_path 
            logger .info ("--resume: continuing from last checkpoint -> %s",last_path )
        elif os .path .exists (best_path ):
            resume_from =best_path 
            logger .warning (
            "--resume: last checkpoint not found (%s) -- falling back to BEST "
            "checkpoint (%s). WARNING: best was saved at an EARLIER epoch than "
            "last, so this re-runs already-completed epochs, resets the optimiser "
            "state, and produces duplicate history entries (the root cause of the "
            "v7 epoch-20 duplicate and subsequent collapse). If this is not "
            "intentional, copy last.ckpt from backup or start fresh.",
            last_path ,best_path )
        else :
            logger .warning ("--resume: no checkpoint at %s or %s -- starting fresh",
            last_path ,best_path )
    if resume_from :
        logger .info ("resuming from %s",resume_from )

    import time 
    t0 =time .time ()
    tr ,va ,tr_reps ,va_reps ,te ,te_reps =train_and_eval (
    args .source ,backbone =args .backbone ,epochs =args .epochs ,
    extra_sources =args .extra_sources or None ,max_bbox_mm =args .max_bbox_mm ,
    keep_prefixes =(args .keep_prefixes .split (",")if args .keep_prefixes else None ),
    val_frac =args .val_frac ,max_parts =args .max_parts ,
    device =args .device ,dist_thresh_mm =args .dist_thresh_mm ,seed =args .seed ,
    op_cache_dir =args .op_cache_dir ,max_gpu_verts =args .max_gpu_verts ,
    best_path =best_path ,last_path =last_path ,history_path =history_path ,
    save_every =args .ckpt_every ,resume_from =resume_from ,
    min_votes =args .min_votes ,heatmap_thresh =args .heatmap_thresh ,
    nms_clearance_mm =args .nms_clearance_mm ,heat_pos_weight =args .heat_pos_weight ,
    heat_loss =args .heat_loss ,focal_gamma =args .focal_gamma ,
    centernet_alpha =args .centernet_alpha ,
    lr_decay_every =args .lr_decay_every ,lr_decay_rate =args .lr_decay_rate ,
    accum_steps =args .accum_steps ,low_memory =args .low_memory ,
    eval_every =args .eval_every ,patience =args .patience ,
    augment =args .augment ,aug_rotate =args .aug_rotate ,
    aug_cube_rotate =args .aug_cube_rotate ,
    aug_jitter_frac =args .aug_jitter_frac ,aug_reflect =args .aug_reflect ,
    aug_dropout_frac =args .aug_dropout_frac ,aug_cable_frac =args .aug_cable_frac ,
    split_group =args .split_group ,test_frac =args .test_frac ,
    cp_surface_tol_frac =args .cp_surface_tol_frac ,
    neg_frac =args .neg_frac ,max_neg =args .max_neg ,
    best_metric =args .best_metric ,resume_strict =args .resume_strict ,
    snapshot_every =args .snapshot_every ,
    w_heat =args .w_heat ,w_off =args .w_off ,w_dir =args .w_dir ,
    lr =args .lr ,weight_decay =args .weight_decay ,lr_schedule =args .lr_schedule ,
    warmup_epochs =args .warmup_epochs ,grad_clip =args .grad_clip ,
    offset_heat_weight =args .offset_heat_weight ,part_weight_mode =args .part_weight ,
    knn_config ={"k":args .knn_k ,"c_width":args .knn_width ,
    "n_layers":args .knn_layers ,"global_feat":args .knn_global ,
    "dropout":args .knn_dropout ,"normals":args .knn_normals ,
    "curvature":args .knn_curvature ,
    "concavity":args .knn_concavity ,
    "edge_dist":args .knn_edge_dist },
    hp_config ={"k":args .knn_k ,
    "widths":tuple (int (v )for v in args .hp_widths .split (",")),
    "ratios":tuple (int (v )for v in args .hp_ratios .split (",")),
    "k_levels":tuple (int (v )for v in args .hp_k_levels .split (",")),
    "layers":tuple (int (v )for v in args .hp_layers .split (",")),
    "dropout":args .hp_dropout ,
    "normals":not args .hp_no_features ,
    "curvature":not (args .hp_no_features or args .hp_no_curvature ),
    "concavity":not (args .hp_no_features or args .hp_no_curvature ),
    "edge_dist":not (args .hp_no_features or args .hp_no_curvature )},
    dn_config ={"n_eig":args .dn_eig ,"c_width":args .dn_width ,
    "n_diffusion_blocks":args .dn_blocks ,"dropout":args .dn_dropout },
    print_points =args .print_points ,amp =args .amp ,
    knn_cache_dir =args .knn_cache_dir ,
    dir_sign_invariant =args .dir_sign_invariant ,
    init_from =args .init_from ,
    hard_mining =args .hard_mining ,
    offset_loss =args .offset_loss ,offset_huber_beta =args .offset_huber_beta ,
    min_delta =args .min_delta ,collapse_gap =args .collapse_gap ,
    fwd_verts =args .fwd_verts ,pin_host_memory =args .pin_host_memory ,
    sign_inv_prefixes =args .sign_inv_prefixes ,split_seed =args .split_seed ,
    family_boost =args .family_boost ,
    exclude_parts_file =args .exclude_parts_file )
    elapsed =time .time ()-t0 
    _print_agg ("TRAIN metrics",tr ,tr_reps )
    _print_agg ("VAL metrics",va ,va_reps )
    if te :
        _print_agg ("TEST metrics (held-out, never tuned)",te ,te_reps )
    print ("\nelapsed: %.0f s (%.2f h)"%(elapsed ,elapsed /3600 ))

    print ("\ncheckpoints:")
    for tag ,p in (("best",best_path ),("last",last_path ),
    ("history",history_path )):
        print ("  %-7s %s%s"%(tag ,p ,
        ""if os .path .exists (p )else "  (not written)"))
    print ("\nto continue this run later (same or another machine):")
    print ("  # checkpoints are BINARY -- share via Git LFS or a drive/artifact store,")
    print ("  # NOT plain `git add` (every epoch rewrites last.ckpt and bloats history).")
    print ("  # one-time LFS setup:  git lfs install && git lfs track '*.ckpt'")
    print ("  #   (commit .gitattributes; then add/commit/push the .ckpt as usual)")
    print ("  # then on the other machine: pull the checkpoints and run:")
    print ("  python train_cp.py %s --backbone %s --resume --epochs <target>"
    %(args .source ,args .backbone ))

    if args .out :
        with open (args .out ,"w")as fh :
            json .dump ({"train_aggregate":tr ,"val_aggregate":va ,
            "test_aggregate":te ,
            "train_per_part":tr_reps ,"val_per_part":va_reps ,
            "test_per_part":te_reps ,
            "elapsed_sec":elapsed },fh ,indent =2 )
        print ("\nmetrics written to",args .out )

    if args .export_model :
    # bundle the decode operating point used for this run so the exported
    # model is self-contained (the weights alone are ambiguous for keypoints)
        decode ={"heatmap_thresh":args .heatmap_thresh ,"min_votes":args .min_votes ,
        "nms_clearance_mm":args .nms_clearance_mm ,
        "dist_thresh_mm":args .dist_thresh_mm }
        src =best_path if os .path .exists (best_path )else last_path 
        if os .path .exists (src ):
            cpr .export_inference_checkpoint (src ,args .export_model ,decode =decode )
            print ("\nexported inference model -> %s\n  (from %s, decode=%s)"
            %(args .export_model ,src ,decode ))
            print ("  run it:  python predict.py %s <mesh-or-corpus> --out preds.json"
            %args .export_model )
        else :
            print ("\n--export-model: no checkpoint found to export (%s)"%src )


if __name__ =="__main__":
    main ()
