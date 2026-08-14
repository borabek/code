#!/usr/bin/env python
"""Train the connection-point detector ten the FULL 479-part corpus within a
wall-clock budget (default 60 min), then stop cleanly.

Why a wrapper instead of just `python train_cp.py ...`:
train_cp has no time limit -- you pick an epoch count and hope it fits. Here we
set a high epoch ceiling and let the CLOCK decide when to stop. The underlying
trainer writes <run>_best.ckpt / <run>_last.ckpt EVERY epoch and flushes its
history ten Ctrl-C, so interrupting it mid-run loses at most the current epoch;
the best-by-val-F1 checkpoint is always safe ten disk.

Backbone: knngraph -- runs natively ten Windows + CUDA (no WSL, no eigenbasis
precompute), which is the only backbone that reliably fits a hard time cap ten
the T1200. Big meshes are subsampled to --max-gpu-verts so the 4 GB GPU never
OOMs (it falls back to CPU per-part if it ever does).

Usage (defaults are tuned for this machine; override any of them):
    python train_1h.py
    python train_1h.py --resume                 # continue the previous hour's run
    python train_1h.py --minutes 90 --augment 4
    python train_1h.py --source "C:/path/to/other/corpus"
"""
import argparse 
import os 
import signal 
import subprocess 
import sys 
import time 

HERE =os .path .dirname (os .path .abspath (__file__ ))
DEFAULT_SOURCE =r"C:\Users\DE00024082\Desktop\JSON"# the real 479-part corpus


def main ():
    ap =argparse .ArgumentParser (description =__doc__ ,
    formatter_class =argparse .RawDescriptionHelpFormatter )
    ap .add_argument ("--source",default =DEFAULT_SOURCE ,
    help ="corpus dir (default: the 479-part JSON folder)")
    ap .add_argument ("--minutes",type =float ,default =60.0 ,
    help ="wall-clock budget in minutes (default 60)")
    ap .add_argument ("--device",default ="cuda",
    help ="cuda (default) or cpu")
    ap .add_argument ("--max-gpu-verts",type =int ,default =7000 ,
    help ="subsample cap. CRITICAL on the 4 GB T1200: the EdgeConv backward "
    "hits a memory cliff AT ~8000 verts (profile_knn.py: bwd 505ms @8000 "
    "vs 92ms @7400), so 8000 sat right ON the cliff and gave only ~23 "
    "epochs/hr. 7000 stays just under it -> ~66 epochs/hr at the SAME "
    "1.0%% recall ceiling (18 GT points at risk either way). Median part "
    "is 5511 verts, so 7000 leaves most parts untouched and only thins "
    "the biggest third. 6000 -> ~71 epochs/hr; do NOT go back to 8000+")
    ap .add_argument ("--epochs",type =int ,default =1000 ,
    help ="epoch CEILING -- the clock normally stops first (default 1000)")
    ap .add_argument ("--augment",type =int ,default =4 ,
    help ="augmented clones per training part. ON by default (4): the raw-xyz "
    "knngraph is pose-overfit without it ([[cp-training-augmentation]]). "
    "Set 0 to disable. Each clone slows the epoch, so you fit fewer/hr")
    ap .add_argument ("--aug-reflect",action ="store_true",default =True ,
    help ="also mirror clones across a random axis (connector geometry is "
    "often mirror-symmetric). On by default with --augment")
    ap .add_argument ("--no-aug-reflect",dest ="aug_reflect",action ="store_false",
    help ="disable reflection augmentation")
    ap .add_argument ("--weight-decay",type =float ,default =5e-4 ,
    help ="AdamW weight decay (default 5e-4 -- regularisation on the small "
    "corpus to fight the val-F1 bounce/overfit; was 1e-4). 0 = plain "
    "Adam. Only takes effect on a FRESH run (a --resume restores the "
    "optimizer's stored weight_decay)")
    ap .add_argument ("--grad-clip",type =float ,default =5.0 ,
    help ="clip gradient norm before opt.step (default 5; 0 = off)")
    ap .add_argument ("--knn-dropout",type =float ,default =0.1 ,
    help ="dropout before the output head (default 0.1). In-network "
    "regularisation for the small corpus -- dampens the val-F1 bounce. "
    "FRESH run only: a --resume rebuilds the architecture from the "
    "checkpoint's stored dropout value")
    ap .add_argument ("--patience",type =int ,default =12 ,
    help ="early stop after this many STALE val evals (val runs every 3 "
    "epochs, so 12 ~= 36 epochs with no new best). 0 disables. The best "
    "checkpoint is kept regardless, so this only saves clock time")
    ap .add_argument ("--run-name",default ="cp_knn_1h")
    ap .add_argument ("--resume",action ="store_true",
    help ="continue the previous run: loads checkpoints/<run-name>_last.ckpt "
    "and keeps training toward --epochs (default: start fresh, which "
    "OVERWRITES any existing checkpoints for this run-name)")
    ap .add_argument ("--accum-steps",type =int ,default =8 ,
    help ="gradient accumulation: average the gradient over this many parts "
    "before opt.step (default 8 = effective batch 8). The model trains "
    "one part at a time, so accum=1 is pure batch-1 SGD -- one odd part "
    "can swing the weights and crash val F1. Accumulating smooths the "
    "update (~180 steps/epoch at 1445 parts). 1 = old per-part behaviour")
    ap .add_argument ("--knn-k",type =int ,default =16 )
    ap .add_argument ("--lr-decay-every",type =int ,default =12 ,
    help ="StepLR step size in epochs (default 12; 0 disables). A flat LR "
    "makes the late-stage val F1 oscillate/crash because the weights "
    "keep taking big steps near the optimum -- decaying it halves the "
    "step every 12 epochs so val settles instead of bouncing (was 20; "
    "tightened after a single high-LR epoch halved val F1). FRESH run "
    "only: a --resume restores the scheduler's stored step size")
    ap .add_argument ("--lr-decay-rate",type =float ,default =0.5 ,
    help ="multiply LR by this at each decay step (default 0.5)")
    ap .add_argument ("--no-global",action ="store_true",
    help ="drop the global-context feature (on by default here)")
    ap .add_argument ("--split-group",choices =["none","prefix","geometry"],
    default ="geometry",
    help ="leakage-safe train/val split (default 'geometry', groups "
    "near-identical meshes by actual geometry -- naming-independent, "
    "and the only mode proven safe for wscad+PXC, see project notes "
    "on 'prefix' breaking on those). 'prefix' groups ABB part-family "
    "variants by stripped PartNr; 'none' = per-PartNr (optimistic)")
    ap .add_argument ("--test-frac",type =float ,default =0.15 ,
    help ="frozen held-out TEST split (default 0.15) -- never trains or "
    "tunes; gives one honest final number alongside val. 0 = off")
    ap .add_argument ("--snapshot-every",type =int ,default =0 ,
    help ="also keep a never-overwritten <run>_ep<N>.ckpt every N epochs "
    "so an earlier good model survives if a later best overfits "
    "(0 = off; only best+last kept)")
    ap .add_argument ("--cp-surface-tol-frac",type =float ,default =None ,
    help ="drop CPs farther than this fraction of the bbox diagonal from "
    "the mesh. Default OFF (keep all): only ~1.8%% of CPs are >10%% "
    "off-surface and they're consistent (recessed terminals), which "
    "the offset channel learns fine. Set 0.10 to A/B-test dropping "
    "the worst ~1.8%%; 0.05 drops ~13%% (likely too aggressive)")
    args =ap .parse_args ()

    if not os .path .isdir (args .source ):
        sys .exit (f"corpus not found: {args .source }")

    cmd =[
    sys .executable ,os .path .join (HERE ,"train_cp.py"),args .source ,
    "--backbone","knngraph",
    "--device",args .device ,
    "--max-gpu-verts",str (args .max_gpu_verts ),
    "--epochs",str (args .epochs ),
    "--val-frac","0.2",
    "--augment",str (args .augment ),
    "--knn-k",str (args .knn_k ),
    "--lr-decay-every",str (args .lr_decay_every ),
    "--lr-decay-rate",str (args .lr_decay_rate ),
    "--weight-decay",str (args .weight_decay ),
    "--grad-clip",str (args .grad_clip ),
    "--knn-dropout",str (args .knn_dropout ),# head dropout (small-corpus regulariser)
    "--patience",str (args .patience ),# early stop ten stale val (best kept)
    "--accum-steps",str (args .accum_steps ),# effective mini-batch (stabler than batch-1)
    "--heat-loss","centernet",# right loss for the sparse CP heatmap
    "--min-votes","1",# GT ceiling: R=0.94 @1 vs 0.79 @2
    "--split-group",args .split_group ,# leakage-safe split (no family bleed)
    "--test-frac",str (args .test_frac ),# frozen held-out -> one honest number
    "--snapshot-every",str (args .snapshot_every ),
    ]
    if args .cp_surface_tol_frac is not None :
        cmd +=["--cp-surface-tol-frac",str (args .cp_surface_tol_frac )]
    cmd +=[
    "--eval-every","3",# val + best-ckpt refresh every 3 epochs
    "--ckpt-every","1",# last.ckpt always current -> safe to kill
    "--run-name",args .run_name ,
    "--out",f"results_{args .run_name }.json",
    "--export-model",f"model_{args .run_name }.pt",
    "--print-points",
    ]
    if not args .no_global :
        cmd .append ("--knn-global")
    if args .aug_reflect and args .augment :
        cmd .append ("--aug-reflect")
    if args .resume :
        cmd .append ("--resume")

    budget_s =args .minutes *60.0 
    print (f"[train_1h] budget {args .minutes :.0f} min | corpus {args .source }")
    print (f"[train_1h] {' '.join (cmd )}\n",flush =True )

    # New process group so we can deliver Ctrl-Break -> the trainer raises
    # KeyboardInterrupt, flushes history, and exits with best/last intact.
    creationflags =subprocess .CREATE_NEW_PROCESS_GROUP if os .name =="nt"else 0 
    t0 =time .time ()
    proc =subprocess .Popen (cmd ,cwd =HERE ,creationflags =creationflags )

    try :
        while proc .poll ()is None :
            if time .time ()-t0 >=budget_s :
                print (f"\n[train_1h] {args .minutes :.0f} min reached -- stopping cleanly "
                f"(best/last checkpoints are saved)",flush =True )
                _interrupt (proc )
                break 
            time .sleep (2.0 )
    except KeyboardInterrupt :
        print ("\n[train_1h] Ctrl-C -- forwarding to trainer",flush =True )
        _interrupt (proc )

    try :
        proc .wait (timeout =120 )
    except subprocess .TimeoutExpired :
        proc .kill ()

    mins =(time .time ()-t0 )/60.0 
    print (f"\n[train_1h] done in {mins :.1f} min")
    print (f"[train_1h] best checkpoint : checkpoints/{args .run_name }_best.ckpt")
    print (f"[train_1h] inference model : model_{args .run_name }.pt")
    print (f"[train_1h] metrics         : results_{args .run_name }.json  "
    f"(see test_aggregate for the held-out number)")
    print ("[train_1h] lock the decode operating point honestly (held-out, not "
    "selected-on-val):")
    print (f"  python sweep_decode.py checkpoints/{args .run_name }_best.ckpt "
    f"\"{args .source }\" --split-group {args .split_group } --holdout-frac 0.3 "
    f"--min-precision 0.5")


def _interrupt (proc ):
    """Ask the child to stop gracefully (Ctrl-Break ten Windows, SIGINT elsewhere)."""
    try :
        if os .name =="nt":
            proc .send_signal (signal .CTRL_BREAK_EVENT )
        else :
            proc .send_signal (signal .SIGINT )
    except Exception :
        proc .terminate ()


if __name__ =="__main__":
    main ()
