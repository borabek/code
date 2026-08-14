"""Snapshot-ensemble the WSCAD->CP model. FULL-COVERAGE lever, no retraining.

The run saves never-overwritten snapshots (snapshot_every=10): ep0009, ep0019,
ep0029, ep0039 ... plus best.ckpt. Different epochs make DIFFERENT errors, so
averaging their per-vertex (heat, offset, direction) outputs cancels uncorrelated
noise and sharpens the consensus -- a classic ensemble, run purely at inference.

The averaging is legitimate because every member is the same architecture ten the
same input: infer_knngraph's spatial patching is deterministic for a given part and
max_gpu_verts, so all members produce an (N, C) array over the SAME vertices in the
SAME frame. We average heat directly (scalar), and offset/direction directly (they
are already in the part's own frame), then decode ONCE.

Run AFTER training. Compares the ensemble to the single best-checkpoint baseline.
Usage:
  python ensemble_snapshots.py --members checkpoints/cp_hp_v31_ftc6_best.ckpt \
      checkpoints/cp_hp_v31_ftc6_best_ep0029.ckpt checkpoints/cp_hp_v31_ftc6_best_ep0039.ckpt \
      --parts _val_parts.txt --thr 0.30
"""
import argparse ,json ,os 
import numpy as np 
import cp_regressor as cpr ,cp_targets as ct ,json_dataset as jd ,metrics as mcp 
import train_cp as tc 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--members",nargs ="+",required =True ,
    help ="checkpoint paths; the FIRST is also the single-model baseline")
    ap .add_argument ("--parts",default ="_val_parts.txt")
    ap .add_argument ("--corpus",default ="wscad_corpus_v8")
    ap .add_argument ("--thr",type =float ,default =0.30 )
    ap .add_argument ("--device",default ="cuda")
    ap .add_argument ("--out",default ="_ensemble_result.json")
    a =ap .parse_args ()

    val =[l .strip ()for l in open (a .parts ,encoding ="utf-8")if l .strip ()]
    members =[]
    for c in a .members :
        if not os .path .exists (c ):
            print (f"  UYARI: {c } none, atlaniyor");continue 
        m ,meta ,_ =cpr .load_model (c ,device =a .device )
        members .append ((c ,m ,meta ))
    print (f"ensemble uyeleri: {len (members )}  |  baseline = {os .path .basename (a .members [0 ])}")
    dedup =tc ._nms_radius (None ,5.0 )

    def infer_arr (model ,meta ,Vn ,scale ,pn ):
        return cpr .infer_knngraph (model ,meta ,Vn ,device =a .device ,max_gpu_verts =14000 ,
        offset_scale =scale ,patch =cpr .is_patch_part (pn ),part_nr =pn )

    base =dict (tp =0 ,fp =0 ,fn =0 )
    ens =dict (tp =0 ,fp =0 ,fn =0 )
    per_part =[]
    for k ,pn in enumerate (val ,1 ):
        f =os .path .join (a .corpus ,pn +".json")
        if not os .path .exists (f ):
            continue 
        p =next (iter (jd .iter_parts (f )))
        _ ,gt ,gd =jd .dedup_connection_points (p )
        if not len (gt ):
            continue 
        V =np .asarray (p .vertices ,float )
        Vn ,_ ,scale =cpr .normalize_vertices (V )

        arrs =[infer_arr (m ,meta ,Vn ,scale ,pn )for _ ,m ,meta in members ]
        # all arrays share shape (same vertices, deterministic patching); average
        shp =min (x .shape [0 ]for x in arrs )# guard against a rare size mismatch
        arrs =[x [:shp ]for x in arrs ]
        arr_ens =np .mean (arrs ,axis =0 )

        # baseline = first member alone
        pb =ct .decode_predictions (V ,arrs [0 ],heatmap_thresh =a .thr ,
        nms_radius_mm =dedup ,min_votes =1 )
        rb =mcp .keypoint_report (pb ,gt ,gd ,dist_thresh_mm =5.0 )
        base ["tp"]+=rb ["tp"];base ["fp"]+=rb ["fp"];base ["fn"]+=rb ["fn"]

        pe =ct .decode_predictions (V ,arr_ens ,heatmap_thresh =a .thr ,
        nms_radius_mm =dedup ,min_votes =1 )
        re =mcp .keypoint_report (pe ,gt ,gd ,dist_thresh_mm =5.0 )
        ens ["tp"]+=re ["tp"];ens ["fp"]+=re ["fp"];ens ["fn"]+=re ["fn"]
        per_part .append ((pn ,rb ["tp"],rb ["fp"],rb ["fn"],re ["tp"],re ["fp"],re ["fn"]))
        if k %30 ==0 :
            print (f"  {k }/{len (val )} parts...",flush =True )

    def jac (d ):return d ["tp"]/(d ["tp"]+d ["fp"]+d ["fn"])if d ["tp"]else 0 
    def f1 (d ):return 2 *d ["tp"]/(2 *d ["tp"]+d ["fp"]+d ["fn"])if d ["tp"]else 0 
    print (f"\n=== SNAPSHOT ENSEMBLE ({len (members )} uye) ===")
    print (f"  BASELINE (tek): TP={base ['tp']} FP={base ['fp']} FN={base ['fn']}  "
    f"Jaccard={jac (base ):.4f} F1={f1 (base ):.4f}")
    print (f"  ENSEMBLE      : TP={ens ['tp']} FP={ens ['fp']} FN={ens ['fn']}  "
    f"Jaccard={jac (ens ):.4f} F1={f1 (ens ):.4f}")
    print (f"  DELTA Jaccard: {jac (ens )-jac (base ):+.4f}")
    json .dump ({"base":base ,"ens":ens ,"per_part":per_part },open (a .out ,"w"))
    print ("DONE")


if __name__ =="__main__":
    main ()
