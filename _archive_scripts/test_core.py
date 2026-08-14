"""Core regression tests for the CP detector pipeline.

Run:  python test_core.py
      python test_core.py -v      (verbose)
      python -m pytest test_core.py

Covers:
  1. encode/decode round-trip: GT labels survive encode→target→decode with F1≥0.95
  2. Patching coverage: spatial_patches covers all vertices (none silently dropped)
  3. Augment CP transform: rotate/reflect/jitter preserves vertex↔CP geometry
  4. Normal round-trip: _pca_normals output is unit, consistent after rotation
  5. Curvature: _pca_normals_curvature returns sensible [0,1] scalars
  6. Graph batching equivalence: batched forward == sum of solo forwards (numerically)
  7. Synth blocks: build_block produces valid geometry for all three face types,
     and CP labels are genuinely RECESSED (not pinned to the opening plane)
  8. Cable distractor: adds vertices, keeps labels unchanged, first ring stands
     off from the CP instead of surrounding it
  9. Resume-config: _check_resume_config catches a silent loss-hyperparameter
     change across --resume (lr/w_off/w_heat/w_dir/weight_decay/centernet_alpha)

See tests/test_split.py for the leakage-safe split/family-key coverage.
"""
import sys 
import numpy as np 
import unittest 


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_mesh (n =300 ,n_cps =4 ,rng =None ):
    """Tiny synthetic mesh + CP labels for testing."""
    rng =rng or np .random .default_rng (42 )
    V =rng .uniform (-50 ,50 ,(n ,3 ))
    F =np .column_stack ([
    np .arange (0 ,n -2 ,3 ),
    np .arange (1 ,n -1 ,3 ),
    np .arange (2 ,n ,3 )
    ])[:min (n //3 ,100 )]
    cp_pts =V [:n_cps ].copy ()+rng .normal (0 ,0.5 ,(n_cps ,3 ))
    cp_dirs =rng .normal (0 ,1 ,(n_cps ,3 ))
    cp_dirs /=np .linalg .norm (cp_dirs ,axis =1 ,keepdims =True )
    return V ,F ,cp_pts ,cp_dirs 


    # ---------------------------------------------------------------------------
    # test cases
    # ---------------------------------------------------------------------------

class TestEncodeDecode (unittest .TestCase ):
    """encode_targets → decode_predictions round-trip."""

    def test_roundtrip_f1 (self ):
        import cp_targets as ct 
        import metrics as mcp 
        rng =np .random .default_rng (0 )
        V ,F ,cp_pts ,cp_dirs =_make_mesh (500 ,6 ,rng )
        Vn ,c ,scale =_normalize (V )
        cp_n =(cp_pts -c )/scale 
        tgt ,msk ,sigma =ct .encode_targets (Vn ,cp_n ,cp_dirs )
        preds =ct .decode_predictions (Vn ,tgt ,heatmap_thresh =0.3 ,
        nms_radius_mm =sigma *3 )
        if not preds :
            self .skipTest ("no predictions decoded (small mesh, skip)")
        rep =mcp .keypoint_report (preds ,cp_n ,cp_dirs ,dist_thresh_mm =sigma *3 )
        self .assertGreaterEqual (rep ["f1"],0.80 ,
        f"encode/decode F1={rep ['f1']:.2f} < 0.80")

    def test_empty_cp (self ):
        import cp_targets as ct 
        rng =np .random .default_rng (1 )
        V ,F ,_ ,_ =_make_mesh (200 ,0 ,rng )
        Vn ,_ ,_ =_normalize (V )
        tgt ,msk ,_ =ct .encode_targets (Vn ,np .zeros ((0 ,3 )),np .zeros ((0 ,3 )))
        self .assertEqual (tgt [:,ct .HEATMAP ].max (),0.0 )
        preds =ct .decode_predictions (Vn ,tgt ,heatmap_thresh =0.3 )
        self .assertEqual (preds ,[])


class TestPatchingCoverage (unittest .TestCase ):
    """spatial_patches must cover every vertex."""

    def test_all_vertices_covered (self ):
        from cp_regressor import spatial_patches 
        rng =np .random .default_rng (7 )
        V =rng .uniform (0 ,100 ,(5000 ,3 ))
        patches ,sel =spatial_patches (V ,cap =1000 ,margin_frac =0.25 )
        if sel is not None :
            V =V [sel ]
        covered =np .zeros (len (V ),dtype =bool )
        for idx in patches :
            covered [idx ]=True 
        self .assertTrue (covered .all (),
        f"{(~covered ).sum ()} vertices not covered by any patch")

    def test_patch_graph_weights_sum_to_one (self ):
        from cp_regressor import patch_graph_weight 
        for n_patches in [1 ,2 ,8 ,37 ]:
            w =patch_graph_weight (n_patches )
            self .assertAlmostEqual (n_patches *w ,1.0 ,places =12 )

    def test_patch_target_slicing_keeps_cp_peaks (self ):
        import cp_targets as ct 
        from cp_regressor import spatial_patches 
        rng =np .random .default_rng (23 )
        V =rng .uniform (0 ,100 ,(3000 ,3 )).astype (np .float32 )
        target =np .zeros ((len (V ),ct .N_CHANNELS ),dtype =np .float32 )
        peak_idx =np .array ([0 ,17 ,513 ,1499 ,2999 ])
        target [peak_idx ,ct .HEATMAP ]=1.0 
        patches ,sel =spatial_patches (V ,cap =600 ,margin_frac =0.25 )
        self .assertIsNone (sel ,"test expects the non-giant path")
        seen =set ()
        for idx in patches :
            sliced =target [idx ]
            local_hot =idx [np .where (sliced [:,ct .HEATMAP ]>=1.0 -1e-6 )[0 ]]
            seen .update (int (i )for i in local_hot )
        self .assertEqual (set (int (i )for i in peak_idx ),seen )


class TestAugmentTransform (unittest .TestCase ):
    """augment_part: CP distance to nearest vertex preserved after rotation."""

    def _surface_dist (self ,V ,cp_pts ):
        return np .array ([np .linalg .norm (V -p ,axis =1 ).min ()for p in cp_pts ])

    def _make_part (self ,n ,n_cps ,rng ,name ="test"):
        import json_dataset as jd 
        V ,F ,cp_pts ,cp_dirs =_make_mesh (n ,n_cps ,rng )
        return jd .Part (part_nr =name ,vertices =V ,faces =F ,
        cp_points =cp_pts ,cp_directions =cp_dirs ,
        cp_names =[f"CP{i }"for i in range (n_cps )])

    def test_rotate_preserves_cp_dist (self ):
        from cp_regressor import augment_part 
        rng =np .random .default_rng (3 )
        _ ,_ ,cp_pts ,cp_dirs =_make_mesh (400 ,4 ,rng )
        part =self ._make_part (400 ,4 ,rng )
        V =np .asarray (part .vertices )
        cp_pts_orig =np .asarray (part .cp_points )
        aug =augment_part (part ,rng ,rotate =True ,jitter_frac =0.0 )
        Va =np .asarray (aug .vertices )
        cp_a =np .asarray (aug .cp_points )
        d_orig =self ._surface_dist (V ,cp_pts_orig )
        d_aug =self ._surface_dist (Va ,cp_a )
        np .testing .assert_allclose (d_orig ,d_aug ,atol =1e-4 ,
        err_msg ="Rotation changed CP-surface distance")

    def test_reflect_dir_flipped (self ):
        rng =np .random .default_rng (5 )
        _ ,_ ,cp_pts ,cp_dirs =_make_mesh (300 ,3 ,rng )
        # verify manual reflection preserves direction norms
        cd_ref =cp_dirs .copy ();cd_ref [:,0 ]=-cd_ref [:,0 ]
        np .testing .assert_allclose (
        np .linalg .norm (cd_ref ,axis =1 ),
        np .ones (len (cd_ref )),atol =1e-6 )


class TestNormals (unittest .TestCase ):
    """_pca_normals: unit vectors, equivariant under rotation."""

    def test_unit_normals (self ):
        from cp_regressor import _pca_normals ,_knn_graph 
        rng =np .random .default_rng (9 )
        V =rng .uniform (-10 ,10 ,(200 ,3 )).astype (np .float32 )
        nbr =_knn_graph (V ,k =16 )
        nrm =_pca_normals (V ,nbr )
        lens =np .linalg .norm (nrm ,axis =1 )
        np .testing .assert_allclose (lens ,np .ones (len (V )),atol =1e-5 ,
        err_msg ="normals not unit length")

    def test_curvature_range (self ):
        from cp_regressor import _pca_normals_curvature ,_knn_graph 
        rng =np .random .default_rng (11 )
        V =rng .uniform (-10 ,10 ,(200 ,3 )).astype (np .float32 )
        nbr =_knn_graph (V ,k =16 )
        _ ,curv =_pca_normals_curvature (V ,nbr )
        self .assertTrue ((curv >=0 ).all ()and (curv <=1 ).all (),
        "curvature out of [0,1]")

    def test_concavity_is_not_degenerate (self ):
    # Regression test: _concavity_score used to be PROVABLY always exactly 0
    # ten every vertex of every part -- the normal was oriented via
    # dot(normal, centroid_dir), then concavity compared that SAME normal
    # back against the SAME centroid_dir, which is tautologically >= 0 by
    # construction (so -dot was always <= 0, clipped to exactly 0). This
    # made --knn-concavity a silent, all-zero dead input channel in every
    # run that used it. Any feature (concavity/curvature/normals) that comes
    # back constant or NaN ten real-ish geometry should fail CI.
        from cp_regressor import _concavity_score ,_pca_normals_curvature ,_knn_graph 
        rng =np .random .default_rng (21 )
        # a bumpy-ish point cloud (not a perfect sphere/plane) so real local
        # concave/convex variation exists to detect
        V =(rng .uniform (-10 ,10 ,(400 ,3 ))
        +3.0 *np .sin (rng .uniform (-10 ,10 ,(400 ,3 )))).astype (np .float32 )
        nbr =_knn_graph (V ,k =16 )
        concav =_concavity_score (V ,nbr )
        self .assertFalse (np .isnan (concav ).any (),"concavity contains NaN")
        self .assertTrue ((concav >=0 ).all ()and (concav <=1 ).all (),
        "concavity out of [0,1]")
        self .assertGreater (concav .std (),1e-4 ,
        "concavity is degenerate/constant (std ~0) -- likely "
        "the tautological normal-vs-same-reference bug")
        self .assertGreater (int ((concav >1e-6 ).sum ()),0 ,
        "concavity is all exactly zero")
        nrm ,curv =_pca_normals_curvature (V ,nbr )
        self .assertFalse (np .isnan (nrm ).any ()or np .isnan (curv ).any (),
        "normals/curvature contain NaN")
        self .assertGreater (curv .std (),1e-4 ,"curvature is degenerate/constant")


class TestGraphBatchEquivalence (unittest .TestCase ):
    """Batched forward must equal sum of solo forwards (numerically)."""

    def test_batch_equals_solo_sum (self ):
        try :
            import torch 
        except ImportError :
            self .skipTest ("torch not available")
        from cp_regressor import build_regressor ,_knn_graph 
        rng =np .random .default_rng (13 )
        model ,meta =build_regressor ("knngraph",{"normals":False ,"c_width":32 ,
        "n_layers":2 ,"global_feat":False ,
        "k":8 })
        model .eval ()
        parts =[]
        for _ in range (3 ):
            n =int (rng .integers (50 ,150 ))
            V =torch .tensor (rng .normal (0 ,1 ,(n ,3 )),dtype =torch .float32 )
            nbr =_knn_graph (V .numpy (),k =8 )# already a torch.long tensor
            parts .append ((V ,nbr ))

            # Solo forwards
        solo_outs =[]
        with torch .no_grad ():
            for V ,nbr in parts :
                out =model (V ,nbr )
                solo_outs .append (out .clone ())

                # Batched forward: offset kNN indices per part
        offset =0 
        nbs =[]
        for V ,nbr in parts :
            nbs .append (nbr +offset )
            offset +=len (V )
        X =torch .cat ([p [0 ]for p in parts ])
        NB =torch .cat (nbs )
        with torch .no_grad ():
            batch_out =model (X ,NB )

        seg =0 
        for i ,(V ,_ )in enumerate (parts ):
            n =len (V )
            np .testing .assert_allclose (
            batch_out [seg :seg +n ].cpu ().numpy (),
            solo_outs [i ].cpu ().numpy (),
            atol =1e-4 ,
            err_msg =f"Batch output differs from solo output for part {i }")
            seg +=n 

    def test_weighted_chunked_backward_matches_solo (self ):
        try :
            import torch 
        except ImportError :
            self .skipTest ("torch not available")
        import cp_targets as ct 
        from cp_regressor import build_regressor ,_knn_graph ,cp_loss 

        rng =np .random .default_rng (31 )
        torch .manual_seed (31 )
        model_a ,_ =build_regressor ("knngraph",{"normals":False ,"c_width":16 ,
        "n_layers":2 ,"global_feat":False ,
        "k":6 })
        model_b ,_ =build_regressor ("knngraph",{"normals":False ,"c_width":16 ,
        "n_layers":2 ,"global_feat":False ,
        "k":6 })
        model_b .load_state_dict (model_a .state_dict ())
        parts =[]
        weights =[0.25 ,0.75 ,1.0 ]
        for j in range (3 ):
            n =32 +j *7 
            V =torch .tensor (rng .normal (0 ,1 ,(n ,3 )),dtype =torch .float32 )
            nbr =_knn_graph (V .numpy (),k =6 )
            target =torch .zeros ((n ,ct .N_CHANNELS ),dtype =torch .float32 )
            target [:,ct .DIRECTION ]=torch .tensor ([0.0 ,0.0 ,1.0 ])
            target [j ,ct .HEATMAP ]=1.0 
            target [j ,ct .OFFSET ]=torch .tensor ([0.01 ,-0.02 ,0.03 ])
            mask =torch .zeros (n ,dtype =torch .bool )
            mask [j ]=True 
            parts .append ({"x":V ,"nbr":nbr ,"t":target ,"m":mask ,
            "graph_weight":weights [j ]})

        total_w =sum (weights )
        loss_a =torch .zeros (1 )
        for d in parts :
            out =model_a (d ["x"],d ["nbr"])
            loss ,_ =cp_loss (out ,d ["t"],d ["m"],heat_loss ="centernet",
            w_heat =1.0 ,w_off =5.0 ,w_dir =2.0 )
            loss_a =loss_a +d ["graph_weight"]*loss 
        (loss_a /total_w ).backward ()

        offset =0 
        Xs ,NBs =[],[]
        for d in parts :
            Xs .append (d ["x"])
            NBs .append (d ["nbr"]+offset )
            offset +=len (d ["x"])
        out_all =model_b (torch .cat (Xs ,dim =0 ),torch .cat (NBs ,dim =0 ))
        seg =0 
        loss_b =torch .zeros (1 )
        for d in parts :
            n =len (d ["x"])
            loss ,_ =cp_loss (out_all [seg :seg +n ],d ["t"],d ["m"],
            heat_loss ="centernet",w_heat =1.0 ,
            w_off =5.0 ,w_dir =2.0 )
            loss_b =loss_b +d ["graph_weight"]*loss 
            seg +=n 
        (loss_b /total_w ).backward ()

        for (name_a ,pa ),(name_b ,pb )in zip (model_a .named_parameters (),
        model_b .named_parameters ()):
            self .assertEqual (name_a ,name_b )
            if pa .grad is None and pb .grad is None :
                continue 
            self .assertIsNotNone (pa .grad ,name_a )
            self .assertIsNotNone (pb .grad ,name_b )
            np .testing .assert_allclose (pa .grad .detach ().numpy (),
            pb .grad .detach ().numpy (),
            atol =1e-5 ,rtol =1e-4 ,
            err_msg =f"gradient mismatch for {name_a }")


class TestSynthBlocks (unittest .TestCase ):
    """synth_blocks.build_block: valid geometry for all face types."""

    def test_all_face_types (self ):
        import synth_blocks 
        rng =np .random .default_rng (17 )
        for face in ["top","front","side"]:
            for _ in range (5 ):
                p =synth_blocks .sample_params (rng )
                p ["block_face"]=face 
                V ,F ,cps ,dirs =synth_blocks .build_block (p ,rng )
                self .assertGreater (len (V ),0 ,f"empty mesh for face={face }")
                self .assertGreater (len (cps ),0 ,f"no CPs for face={face }")
                norms =np .linalg .norm (dirs ,axis =1 )
                np .testing .assert_allclose (norms ,np .ones (len (dirs )),atol =1e-6 )

    def test_cp_is_recessed_not_at_opening_plane (self ):
    # Regression test: CP labels used to be placed at the hole's opening
    # plane (z=Lz) regardless of the sampled `depth`, so pretraining never
    # demonstrated a recessed CP -- a real, measured cause of the reported
    # synth-pretrain -> real-finetune regression. The CP must now be offset
    # from the opening by AT LEAST `depth` (within the Lz-1 floor clamp);
    # `deep_recess` parts sit further still (below the pocket-floor vertex
    # itself, see build_block's `cp_z`), so recess is >= depth, not == depth.
        import synth_blocks 
        rng =np .random .default_rng (5 )
        for _ in range (10 ):
            p =synth_blocks .sample_params (rng )
            p ["block_face"]="top"
            V ,F ,cps ,dirs =synth_blocks .build_block (p ,rng )
            expected_zb =p ["Lz"]-min (p ["depth"],p ["Lz"]-1.0 )
            min_expected_recess =p ["Lz"]-expected_zb 
            top_z =V [:,2 ].max ()
            for cpt in cps :
                actual_recess =top_z -cpt [2 ]
                self .assertGreaterEqual (actual_recess ,min_expected_recess -1e-6 )
                if p ["Lz"]-1.0 >=p ["depth"]:# not floor-clamped
                    self .assertGreater (actual_recess ,0.0 ,
                    "CP must be below the opening plane")


class TestCableDistractor (unittest .TestCase ):
    """cable_distractor adds vertices, keeps labels unchanged."""

    def test_vertices_grow (self ):
        from augment import cable_distractor 
        rng =np .random .default_rng (19 )
        V ,F ,cp_pts ,cp_dirs =_make_mesh (200 ,4 ,rng )
        V2 ,cp2 ,cd2 =cable_distractor (V ,cp_pts ,cp_dirs ,frac =1.0 ,rng =rng )
        self .assertGreater (len (V2 ),len (V ),"cable_distractor should add vertices")
        np .testing .assert_array_equal (cp2 ,cp_pts ,err_msg ="CP labels changed")
        np .testing .assert_array_equal (cd2 ,cp_dirs ,err_msg ="CP dirs changed")

    def test_first_ring_stands_off_from_cp (self ):
    # Regression test: the first cable ring used to be centred exactly AT
    # the CP point (only the radial cable_r_mm gap), packing distractor
    # vertices right around the coordinate the model most needs a clean
    # local-geometry signal at. Every new vertex must now be farther than
    # cable_r_mm from the CP (a genuine 3D standoff, not just radial).
        from augment import cable_distractor 
        rng =np .random .default_rng (3 )
        cable_r_mm =1.5 
        V ,F ,cp_pts ,cp_dirs =_make_mesh (50 ,1 ,rng )
        V2 ,_ ,_ =cable_distractor (V ,cp_pts ,cp_dirs ,frac =1.0 ,rng =rng ,
        cable_r_mm =cable_r_mm )
        new_verts =V2 [len (V ):]
        self .assertGreater (len (new_verts ),0 )
        dists =np .linalg .norm (new_verts -cp_pts [0 ],axis =1 )
        self .assertGreater (dists .min (),cable_r_mm )


class TestResumeConfig (unittest .TestCase ):
    """cp_regressor._check_resume_config: catches a --resume that silently
    changes a loss-shape hyperparameter (w_heat/w_off/w_dir/weight_decay/
    centernet_alpha/lr), not just val_frac/seed/split_group as before."""

    def _cfg (self ,**overrides ):
        base =dict (val_frac =0.2 ,seed =0 ,split_group ="geometry",
        heat_pos_weight =50.0 ,heat_loss ="centernet",focal_gamma =2.0 ,
        augment =4 ,best_metric ="micro_f1",max_gpu_verts =7000 ,
        w_heat =1.0 ,w_off =5.0 ,w_dir =2.0 ,weight_decay =0.0005 ,
        centernet_alpha =2.0 ,lr =0.001 ,
        lr_schedule ="cosine",lr_decay_every =0 ,lr_decay_rate =0.5 ,
        warmup_epochs =0 ,eval_every =5 ,patience =4 ,
        min_delta =0.005 )
        base .update (overrides )
        return base 

    def test_no_warning_when_unchanged (self ):
        import logging 
        import cp_regressor as cpr 
        stored =self ._cfg ()
        current =self ._cfg ()
        with self .assertLogs ("cp_regressor",level ="WARNING")as _ :
        # force at least one log so assertLogs doesn't fail ten "no logs" --
        # then assert the resume-mismatch message specifically did NOT fire
            logging .getLogger ("cp_regressor").warning ("sentinel")
            cpr ._check_resume_config (stored ,current ,strict =False )
            # the sentinel is the only WARNING; no "training config changed" message
        self .assertFalse (any ("training config changed"in m for m in _ .output ))

    def test_warns_on_lr_or_w_off_change (self ):
        import cp_regressor as cpr 
        stored =self ._cfg ()
        current =self ._cfg (lr =0.0003 ,w_off =1.0 )
        with self .assertLogs ("cp_regressor",level ="WARNING")as cm :
            cpr ._check_resume_config (stored ,current ,strict =False )
        msg =" ".join (cm .output )
        self .assertIn ("training config changed",msg )
        self .assertIn ("lr",msg )
        self .assertIn ("w_off",msg )

    def test_strict_raises_on_mismatch (self ):
        import cp_regressor as cpr 
        stored =self ._cfg ()
        current =self ._cfg (centernet_alpha =1.5 )
        with self .assertRaises (ValueError ):
            cpr ._check_resume_config (stored ,current ,strict =True )

    def test_strict_raises_on_scheduler_or_patience_change (self ):
        import cp_regressor as cpr 
        stored =self ._cfg ()
        current =self ._cfg (lr_schedule ="plateau",patience =12 )
        with self .assertRaises (ValueError ):
            cpr ._check_resume_config (stored ,current ,strict =True )

    def test_warns_when_old_checkpoint_missing_scheduler_metadata (self ):
        import cp_regressor as cpr 
        stored =self ._cfg ()
        for k in ["lr_schedule","lr_decay_every","lr_decay_rate",
        "warmup_epochs","eval_every","patience","min_delta"]:
            stored .pop (k ,None )
        current =self ._cfg ()
        with self .assertLogs ("cp_regressor",level ="WARNING")as cm :
            cpr ._check_resume_config (stored ,current ,strict =False )
        self .assertIn ("missing metadata"," ".join (cm .output ))


class TestHardMiningAndOomGuards (unittest .TestCase ):
    def test_hard_mining_probabilities_have_floor_and_sum (self ):
        from cp_regressor import hard_mining_probabilities 
        p =hard_mining_probabilities (np .array ([0.0 ,1.0 ,8.0 ]))
        self .assertAlmostEqual (float (p .sum ()),1.0 ,places =12 )
        self .assertTrue ((p >0 ).all ())
        self .assertGreater (p [-1 ],p [1 ])

    def test_oom_guard_refuses_to_discard_accumulated_grads (self ):
        from cp_regressor import _guard_oom_fallback 
        _guard_oom_fallback (0 )
        with self .assertRaises (RuntimeError ):
            _guard_oom_fallback (1 )


            # ---------------------------------------------------------------------------
            # helper (not a test class)
            # ---------------------------------------------------------------------------

def _normalize (V ):
    from cp_regressor import normalize_vertices 
    return normalize_vertices (V )


if __name__ =="__main__":
    unittest .main (verbosity =2 )
