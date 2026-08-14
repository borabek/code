# Connection-point target encoding/decoding for direct regression (Option B).
#
# The detector is a per-vertex DiffusionNet (outputs_at="vertices"). Instead of
# the 5-class segmentation head, the connection-point regressor predicts, per
# vertex, a 7-channel target:
#
#   [0]     heatmap   h in [0,1]   = exp(-d^2 / (2 sigma^2)), d = dist to nearest CP
#   [1:4]   offset    that (mm)       = (CP_point - vertex)            (vector to that CP)
#   [4:7]   direction u (unit)     = InsertDirection of that CP     (OUTWARD)
#
# Decoding inverts this: high-heatmap vertices vote for a CP at
# (vertex + offset); votes are clustered with a greedy NMS whose radius is the
# robot tool clearance, giving one prediction per real connection point with a
# sub-vertex-accurate location and an averaged outward direction.
#
# Reference: Scheffler (2022) backbone (§5.3.3); heatmap/offset keypoint
# regression is the Option-B head chosen for the ABB connection-point data.

import logging 
import numpy as np 

logger =logging .getLogger (__name__ )

HEATMAP =0 
OFFSET =slice (1 ,4 )
DIRECTION =slice (4 ,7 )
N_CHANNELS =7 

# Heatmap Gaussian width as a fraction of the bbox diagonal. SINGLE SOURCE OF
# TRUTH: encode_targets (training targets) and predict.py (inference NMS radius)
# both derive sigma from sigma_for(), so they can never silently drift apart.
SIGMA_FRAC =0.02 

# The radius at which two nearby openings are treated as ONE.
#
# The label side (step_openings, LABEL_MERGE_CEILING_MM) and the decode side
# (decode_predictions, NMS_CEILING_MM) deliberately DIFFER today, and the gap is
# a known, measured limitation rather than an oversight:
#
#   * Decode merges detections within min(5.0mm, 0.04*bbox_diag). On a part with
#     a >100mm diagonal that ceiling binds, so two detections 4-5mm apart collapse
#     into one -- the decoder cannot resolve them.
#   * It is tempting to "unify" by coarsening the LABELS to 5.0mm as well. That is
#     WRONG: measured ten the human ground truth, 37 pairs of REAL terminals ten 8
#     parts sit 4-5mm apart ten >100mm bodies. Merging them at label time would
#     teach the model that two wire entries are one point (the robot would then
#     only ever be handed one of them) and would silently turn those unavoidable
#     false negatives into "true positives" -- inflating F1 without improving the
#     product.
#
# So labels stay at the finer 4.0mm (truthful), decode stays at 5.0mm (its real
# resolution), and the residual FN ten those few parts is PRICED INTO the reported
# numbers -- exactly as it was for the v28 baseline, keeping runs comparable.
# The correct fix is ten the DECODE side (a finer ceiling, or vote-density-aware
# clustering) and must be measured before it ships: see RESULTS 11t.
NMS_CEILING_MM =5.0 # decode / eval NMS
LABEL_MERGE_CEILING_MM =4.0 # CAD face merge at label time
NMS_SCALE_FRAC =0.04 


def merge_radius_mm (vertices_or_bbox_diag ,ceiling =NMS_CEILING_MM ,
frac =NMS_SCALE_FRAC ):
    """Scale-adaptive merge/NMS radius = min(ceiling, frac * bbox diagonal).

    Accepts either a vertex array or a pre-computed bbox diagonal (mm), so the
    CAD label path (which has a B-rep bbox, not a mesh) and the decode path
    (which has vertices) can share one rule."""
    if np .isscalar (vertices_or_bbox_diag ):
        diag =float (vertices_or_bbox_diag )
    else :
        V =np .asarray (vertices_or_bbox_diag ,dtype =np .float64 )
        diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))if len (V )else 0.0 
    if diag <=0 :
        return float (ceiling )
    return min (float (ceiling ),max (frac *diag ,1e-6 ))


def sigma_for (vertices ,frac =SIGMA_FRAC ):
    """Scale-adaptive heatmap sigma (mm) = frac * bbox diagonal, floored > 0."""
    V =np .asarray (vertices ,dtype =np .float64 )
    if len (V )==0 :
        return 1e-6 
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
    return max (diag *frac ,1e-6 )


    # ---------------------------------------------------------------------------
    # CP-ten-surface validation: a GT connection point should sit ON the mesh
    # ---------------------------------------------------------------------------

def cp_surface_distances (vertices ,cp_points ):
    """For each CP, the distance (mm) to the NEAREST vertex.

    encode_targets snaps each CP's heat peak onto its nearest vertex regardless of
    how far that is, so a CP floating off the surface (mislabel / wrong frame)
    becomes a phantom target the model cannot learn. This exposes that distance so
    prepare_sample can warn ten / drop such CPs.
    """
    V =np .asarray (vertices ,dtype =np .float64 )
    P =np .asarray (cp_points ,dtype =np .float64 )
    if len (P )==0 or len (V )==0 :
        return np .zeros (len (P ),dtype =np .float64 )
    out =np .empty (len (P ),dtype =np .float64 )
    for j in range (len (P )):
        out [j ]=float (np .linalg .norm (V -P [j ],axis =1 ).min ())
    return out 


    # ---------------------------------------------------------------------------
    # encode: (vertices, cp_points, cp_directions) -> (N, 7) target
    # ---------------------------------------------------------------------------

def _nearest_cp (vertices ,cp_points ):
    """For each vertex, index of and distance to the nearest CP. Chunked to
    keep the (N x K) distance matrix small ten large meshes."""
    N =len (vertices )
    nearest =np .zeros (N ,dtype =np .int64 )
    dist =np .zeros (N ,dtype =np .float64 )
    chunk =4096 
    for s in range (0 ,N ,chunk ):
        v =vertices [s :s +chunk ]# (c, 3)
        d =np .linalg .norm (v [:,None ,:]-cp_points [None ,:,:],axis =2 )# (c, K)
        j =np .argmin (d ,axis =1 )
        nearest [s :s +chunk ]=j 
        dist [s :s +chunk ]=d [np .arange (len (v )),j ]
    return nearest ,dist 


def encode_targets (vertices ,cp_points ,cp_directions ,sigma_mm =None ,
offset_radius_mm =None ,max_sigma_spacing_frac =0.5 ):
    """Build the (N, 7) per-vertex regression target.

    sigma_mm         : Gaussian width of the heatmap. Default = 2% of the mesh
                       bounding-box diagonal (scale-adaptive), but CAPPED below so
                       neighbouring connection points stay separable.
    offset_radius_mm : offset+direction channels are only meaningful near a CP;
                       this radius defines the supervision mask returned to the
                       loss. Default = 3 * sigma_mm (after the cap).
    max_sigma_spacing_frac : sigma is capped to this fraction of the CLOSEST CP
                       spacing. The bbox-diagonal sigma is dominated by part LENGTH,
                       so ten an elongated part it can exceed the gap between CPs and
                       two distinct points smear into one merged "super-blob" -- the
                       network then can't separate them and recall collapses. Capping
                       at <=0.5x spacing guarantees the blobs do not overlap.

    Returns (target (N,7) float32, mask (N,) bool, sigma_mm). `mask` marks the
    vertices whose offset/direction channels should be supervised.
    """
    vertices =np .asarray (vertices ,dtype =np .float64 )
    N =len (vertices )
    target =np .zeros ((N ,N_CHANNELS ),dtype =np .float32 )

    if len (cp_points )==0 :
        mask =np .zeros (N ,dtype =bool )
        return target ,mask ,(sigma_mm or 0.0 )

    if sigma_mm is None :
        sigma_mm =sigma_for (vertices )
    uncapped_sigma_mm =sigma_mm 

    # CAP sigma so neighbouring CP heat blobs stay SEPARABLE. Without this, an
    # elongated part's bbox-diagonal sigma can exceed the CP gap -> the two
    # Gaussians merge into one peak in the target, the model is trained to predict a
    # merged blob, and at decode the two points collapse into one (lost recall).
    min_spacing =float ("inf")
    if len (cp_points )>=2 :
        cpp =np .asarray (cp_points ,dtype =np .float64 )
        dd =np .linalg .norm (cpp [:,None ,:]-cpp [None ,:,:],axis =2 )
        np .fill_diagonal (dd ,np .inf )
        min_spacing =float (dd .min ())
        cap =max_sigma_spacing_frac *min_spacing 
        if np .isfinite (min_spacing )and sigma_mm >cap and cap >0 :
            logger .info ("encode_targets: sigma %.2fmm > %.2fx closest CP spacing "
            "%.2fmm -> capping to %.2fmm so heat blobs stay separable",
            sigma_mm ,max_sigma_spacing_frac ,min_spacing ,cap )
            sigma_mm =cap 

    if offset_radius_mm is None :
    # DECOUPLED from the (possibly capped) heatmap sigma. Capping sigma keeps
    # heat BLOBS separable, but the offset/direction mask doesn't need the same
    # strict separation -- each vertex's target already points at its OWN
    # nearest CP (_nearest_cp below), so a WIDER mask only adds more supervised
    # vertices; it cannot misassign one to the wrong CP (nearest-distance still
    # resolves ties correctly), so there is no correctness reason to cap it by
    # neighbour spacing the way sigma must be. Using the CAPPED sigma for this
    # (the old behaviour, `3*sigma_mm` after capping) meant tightly-spaced-CP
    # parts got an offset radius as narrow as ~3 local-vertex-hops -- measured
    # ten the real corpus, 26% of multi-CP parts hit the sigma cap, and ten THOSE
    # parts the offset loss's effective supervised-vertex count collapsed to a
    # handful (22.5% of ALL real parts have an effective sample size <5),
    # starving the offset regressor of smooth supervision exactly ten the tight-
    # spacing parts most prone to a near-miss false positive after NMS (measured
    # ten a real trained model: 66% of FPs are near-miss, median ~4.6mm off).
    # Use the UNCAPPED (bbox-scale) sigma for the radius instead -- never
    # narrower than the old capped-sigma value, so this can only widen
    # supervision, never shrink it.
        offset_radius_mm =max (3.0 *uncapped_sigma_mm ,3.0 *sigma_mm )

    nearest ,dist =_nearest_cp (vertices ,cp_points )
    target [:,HEATMAP ]=np .exp (-(dist **2 )/(2.0 *sigma_mm **2 ))
    target [:,OFFSET ]=(cp_points [nearest ]-vertices ).astype (np .float32 )
    target [:,DIRECTION ]=cp_directions [nearest ].astype (np .float32 )

    mask =dist <=offset_radius_mm 

    # Guarantee one EXACT peak (heat=1) per CP -- the nearest vertex to it. On a
    # coarse mesh the Gaussian skirt may never reach ~1 at any vertex, so this
    # snaps the true peak. Penalty-reduced focal losses (cp_regressor heat_loss
    # 'centernet') key their positives off heat==1 and collapse without it.
    for j in range (len (cp_points )):
        i =int (np .argmin (np .linalg .norm (vertices -cp_points [j ],axis =1 )))
        target [i ,HEATMAP ]=1.0 
        # ALSO force the peak vertex into the offset/direction supervision mask. For
        # a RECESSED / off-surface CP the nearest vertex can be farther than the
        # offset_radius (3*sigma), so without this it would get a heat=1 peak but NO
        # offset target -- the model could never learn the (large) vertex->CP offset
        # and the decoded vote would land ten the surface, mis-localising the CP.
        mask [i ]=True 
    return target ,mask ,sigma_mm 


    # ---------------------------------------------------------------------------
    # decode: (N, 7) prediction -> list of connection points
    # ---------------------------------------------------------------------------

def decode_predictions (vertices ,pred ,heatmap_thresh =0.3 ,
nms_radius_mm =NMS_CEILING_MM ,min_votes =1 ,
snap_to_surface =False ,nms_scale_frac =NMS_SCALE_FRAC ):
    """Turn a (N, 7) prediction into discrete connection points.

    Each vertex with heatmap > thresh votes for a CP at (vertex + offset),
    weighted by its heatmap value. Votes are greedily clustered: the highest
    remaining vote seeds a cluster, all votes within the EFFECTIVE merge radius
    are merged into it (weighted mean of position and direction), and removed.

    nms_radius_mm is a CEILING (e.g. robot tool clearance), not used directly:
    the effective radius is min(nms_radius_mm, nms_scale_frac * bbox_diagonal).
    Measured ten the real corpus, ~5% of parts have two genuine connection points
    closer together than the flat 5mm clearance default (down to 1.8mm ten small
    parts) -- a flat radius merges those into one lost detection regardless of
    model quality. Scaling the cap to the part's own size fixes this for SMALL
    parts (where the tight spacing is a large fraction of the part), which is
    most of the affected cases; it does NOT fully fix a few large parts with an
    absolute-tight spacing that's still a tiny fraction of their bbox (e.g. a
    3mm gap ten a 460mm part) -- untangling those needs vote-density-aware
    clustering, not a bigger global constant, and is a known residual gap.
    Pass nms_scale_frac=None to disable adaptive capping (old fixed-radius
    behaviour, e.g. for exact reproduction of a prior run's decode).

    The decoded point is a weighted mean of (vertex+offset) votes and so is not
    guaranteed to lie ON the mesh. `surface_dist` (mm to the nearest vertex) is
    always reported; snap_to_surface=True additionally moves the point onto that
    nearest vertex (useful when a robot must contact the surface).

    Returns list of dicts: {point (3,), direction (3,) unit OUTWARD,
    insertion_axis (3,) unit INWARD, score, n_votes, surface_dist}.
    `score` is the raw max heatmap (sigmoid) of the cluster -- an UNCALIBRATED
    confidence, not a probability, and not comparable across parts.
    """
    vertices =np .asarray (vertices ,dtype =np .float64 )
    pred =np .asarray (pred ,dtype =np .float64 )
    if nms_scale_frac is not None and len (vertices ):
        diag =float (np .linalg .norm (vertices .max (0 )-vertices .min (0 )))
        nms_radius_mm =min (float (nms_radius_mm ),max (nms_scale_frac *diag ,1e-6 ))
    h =pred [:,HEATMAP ]
    keep =np .where (h >heatmap_thresh )[0 ]
    if len (keep )==0 :
        return []

    votes =vertices [keep ]+pred [keep ,OFFSET ]# (n, 3) predicted CP location
    dirs =pred [keep ,DIRECTION ]
    scores =h [keep ]
    order =np .argsort (-scores )# high score first

    used =np .zeros (len (keep ),dtype =bool )
    out =[]
    for oi in order :
        if used [oi ]:
            continue 
        seed =votes [oi ]
        d =np .linalg .norm (votes -seed ,axis =1 )
        members =(d <=nms_radius_mm )&(~used )
        used [members ]=True 
        if members .sum ()<min_votes :
            continue 
        w =scores [members ]
        wsum =w .sum ()
        pt =(votes [members ]*w [:,None ]).sum (0 )/wsum 
        dv =(dirs [members ]*w [:,None ]).sum (0 )
        nrm =np .linalg .norm (dv )
        dv =dv /nrm if nrm >0 else np .array ([0.0 ,0.0 ,1.0 ])
        # distance from the decoded point to the mesh (nearest vertex)
        nv =int (np .argmin (np .linalg .norm (vertices -pt ,axis =1 )))
        surface_dist =float (np .linalg .norm (vertices [nv ]-pt ))
        if snap_to_surface :
            pt =vertices [nv ]
        out .append ({
        "point":pt ,
        "direction":dv ,# OUTWARD (= approach_vector)
        "insertion_axis":-dv ,# INWARD (travel into the part)
        "score":float (w .max ()),
        "n_votes":int (members .sum ()),
        "surface_dist":surface_dist ,
        })
    return out 


    # ---------------------------------------------------------------------------
    # export decoded connection points -> repo robot-ready JSON node schema
    # ---------------------------------------------------------------------------
    # Mirrors connector3d.build_connector_graph() node fields so the regression
    # detector emits the same robot-ready output as the segmentation pipeline.

def decoded_to_robot_nodes (preds ,names_per_block =None ,approach_distance_mm =0.0 ,
robot_class ="TerminalContact"):
    """Convert decode_predictions() output -> list of robot-ready node dicts.

    preds                : list from decode_predictions (point/direction/insertion_axis/score).
    names_per_block      : optional list[list[str]] of terminal names per block,
                           aligned to preds order (carried as 'terminal_names').
    approach_distance_mm : standoff distance along the outward approach vector.

    'confidence_score' is the raw max heatmap (sigmoid) for the cluster -- an
    UNCALIBRATED score, NOT a probability and not comparable across parts; treat
    it only as a relative within-part ranking. 'surface_dist_mm' reports how far
    the decoded point is from the mesh (sanity check for floating detections).
    """
    def vec (v ):
        return None if v is None else [round (float (x ),5 )for x in np .asarray (v )]

    nodes =[]
    for i ,p in enumerate (preds ):
        entry =np .asarray (p ["point"],dtype =float )
        approach =np .asarray (p ["direction"],dtype =float )# OUTWARD
        nodes .append ({
        "id":i ,
        "robot_class":robot_class ,
        "entry_point":vec (entry ),
        "approach_vector":vec (approach ),
        "insertion_axis":vec (p .get ("insertion_axis",-approach )),# INWARD travel
        "standoff_point":vec (entry +approach *float (approach_distance_mm )),
        "approach_distance_mm":round (float (approach_distance_mm ),5 ),
        "confidence_score":round (float (p .get ("score",1.0 )),4 ),# raw, uncalibrated
        "surface_dist_mm":round (float (p .get ("surface_dist",0.0 )),4 ),
        "n_votes":int (p .get ("n_votes",0 )),
        "terminal_names":list (names_per_block [i ])if names_per_block else None ,
        "n_terminals":(len (names_per_block [i ])if names_per_block else None ),
        })
    return nodes 


    # ---------------------------------------------------------------------------
    # selftest: encode a real-ish part, decode, check we recover the CPs
    # ---------------------------------------------------------------------------

def _selftest ():
# a flat-ish plate of vertices in a 100mm square, z=0
    xs ,ys =np .meshgrid (np .linspace (0 ,100 ,60 ),np .linspace (0 ,100 ,60 ))
    V =np .column_stack ([xs .ravel (),ys .ravel (),np .zeros (xs .size )])
    # 3 ground-truth CPs, each with an outward (+z) direction
    cp_pts =np .array ([[25 ,25 ,0 ],[75 ,30 ,0 ],[50 ,80 ,0 ]],dtype =float )
    cp_dir =np .array ([[0 ,0 ,1.0 ]]*3 )

    tgt ,mask ,sigma =encode_targets (V ,cp_pts ,cp_dir )
    assert tgt .shape ==(len (V ),7 )
    # decode the *ground-truth* target -> should recover the 3 CPs precisely
    got =decode_predictions (V ,tgt ,heatmap_thresh =0.3 ,nms_radius_mm =10.0 )
    assert len (got )==3 ,f"expected 3 CPs, got {len (got )}"
    # match each prediction to nearest GT and check error
    errs =[]
    for g in got :
        dd =np .linalg .norm (cp_pts -g ["point"],axis =1 )
        errs .append (dd .min ())
        assert g ["direction"][2 ]>0.99 # outward +z
        assert g ["insertion_axis"][2 ]<-0.99 # inward -z
        # exercise the robot-ready exporter
    nodes =decoded_to_robot_nodes (got ,names_per_block =[["a"],["b"],["c"]],
    approach_distance_mm =10.0 )
    assert len (nodes )==3 and nodes [0 ]["robot_class"]=="TerminalContact"
    assert nodes [0 ]["insertion_axis"][2 ]<-0.99 # inward = -approach
    assert nodes [0 ]["terminal_names"]is not None 
    print (f"cp_targets selftest OK: sigma={sigma :.2f}mm, "
    f"recovered {len (got )} CPs, max loc err={max (errs ):.3f}mm, "
    f"robot nodes={len (nodes )}")


if __name__ =="__main__":
    logging .basicConfig (level =logging .INFO )
    _selftest ()
