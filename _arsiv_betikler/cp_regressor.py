# Connection-point regressor (Option B): per-vertex 7-channel head + losses +
# training loop. Predicts (heatmap, offset xyz, direction xyz) per vertex; the
# decoder in cp_targets turns that into discrete terminal-block detections.
#
# Two backbones are provided:
#   * build_diffusionnet_regressor() – the production head: a DiffusionNet with
#     C_out=7 (reuses diffusionnet.precompute_operators / _forward). Needs the
#     `diffusion_net` package.
#   * CPMLP – a light per-vertex MLP ten normalised xyz that needs only torch.
#     Used for smoke-tests / environments without the diffusion_net compile
#     chain. It has no geometric receptive field, so it is NOT the production
#     model, but it exercises the full target/loss/decode pipeline.
#
# torch is imported lazily (like diffusionnet.py) so the data utilities remain
# importable without it.

import hashlib 
import logging 
import os # _disk_has_room/_knn_graph_disk use it; without a module-level
# import they raise NameError the first time a cached-graph path
# runs -- which is the VAL eval, i.e. ~1.5h into a run that then
# dies with no best checkpoint (2026-07-14, v31 epoch 5).
import numpy as np 

import cp_targets as ct 

logger =logging .getLogger (__name__ )

N_CHANNELS =ct .N_CHANNELS # 7


def _require_torch ():
    try :
        import torch 
        return torch 
    except ImportError as exc :
        raise ImportError ("'torch' is required for cp_regressor "
        "(pip install torch)")from exc 


        # ---------------------------------------------------------------------------
        # normalisation: center + scale vertices so the MLP/optimiser see ~unit coords
        # ---------------------------------------------------------------------------

def normalize_vertices (V ):
    """Center ten centroid and scale by bounding-box diagonal. Returns (Vn, center, scale)."""
    V =np .asarray (V ,dtype =np .float64 )
    center =V .mean (0 )
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
    scale =diag if diag >0 else 1.0 
    return (V -center )/scale ,center ,scale 


    # ---------------------------------------------------------------------------
    # light MLP backbone (smoke-test / no diffusion_net)
    # ---------------------------------------------------------------------------

def build_cpmlp (width =128 ,depth =4 ,c_in =3 ):
    """Per-vertex MLP: c_in -> ... -> 7. torch.nn.Module."""
    _require_torch ()
    import torch .nn as nn 

    class CPMLP (nn .Module ):
        def __init__ (self ):
            super ().__init__ ()
            layers =[nn .Linear (c_in ,width ),nn .ReLU ()]
            for _ in range (depth -1 ):
                layers +=[nn .Linear (width ,width ),nn .ReLU ()]
            layers +=[nn .Linear (width ,N_CHANNELS )]
            self .net =nn .Sequential (*layers )

        def forward (self ,x ):
            return self .net (x )

    return CPMLP ()


    # ---------------------------------------------------------------------------
    # production backbone: DiffusionNet with a 7-channel regression head
    # ---------------------------------------------------------------------------

def build_diffusionnet_regressor (config =None ):
    """Build a DiffusionNet with C_out=7 (regression). Returns (model, meta)."""
    import diffusionnet as dnmod 
    _require_torch ()
    dn =dnmod ._require ("diffusion_net","pip install git+https://github.com/nmwsharp/diffusion-net")
    cfg ={**dnmod .DEFAULTS ,**(config or {})}
    c_in =3 if cfg .get ("input_features","xyz")=="xyz"else 16 
    model =dn .layers .DiffusionNet (
    C_in =c_in ,C_out =N_CHANNELS ,
    C_width =int (cfg .get ("c_width",128 )),
    N_block =int (cfg .get ("n_diffusion_blocks",4 )),
    last_activation =None ,# raw outputs; we apply sigmoid/normalise in post
    outputs_at ="vertices",
    dropout =float (cfg .get ("dropout",0.0 ))>0.0 ,
    )
    meta ={"input_features":cfg .get ("input_features","xyz"),
    "k_eig":int (cfg .get ("n_eig",128 )),"hks_count":16 ,
    # structural params -> a checkpoint is self-describing and rebuildable
    "c_in":c_in ,"c_width":int (cfg .get ("c_width",128 )),
    "n_block":int (cfg .get ("n_diffusion_blocks",4 )),
    "dropout":float (cfg .get ("dropout",0.0 ))}
    return model ,meta 


    # ---------------------------------------------------------------------------
    # checkpointing: one .ckpt format for ALL backbones, with full training state
    # ---------------------------------------------------------------------------
    # A checkpoint is a dict with at least {state_dict, meta, backbone}. meta
    # carries the structural params so the exact architecture can be rebuilt
    # before loading weights -- no separate config needed for inference later.
    #
    # Training checkpoints additionally carry {optimizer, scheduler, epoch,
    # best_f1, history, rng} so a run can RESUME exactly where it stopped --
    # including ten a different machine: commit/push the .ckpt, pull it ten the
    # other device, and run train_cp.py --resume. torch.save files are portable
    # across OS/CPU/GPU (tensors are remapped via map_location ten load).

CKPT_VERSION =1 

# Bumped whenever the encode/decode SEMANTICS change (target channel meaning,
# sigma rule, vote/NMS math) so an old exported .pt that would now mis-decode is
# flagged at load instead of silently producing wrong points. Independent of
# CKPT_VERSION, which tracks the checkpoint container format.
# v2: sigma capped to <=0.5x closest CP spacing in encode_targets; decode NMS
#     radius decoupled from sigma (fixed tool clearance) so close CPs resolve.
DECODE_SCHEMA =2 

# Decode operating point bundled with an inference artifact. For keypoint
# detection these three knobs are part of "the model": the same weights yield a
# very different precision/recall depending ten them, so an export carries them
# explicitly. dist_thresh_mm is the match radius used when scoring, kept for
# provenance. Older checkpoints without a "decode" key fall back to these.
# min_votes=1 matches the deploy default everywhere else (the GT-decode ceiling
# is R=0.94 @1 vs 0.79 @2 ten the real corpus -- see diag_encoding.py).
DEFAULT_DECODE ={"heatmap_thresh":0.3 ,"min_votes":1 ,
"nms_clearance_mm":5.0 ,"dist_thresh_mm":5.0 }


# ---------------------------------------------------------------------------
# vertex subsampling -- ONE strategy shared by training and inference
# ---------------------------------------------------------------------------
# A part above the model's vertex cap is uniformly subsampled to that cap so the
# kNN graph keeps the density the model trained ten. CRITICAL: inference is
# GT-free, so it CANNOT keep CP-peak vertices -- therefore TRAINING must use the
# same plain-uniform sample (and re-snap its heat peaks onto the kept vertices),
# otherwise the model trains ten a vertex distribution inference never reproduces
# and recall silently drops ten big parts.

def uniform_subsample_idx (n ,cap ,seed =0 ):
    """Deterministic uniform subsample: sorted indices of `cap` of `n` vertices.
    Same routine ten both sides so train and inference match in distribution."""
    if not cap or n <=cap :
        return None 
    return np .sort (np .random .RandomState (seed ).choice (n ,cap ,replace =False ))


def part_subsample_seed (seed ,part_nr ):
    """Deterministic per-part uniform-subsample seed: XOR the run seed with a hash
    of the part number, so different parts get different (but each reproducible)
    subsamples. SINGLE SOURCE OF TRUTH shared by train_knngraph_regressor's ABB/
    general (non-patch) vertex selection AND infer_knngraph: training picks a
    part-specific subsample for any part over max_gpu_verts, so inference/eval
    MUST reconstruct the SAME subsample for that exact part or the model is being
    evaluated/deployed ten a vertex distribution it never trained ten for that part
    (a real, previously-silent train/inference mismatch -- see infer_knngraph)."""
    return int (seed )^(int (hashlib .sha256 (str (part_nr ).encode ()).hexdigest (),16 )
    &0x7FFFFFFF )


    # fraction of the (normalised) bbox diagonal by which each spatial patch is grown
    # so a connection-point hole sitting ten a patch boundary is fully seen by both
    # neighbouring patches (overlap is averaged at inference, harmlessly duplicated at
    # training). ~0.05 of a ~70 mm part is ~3.5 mm > a terminal hole.
PATCH_MARGIN_FRAC =0.06 


def is_patch_part (part_nr ):
    """True for TERMINAL BLOCKS (small recessed wire-entry CP holes that a uniform
    subsample erases -> they need full-density SPATIAL PATCHES). In this corpus those
    are wscad (wscaduniverse) and Phoenix Contact (PXC) parts. Contactors / drives /
    breakers have raised screw terminals that survive subsampling -> v8 path. Used by
    training prep AND every inference caller (predict/eval/tta) so the per-part vertex
    selection matches between train and test."""
    return str (part_nr ).startswith (("wscaduniverse","PXC"))


def spatial_patches (V ,cap ,margin_frac =PATCH_MARGIN_FRAC ,seed =0 ,max_patches =8 ):
    """Partition vertices into spatial blocks, each <= cap verts at FULL density.

    The fix for big parts (40k-70k verts): a single uniform subsample to `cap`
    throws away ~85-90 % of the mesh, so the small cylindrical CP openings lose the
    very geometry that identifies them and the model cannot learn (or detect) them.
    Instead we KD-median-split the longest axis until each block has <= cap verts,
    keeping every vertex inside a block -- so the hole geometry survives intact and
    the model sees the same full-density local neighbourhood it must key ten. Used ten
    BOTH sides (training prep and infer_knngraph) so train/inference stay matched.

    Returns a list of index arrays into V. A part with <= cap verts -> one block
    (all indices), identical to the old single-pass behaviour. Each core block is
    grown by `margin_frac` of the bbox diagonal so boundary holes are covered twice.

    max_patches bounds cost: a part larger than max_patches*cap verts (e.g. the
    471k-vertex ABB assembly) is first uniformly subsampled to that budget, then
    patched -- so no single part can explode into dozens of graphs. The index arrays
    then point into the SUBSAMPLED set, so callers must apply the same subsample.
    Returns (patches, sel) where sel is the kept-index array into the ORIGINAL V
    (or None when no subsample happened)."""
    V =np .asarray (V ,dtype =np .float64 )
    n =len (V )
    if not cap or n <=cap :
        return [np .arange (n )],None 
    sel =None 
    budget =int (max_patches )*cap if max_patches else 0 
    if budget and n >budget :
        sel =uniform_subsample_idx (n ,budget ,seed =0 )# giant part: cap total verts
        V =V [sel ]
        n =len (V )
    thr =cap # core blocks up to cap; margin trimmed to cap
    cores =[]
    stack =[np .arange (n )]
    while stack :
        idx =stack .pop ()
        if len (idx )<=thr :
            cores .append (idx )
            continue 
        sub =V [idx ]
        ax =int (np .argmax (sub .max (0 )-sub .min (0 )))
        med =np .median (sub [:,ax ])
        left =idx [sub [:,ax ]<=med ]
        right =idx [sub [:,ax ]>med ]
        if len (left )==0 or len (right )==0 :
        # median did not separate (many coincident coords ten this axis): split by
        # sorted position so every vertex stays covered and the recursion still
        # makes progress (never drop verts -- dropping = silent coverage holes).
            order =idx [np .argsort (sub [:,ax ],kind ="stable")]
            h =len (order )//2 
            stack .append (order [:h ])
            stack .append (order [h :])
            continue 
        stack .append (left )
        stack .append (right )
    if not margin_frac or margin_frac <=0 :
        return [np .sort (c )for c in cores ],sel 
    diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))or 1.0 
    m =margin_frac *diag 
    rng =np .random .RandomState (seed )
    out =[]
    for idx in cores :
        lo =V [idx ].min (0 )-m 
        hi =V [idx ].max (0 )+m 
        box =np .where (((V >=lo )&(V <=hi )).all (1 ))[0 ]
        if len (box )>cap :
            core =set (idx .tolist ())
            ring =np .array ([i for i in box if i not in core ],dtype =np .int64 )
            room =cap -len (idx )
            if room >0 and len (ring )>room :
                ring =rng .choice (ring ,room ,replace =False )
            box =np .concatenate ([idx ,ring ])if room >0 else idx 
        out .append (np .sort (box ))
    return out ,sel 


def subsample_coverage (verts_norm ,gt_points ,center ,scale ,cap ,seed =0 ,
sigma_frac =0.02 ):
    """How many GT points become UNRECOVERABLE under the inference vertex selection.

    A GT point whose nearest *kept* vertex is farther than ~3 sigma gets ~0 heat
    everywhere, so no vertex can vote for it -> a guaranteed false negative. With
    SPATIAL PATCHES the kept set is the union of all patches: for an ordinary big
    part that union is ALL vertices (full coverage -> 0 at risk), and only a giant
    part (>max_patches*cap, uniform-capped before patching) can drop a GT region.
    Returns (n_gt, n_at_risk). GT-aware -> diagnostics only, never feeds decode."""
    n =len (verts_norm )
    if not cap or n <=cap or len (gt_points )==0 :
        return len (gt_points ),0 
    V =np .asarray (verts_norm ,dtype =np .float64 )
    _patches ,sel =spatial_patches (V ,cap )# same selection as infer_knngraph
    kept =V if sel is None else V [sel ]# union of all patches
    gtn =(np .asarray (gt_points ,dtype =np .float64 )-center )/scale 
    thr =3.0 *sigma_frac # normalised diag ~= 1
    at_risk =0 
    for g in gtn :
        if np .linalg .norm (kept -g ,axis =1 ).min ()>thr :
            at_risk +=1 
    return len (gt_points ),at_risk 


def _meta_to_config (backbone ,meta ):
    """Map a checkpoint's meta back to the builder config for that backbone."""
    if backbone =="diffusionnet":
        return {"input_features":meta ["input_features"],"c_width":meta ["c_width"],
        "n_diffusion_blocks":meta ["n_block"],"n_eig":meta ["k_eig"],
        "dropout":meta .get ("dropout",0.0 )}
    if backbone =="knngraph":
        return {"c_width":meta ["c_width"],"n_layers":meta ["n_layers"],
        "k":meta ["k"],"global_feat":meta .get ("global_feat",False ),
        "dropout":meta .get ("dropout",0.0 ),
        "normals":meta .get ("normals",False ),
        "curvature":meta .get ("curvature",False ),
        "concavity":meta .get ("concavity",False ),
        "edge_dist":meta .get ("edge_dist",False )}
    if backbone =="mlp":
        return {"width":meta .get ("width",128 ),"depth":meta .get ("depth",4 )}
    if backbone =="hierpoint":
        import hierpoint as hp 
        cfg ={k :meta [k ]for k in ("widths","ratios","k_levels","layers",
        "group_k","up_k","k","dropout",
        "normals","curvature","concavity",
        "edge_dist")if k in meta }
        return {**hp .HP_DEFAULTS ,**cfg }
    raise ValueError (f"unknown backbone {backbone !r }")


def build_regressor (backbone ,config =None ):
    """Build any of the four backbones. Returns (model, meta)."""
    if backbone =="diffusionnet":
        return build_diffusionnet_regressor (config )
    if backbone =="knngraph":
        return build_knngraph_regressor (config )
    if backbone =="hierpoint":
        import hierpoint as hp 
        return hp .build_hierpoint_regressor (config )
    if backbone =="mlp":
        cfg ={"width":128 ,"depth":4 ,**(config or {})}
        model =build_cpmlp (width =int (cfg ["width"]),depth =int (cfg ["depth"]))
        meta ={"backbone":"mlp","input_features":"xyz","c_in":3 ,
        "width":int (cfg ["width"]),"depth":int (cfg ["depth"])}
        return model ,meta 
    raise ValueError (f"unknown backbone {backbone !r }")


def _capture_rng ():
    """Snapshot python/numpy/torch(/cuda) RNG states for exact resume."""
    torch =_require_torch ()
    import random 
    st ={"python":random .getstate (),"numpy":np .random .get_state (),
    "torch":torch .get_rng_state ()}
    if torch .cuda .is_available ():
        st ["cuda"]=torch .cuda .get_rng_state_all ()
    return st 


def _restore_rng (st ):
    """Best-effort RNG restore (a resume ten different hardware still works --
    it just reshuffles from the seed instead of the exact saved stream)."""
    if not st :
        return 
    torch =_require_torch ()
    import random 
    try :
        random .setstate (st ["python"])
        np .random .set_state (st ["numpy"])
        torch .set_rng_state (st ["torch"].cpu ().to (torch .uint8 ))
        cuda =st .get ("cuda")
        if cuda and torch .cuda .is_available ()and len (cuda )==torch .cuda .device_count ():
            torch .cuda .set_rng_state_all ([t .cpu ().to (torch .uint8 )for t in cuda ])
    except Exception as exc :# noqa: BLE001
        logger .warning ("could not restore RNG state (%s) -- continuing",exc )


        # cap ten how many per-eval history records are EMBEDDED in a checkpoint. The full
        # history lives in the sibling history JSON (history_path); the checkpoint keeps
        # only a tail for self-containment so per-epoch saves do not grow without bound.
CKPT_HISTORY_CAP =200 


def save_checkpoint (path ,model ,meta ,backbone ,optimizer =None ,scheduler =None ,
epoch =None ,best_f1 =None ,best_epoch =None ,history =None ,
decode =None ,train_config =None ,best_metric =None ,quiet =False ):
    """Write a (resumable) checkpoint atomically.

    With only model+meta this is an inference checkpoint (what the legacy
    save_diffusionnet/save_knngraph wrote); passing optimizer/scheduler/epoch
    makes it a full training checkpoint that train loops can resume from.
    `epoch` is the last COMPLETED epoch (0-based). `decode` (a dict of the
    operating-point knobs) is stored when given so the artifact is unambiguous.
    `train_config` (val_frac/seed/loss/augment/...) is stored so a --resume can
    detect a mismatched continuation. Embedded history is capped (full log lives
    in the history JSON). Atomic write (tmp+replace) so a crash mid-save never
    corrupts an existing checkpoint.
    """
    torch =_require_torch ()
    import os 
    d =os .path .dirname (os .path .abspath (path ))
    if d :
        os .makedirs (d ,exist_ok =True )
    ckpt ={"ckpt_version":CKPT_VERSION ,"decode_schema":DECODE_SCHEMA ,
    "backbone":backbone ,"state_dict":model .state_dict (),"meta":meta }
    if decode is not None :
        ckpt ["decode"]=dict (decode )
    if train_config is not None :
        ckpt ["train_config"]=dict (train_config )
    if best_metric is not None :
        ckpt ["best_metric"]=str (best_metric )
    if optimizer is not None :
        ckpt ["optimizer"]=optimizer .state_dict ()
    if scheduler is not None :
        ckpt ["scheduler"]=scheduler .state_dict ()
    if epoch is not None :
        ckpt ["epoch"]=int (epoch )
        ckpt ["rng"]=_capture_rng ()
    if best_f1 is not None :
        ckpt ["best_f1"]=float (best_f1 )
    if best_epoch is not None :
        ckpt ["best_epoch"]=int (best_epoch )
    if history is not None :
        ckpt ["history"]=list (history )[-CKPT_HISTORY_CAP :]
    tmp =path +".tmp"
    torch .save (ckpt ,tmp )
    os .replace (tmp ,path )
    # quiet: the rolling per-epoch 'last' snapshot would otherwise spam one INFO
    # line every epoch and drown out the train-loss/val output.
    (logger .debug if quiet else logger .info )(
    "saved %s checkpoint -> %s%s",backbone ,path ,
    f" (epoch {epoch })"if epoch is not None else "")


_SAFE_GLOBALS_DONE =False 


def _allow_numpy_safe_globals (torch ):
    """Allowlist the numpy unpickling primitives that appear in our `meta` dicts
    (plain arrays/dtypes) so a weights_only=True load SUCCEEDS for inference
    checkpoints instead of falling back to the unsafe full load. These globals
    reconstruct numpy arrays only -- they execute no arbitrary code."""
    global _SAFE_GLOBALS_DONE 
    if _SAFE_GLOBALS_DONE :
        return 
    try :
        allow =[np .ndarray ,np .dtype ]
        for modname in ("numpy._core.multiarray","numpy.core.multiarray"):
            try :
                m =__import__ (modname ,fromlist =["_reconstruct","scalar"])
                allow .append (m ._reconstruct )
                if hasattr (m ,"scalar"):
                    allow .append (m .scalar )
            except Exception :# noqa: BLE001
                pass 
                # every concrete numpy dtype class (Float64DType, UInt32DType, ...) plus
                # the scalar types -- all pure data reconstructors, no code execution.
        try :
            import numpy .dtypes as _ndt 
            allow +=[getattr (_ndt ,n )for n in dir (_ndt )
            if isinstance (getattr (_ndt ,n ),type )]
        except Exception :# noqa: BLE001
            pass 
        for nm in ("float64","float32","float16","int64","int32","int16",
        "int8","uint64","uint32","uint16","uint8","bool_"):
            if hasattr (np ,nm ):
                allow .append (getattr (np ,nm ))
        torch .serialization .add_safe_globals (allow )
    except Exception :# noqa: BLE001
        pass 
    _SAFE_GLOBALS_DONE =True 


def load_checkpoint (path ,device ="cpu",weights_only =False ):
    """Load a checkpoint dict (any backbone, training or inference-only).

    weights_only=True is the SAFE load for deployment: it refuses to unpickle
    arbitrary code (these .ckpt travel between machines via git, so a tampered
    file is a real supply-chain risk). It works for inference checkpoints
    (weights + plain-dict/numpy meta/decode) but not for resume checkpoints whose
    optimizer/scheduler state needs full unpickling -- those fall back to False.
    """
    torch =_require_torch ()
    if isinstance (device ,str )and device .startswith ("cuda")and not torch .cuda .is_available ():
        device ="cpu"
    if weights_only :
        _allow_numpy_safe_globals (torch )
    try :
        ckpt =torch .load (path ,map_location =device ,weights_only =weights_only )
    except Exception as exc :# noqa: BLE001
        if not weights_only :
            raise 
        logger .warning ("safe (weights_only) load of %s failed (%s) -- falling back "
        "to a full load; only do this for checkpoints you trust",path ,exc )
        ckpt =torch .load (path ,map_location =device ,weights_only =False )
    if "state_dict"not in ckpt or "meta"not in ckpt :
        raise ValueError (f"{path } is not a cp_regressor checkpoint")
    v =ckpt .get ("ckpt_version")
    if v is not None and v !=CKPT_VERSION :
        logger .warning ("checkpoint %s is format version %s but this code expects %d "
        "-- regenerate/re-export it if loading fails",path ,v ,CKPT_VERSION )
    return ckpt 


def _model_from_ckpt (ckpt ,device ="cpu"):
    """Rebuild the exact architecture from a loaded checkpoint dict and load its
    weights. Returns (model ten `device`, in eval mode; meta; backbone)."""
    backbone =ckpt .get ("backbone")or ckpt ["meta"].get ("backbone","diffusionnet")
    model ,_ =build_regressor (backbone ,_meta_to_config (backbone ,ckpt ["meta"]))
    try :
        model .load_state_dict (ckpt ["state_dict"])
    except RuntimeError as exc :
    # The model definition changed since this checkpoint was written (e.g. the
    # knngraph EdgeConv gained LayerNorm), so the state_dict keys no longer line
    # up. Surface that plainly instead of a wall of "Missing/Unexpected key(s)".
        raise RuntimeError (
        f"'{backbone }' checkpoint is from an OLDER architecture and can't be "
        f"loaded by the current code -- retrain it, or check out the matching "
        f"code revision. (state_dict mismatch: {str (exc ).splitlines ()[0 ]})"
        )from exc 
    model =model .to (device )
    model .eval ()
    return model ,ckpt ["meta"],backbone 


def load_model (path ,device ="cpu"):
    """Rebuild a trained regressor from any checkpoint for inference.

    Works for all backbones and both old ({state_dict, meta}) and new
    full-training checkpoints. Returns (model, meta, backbone); model is ten
    `device` and in eval mode.
    """
    ckpt =load_checkpoint (path ,device =device ,weights_only =True )
    model ,meta ,backbone =_model_from_ckpt (ckpt ,device =device )
    logger .info ("loaded %s checkpoint <- %s",backbone ,path )
    return model ,meta ,backbone 


def load_inference (path ,device ="cpu"):
    """Like load_model, but also returns the decode operating point bundled with
    the artifact. Returns (model, meta, backbone, decode) where decode is a dict
    with heatmap_thresh / min_votes / nms_clearance_mm / dist_thresh_mm --
    DEFAULT_DECODE for older checkpoints that predate the bundled params.
    """
    ckpt =load_checkpoint (path ,device =device ,weights_only =True )
    model ,meta ,backbone =_model_from_ckpt (ckpt ,device =device )
    decode ={**DEFAULT_DECODE ,**(ckpt .get ("decode")or {})}
    # warn (don't crash) when the artifact predates / postdates the current
    # encode/decode semantics, so a stale .pt that would mis-decode is visible.
    schema =ckpt .get ("decode_schema")
    if schema is None :
        logger .warning ("model %s has no decode_schema (pre-versioning export) -- "
        "verify it was trained with the current encode/decode before "
        "trusting its points",path )
    elif schema !=DECODE_SCHEMA :
        logger .warning ("model %s decode_schema=%s != current %d -- encode/decode "
        "semantics changed; re-export or re-train this model",
        path ,schema ,DECODE_SCHEMA )
    logger .info ("loaded %s checkpoint <- %s (decode=%s)",backbone ,path ,decode )
    return model ,meta ,backbone ,decode 


def export_inference_checkpoint (src_path ,dst_path ,decode =None ,device ="cpu"):
    """Write a lean inference-only artifact from a (possibly fat training)
    checkpoint.

    Keeps weights + meta + backbone (+ decode operating point); drops the
    optimizer/scheduler/epoch/RNG/history a resumable checkpoint carries. This
    roughly halves the file and makes it an unambiguous deployment model. If
    `decode` is None the source checkpoint's own decode (if any) is preserved.
    Atomic write. Returns dst_path.
    """
    torch =_require_torch ()
    import os 
    ckpt =load_checkpoint (src_path ,device =device ,weights_only =True )
    backbone =ckpt .get ("backbone")or ckpt ["meta"].get ("backbone","diffusionnet")
    lean ={"ckpt_version":CKPT_VERSION ,"decode_schema":DECODE_SCHEMA ,
    "backbone":backbone ,
    "state_dict":ckpt ["state_dict"],"meta":ckpt ["meta"]}
    if decode is not None :
        lean ["decode"]=dict (decode )
    elif ckpt .get ("decode"):
        lean ["decode"]=dict (ckpt ["decode"])
    d =os .path .dirname (os .path .abspath (dst_path ))
    if d :
        os .makedirs (d ,exist_ok =True )
    tmp =dst_path +".tmp"
    torch .save (lean ,tmp )
    os .replace (tmp ,dst_path )
    logger .info ("exported lean %s inference model -> %s%s",backbone ,dst_path ,
    f" (decode={lean ['decode']})"if "decode"in lean else "")
    return dst_path 


    # Backward-compatible wrappers (the readme and older scripts reference these).

def save_diffusionnet (model ,meta ,path ):
    """Save a trained DiffusionNet regressor (weights + meta) to `path`."""
    save_checkpoint (path ,model ,meta ,"diffusionnet")


def load_diffusionnet (path ,device ="cpu"):
    """Rebuild a DiffusionNet regressor from a checkpoint. Returns (model, meta)."""
    model ,meta ,_ =load_model (path ,device =device )
    return model ,meta 


    # ---------------------------------------------------------------------------
    # shared training plumbing: setup/resume + per-epoch bookkeeping
    # ---------------------------------------------------------------------------

    # train_config keys that MUST match for a resume to be coherent. heat_pos_weight
    # changes the loss; val_frac/seed/split_group change the val set best_f1 is
    # measured ten; best_metric changes what best_f1 even means; augment changes the
    # fit set. w_heat/w_off/w_dir/weight_decay/centernet_alpha/lr all change the loss
    # surface being optimised -- a resume that silently changes any of these keeps
    # the old best_f1/optimizer momentum but starts training a DIFFERENT objective,
    # with no warning (previously missing from this tuple despite already being
    # stored in train_config). Continuing across a change to any of these compares
    # apples to oranges.
_RESUME_CRITICAL =("val_frac","seed","split_seed","split_group",
"family_boost","exclude_parts_file","heat_pos_weight",
"heat_loss","focal_gamma","augment","best_metric",
"max_gpu_verts","w_heat","w_off","w_dir",
"weight_decay","centernet_alpha","lr",
"lr_schedule","lr_decay_every","lr_decay_rate",
"warmup_epochs","eval_every","patience","min_delta",
"offset_loss","offset_huber_beta",
# which parts train with sign-invariant direction loss changes
# the loss surface exactly like a w_dir change would
"sign_inv_prefixes","dir_sign_invariant")

_RESUME_METADATA_FIELDS =("epochs","eval_every","patience","lr_schedule",
"lr_decay_every","lr_decay_rate","warmup_epochs",
"ckpt_every","snapshot_every","min_delta")


def _check_resume_config (stored ,current ,strict ):
    """Warn (or, if strict, raise) when a --resume changes a critical hyperparam,
    so a continuation never silently compares against an incomparable best_f1."""
    if not stored :
        logger .warning ("resume: checkpoint has no stored train_config -- cannot "
        "verify the continuation matches; proceeding")
        return 
    missing =[k for k in _RESUME_METADATA_FIELDS if k in current and k not in stored ]
    if missing :
        logger .warning ("resume: checkpoint train_config is missing metadata field(s) "
        "%s -- scheduler/early-stop provenance may be incomplete",
        ", ".join (missing ))
    diffs =[]
    for k in _RESUME_CRITICAL :
        if k in stored and k in current and stored [k ]!=current [k ]:
            diffs .append ("%s %r->%r"%(k ,stored [k ],current [k ]))
    if diffs :
        msg ="resume: training config changed vs the checkpoint ("+"; ".join (diffs )+") -- best_f1/history are no longer comparable"
        if strict :
            raise ValueError (msg +" [--strict-resume]")
        logger .warning (msg +" -- continuing anyway (pass --strict-resume to refuse)")


def patch_graph_weight (n_patches ):
    """Per-patch loss weight for one original part split into N spatial patches."""
    return 1.0 /max (1 ,int (n_patches ))


def hard_mining_probabilities (loss_ema ):
    """Sampling probabilities for hard-mining prepared graphs/patches.

    The unit is deliberately the prepared graph, not the original part. Patch
    graph weights still keep a patched part's total gradient budget normalized.
    """
    w =np .asarray (loss_ema ,dtype =np .float64 ).copy ()
    if w .size ==0 :
        return w 
    mean =float (w .mean ())
    floor =mean /4.0 if mean >0 else 1.0 
    w =np .maximum (w ,floor )
    total =float (w .sum ())
    if not np .isfinite (total )or total <=0 :
        return np .full_like (w ,1.0 /len (w ),dtype =np .float64 )
    return w /total 


def _format_lrs (opt ):
    if opt is None :
        return "n/a"
    vals =[]
    for group in opt .param_groups :
        lr =float (group .get ("lr",0.0 ))
        if not vals or abs (vals [-1 ]-lr )>1e-15 :
            vals .append (lr )
    return ",".join (f"{v :.3g}"for v in vals )


def _guard_oom_fallback (accum_since_step ):
    """Abort instead of silently discarding previous accumulated gradients."""
    if accum_since_step :
        raise RuntimeError (
        "CUDA OOM occurred after previous accumulated gradients in this "
        "optimizer window; aborting to avoid silently discarding gradients. "
        "Rerun with --accum-steps 1 or lower --max-gpu-verts.")


def _build_scheduler (opt ,lr_schedule ,lr_decay_every ,lr_decay_rate ,
warmup_epochs ,total_epochs ):
    """Return (scheduler, is_metric_driven). 'step' = StepLR (the default);
    'plateau' = ReduceLROnPlateau ten the val metric (stepped by the bookkeeper);
    'cosine' = cosine decay, both with an optional linear warmup prepended."""
    torch =_require_torch ()
    sl =torch .optim .lr_scheduler 
    warm =int (warmup_epochs or 0 )
    if lr_schedule =="plateau":
    # mode='max': we maximise val F1; patience/factor are gentle defaults
        return sl .ReduceLROnPlateau (opt ,mode ="max",factor =float (lr_decay_rate ),
        patience =max (1 ,int (lr_decay_every or 5 ))),True 
    if lr_schedule =="cosine":
        T =max (1 ,int (total_epochs or 100 )-warm )
        cos =sl .CosineAnnealingLR (opt ,T_max =T )
        if warm :
            wu =sl .LinearLR (opt ,start_factor =0.01 ,total_iters =warm )
            return sl .SequentialLR (opt ,[wu ,cos ],milestones =[warm ]),False 
        return cos ,False 
        # default: step decay (optionally warmed up)
    if not (lr_decay_every and lr_decay_every >0 ):
        if warm :
            return sl .LinearLR (opt ,start_factor =0.01 ,total_iters =warm ),False 
        return None ,False 
    step =sl .StepLR (opt ,step_size =int (lr_decay_every ),gamma =float (lr_decay_rate ))
    if warm :
        wu =sl .LinearLR (opt ,start_factor =0.01 ,total_iters =warm )
        return sl .SequentialLR (opt ,[wu ,step ],milestones =[warm ]),False 
    return step ,False 


def _init_training (backbone ,config ,resume_from ,device ,lr ,
lr_decay_every ,lr_decay_rate ,seed ,
train_config =None ,resume_strict =False ,history_path =None ,
weight_decay =0.0 ,lr_schedule ="step",warmup_epochs =0 ,
total_epochs =None ,init_from =None ):
    """Common setup for every train loop: seed the RNGs, build the model and
    optimizer/scheduler -- or restore all of them from `resume_from` (a path or
    a pre-loaded checkpoint dict) so training continues exactly where it
    stopped. Returns (model, meta, opt, sched, start_epoch, best_f1, history,
    sched_is_metric_driven).

    init_from (path): FINE-TUNE init -- load ONLY the model weights from a
    (pretrained) checkpoint, then start a FRESH run (epoch 0, fresh optimizer/
    scheduler, empty history, best reset). This is the pretrain->finetune recipe:
    pretrain ten the large synthetic set, then `init_from` those weights and
    fine-tune ten the real corpus. Ignored when resume_from is given (resume wins).
    The architecture is rebuilt from the pretrained checkpoint's meta, so the
    fine-tune MUST use the same backbone hyperparameters (k / width / layers /
    normals) -- mismatched shapes raise ten load_state_dict.
    """
    import random 
    torch =_require_torch ()
    random .seed (seed );np .random .seed (seed );torch .manual_seed (seed )

    # WDDM (Windows) NVIDIA drivers satisfy over-VRAM cudaMalloc from SYSTEM
    # memory ("sysmem fallback") instead of raising OOM. That defeats every
    # OOM->CPU fallback in this file: nothing ever fails, the allocator pool
    # just spills over PCIe and the run quietly crawls 10-100x slower
    # (observed: 10.6 GB "shared GPU memory", ~2 h/epoch ten a 4 GB T1200).
    # Capping the caching allocator BELOW physical VRAM restores the designed
    # behaviour: it flushes its cache and retries first, and only then raises
    # a REAL OOM that the existing handlers catch.
    if str (device ).startswith ("cuda")and torch .cuda .is_available ():
        try :
            torch .cuda .set_per_process_memory_fraction (
            0.93 ,torch .device (device ).index or 0 )
        except Exception as exc :# noqa: BLE001
            logger .warning ("could not cap the CUDA allocator (%s)",exc )

            # FINE-TUNE init: weights only, everything else fresh (only when not resuming)
    if init_from and not resume_from :
        pre =(load_checkpoint (init_from ,device =device )
        if isinstance (init_from ,str )else init_from )
        pre_backbone =pre .get ("backbone")or pre ["meta"].get ("backbone","diffusionnet")
        if pre_backbone !=backbone :
            raise ValueError (f"--init-from checkpoint is for backbone "
            f"{pre_backbone !r }, but --backbone {backbone !r } requested")
        model ,meta =build_regressor (backbone ,_meta_to_config (backbone ,pre ["meta"]))
        model =model .to (device )
        missing =model .load_state_dict (pre ["state_dict"],strict =True )
        opt =torch .optim .AdamW (model .parameters (),lr =lr ,
        weight_decay =float (weight_decay ))
        sched ,sched_metric =_build_scheduler (opt ,lr_schedule ,lr_decay_every ,
        lr_decay_rate ,warmup_epochs ,total_epochs )
        logger .info ("fine-tune: loaded pretrained weights from %s (fresh optimizer/"
        "epoch/history)",init_from )
        return model ,meta ,opt ,sched ,0 ,-1.0 ,[],sched_metric 

    state =None 
    if resume_from :
        state =(load_checkpoint (resume_from ,device =device )
        if isinstance (resume_from ,str )else resume_from )
        ck_backbone =state .get ("backbone")or state ["meta"].get ("backbone",
        "diffusionnet")
        if ck_backbone !=backbone :
            raise ValueError (f"checkpoint is for backbone {ck_backbone !r }, "
            f"but --backbone {backbone !r } was requested")
        if train_config is not None :
            _check_resume_config (state .get ("train_config"),train_config ,resume_strict )
            # rebuild the EXACT architecture from the checkpoint meta, not from
            # the (possibly different) fresh config
        model ,_ =build_regressor (backbone ,_meta_to_config (backbone ,state ["meta"]))
        meta =state ["meta"]
    else :
        model ,meta =build_regressor (backbone ,config )

    model =model .to (device )
    if state is not None :
        model .load_state_dict (state ["state_dict"])
        # AdamW so --weight-decay is real L2 regularisation (decoupled); wd=0 == Adam.
    opt =torch .optim .AdamW (model .parameters (),lr =lr ,weight_decay =float (weight_decay ))
    sched ,sched_metric =_build_scheduler (opt ,lr_schedule ,lr_decay_every ,
    lr_decay_rate ,warmup_epochs ,total_epochs )

    start_epoch ,best_f1 ,history =0 ,-1.0 ,[]
    if state is not None :
        if state .get ("optimizer"):
            opt .load_state_dict (state ["optimizer"])# state tensors are
        if sched is not None and state .get ("scheduler"):# remapped to the
            sched .load_state_dict (state ["scheduler"])# params' device
        start_epoch =int (state .get ("epoch",-1 ))+1 
        best_f1 =float (state .get ("best_f1",-1.0 ))
        # prefer the FULL history from the sibling JSON (the checkpoint only
        # embeds a capped tail); fall back to the embedded copy.
        history =_load_history_json (history_path )or list (state .get ("history")or [])
        _restore_rng (state .get ("rng"))
        logger .info ("resumed %s checkpoint (next epoch %d, best %s=%.4f)",
        backbone ,start_epoch ,
        (train_config or {}).get ("best_metric","val F1"),best_f1 )
    return model ,meta ,opt ,sched ,start_epoch ,best_f1 ,history ,sched_metric 


def _load_history_json (history_path ):
    """Read the full per-eval history list from the sibling JSON, or None."""
    if not history_path :
        return None 
    import os 
    if not os .path .exists (history_path ):
        return None 
    try :
        import json as _json 
        with open (history_path ,encoding ="utf-8")as fh :
            data =_json .load (fh )
        return data if isinstance (data ,list )else None 
    except Exception :# noqa: BLE001
        return None 


class _Bookkeeper :
    """Per-epoch bookkeeping shared by all backbones: log the train loss, run
    validation metrics every `eval_every` epochs (accuracy / precision /
    recall / F1 / localisation / angular error), keep the best-by-val-F1 full
    checkpoint at `best_path`, roll a resumable `last` checkpoint at
    `last_path` every `save_every` epochs, persist the metric history to
    `history_path` (JSON), and early-stop after `patience` stale evals."""

    def __init__ (self ,model ,meta ,backbone ,opt ,sched ,epochs ,*,
    metrics_fn =None ,eval_every =5 ,patience =0 ,
    best_path =None ,last_path =None ,save_every =1 ,
    history_path =None ,log_every =1 ,best_f1 =-1.0 ,history =None ,
    decode =None ,train_config =None ,best_metric ="micro_f1",
    snapshot_every =0 ,snapshot_stem =None ,sched_metric =False ,
    min_delta =0.0 ,collapse_gap =0.03 ):
        self .model ,self .meta ,self .backbone =model ,meta ,backbone 
        self .opt ,self .sched ,self .epochs =opt ,sched ,epochs 
        # a ReduceLROnPlateau scheduler is stepped HERE with the val metric (the
        # train loop must NOT step it per-epoch); others step per-epoch in the loop
        self .sched_metric =bool (sched_metric )
        self .metrics_fn ,self .eval_every =metrics_fn ,max (1 ,int (eval_every ))
        self .patience =int (patience )
        self .best_path ,self .last_path =best_path ,last_path 
        self .save_every =int (save_every )
        self .history_path ,self .log_every =history_path ,log_every 
        self .best_f1 =float (best_f1 )
        self .history =history if history is not None else []
        self .min_delta =max (0.0 ,float (min_delta or 0.0 ))
        self .collapse_gap =max (0.0 ,float (collapse_gap or 0.0 ))
        # decode + train_config are bundled into every saved checkpoint so a raw
        # .ckpt deploys with the run's real operating point (not DEFAULT_DECODE)
        # and a --resume can detect a mismatched continuation.
        self .decode =decode 
        self .train_config =train_config 
        self .best_metric =best_metric # micro_f1 (deploy metric) by default
        self .snapshot_every =int (snapshot_every )
        # default snapshot name stem = best_path without its extension
        if snapshot_stem is None and best_path :
            import os 
            snapshot_stem =os .path .splitext (best_path )[0 ]
        self .snapshot_stem =snapshot_stem 
        self .stale =0 
        self .saved_best =False 
        self .had_valid_eval =False 
        self .last_epoch_done =-1 
        self .best_epoch =self ._infer_best_epoch ()

    def _metric_from_record (self ,rec ):
        v =rec .get ("val_"+self .best_metric )
        if isinstance (v ,(int ,float ))and np .isfinite (float (v )):
            return float (v )
        return None 

    def _metric_records (self ):
        rows =[]
        for rec in self .history :
            v =self ._metric_from_record (rec )
            if v is not None :
                rows .append ((int (rec .get ("epoch",-1 )),v ))
        return rows 

    def _infer_best_epoch (self ):
        rows =self ._metric_records ()
        if not rows :
            return None 
        ep ,val =max (rows ,key =lambda x :x [1 ])
        if self .best_f1 <0 or abs (val -self .best_f1 )<=max (self .min_delta ,1e-9 ):
            return ep 
        return None 

    def _save (self ,path ,epoch ,quiet =True ):
        save_checkpoint (path ,self .model ,self .meta ,self .backbone ,
        optimizer =self .opt ,scheduler =self .sched ,epoch =epoch ,
        best_f1 =self .best_f1 ,best_epoch =self .best_epoch ,
        history =self .history ,
        decode =self .decode ,train_config =self .train_config ,
        best_metric =self .best_metric ,quiet =quiet )

    def flush_history (self ):
        if not self .history_path :
            return 
        import json as _json 
        import math 
        import os 
        d =os .path .dirname (os .path .abspath (self .history_path ))
        if d :
            os .makedirs (d ,exist_ok =True )
            # NaN (e.g. loc/ang error with zero matches) is not valid strict JSON
        clean =[{k :(None if isinstance (v ,float )and not math .isfinite (v )else v )
        for k ,v in rec .items ()}for rec in self .history ]
        with open (self .history_path ,"w",encoding ="utf-8")as fh :
            _json .dump (clean ,fh ,indent =2 )

    def after_epoch (self ,ep ,mean_loss ):
        """End-of-epoch hook. Returns True when training should stop early."""
        self .last_epoch_done =ep 
        if self .log_every and (ep %self .log_every ==0 or ep ==self .epochs -1 ):
            logger .info ("epoch %3d/%d  train loss=%.5f  lr=%s",
            ep ,self .epochs ,mean_loss ,_format_lrs (self .opt ))

        stop =False 
        if self .metrics_fn is not None and (
        (ep %self .eval_every ==0 and ep >0 )
        or ep ==self .epochs -1 ):
            logger .info ("  epoch %d: running validation eval "
            "(slow on large parts) ...",ep )
            m =self .metrics_fn (self .model ,self .meta )or {}
            rec ={"epoch":ep ,"train_loss":float (mean_loss )}
            rec .update ({"val_"+k :v for k ,v in m .items ()})
            self .history .append (rec )
            nan =float ("nan")
            # select the BEST checkpoint ten the same metric we report/deploy
            # (micro_f1 by default), not the per-part macro mean.
            sel =m .get (self .best_metric ,nan )
            pct =lambda v :v *100.0 if isinstance (v ,float )else nan 
            logger .info ("  epoch %d  VAL  accuracy=%.1f%%  precision=%.1f%%  "
            "recall=%.1f%%  F1=%.1f%% (micro %.1f%%)  loc=%.2f mm  ang=%.1f deg",
            ep ,pct (m .get ("accuracy",nan )),pct (m .get ("precision",nan )),
            pct (m .get ("recall",nan )),pct (m .get ("f1",nan )),
            pct (m .get ("micro_f1",nan )),
            m .get ("mean_loc_err_mm",nan ),m .get ("mean_ang_err_deg",nan ))
            if isinstance (sel ,float )and not np .isnan (sel ):
                self .had_valid_eval =True 
                # metric-driven LR schedule (ReduceLROnPlateau) steps ten the val metric
                if self .sched_metric and self .sched is not None :
                    self .sched .step (sel )
                improvement =float (sel )-self .best_f1 
                # TWO SEPARATE QUESTIONS, previously conflated into one min_delta test:
                #
                #   (a) "is this the best model so far?"  -> ANY improvement counts.
                #   (b) "is the run still making progress?" -> only a min_delta-sized
                #       gain counts (that is what patience is for).
                #
                # Gating (a) ten min_delta silently throws away the real best: with
                # min_delta=0.005 a run climbing 0.890 -> 0.893 -> 0.895 -> 0.897 saves
                # NOTHING and reports 0.890 as its best -- and that is exactly the regime
                # near the peak, where the per-eval gains get small. It also froze
                # best_f1, so every later comparison was against a stale number.
                is_best =self .best_f1 <0 or improvement >0.0 
                progressed =self .best_f1 <0 or improvement >=max (self .min_delta ,1e-12 )
                if is_best :
                    self .best_f1 ,self .best_epoch =float (sel ),ep 
                    if self .best_path :
                        self ._save (self .best_path ,ep )
                        self .saved_best =True 
                        logger .info ("  new best %s=%.4f -> %s",self .best_metric ,
                        sel ,self .best_path )
                        # patience counts STALE evals, i.e. evals that did not advance the metric
                        # by min_delta -- independently of whether the checkpoint was refreshed.
                if progressed :
                    self .stale =0 
                else :
                    self .stale +=1 
                    logger .info ("  no %s progress for %d eval(s) "
                    "(best=%.4f, min_delta=%.4f%s)",
                    self .best_metric ,self .stale ,self .best_f1 ,
                    self .min_delta ,
                    "; best ckpt still refreshed"if is_best else "")
                stop =bool (self .patience )and self .stale >=self .patience 
            self .flush_history ()

            # periodic, never-overwritten snapshots so an earlier good model is
            # recoverable if a later "best" turns out to overfit.
        if (self .snapshot_every and self .snapshot_stem 
        and (ep +1 )%self .snapshot_every ==0 ):
            snap ="%s_ep%04d.ckpt"%(self .snapshot_stem ,ep )
            self ._save (snap ,ep ,quiet =False )
            logger .info ("  snapshot -> %s",snap )

        if self .last_path and ((self .save_every >0 and (ep +1 )%self .save_every ==0 )
        or ep ==self .epochs -1 or stop ):
            self ._save (self .last_path ,ep )
            # visible confirmation every 10 epochs (and at the end / early stop) so
            # you know there's a fresh resumable checkpoint and it's safe to stop.
            if (ep +1 )%10 ==0 or ep ==self .epochs -1 or stop :
                logger .info ("  checkpoint saved at epoch %d -> %s  "
                "[safe to stop; resume with --resume]",ep ,self .last_path )
        return stop 

    def finalize (self ):
        """After the loop: make sure a best checkpoint exists even without a
        validation set (no metrics_fn) and the history file is written. When
        valid evals DID happen but none beat the resumed best_f1, the existing
        best checkpoint is deliberately left untouched -- the final model is
        worse than the historical best."""
        if (self .best_path and not self .saved_best and not self .had_valid_eval 
        and self .last_epoch_done >=0 ):
        # no eval ever produced a valid metric (no val set, or every eval was
        # NaN because TP+FP+FN=0) -- still leave a usable best from the final
        # epoch so deployment is never left without an artifact.
            if self .metrics_fn is not None :
                logger .warning ("no valid validation metric was ever produced "
                "(all NaN?) -- saving the final-epoch model as best")
            self .best_epoch =self .last_epoch_done 
            self ._save (self .best_path ,self .last_epoch_done )
        rows =self ._metric_records ()
        if rows :
            if self .best_epoch is None :
                hist_best_ep ,hist_best =max (rows ,key =lambda x :x [1 ])
                if hist_best >=self .best_f1 :
                    self .best_epoch ,self .best_f1 =hist_best_ep ,hist_best 
            last_ep ,last_val =rows [-1 ]
            gap =self .best_f1 -last_val 
            best_ep ="unknown"if self .best_epoch is None else str (self .best_epoch )
            logger .info ("validation summary: best %s=%.4f at epoch %s; "
            "last eval epoch %d %s=%.4f; gap=%.4f",
            self .best_metric ,self .best_f1 ,best_ep ,last_ep ,
            self .best_metric ,last_val ,gap )
            if gap >self .collapse_gap :
                logger .warning ("validation collapse detected: last %s is %.4f below "
                "best (threshold %.4f). Promote best.ckpt, not last.ckpt.",
                self .best_metric ,gap ,self .collapse_gap )
        self .flush_history ()


        # ---------------------------------------------------------------------------
        # loss: heatmap BCE + masked offset L1 + masked direction cosine
        # ---------------------------------------------------------------------------

def cp_loss (pred ,target ,mask ,w_heat =1.0 ,w_off =5.0 ,w_dir =1.0 ,
heat_pos_weight =50.0 ,heat_loss ="bce",focal_gamma =2.0 ,
centernet_alpha =2.0 ,
offset_heat_weight =True ,part_weight =1.0 ,
dir_sign_invariant =False ,offset_loss ="l1",offset_huber_beta =0.01 ,
as_tensors =False ):
    """Combined regression loss.

    pred   : (N,7) raw model output (heatmap channel is a logit).
    target : (N,7) from prepare_sample (heatmap in [0,1]; offset in NORMALISED
             units, i.e. mm/scale -- see prepare_sample). Because the offset is
             already scale-normalised, no per-mesh rescaling happens here.
    mask   : (N,) bool, vertices near a CP (offset/direction supervised there).
    offset_heat_weight : weight each masked vertex's offset/direction error by its
             TARGET heat, so the vertices that will actually VOTE at decode (high
             heat) are the ones whose offsets are pushed most accurate -- couples
             the otherwise-independent heat and offset heads.
    part_weight : scalar multiplier ten this part's total loss (e.g. #keypoints) so
             a multi-CP part is not down-weighted to the same step as a 1-CP part.
    offset_loss : "l1" (default, plain mean-abs-error -- constant-magnitude gradient
             regardless of error size) or "huber" (smooth-L1: quadratic below
             offset_huber_beta, linear above). Measured ten a real trained model:
             66% of false positives are "near-miss" (within 10mm of a real CP but
             outside the 5mm match radius, median ~4.6mm off) -- a mixed population
             of easy/near-correct and hard/far-off offset predictions, which plain
             L1's constant gradient doesn't preferentially "polish" before roughly
             right. Huber concentrates gradient ten shrinking the last few mm of an
             already-close prediction (quadratic regime) while staying robust to
             genuinely bad predictions (linear regime, same slope as L1 there) --
             offset_huber_beta=0.01 (normalised units) is calibrated to the typical
             observed offset-error scale (~4-6mm at typical part scale ~500-600mm).
    heat_loss : "bce"   -> per-vertex BCE weighted by (1 + heat_pos_weight*h) so
                           the few high-heat vertices are not drowned out;
                "focal" -> quality-focal loss |h - sigmoid(logit)|^gamma * BCE,
                           which down-weights the easy ~0 background and tends to
                           improve precision ten the sparse-keypoint heatmap.
                "centernet" -> penalty-reduced focal loss (CornerNet/CenterNet,
                           Law&Deng 2018 / Zhou 2019) NORMALISED BY #KEYPOINTS, not
                           by #vertices. The heat target is ~99% background, so a
                           per-vertex mean() (bce/focal) dilutes the handful of
                           positive vertices into the noise -- the model then either
                           collapses to ~0 (focal/low pos_weight) or over-fires
                           broadly (high pos_weight). Dividing the loss by the number
                           of positive (peak) vertices removes that dilution, which
                           is what lets the heatmap learn sharp, selective peaks.
    """
    torch =_require_torch ()
    import torch .nn .functional as Fnn 
    heat_logit =pred [:,ct .HEATMAP ]
    heat_tgt =target [:,ct .HEATMAP ]
    if heat_loss =="centernet":
    # alpha: focuses ten hard examples. Lower alpha (1.0-1.5) penalises FN
    # less harshly → model fires more freely → higher recall at cost of
    # precision. Standard CenterNet uses alpha=2 (precision-biased). Use
    # --centernet-alpha 1.5 to shift the recall/precision balance.
    # beta: reduces penalty ten the Gaussian skirt (near-positive vertices
    # not punished as hard negatives).
        alpha ,beta =float (centernet_alpha ),4.0 
        p =torch .sigmoid (heat_logit ).clamp (1e-6 ,1.0 -1e-6 )
        pos =(heat_tgt >=1.0 -1e-4 ).float ()# exact peaks (encode_targets snaps them)
        neg =1.0 -pos 
        neg_w =(1.0 -heat_tgt ).pow (beta )
        pos_loss =((1.0 -p ).pow (alpha )*torch .log (p ))*pos 
        neg_loss =(p .pow (alpha )*torch .log (1.0 -p ))*neg_w *neg 
        n_pos =pos .sum ().clamp (min =1.0 )
        loss_heat =-(pos_loss .sum ()+neg_loss .sum ())/n_pos 
    elif heat_loss =="focal":
        bce =Fnn .binary_cross_entropy_with_logits (heat_logit ,heat_tgt ,
        reduction ="none")
        mod =(heat_tgt -torch .sigmoid (heat_logit )).abs ().pow (focal_gamma )
        loss_heat =(mod *bce ).mean ()
    else :
        heat_w =1.0 +heat_pos_weight *heat_tgt 
        loss_heat =Fnn .binary_cross_entropy_with_logits (
        heat_logit ,heat_tgt ,weight =heat_w )

    if mask .any ():
        m =mask 
        if offset_loss =="huber":
            off_err =Fnn .smooth_l1_loss (pred [m ][:,ct .OFFSET ],target [m ][:,ct .OFFSET ],
            reduction ="none",beta =offset_huber_beta ).mean (dim =-1 )
        else :
            off_err =(pred [m ][:,ct .OFFSET ]-target [m ][:,ct .OFFSET ]).abs ().mean (dim =-1 )
        dir_pred =Fnn .normalize (pred [m ][:,ct .DIRECTION ],dim =-1 ,eps =1e-8 )
        dir_tgt =Fnn .normalize (target [m ][:,ct .DIRECTION ],dim =-1 ,eps =1e-8 )
        cos_sim =(dir_pred *dir_tgt ).sum (-1 )
        # sign-invariant: treat dir and -dir as equally correct (for labels where
        # the insert-axis sign is ambiguous, e.g. step_openings ten symmetric holes)
        dir_err =1.0 -(cos_sim .abs ()if dir_sign_invariant else cos_sim )
        if offset_heat_weight :
            w =heat_tgt [m ].clamp (min =1e-3 )# vote weight = target heat
            wsum =w .sum ().clamp (min =1e-6 )
            loss_off =(w *off_err ).sum ()/wsum 
            loss_dir =(w *dir_err ).sum ()/wsum 
        else :
            loss_off =off_err .mean ()
            loss_dir =dir_err .mean ()
    else :
        loss_off =pred .sum ()*0.0 
        loss_dir =pred .sum ()*0.0 

    total =(w_heat *loss_heat +w_off *loss_off +w_dir *loss_dir )*part_weight 
    n_pos_t =(heat_tgt >=1.0 -1e-4 ).sum ()
    if as_tensors :
    # Detached 0-dim tensors instead of floats: float(cuda_tensor) forces a
    # GPU->CPU sync, and the training hot loop calls this per PART (~14k
    # syncs/epoch measured ten v19). Callers that need numbers stack these and
    # materialise ONCE per epoch.
        return total ,{"heat":loss_heat .detach (),"off":loss_off .detach (),
        "dir":loss_dir .detach (),"total":total .detach (),
        "n_pos":n_pos_t }
    return total ,{"heat":float (loss_heat .detach ()),"off":float (loss_off .detach ()),
    "dir":float (loss_dir .detach ()),"total":float (total .detach ()),
    "n_pos":float (n_pos_t )}


def pred_to_array (pred ,offset_scale =1.0 ):
    """Convert raw model output -> (N,7) numpy ready for cp_targets.decode_predictions
    (sigmoid ten heatmap, unit-normalise direction).

    offset_scale : the model predicts offsets in NORMALISED units; multiply by the
    mesh scale (mm) here so decode_predictions, which adds the offset to the raw
    mm vertices, sees millimetres. Pass the sample's 'scale'.
    """
    torch =_require_torch ()
    import torch .nn .functional as Fnn 
    out =pred .detach ().clone ()
    out [:,ct .HEATMAP ]=torch .sigmoid (out [:,ct .HEATMAP ])
    out [:,ct .OFFSET ]=out [:,ct .OFFSET ]*float (offset_scale )
    out [:,ct .DIRECTION ]=Fnn .normalize (out [:,ct .DIRECTION ],dim =-1 ,eps =1e-8 )
    return out .cpu ().numpy ()


    # ---------------------------------------------------------------------------
    # MLP training loop (smoke-test path). Samples: dicts with verts, target, mask, scale.
    # ---------------------------------------------------------------------------

def train_cpmlp (samples ,epochs =300 ,lr =1e-3 ,width =128 ,depth =4 ,
device ="cpu",log_every =50 ,metrics_fn =None ,eval_every =10 ,
patience =0 ,best_path =None ,last_path =None ,save_every =1 ,
history_path =None ,resume_from =None ,seed =0 ,
decode =None ,train_config =None ,best_metric ="micro_f1",
resume_strict =False ,snapshot_every =0 ,weight_decay =0.0 ,
offset_heat_weight =True ,part_weight_mode ="none",grad_clip =0.0 ,
lr_schedule ="step",warmup_epochs =0 ,min_delta =0.0 ,
collapse_gap =0.03 ):
    """Overfit/smoke train the MLP backbone ten prepared samples.

    samples: list of {'verts_norm' (N,3), 'target' (N,7), 'mask' (N,), 'scale'}.
    Supports the same per-epoch val metrics + best/last full-state .ckpt +
    resume machinery as the graph backbones (see train_diffusionnet_regressor
    for the parameter docs). Returns the trained model.
    """
    torch =_require_torch ()
    model ,meta ,opt ,_ ,start_epoch ,best_f1 ,history ,_sm =_init_training (
    "mlp",{"width":width ,"depth":depth },resume_from ,device ,lr ,
    0 ,0.5 ,seed ,train_config =train_config ,resume_strict =resume_strict ,
    history_path =history_path ,weight_decay =weight_decay )
    if start_epoch >=epochs :
        logger .info ("checkpoint already at epoch %d >= --epochs %d; nothing to "
        "train (raise --epochs to continue)",start_epoch ,epochs )
        return model 
    tensors =[]
    for s in samples :
        tensors .append ({
        "x":torch .tensor (s ["verts_norm"],dtype =torch .float32 ,device =device ),
        "t":torch .tensor (s ["target"],dtype =torch .float32 ,device =device ),
        "m":torch .tensor (s ["mask"],dtype =torch .bool ,device =device ),
        "scale":float (s ["scale"]),
        })
    bk =_Bookkeeper (model ,meta ,"mlp",opt ,None ,epochs ,
    metrics_fn =metrics_fn ,eval_every =eval_every ,
    patience =patience ,best_path =best_path ,last_path =last_path ,
    save_every =save_every ,history_path =history_path ,
    log_every =log_every ,best_f1 =best_f1 ,history =history ,
    decode =decode ,train_config =train_config ,
    best_metric =best_metric ,snapshot_every =snapshot_every ,
    min_delta =min_delta ,collapse_gap =collapse_gap )
    try :
        for ep in range (start_epoch ,epochs ):
            model .train ()
            tot =0.0 
            for s in tensors :
                opt .zero_grad (set_to_none =True )
                out =model (s ["x"])
                pw =(max (1.0 ,float ((s ["t"][:,ct .HEATMAP ]>=1.0 -1e-4 ).sum ()))
                if part_weight_mode =="keypoints"else 1.0 )
                loss ,parts =cp_loss (out ,s ["t"],s ["m"],
                offset_heat_weight =offset_heat_weight ,
                part_weight =pw )
                loss .backward ()
                if grad_clip and grad_clip >0 :
                    torch .nn .utils .clip_grad_norm_ (model .parameters (),grad_clip )
                opt .step ()
                tot +=parts ["total"]
            if bk .after_epoch (ep ,tot /max (1 ,len (tensors ))):
                logger .info ("early stopping at epoch %d",ep )
                break 
    except KeyboardInterrupt :
        bk .flush_history ()
        logger .warning ("training interrupted -- resume from the last checkpoint%s",
        f" ({last_path })"if last_path else "")
        raise 
    bk .finalize ()
    return model 


def prepare_sample (part ,dedup =True ,cp_surface_tol_frac =None ,quiet =False ):
    """Part -> training sample dict (normalised verts, encoded target, mask, scale).

    Uses deduped terminal-block locations as the regression targets. Carries the
    raw verts/faces too so the DiffusionNet backbone can build mesh operators.

    cp_surface_tol_frac: if set, DROP any CP whose nearest vertex is farther than
    this fraction of the bbox diagonal (a CP floating off the surface is likely a
    mislabel). When None, off-surface CPs are only reported and kept.
    quiet: suppress the per-part validation / dedup / sigma-cap log lines. Set for
    AUGMENTED clones -- a clone is a rigid (distance-preserving) copy of its base,
    so its warnings are IDENTICAL to the original's and re-printing them before per
    clone just floods the terminal (and made augmentation look like the cause of
    off-surface CPs, which it provably is not -- rotation is an isometry).
    """
    import logging as _logging 
    _prev_disable =_logging .root .manager .disable 
    if quiet :# silence redundant clone logs (dedup/sigma/surface)
        _logging .disable (_logging .WARNING )
    try :
        return _prepare_sample_impl (part ,dedup ,cp_surface_tol_frac )
    finally :
        if quiet :
            _logging .disable (_prev_disable )


def _prepare_sample_impl (part ,dedup ,cp_surface_tol_frac ):
    import json_dataset as jd 
    if dedup :
        _ ,bp ,bd =jd .dedup_connection_points (part )
    else :
        bp =np .asarray (part .cp_points ,dtype =np .float64 )
        bd =np .asarray (part .cp_directions ,dtype =np .float64 )

    V =np .asarray (part .vertices ,dtype =np .float64 )
    # CP-ten-surface check. Most off-surface CPs are RECESSED terminals (a contact a
    # few mm inside a socket) -- legitimate and now learnable (encode_targets forces
    # the peak vertex into the offset mask). Only a CP very far from the mesh is a
    # likely real mislabel (wrong frame/units), so tier the two instead of crying
    # "mislabel" at every recessed terminal.
    if len (bp )and len (V ):
        diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))or 1.0 
        sdist =ct .cp_surface_distances (V ,bp )
        recessed =(sdist >0.05 *diag )&(sdist <=0.15 *diag )
        mislabel =sdist >0.15 *diag 
        if recessed .any ():
            logger .info ("part %s: %d CP(s) recessed 5-15%% of bbox from the surface "
            "(expected for socketed terminals; kept)",
            part .part_nr ,int (recessed .sum ()))
        if mislabel .any ():
            logger .warning ("part %s: %d/%d CP(s) lie >15%% of bbox from any vertex "
            "(max %.1fmm) -- LIKELY MISLABEL (wrong frame/units). Drop "
            "with --cp-surface-tol-frac 0.15",part .part_nr ,
            int (mislabel .sum ()),len (bp ),float (sdist .max ()))
        if cp_surface_tol_frac is not None :
            keep =sdist <=cp_surface_tol_frac *diag 
            if not keep .all ():
                logger .warning ("part %s: dropping %d off-surface CP(s) (> %.1f%% bbox)",
                part .part_nr ,int ((~keep ).sum ()),
                100.0 *cp_surface_tol_frac )
                bp ,bd =bp [keep ],bd [keep ]

    Vn ,center ,scale =normalize_vertices (part .vertices )
    target ,mask ,sigma =ct .encode_targets (part .vertices ,bp ,bd )
    # #3: regress offsets in NORMALISED units (mm/scale) so the target is
    # scale-invariant and matches the scale-normalised input features. The mm
    # offset is recovered at decode via pred_to_array(offset_scale=scale).
    target =target .copy ()
    target [:,ct .OFFSET ]=target [:,ct .OFFSET ]/scale 
    return {"part_nr":str (part .part_nr ),
    "verts_norm":Vn ,"verts":np .asarray (part .vertices ,dtype =np .float64 ),
    "faces":np .asarray (part .faces ,dtype =np .int64 ),
    "target":target ,"mask":mask ,"scale":scale ,
    "center":center ,"sigma":sigma ,"gt_points":bp ,"gt_directions":bd }


_OCTA_CACHE =None 


def _octahedral_matrices ():
    """The 24 proper signed-axis-permutation rotations (det=+1) that map the axes
    onto themselves -- cube rotations. Unlike a continuous rotation these PRESERVE
    axis-alignment (a +-X/+-Y/+-Z CP direction maps to another +-axis), so they
    teach frame/orientation invariance WITHOUT erasing the 98%-axis-aligned prior."""
    global _OCTA_CACHE 
    if _OCTA_CACHE is None :
        import itertools 
        mats =[]
        for perm in itertools .permutations (range (3 )):
            for signs in itertools .product ((-1.0 ,1.0 ),repeat =3 ):
                M =np .zeros ((3 ,3 ))
                for i in range (3 ):
                    M [i ,perm [i ]]=signs [i ]
                if abs (np .linalg .det (M )-1.0 )<1e-9 :
                    mats .append (M )
        _OCTA_CACHE =mats # exactly 24
    return _OCTA_CACHE 


def augment_part (part ,rng ,rotate =True ,jitter_frac =0.0 ,reflect =False ,
dropout_frac =0.0 ,cable_frac =0.0 ,cube_rotate =False ):
    """Return a randomly augmented clone of a Part for training.

    A *rigid* rotation (uniform random, about the vertex centroid) is applied to
    the whole part at before -- vertices, CP points AND CP directions -- so the
    geometry/label relationship is preserved exactly. This is the lever that
    teaches the raw-xyz backbones (mlp/knngraph, which have NO built-in pose
    invariance) to detect connection points at any orientation; without it they
    overfit to the single canonical pose each part ships in.

    reflect (>0 prob): with probability 0.5, also mirror the part across a random
    axis through the centroid. Connector geometry is frequently mirror-symmetric,
    so reflections are plausible unseen parts. A reflection is improper (det=-1),
    so vertices AND CP points/directions are all mirrored together, and face
    winding is flipped to keep outward normals consistent for the mesh backbones.

    jitter_frac (>0) adds Gaussian vertex noise with std = jitter_frac * bbox
    diagonal to the VERTICES ONLY -- the GT CP points stay put, so the network
    learns to localise the true connection point from a noisy/imperfect mesh
    (encode_targets is recomputed downstream against the jittered vertices).

    dropout_frac (>0): drop a spherical region of vertices whose radius =
    dropout_frac * bbox_diagonal (simulates a cable/housing occluding part of the
    scan, or missing data in a real point cloud). Only the vertices are dropped;
    CP GT points are kept unchanged (a partly-occluded terminal is still a target).
    Faces become invalid after dropout -- only safe for knngraph/mlp (point-cloud
    backbones). DiffusionNet parts are NOT dropped (face topology needed). Applied
    AFTER rotation/reflect/jitter so the dropped region is in the final pose.

    terminal names are unchanged. Used to expand the TRAIN set only; the
    validation set is never augmented (so val metrics stay deployment-real).
    """
    import json_dataset as jd 
    import augment as aug 
    V =np .asarray (part .vertices ,dtype =np .float64 )
    cp =np .asarray (part .cp_points ,dtype =np .float64 )
    cd =np .asarray (part .cp_directions ,dtype =np .float64 )
    F =np .asarray (part .faces ,dtype =np .int64 )
    if cube_rotate and len (V ):
        R =_octahedral_matrices ()[int (rng .integers (24 ))]# axis-preserving cube rotation
        c =V .mean (0 )
        V =(V -c )@R .T +c 
        if len (cp ):
            cp =(cp -c )@R .T +c 
            cd =cd @R .T 
    elif rotate and len (V ):
        R =aug ._random_rotation_matrix (rng )
        c =V .mean (0 )
        V =(V -c )@R .T +c 
        if len (cp ):
            cp =(cp -c )@R .T +c 
            cd =cd @R .T # directions rotate, stay unit
    if reflect and len (V )and rng .random ()<0.5 :
        ax =int (rng .integers (3 ))# mirror across one random axis plane
        c =V .mean (0 )
        V =V .copy ();V [:,ax ]=2.0 *c [ax ]-V [:,ax ]
        if len (cp ):
            cp =cp .copy ();cp [:,ax ]=2.0 *c [ax ]-cp [:,ax ]
            cd =cd .copy ();cd [:,ax ]=-cd [:,ax ]# direction mirrors too
        if F .size :# reflection flips winding -> swap 2 cols
            F =F [:,[0 ,2 ,1 ]]
    if jitter_frac and jitter_frac >0.0 and len (V ):
        diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))or 1.0 
        V =V +rng .normal (0.0 ,jitter_frac *diag ,size =V .shape )
    if dropout_frac and dropout_frac >0.0 and len (V )>1 :
        keep ,V =aug .point_dropout (V ,dropout_frac ,rng =rng )
        F =np .empty ((0 ,3 ),dtype =np .int64 )# faces invalid after vertex drop
    if cable_frac and cable_frac >0.0 and len (cp )>0 :
        V ,cp ,cd =aug .cable_distractor (V ,cp ,cd ,frac =cable_frac ,rng =rng )
        F =np .empty ((0 ,3 ),dtype =np .int64 )
    return jd .Part (part_nr =str (part .part_nr )+"~aug",vertices =V ,
    faces =F ,cp_points =cp ,cp_directions =cd ,
    cp_names =list (part .cp_names ))


    # ---------------------------------------------------------------------------
    # production backbone: DiffusionNet training / inference (needs diffusion_net)
    # ---------------------------------------------------------------------------
    # The eigenbasis (mesh operators) is the expensive step at 479-file scale; it is
    # computed ONCE per part and cached to op_cache_dir (diffusion_net hashes the
    # vertices to key the cache), then reused across epochs and runs. Operators are
    # built ten the raw mm mesh; the *input features* are the normalised xyz so the
    # first layer is well-conditioned regardless of part size. Offset targets are
    # scale-normalised (mm/scale) to match the normalised input; pred_to_array
    # multiplies them back by the mesh scale before decoding.

def _cp_operators (verts ,faces ,k_eig =128 ,op_cache_dir =None ):
    """Mesh operators (mass/Laplacian/eigenbasis/gradients) for one part.

    Cached to op_cache_dir when given (recommended for the 479-file corpus).
    """
    torch =_require_torch ()
    import diffusionnet as dnmod 
    dn =dnmod ._require (
    "diffusion_net",
    "clone https://github.com/nmwsharp/diffusion-net and add its src/ to "
    "PYTHONPATH; needs robust_laplacian + potpourri3d + scikit-learn")
    k_eig =int (min (int (k_eig ),dnmod .MAX_K_EIG ))
    V =torch .tensor (np .asarray (verts ),dtype =torch .float32 )
    F =torch .tensor (np .asarray (faces ),dtype =torch .long )
    _ ,mass ,L ,evals ,evecs ,gradX ,gradY =dn .geometry .get_operators (
    V ,F ,k_eig =k_eig ,op_cache_dir =op_cache_dir )
    return {"verts":V ,"faces":F ,"mass":mass ,"L":L ,"evals":evals ,
    "evecs":evecs ,"gradX":gradX ,"gradY":gradY }


def _to_device_ops (ops ,device ):
    return {k :(v .to (device )if hasattr (v ,"to")else v )for k ,v in ops .items ()}


def train_diffusionnet_regressor (train_samples ,config =None ,epochs =200 ,lr =1e-3 ,
device ="cpu",op_cache_dir =None ,log_every =1 ,
metrics_fn =None ,eval_every =5 ,patience =0 ,
w_heat =1.0 ,w_off =5.0 ,w_dir =1.0 ,
heat_pos_weight =50.0 ,heat_loss ="bce",
focal_gamma =2.0 ,max_gpu_verts =60000 ,
best_path =None ,last_path =None ,save_every =1 ,
history_path =None ,resume_from =None ,seed =0 ,
lr_decay_every =0 ,lr_decay_rate =0.5 ,
accum_steps =1 ,low_memory =False ,
decode =None ,train_config =None ,
best_metric ="micro_f1",resume_strict =False ,
snapshot_every =0 ,weight_decay =0.0 ,
lr_schedule ="step",warmup_epochs =0 ,
offset_heat_weight =True ,part_weight_mode ="none",
grad_clip =0.0 ,min_delta =0.0 ,
collapse_gap =0.03 ):
    """Train the production DiffusionNet regressor (C_out=7).

    Each sample needs 'verts','faces','verts_norm','target','mask','scale'
    (see prepare_sample). Operators are precomputed/cached before per part, each
    part is moved to the device for its step then freed. Returns (model, meta).

    seed                 : seeds torch/np/random for reproducible runs (#4).
    lr_decay_every/rate  : StepLR schedule (epochs / gamma); 0 disables it (#4).
    accum_steps          : gradient accumulation over N parts before opt.step.
    metrics_fn(model,meta): optional; runs every `eval_every` epochs and returns
                           the val metric dict (accuracy/precision/recall/f1/
                           loc/ang). Drives best-by-F1 checkpointing to
                           `best_path` and early stopping after `patience`
                           stale evals.
    best_path/last_path  : full-state .ckpt files -- best-by-val-F1 and a rolling
                           resumable snapshot (every `save_every` epochs and at
                           the end). Both contain model+optimizer+scheduler+
                           epoch+RNG+history, so EITHER can be resumed from.
    history_path         : per-eval metric history as JSON (plot/inspect later).
    resume_from          : path (or loaded dict) of a previous .ckpt; training
                           continues at its next epoch toward `epochs` total --
                           works across machines (commit the .ckpt, pull, resume).
    max_gpu_verts        : parts above this run ten CPU (4 GB GPU OOMs ten big
                           meshes); grads from the temp CPU copy are added back to
                           the GPU optimiser, identical to a GPU step.
    low_memory           : do NOT hold every part's eigenbasis resident; reload it
                           from op_cache_dir each step (bounded RAM for corpora
                           larger than memory, at the cost of cache I/O) (#7).
    """
    import copy 
    torch =_require_torch ()
    import diffusionnet as dnmod 
    model ,meta ,opt ,sched ,start_epoch ,best_f1 ,history ,sched_metric =_init_training (
    "diffusionnet",config ,resume_from ,device ,lr ,
    lr_decay_every ,lr_decay_rate ,seed ,train_config =train_config ,
    resume_strict =resume_strict ,history_path =history_path ,
    weight_decay =weight_decay ,lr_schedule =lr_schedule ,
    warmup_epochs =warmup_epochs ,total_epochs =epochs )
    if start_epoch >=epochs :
        logger .info ("checkpoint already at epoch %d >= --epochs %d; nothing to "
        "train (raise --epochs to continue)",start_epoch ,epochs )
        return model ,meta 
    accum =max (1 ,int (accum_steps ))

    def _loss (out ,t ,m ):
        pw =1.0 
        if part_weight_mode =="keypoints":
            pw =max (1.0 ,float ((t [:,ct .HEATMAP ]>=1.0 -1e-4 ).sum ()))
        return cp_loss (out ,t ,m ,w_heat =w_heat ,w_off =w_off ,w_dir =w_dir ,
        heat_pos_weight =heat_pos_weight ,heat_loss =heat_loss ,
        focal_gamma =focal_gamma ,offset_heat_weight =offset_heat_weight ,
        part_weight =pw )

    def _accumulate (ops ,d ,run_device ,src_device ,loss_div ):
        """Forward+backward for one part, accumulating gradient into `model`
        (no zero_grad / no step here -- the caller steps at accumulation
        boundaries). Oversized/OOM parts run ten a temporary copy ten run_device
        and copy their gradient back to the model ten src_device."""
        ops_d =_to_device_ops (ops ,run_device )
        x =(d ["x"].to (run_device )if meta ["input_features"]=="xyz"
        else dnmod ._model_input (ops_d ,meta ))
        t =d ["t"].to (run_device );m =d ["m"].to (run_device )
        if run_device ==src_device :
            out =dnmod ._forward (model ,ops_d ,x )
            loss ,parts =_loss (out ,t ,m )
            (loss /loss_div ).backward ()
        else :
            cm =copy .deepcopy (model ).to (run_device )
            out =dnmod ._forward (cm ,ops_d ,x )
            loss ,parts =_loss (out ,t ,m )
            (loss /loss_div ).backward ()
            for pg ,pc in zip (model .parameters (),cm .parameters ()):
                if pc .grad is not None :
                    g =pc .grad .detach ().to (src_device )
                    pg .grad =g if pg .grad is None else (pg .grad +g )
            del cm 
        del ops_d ,x ,t ,m ,out ,loss 
        return parts 

        # #1: precompute operators with a per-part guard -- one malformed mesh
        # (degenerate faces, eigensolver non-convergence) must not kill the whole run.
    logger .info ("DiffusionNet regressor: preparing operators for %d parts "
    "(k_eig=%d, cache=%s, low_memory=%s) ...",len (train_samples ),
    meta ["k_eig"],op_cache_dir ,low_memory )
    prepared ,skipped =[],0 
    for s in train_samples :
        item ={"x":torch .tensor (s ["verts_norm"],dtype =torch .float32 ),
        "t":torch .tensor (s ["target"],dtype =torch .float32 ),
        "m":torch .tensor (s ["mask"],dtype =torch .bool ),
        "scale":float (s ["scale"]),
        "verts":s ["verts"],"faces":s ["faces"]}
        try :
            ops =_cp_operators (s ["verts"],s ["faces"],meta ["k_eig"],op_cache_dir )
        except Exception as exc :# noqa: BLE001 (skip bad mesh)
            skipped +=1 
            logger .warning ("skipping part (operator build failed, %d verts): %s",
            len (np .asarray (s ["verts"])),exc )
            continue 
        if not low_memory :
            item ["ops"]=ops # keep resident
        del ops # low_memory: only warmed cache
        prepared .append (item )
    if skipped :
        logger .warning ("skipped %d/%d parts with bad geometry",skipped ,
        len (train_samples ))
    if not prepared :
        raise RuntimeError ("no usable parts after operator precompute")

    import random 
    bk =_Bookkeeper (model ,meta ,"diffusionnet",opt ,sched ,epochs ,
    metrics_fn =metrics_fn ,eval_every =eval_every ,
    patience =patience ,best_path =best_path ,last_path =last_path ,
    save_every =save_every ,history_path =history_path ,
    log_every =log_every ,best_f1 =best_f1 ,history =history ,
    decode =decode ,train_config =train_config ,
    best_metric =best_metric ,snapshot_every =snapshot_every ,
    sched_metric =sched_metric ,min_delta =min_delta ,
    collapse_gap =collapse_gap )
    order =list (range (len (prepared )))
    try :
        for ep in range (start_epoch ,epochs ):
            model .train ()
            random .shuffle (order )
            opt .zero_grad (set_to_none =True )
            tot =0.0 
            for k ,i in enumerate (order ,1 ):
                d =prepared [i ]
                ops =(_cp_operators (d ["verts"],d ["faces"],meta ["k_eig"],op_cache_dir )
                if low_memory else d ["ops"])
                n =d ["x"].shape [0 ]
                big =device !="cpu"and max_gpu_verts and n >max_gpu_verts 
                try :
                    parts =_accumulate (ops ,d ,"cpu"if big else device ,device ,accum )
                except (torch .cuda .OutOfMemoryError ,RuntimeError )as exc :
                    if "out of memory"not in str (exc ).lower ():
                        raise 
                    torch .cuda .empty_cache ()
                    logger .warning ("CUDA OOM on %d-vertex part -> CPU fallback",n )
                    parts =_accumulate (ops ,d ,"cpu",device ,accum )
                if low_memory :
                    del ops 
                if k %accum ==0 or k ==len (order ):
                    if grad_clip and grad_clip >0 :
                        torch .nn .utils .clip_grad_norm_ (model .parameters (),grad_clip )
                    opt .step ()
                    opt .zero_grad (set_to_none =True )
                tot +=parts ["total"]
            if sched is not None and not sched_metric :
                sched .step ()
            if bk .after_epoch (ep ,tot /max (1 ,len (prepared ))):
                logger .info ("early stopping at epoch %d",ep )
                break 
    except KeyboardInterrupt :
        bk .flush_history ()
        logger .warning ("training interrupted -- resume from the last checkpoint%s",
        f" ({last_path })"if last_path else "")
        raise 
    bk .finalize ()
    return model ,meta 


def infer_diffusionnet (model ,meta ,verts ,faces ,verts_norm ,
op_cache_dir =None ,device ="cpu",max_gpu_verts =60000 ,
offset_scale =1.0 ):
    """Run a trained DiffusionNet regressor ten one part -> (N,7) numpy array
    ready for cp_targets.decode_predictions.

    offset_scale : mesh scale (mm); the model emits normalised offsets, so this
    converts them back to millimetres for the decoder (pass the sample 'scale').
    Like training, oversized meshes (or a CUDA OOM) fall back to the CPU so a
    4 GB GPU does not crash ten the largest parts; results are identical.
    """
    torch =_require_torch ()
    import diffusionnet as dnmod 
    import copy 
    n =len (np .asarray (verts ))
    run_device =("cpu"if (device !="cpu"and max_gpu_verts and n >max_gpu_verts )
    else device )

    def _run (rd ):
        ops =_to_device_ops (_cp_operators (verts ,faces ,meta ["k_eig"],op_cache_dir ),rd )
        x =(torch .tensor (np .asarray (verts_norm ),dtype =torch .float32 ,device =rd )
        if meta ["input_features"]=="xyz"else dnmod ._model_input (ops ,meta ))
        mdl =model if rd ==device else copy .deepcopy (model ).to (rd )
        mdl .eval ()
        with torch .no_grad ():
            out =dnmod ._forward (mdl ,ops ,x )
        return pred_to_array (out ,offset_scale =offset_scale )

    try :
        return _run (run_device )
    except (torch .cuda .OutOfMemoryError ,RuntimeError )as exc :
        if "out of memory"not in str (exc ).lower ():
            raise 
        torch .cuda .empty_cache ()
        logger .warning ("CUDA OOM during inference on %d-vertex part -> CPU",n )
        return _run ("cpu")


        # ===========================================================================
        # kNN-graph backbone (native-Windows: NO robust_laplacian / potpourri3d)
        # ===========================================================================
        # A torch-only geometric backbone for environments where the DiffusionNet native
        # wheels segfault (Windows). The "operator" is a k-nearest-neighbour graph built
        # with scipy.spatial.cKDTree; message passing is plain torch (EdgeConv: each
        # vertex aggregates an MLP of (h_i, h_j - h_i) over its neighbours, max-pooled).
        # Stacking blocks with residuals grows a real geometric receptive field, so --
        # unlike the bare MLP -- it generalises across parts. It reuses the SAME target
        # encoding / cp_loss / decode / metrics as the other backbones; only the network
        # and the operator differ. Deps: torch + scipy + numpy (all clean Windows wheels).

KNN_DEFAULTS ={"c_width":128 ,"n_layers":4 ,"k":16 ,"global_feat":False ,
"dropout":0.0 ,"normals":False }


def _knn_graph (verts ,k =16 ):
    """(N,k) neighbour-index tensor from a point set, via scipy cKDTree.

    Built ten the (uniformly) normalised coords -- uniform scale/translation
    preserves nearest neighbours, so the graph is scale-invariant. k is clamped
    to N-1 for tiny meshes. Needs no mesh faces (works ten point clouds too).

    workers=-1 parallelises the neighbour search across all cores -- a big win at
    479-parts x (1 + --augment) graphs; each is built before and cached.

    Returned as int32, NOT int64: these graphs are the single biggest resident
    host structure in a run (one (N,k) matrix per part, all held for the whole
    training loop -- ~13GB at int64 ten the 1749-part corpus, which is what drove
    the repeated Windows OOM kills). Indices address at most `max_gpu_verts`
    (14k) vertices, so int32 is ample and halves that to ~6.5GB. Consumers widen
    to int64 AFTER the (now half-size) device copy -- the same trick hierpoint's
    collate() already uses.
    """
    torch =_require_torch ()
    from scipy .spatial import cKDTree # type: ignore[attr-defined]
    V =np .asarray (verts ,dtype =np .float64 )
    n =len (V )
    kq =int (min (k +1 ,n ))# +1: query returns the point itself
    tree =cKDTree (V )
    _ ,idx =tree .query (V ,k =kq ,workers =-1 )
    idx =np .atleast_2d (idx )
    if idx .shape [1 ]>1 :
        idx =idx [:,1 :]# drop self-neighbour (column 0)
    return torch .tensor (np .ascontiguousarray (idx ,dtype =np .int32 ),
    dtype =torch .int32 )


def _hier_worker (args ):
    """Build ONE pooling hierarchy in a CHILD PROCESS (picklable, module level).

    The prebuild used to run in a ThreadPoolExecutor, but hierpoint._fps is a pure
    Python loop -- the GIL pins it to ~2 effective cores no matter how many threads
    are configured, which is why 7118 patches took 6.2 HOURS ten the corpus-v4 run.
    build_hierarchy is a deterministic pure function of the vertices (FPS from a
    fixed start, then kNN groupings), so moving it into processes changes the wall
    clock and NOTHING else: byte-identical structures, byte-identical training.
    """
    verts ,hier_kw ,cache_dir =args 
    import hierpoint as _hp 
    return _hp .build_hierarchy (verts ,cache_dir =cache_dir ,**hier_kw )


def _disk_has_room (path ,min_free_gb =3.0 ):
    """True if `path`'s drive has at least min_free_gb free.

    The prep caches are an OPTIMISATION; filling the disk with them would kill
    the training run they are meant to speed up (checkpoints/pagefile need the
    space -- this project lost three nights to disk-full/OOM cascades). So every
    cache WRITE is gated ten headroom, and a full disk just means "no cache",
    never a crash."""
    import shutil 
    try :
        return shutil .disk_usage (
        path if os .path .isdir (path )else os .path .dirname (os .path .abspath (path ))
        ).free >=min_free_gb *(1 <<30 )
    except OSError :
        return False 


def _knn_graph_disk (verts ,k ,cache_dir ):
    """_knn_graph with a disk cache keyed by SHA256(float32 vertex bytes) + k.

    On a resume, the val (and optionally train) graphs are loaded from disk
    instead of rebuilt from scratch -- saves ~3s per val set and noticeable time
    ten large augmented train sets. Atomic write (tmp+replace) guards against
    corrupt entries from a crash mid-save. A corrupt/unreadable file is silently
    rebuilt and overwritten.
    """
    import os ,hashlib 
    torch =_require_torch ()
    key =hashlib .sha256 (
    np .asarray (verts ,dtype =np .float32 ).tobytes ()).hexdigest ()[:24 ]
    path =os .path .join (cache_dir ,f"knn_k{k }_{key }.npy")
    if os .path .exists (path ):
        try :
        # .to(int32): entries written before the int32 switch are int64 and
        # would otherwise silently reintroduce the doubled residency.
            return torch .from_numpy (np .load (path )).to (torch .int32 )
        except Exception :# noqa: BLE001 – corrupt entry
            pass 
    nbr =_knn_graph (verts ,k )
    if not _disk_has_room (cache_dir ):
        return nbr # no headroom -> skip caching
    os .makedirs (cache_dir ,exist_ok =True )
    # np.save() always appends ".npy" unless the name already ends in ".npy",
    # so the tmp name must end in ".npy" too or os.replace(tmp, path) never
    # finds the file it just wrote (silently swallowed below -> cache never
    # persists, and a stray "*.npy.tmp.npy" is left behind every call).
    tmp =path +".tmp.npy"
    try :
        np .save (tmp ,nbr .numpy ())
        os .replace (tmp ,path )
    except Exception :# noqa: BLE001 – disk full / RO
        try :
            os .remove (tmp )
        except OSError :
            pass 
    return nbr 


def _pca_normals_curvature (V ,nbr ):
    """Per-vertex PCA over the kNN neighbourhood. Returns (normals (N,3), curvature (N,1)).
    curvature = λ₀/(λ₀+λ₁+λ₂): near 0 ten flat faces, higher in curved/concave pockets —
    the direct signal for wire-entry openings."""
    V =np .asarray (V ,dtype =np .float64 )
    nb =np .asarray (nbr .cpu ().numpy ()if hasattr (nbr ,"cpu")else nbr )
    P =V [nb ]# (N, k, 3)
    Q =P -P .mean (1 ,keepdims =True )
    cov =np .einsum ("nki,nkj->nij",Q ,Q )/max (1 ,Q .shape [1 ])
    _w ,vec =np .linalg .eigh (cov )# ascending eigenvalues
    nrm =vec [:,:,0 ]# smallest-eigenvalue direction
    s =np .einsum ("ni,ni->n",nrm ,V -V .mean (0 ))# orient outward from centroid
    s [s ==0 ]=1.0 
    nrm =nrm *np .sign (s )[:,None ]
    curv =(_w [:,0 ]/(_w .sum (1 )+1e-8 )).reshape (-1 ,1 )
    return nrm .astype (np .float32 ),curv .astype (np .float32 )


def _pca_normals (V ,nbr ):
    """Per-vertex unit normal via PCA (see _pca_normals_curvature for full docs)."""
    return _pca_normals_curvature (V ,nbr )[0 ]


def _concavity_score (V ,nbr ):
    """Per-vertex concavity: does the LOCAL neighbourhood extend further in the
    outward direction than this vertex does (recessed = concave), or does this
    vertex poke out past its neighbours (convex)? Normalised to [0,1]: 1=fully
    concave, 0=convex/flat. Strong signal for CP openings (recessed pockets).

    BUG THIS REPLACES: the previous formula was dot(normal, centroid_dir) where
    `normal` is _pca_normals's PCA-eigenvector normal, ALREADY sign-flipped by
    _pca_normals_curvature to agree with `centroid_dir` (`nrm * sign(dot(nrm,
    V-centroid))`, see _pca_normals_curvature). Comparing that normal back against
    the SAME centroid_dir it was just oriented to agree with is tautological --
    dot(nrm, centroid_dir) >= 0 by construction for every vertex, every part, so
    concav = clip(-dot, 0, 1) was PROVABLY always exactly 0 (confirmed empirically:
    0/N nonzero ten every real part checked). This silently made --knn-concavity a
    dead, all-zero input channel in every run that used it.

    FIX: reuse the same globally-oriented `nrm` (still useful as a rough "outward"
    reference -- keeping it means no retraining-breaking API change), but compare
    it to something that is NOT the same reference used to orient it: the mean
    direction from this vertex to its actual k-NN NEIGHBOURS. If neighbours sit
    further along +nrm than V does, V is set back into a pocket relative to them
    -- concave. This is genuinely local information (distinct from the global
    centroid heuristic), so it is not tautologically zero."""
    V =np .asarray (V ,dtype =np .float64 )
    nrm =_pca_normals (V ,nbr ).astype (np .float64 )# (N,3), globally outward-oriented
    nb =np .asarray (nbr .cpu ().numpy ()if hasattr (nbr ,"cpu")else nbr )
    P =V [nb ]# (N, k, 3) neighbour positions
    offs =P -V [:,None ,:]# (N, k, 3) vertex -> neighbour
    dist =np .linalg .norm (offs ,axis =-1 ,keepdims =True )
    dirs =offs /np .where (dist <1e-8 ,1.0 ,dist )# unit directions to each neighbour
    proj =np .einsum ("nki,ni->nk",dirs ,nrm )# (N, k): neighbour-dir . own normal
    mean_proj =proj .mean (1 )# >0: neighbours lie along +nrm (V recessed)
    concav =np .clip (mean_proj ,0.0 ,1.0 ).astype (np .float32 ).reshape (-1 ,1 )
    return concav 


def _edge_distance (V ,nbr ):
    """Per-vertex approximate distance to the mesh boundary (normalised by bbox diag).
    Boundary vertices = those with the fewest kNN neighbours ten only ONE side of the surface.
    Heuristic: boundary ~ vertices where the kNN neighbourhood has high normal variance."""
    nrm =_pca_normals (V ,nbr )
    nb =np .asarray (nbr .cpu ().numpy ()if hasattr (nbr ,"cpu")else nbr )
    # For each vertex: mean angular deviation of neighbour normals from own normal.
    # High deviation = surface fold / boundary = low edge_dist.
    nrm_nbr =nrm [nb ]# (N, k, 3)
    cos_sim =np .einsum ("ni,nki->nk",nrm ,nrm_nbr ).clip (-1 ,1 )# (N, k)
    ang_dev =np .arccos (np .abs (cos_sim )).mean (1 )# (N,)  in radians
    # Normalise: high ang_dev → low edge_dist (near boundary)
    # Scale to [0,1] relative to max deviation seen in the part
    amax =float (ang_dev .max ())or 1.0 
    edge_dist =(1.0 -ang_dev /amax ).astype (np .float32 ).reshape (-1 ,1 )
    return edge_dist 


def _knn_feature_tensor (V ,nbr ,use_normals ,device ,use_curvature =False ,
use_concavity =False ,use_edge_dist =False ):
    """Build the model input tensor.
    xyz (3) | +normal (6) | +curvature (7) | +concavity (8) | +edge_dist (9)."""
    import torch 
    Vf =np .asarray (V ,dtype =np .float32 )
    if not use_normals :
        return torch .tensor (Vf ,dtype =torch .float32 ,device =device )
    parts =[Vf ]
    if use_curvature :
        nrm ,curv =_pca_normals_curvature (V ,nbr )
        parts .extend ([nrm ,curv ])
    else :
        parts .append (_pca_normals (V ,nbr ))
    if use_concavity :
        parts .append (_concavity_score (V ,nbr ))
    if use_edge_dist :
        parts .append (_edge_distance (V ,nbr ))
    feat =np .concatenate (parts ,axis =1 )
    return torch .tensor (feat ,dtype =torch .float32 ,device =device )


def build_knngraph_regressor (config =None ):
    """Build the EdgeConv kNN-graph regressor (C_out=7). Returns (model, meta)."""
    _require_torch ()
    import torch 
    import torch .nn as nn 
    cfg ={**KNN_DEFAULTS ,**(config or {})}
    use_normals =bool (cfg .get ("normals",False ))
    use_curvature =bool (cfg .get ("curvature",False ))and use_normals 
    use_concavity =bool (cfg .get ("concavity",False ))and use_normals 
    use_edge_dist =bool (cfg .get ("edge_dist",False ))and use_normals 
    width =int (cfg ["c_width"]);n_layers =int (cfg ["n_layers"])
    # c_in: 3 (xyz) | 6 (+nrm) | 7 (+curv) | 8 (+concav) | 9 (+edgedist)
    c_in =3 
    if use_normals :c_in +=3 # normals
    if use_curvature :c_in +=1 
    if use_concavity :c_in +=1 
    if use_edge_dist :c_in +=1 
    global_feat =bool (cfg ["global_feat"])
    dropout =float (cfg .get ("dropout",0.0 ))

    class EdgeConv (nn .Module ):
        """h_i' = max_j MLP([h_i, h_j - h_i]) over the kNN neighbours j of i."""
        def __init__ (self ,c_in ,c_out ):
            super ().__init__ ()
            self .mlp =nn .Sequential (nn .Linear (2 *c_in ,c_out ),
            nn .LayerNorm (c_out ),nn .ReLU (),
            nn .Linear (c_out ,c_out ),
            nn .LayerNorm (c_out ))

        def forward (self ,h ,nbr_idx ):
            hj =h [nbr_idx ]# (N, k, C)
            hi =h .unsqueeze (1 ).expand_as (hj )# (N, k, C)
            edge =torch .cat ([hi ,hj -hi ],dim =-1 )# (N, k, 2C)
            return self .mlp (edge ).amax (dim =1 )# (N, C') max aggregation

    class KNNGraphNet (nn .Module ):
        def __init__ (self ):
            super ().__init__ ()
            self .inp =nn .Sequential (nn .Linear (c_in ,width ),nn .ReLU ())
            self .blocks =nn .ModuleList ([EdgeConv (width ,width )for _ in range (n_layers )])
            self .global_feat =global_feat 
            # dropout before the head regularises the per-vertex features (the main
            # in-network defence against overfitting ten a small corpus)
            drop =nn .Dropout (dropout )if dropout >0 else nn .Identity ()
            if global_feat :
            # PointNet-style global context: each vertex sees a summary of the
            # WHOLE part, which helps disambiguate sparse connection points
            # (a salient bump is a CP only relative to the rest of the housing).
                self .head =nn .Sequential (drop ,nn .Linear (2 *width ,width ),nn .ReLU (),
                nn .Linear (width ,N_CHANNELS ))
            else :
                self .head =nn .Sequential (drop ,nn .Linear (width ,N_CHANNELS ))

        def forward (self ,x ,nbr_idx ):
            h =self .inp (x )
            for blk in self .blocks :
                h =h +blk (h ,nbr_idx )# residual -> stable depth
            if self .global_feat :
                g =h .amax (dim =0 ,keepdim =True ).expand_as (h )# global max-pool
                h =torch .cat ([h ,g ],dim =-1 )
            return self .head (h )

    _feats =(["xyz"]+(["normal"]if use_normals else [])+
    (["curv"]if use_curvature else [])+
    (["concav"]if use_concavity else [])+
    (["edgedist"]if use_edge_dist else []))
    meta ={"backbone":"knngraph",
    "input_features":"_".join (_feats ),"c_in":c_in ,
    "normals":use_normals ,"curvature":use_curvature ,
    "concavity":use_concavity ,"edge_dist":use_edge_dist ,
    "c_width":width ,"n_layers":n_layers ,"k":int (cfg ["k"]),
    "global_feat":global_feat ,"dropout":dropout }
    return KNNGraphNet (),meta 


def save_knngraph (model ,meta ,path ):
    """Save a trained kNN-graph regressor (weights + meta) to `path`."""
    save_checkpoint (path ,model ,meta ,"knngraph")


def load_knngraph (path ,device ="cpu"):
    """Rebuild a kNN-graph regressor from a checkpoint. Returns (model, meta)."""
    model ,meta ,_ =load_model (path ,device =device )
    return model ,meta 


def train_knngraph_regressor (train_samples ,config =None ,epochs =200 ,lr =1e-3 ,
device ="cpu",log_every =1 ,metrics_fn =None ,
eval_every =5 ,patience =0 ,
w_heat =1.0 ,w_off =5.0 ,w_dir =1.0 ,heat_pos_weight =50.0 ,
heat_loss ="bce",focal_gamma =2.0 ,centernet_alpha =2.0 ,
max_gpu_verts =60000 ,
best_path =None ,last_path =None ,save_every =1 ,
history_path =None ,resume_from =None ,seed =0 ,
lr_decay_every =0 ,lr_decay_rate =0.5 ,accum_steps =1 ,
amp =False ,decode =None ,train_config =None ,
best_metric ="micro_f1",resume_strict =False ,
snapshot_every =0 ,weight_decay =0.0 ,
lr_schedule ="step",warmup_epochs =0 ,
offset_heat_weight =True ,part_weight_mode ="none",
grad_clip =0.0 ,knn_cache_dir =None ,
dir_sign_invariant =False ,init_from =None ,
hard_mining =False ,offset_loss ="l1",
offset_huber_beta =0.01 ,min_delta =0.0 ,
collapse_gap =0.03 ,fwd_verts =None ,
pin_host_memory =None ,sign_inv_prefixes ="",
backbone ="knngraph"):
    """Train the kNN-graph regressor. Mirrors train_diffusionnet_regressor (seed,
    LR schedule, gradient accumulation, GPU/CPU-hybrid for big meshes, per-epoch
    val metrics, best/last full-state .ckpt + resume, early stop) but the per-part
    operator is a cheap kNN graph held in RAM (no eigenbasis, no op cache).
    Returns (model, meta). See train_diffusionnet_regressor for the checkpoint/
    resume parameter docs.

    weight_decay : AdamW L2 regularisation. lr_schedule : step/plateau/cosine.
    offset_heat_weight : couple offset loss to target heat. part_weight_mode :
    'none' or 'keypoints' (weight a part's loss by its #CPs). grad_clip : max grad
    norm before opt.step (0 = off)."""
    import copy 
    torch =_require_torch ()
    # Early-warn about known bad hyperparameter combinations before spending hours
    # ten a run that will collapse:
    if w_heat >2.0 and heat_loss =="centernet":
        logger .warning (
        "w_heat=%.1f with heat_loss=centernet is likely to cause direction-head "
        "collapse (ang>90 deg, as seen in v7 at epoch 25). CenterNet normalises "
        "the heatmap loss by #keypoints, so once the heatmap fits its gradient "
        "approaches 0; a high w_heat then starves the direction/offset heads of "
        "gradient signal and they can invert. Recommend w_heat=1.0 (default).",
        w_heat )
    if lr_schedule =="step"and not (lr_decay_every and lr_decay_every >0 ):
        logger .warning (
        "no LR decay configured (--lr-schedule step, --lr-decay-every 0): "
        "LR stays constant for all %d epochs. The model will keep updating past "
        "its best checkpoint and may collapse. Use --lr-decay-every 25 or "
        "--lr-schedule cosine.",epochs )
        # backbone: "knngraph" (flat EdgeConv, model(x, nbr)) or "hierpoint"
        # (hierarchical point U-Net, model(x, hier) -- hierpoint.py). Everything else
        # in this trainer (prep/patching, graph batching, chunked accumulation, OOM
        # fallback, bookkeeping) is shared; only graph construction and the forward
        # call differ.
    is_hier =backbone =="hierpoint"
    _hp =None 
    if is_hier :
        import hierpoint as _hp_mod 
        _hp =_hp_mod 
    model ,meta ,opt ,sched ,start_epoch ,best_f1 ,history ,sched_metric =_init_training (
    backbone ,config ,resume_from ,device ,lr ,
    lr_decay_every ,lr_decay_rate ,seed ,train_config =train_config ,
    resume_strict =resume_strict ,history_path =history_path ,
    weight_decay =weight_decay ,lr_schedule =lr_schedule ,
    warmup_epochs =warmup_epochs ,total_epochs =epochs ,init_from =init_from )
    if start_epoch >=epochs :
        logger .info ("checkpoint already at epoch %d >= --epochs %d; nothing to "
        "train (raise --epochs to continue)",start_epoch ,epochs )
        return model ,meta 
    accum =max (1 ,int (accum_steps ))
    # Record the subsample cap as a model property: inference must build the kNN
    # graph at the SAME vertex density it was trained ten, so infer_knngraph reads
    # this from meta rather than a (possibly different) CLI value. 0 = no cap.
    meta ["subsample"]=int (max_gpu_verts or 0 )
    # Record the training seed too: infer_knngraph needs it (+ the part_nr) to
    # reconstruct the SAME per-part subsample training picked (part_subsample_seed),
    # not a fixed seed=0 that silently differs from what the model was trained ten.
    meta ["seed"]=int (seed )
    use_normals =bool (meta .get ("normals",False ))
    use_curvature =bool (meta .get ("curvature",False ))
    use_concavity =bool (meta .get ("concavity",False ))
    use_edge_dist =bool (meta .get ("edge_dist",False ))

    # Mixed precision (fp16 autocast + GradScaler), CUDA only, OPT-IN (amp=True).
    # EdgeConv here is gather/bandwidth-bound, not compute-bound, so ten a T1200 the
    # fp16 cast + scaler overhead actually makes BELOW-cliff caps (<=~12k verts)
    # SLOWER than fp32; it only claws back ~30% in the deep memory-thrash regime
    # (cap ~20k) and even then can't beat just capping at 8k. Kept for high-cap runs
    # or a stronger GPU. When off, every scaler call is a no-op so the fp32 path is
    # byte-for-byte unchanged. Scaler state is intentionally NOT checkpointed -- it
    # re-warms its loss scale within a few steps after a resume.
    use_amp =bool (amp )and device =="cuda"
    from torch .amp .grad_scaler import GradScaler # defining module -> pyright-clean
    scaler =GradScaler ("cuda",enabled =use_amp )
    if use_amp :
        logger .info ("knngraph: mixed precision ON (fp16 autocast + GradScaler)")

        # one persistent CPU replica for the big-part / OOM fallback, re-synced from the
        # live model each use -- avoids rebuilding the whole module (copy.deepcopy) per
        # big part per epoch.
        # dirty=True: weights need to be synced from the GPU model before next use.
        # Marked True after every opt.step(); cleared ten sync. This means load_state_dict
        # (an expensive GPU→CPU copy of all weights) happens at most ONCE per accumulation
        # window instead of before per big part -- critical when many big parts hit CPU each
        # epoch (ten a T1200 with max_gpu_verts=7000, most parts go CPU).
    _cpu ={"m":None ,"dirty":True }

    def _cpu_replica ():
        if _cpu ["m"]is None :
            mm ,_ =build_regressor (backbone ,_meta_to_config (backbone ,meta ))
            _cpu ["m"]=mm .to ("cpu")
        if _cpu ["dirty"]:
            _cpu ["m"].load_state_dict (
            {k :v .detach ().cpu ()for k ,v in model .state_dict ().items ()})
            _cpu ["dirty"]=False 
        _cpu ["m"].train ()
        return _cpu ["m"]

    def _loss (out ,t ,m ,sign_inv =False ,pw =None ,as_tensors =False ):
    # offset in NORMALISED mm can be tiny -- compute the loss in fp32 (outside
    # autocast) so AMP fp16 never underflows the offset/dir terms.
    # pw: precomputed per-part weight (d["pw"], from the numpy target at prep
    # time). Computing it here from the GPU tensor would be a hidden per-part
    # device sync -- only fall back to that for callers without prep dicts.
        if pw is None :
            pw =1.0 
            if part_weight_mode =="keypoints":
                pw =max (1.0 ,float ((t [:,ct .HEATMAP ]>=1.0 -1e-4 ).sum ()))
        return cp_loss (out .float (),t ,m ,w_heat =w_heat ,w_off =w_off ,w_dir =w_dir ,
        heat_pos_weight =heat_pos_weight ,heat_loss =heat_loss ,
        focal_gamma =focal_gamma ,centernet_alpha =centernet_alpha ,
        offset_heat_weight =offset_heat_weight ,
        part_weight =pw ,
        dir_sign_invariant =dir_sign_invariant or sign_inv ,
        offset_loss =offset_loss ,offset_huber_beta =offset_huber_beta ,
        as_tensors =as_tensors )

    _hier_kw =_hp .hier_params_from_meta (meta )if is_hier else None 

    # CP_LAZY_HIER=1: read each pooling hierarchy back from the DISK cache at batch
    # time instead of pinning all of them in RAM. Corpus v8 spatial-patches into
    # 12,556 graphs; holding every hierarchy resident drove commit to 99% of the
    # limit, at which point Windows grew the pagefile until the disk hit 5GB and the
    # run had to be killed (2026-07-13). build_hierarchy is a pure function of the
    # patch geometry and is already disk-cached, so re-reading it costs I/O and
    # NOTHING else: byte-identical structures, byte-identical training.
    # Requires knn_cache_dir -- without it a "lazy" read would REBUILD from scratch
    # (the historical 6.2h item), so it is refused rather than silently ruinous.
    import os as _os_lazy 
    _lazy_hier =_os_lazy .environ .get ("CP_LAZY_HIER")=="1"and is_hier 
    if _lazy_hier and not knn_cache_dir :
        raise ValueError ("CP_LAZY_HIER=1 needs --prep-cache-dir: without the disk "
        "cache every batch would REBUILD its hierarchies from "
        "scratch (6.2h/epoch territory), not read them back.")
    if _lazy_hier :
        logger .info ("hierarchies: LAZY (read per batch from %s, not held in RAM)",
        knn_cache_dir )

    def _get_hier (d ):
    # pooling structure (FPS levels + grouping/upsample indices) is a pure
    # function of the patch geometry -- built before, cached ten the dict (and
    # ten disk next to the kNN graphs when a cache dir is configured). In lazy
    # mode it is NOT retained ten the dict: the disk cache is the store.
        h =d .get ("hier")
        if h is None :
            h =_hp .build_hierarchy (d ["verts_norm"],
            cache_dir =knn_cache_dir ,**_hier_kw )
            if not _lazy_hier :
                d ["hier"]=h 
        return h 

    def _ensure_hier (d ):
        if not _lazy_hier :
            _get_hier (d )

    def _accumulate (d ,run_device ,src_device ,loss_div ,sign_inv =False ,
    weight_div =1.0 ):
    # weight_div: the batch's summed graph_weight. The fused GPU path divides
    # its batch loss by total_w (a weighted MEAN over the batch), so the
    # solo/CPU path -- which the big-part and CUDA-OOM fallbacks run
    # part-by-part -- must divide by the same total_w or it contributes a
    # weighted SUM instead: a gradient ~batch-size times too large, injected
    # exactly when memory is already tight. Default 1.0 keeps single-part
    # callers unchanged.
    # input features: xyz, or [xyz, pca_normal] when use_normals. Normals need the
    # kNN graph, so build before per part (after nbr exists) and cache ten the dict.
        if use_normals :
            if d .get ("featx")is None :
                d ["featx"]=_knn_feature_tensor (
                d ["verts_norm"],d ["nbr"],True ,"cpu",
                use_curvature ,use_concavity ,use_edge_dist )
            x =d ["featx"].to (run_device )
        else :
            x =d ["x"].to (run_device )
        if is_hier :
            nb =_hp .collate ([_get_hier (d )],[d ["nbr"]],device =run_device )
        else :
        # host copy is int32 (half the RAM); widen AFTER the device copy --
        # EdgeConv's h[nbr_idx] gather needs int64 ten device
            nb =d ["nbr"].to (run_device ).long ()
        t =d ["t"].to (run_device );m =d ["m"].to (run_device )
        gw =d .get ("graph_weight",1.0 )
        if run_device ==src_device :
            with torch .autocast (device_type ="cuda",dtype =torch .float16 ,
            enabled =(use_amp and run_device =="cuda")):
                out =model (x ,nb )
            loss ,parts =_loss (out ,t ,m ,sign_inv =sign_inv ,pw =d .get ("pw"))
            scaler .scale (gw *loss /max (weight_div ,1e-8 )/loss_div ).backward ()
        else :
        # CPU fallback (oversized part or CUDA OOM): always fp32, no autocast.
            cm =_cpu_replica ()
            out =cm (x ,nb )
            loss ,parts =_loss (out ,t ,m ,sign_inv =sign_inv ,pw =d .get ("pw"))
            # Scale these CPU grads by the SAME factor the scaler will later unscale
            # ten the GPU model, so a mixed GPU/CPU accumulation window stays
            # consistent (get_scale()==1.0 when amp is off -> identical to fp32).
            (scaler .get_scale ()*gw *loss 
            /max (weight_div ,1e-8 )/loss_div ).backward ()
            for pg ,pc in zip (model .parameters (),cm .parameters ()):
                if pc .grad is not None :
                    g =pc .grad .detach ().to (src_device )
                    pg .grad =g if pg .grad is None else (pg .grad +g )
            cm .zero_grad (set_to_none =True )# replica is reused, clear its grads
        del x ,nb ,t ,m ,out ,loss 
        return parts 

        # kNN graphs are built lazily ten first access and cached -- training starts
        # immediately, and each graph is built only before (not before per epoch).
        # Big meshes (>max_gpu_verts) are split into SPATIAL PATCHES, each <= cap verts
        # at FULL density (spatial_patches), so the kNN-graph density inside a patch
        # matches what infer_knngraph builds (it patches identically) AND the CP-hole
        # geometry survives -- train and inference therefore see the same density and the
        # same intact features. A giant part (>max_patches*cap) is uniform-capped first.
    logger .info ("kNN-graph regressor: %d parts (k=%d, subsample=%d), graphs cached "
    "on first use",len (train_samples ),meta ["k"],meta ["subsample"])
    def _resnap_peaks (verts_full ,tgt_full ,base ,tgt_base ,msk_base ):
    # Re-snap one heat=1 peak per CP onto the nearest KEPT vertex after a uniform
    # subsample dropped the original peak: restores the centernet positive AND
    # recomputes its offset (FROM the new vertex TO the CP) so offset/dir loss
    # fires correctly. Without it big subsampled parts lose recall.
        peaks =np .where (tgt_full [:,ct .HEATMAP ]>=1.0 -1e-4 )[0 ]
        if not len (peaks ):
            return 
        cp_norm =verts_full [peaks ]+tgt_full [peaks ][:,ct .OFFSET ]
        for c in cp_norm :
            j =int (np .argmin (np .linalg .norm (base -c ,axis =1 )))
            tgt_base [j ,ct .HEATMAP ]=1.0 
            tgt_base [j ,ct .OFFSET ]=c -base [j ]
            msk_base [j ]=True 

    prepared =[]
    n_patched =n_subsampled =0 
    # Per-part sign-invariant direction loss is an ESCAPE HATCH for corpora whose
    # insert-axis sign is untrusted, not a default: training 1-|cos| while the
    # eval angle is SIGNED is exactly the v19 ~100-deg failure (79% of graphs were
    # wscad and never penalised for a 180-deg flip). Since wscad_corpus_v2 labels
    # carry mesh-clearance-verified directions, the default is signed everywhere;
    # pass sign_inv_prefixes="wscaduniverse" to restore the old behaviour for
    # old-label corpora.
    _sign_inv_pfx =[p .strip ()for p in str (sign_inv_prefixes or "").split (",")
    if p .strip ()]
    for s in train_samples :
        verts =s ["verts_norm"]
        tgt =s ["target"]
        msk =s ["mask"]
        n =len (verts )
        # GATE by part type. TERMINAL BLOCKS (wscad + PXC) have small recessed CP
        # holes a uniform subsample erases, so they need SPATIAL PATCHES (full
        # density). Contactors / drives / breakers have raised screw terminals that
        # survive subsampling -> the proven v8 single-subsample path; mixing the two
        # cleanly avoids the patch-floods-big-parts conflict. sign_inv is SEPARATE
        # and driven by sign_inv_prefixes (default: none -- signed loss everywhere;
        # see the escape-hatch comment above the loop).
        pn =str (s .get ("part_nr",""))
        do_patch =is_patch_part (pn )
        sign_inv =any (pn .startswith (p )for p in _sign_inv_pfx )
        if do_patch :
        # terminal block: spatial patches, whole-part target sliced per patch
        # (per-vertex heat/offset/dir valid under slicing; every CP peak kept).
            patches ,sel =spatial_patches (verts ,max_gpu_verts ,
            margin_frac =PATCH_MARGIN_FRAC )
            if sel is not None :# giant part: capped, re-snap peaks
                base ,tbase ,mbase =verts [sel ].copy (),tgt [sel ].copy (),msk [sel ].copy ()
                _resnap_peaks (verts ,tgt ,base ,tbase ,mbase )
            else :
                base ,tbase ,mbase =verts ,tgt ,msk 
            if len (patches )>1 :
                n_patched +=1 
                # Per-patch loss weight = 1/n_patches so a WHOLE patched part contributes
                # the same total per-epoch gradient weight as a single non-patched part,
                # regardless of how many patches it was split into. Without this, patched
                # (wscad/PXC) parts silently dominate the gradient budget in proportion to
                # their patch count (measured: wscaduniverse ~7-38 patches/part -> ~79% of
                # ALL training graphs from only ~34% of train parts ten the real corpus),
                # starving every non-patched family (ABB/SIE/A-B/RIT/FES) of learning
                # signal and explaining a catastrophic per-family FN-rate cliff (cp_knn_v17:
                # wscad 35% FN vs 48-100% FN everywhere else). See part_subsample_seed for
                # the analogous "single source of truth" pattern this mirrors.
            graph_w =patch_graph_weight (len (patches ))
            for idx in patches :
                pv =base [idx ];pt =tbase [idx ];pm =mbase [idx ]
                # per-part loss weight, precomputed ten the numpy target so the hot
                # loop never syncs the GPU to derive it (see _loss)
                pw =(max (1.0 ,float ((pt [:,ct .HEATMAP ]>=1.0 -1e-4 ).sum ()))
                if part_weight_mode =="keypoints"else 1.0 )
                prepared .append ({"verts_norm":np .ascontiguousarray (pv ),"nbr":None ,
                "x":torch .tensor (pv ,dtype =torch .float32 ),
                "t":torch .tensor (pt ,dtype =torch .float32 ),
                "m":torch .tensor (pm ,dtype =torch .bool ),
                "sign_inv":sign_inv ,"graph_weight":graph_w ,
                "pw":pw })
        else :
        # ABB / general: v8 path -- single uniform subsample to cap + peak re-snap.
            if max_gpu_verts and n >max_gpu_verts :
                pid =str (s .get ("part_nr",id (s )))
                psd =part_subsample_seed (seed ,pid )
                idx =uniform_subsample_idx (n ,max_gpu_verts ,psd )
                base ,tbase ,mbase =verts [idx ].copy (),tgt [idx ].copy (),msk [idx ].copy ()
                _resnap_peaks (verts ,tgt ,base ,tbase ,mbase )
                n_subsampled +=1 
            else :
                base ,tbase ,mbase =verts ,tgt ,msk 
            pw =(max (1.0 ,float ((tbase [:,ct .HEATMAP ]>=1.0 -1e-4 ).sum ()))
            if part_weight_mode =="keypoints"else 1.0 )
            prepared .append ({"verts_norm":np .ascontiguousarray (base ),"nbr":None ,
            "x":torch .tensor (base ,dtype =torch .float32 ),
            "t":torch .tensor (tbase ,dtype =torch .float32 ),
            "m":torch .tensor (mbase ,dtype =torch .bool ),
            "sign_inv":sign_inv ,"graph_weight":1.0 ,
            "pw":pw })
            # Free this sample's FULL-RESOLUTION source arrays before its patches/
            # subsample are copied into `prepared` (which holds independent copies --
            # np.ascontiguousarray + torch.tensor both copy). ONLY for augmented
            # clones: they live exclusively in the trainer's fit_s list and are never
            # touched again after this point. The NON-clone originals are shared with
            # train_cp.py's train_s (needed for the end-of-run train report), so they
            # are left intact. This is math-neutral -- the model trains ten `prepared`,
            # which is byte-identical either way -- it just returns the clones' full-
            # res RAM (big wscad clones are 26-85k verts each; ~452 of them here)
            # instead of pinning it for the whole run, which ten a 31GB box is the
            # difference between the working set staying resident and Windows trimming
            # it into page-fault thrash (~2.6x slower/epoch).
            # ... and, with CP_FREE_TRAIN_SOURCE=1, for the ORIGINALS too. The clause
            # above frees only clones, so a run with --augment 0 frees NOTHING: every
            # part pins its full-res verts (float64), faces (int64), verts_norm, target
            # and mask for the whole run. On corpus v8 that is ~30GB of source the
            # hierpoint/knngraph path never reads again -- it trains ten `prepared`,
            # which holds independent copies -- and it is what put commit at 88% of the
            # limit while still only PRE-BUILDING kNN graphs (2026-07-13, 12556 graphs).
            # The ONLY cost is train_cp's end-of-run TRAIN report (val/test reports use
            # their own untouched samples), which skips the freed parts.
        import os as _os_env # cp_regressor has no module-level os import
        if "~aug"in pn or _os_env .environ .get ("CP_FREE_TRAIN_SOURCE")=="1":
            s ["verts_norm"]=None ;s ["target"]=None ;s ["mask"]=None 
            s ["verts"]=None ;s ["faces"]=None 
    if not prepared :
        raise RuntimeError ("no usable parts for kNN-graph training")
    if n_patched or n_subsampled :
        logger .info ("vertex selection: %d terminal-block part(s) spatial-patched, %d "
        "other part(s) uniform-subsampled -> %d total training graphs",
        n_patched ,n_subsampled ,len (prepared ))

        # Warn when the cached kNN graphs will eat significant RAM. Each graph is
        # (cap × k) int32 values; with 7000-vert cap, k=16: ~435 KB per part.
        # With 1367 real parts + 4 augmented clones = 6835 graphs ≈ 3.0 GB.
    if max_gpu_verts and meta ["k"]:
        n_aug_parts =sum (
        1 for s in train_samples if "~aug"in str (s .get ("part_nr","")))
        # graph count is now driven by spatial patching AND augmentation, so estimate
        # from the actual prepared-graph count regardless of whether clones are present.
        # 4 bytes/index since _knn_graph returns int32 (was int64 -> this estimate,
        # and the residency itself, used to be exactly double).
        est_mb =len (prepared )*max_gpu_verts *meta ["k"]*4 /1_000_000 
        if est_mb >2000 :
            logger .warning (
            "kNN graph RAM estimate: %d graphs (%d incl. augmented clones) × "
            "%d verts × k=%d ≈ %.0f MB held in RAM. If OOM, reduce --augment or "
            "--max-gpu-verts.",len (prepared ),n_aug_parts ,
            max_gpu_verts ,meta ["k"],est_mb )

    import random 
    import time 
    bk =_Bookkeeper (model ,meta ,backbone ,opt ,sched ,epochs ,
    metrics_fn =metrics_fn ,eval_every =eval_every ,
    patience =patience ,best_path =best_path ,last_path =last_path ,
    save_every =save_every ,history_path =history_path ,
    log_every =log_every ,best_f1 =best_f1 ,history =history ,
    decode =decode ,train_config =train_config ,
    best_metric =best_metric ,snapshot_every =snapshot_every ,
    sched_metric =sched_metric ,min_delta =min_delta ,
    collapse_gap =collapse_gap )
    order =list (range (len (prepared )))

    if is_hier and not knn_cache_dir :
        logger .warning (
        "hierpoint WITHOUT --prep-cache-dir: the pooling-hierarchy prebuild "
        "(%d graphs) is NOT cached and will be recomputed from scratch on "
        "every launch AND every --resume -- measured at 2-6 HOURS on this "
        "corpus. Pass --prep-cache-dir <dir> to pay it once.",len (prepared ))

        # Pre-build all kNN graphs in parallel BEFORE the first epoch so epoch-1 is not
        # silently slow (building 7000+ graphs serially looked like a hang).
        # scipy.cKDTree.query releases the GIL -> ThreadPoolExecutor gives real speedup.
        # Each d["nbr"] write is to a distinct dict so there are no data races.
    _to_build =[d for d in prepared if d ["nbr"]is None ]
    if _to_build :
        import os as _os 
        from concurrent .futures import ThreadPoolExecutor as _TPE 

        def _build_one (d ):
            try :
            # kNN graphs are NOT disk-cached in the fit path ten purpose: the
            # whole corpus builds in ~48s (cKDTree releases the GIL, so the
            # thread pool is genuinely parallel here), while caching them would
            # cost ~6.4GB of disk that the pooling hierarchies -- the 6-HOUR
            # item -- need far more.
                d ["nbr"]=_knn_graph (d ["verts_norm"],meta ["k"])
            except Exception as _exc :# noqa: BLE001
                logger .warning ("kNN graph build failed (part will be skipped): %s",
                _exc )

                # CP_PREP_WORKERS caps prebuild concurrency: each in-flight part holds
                # transient float64 gather buffers, so ten a commit-tight box (31GB RAM,
                # pagefile pinned by a full disk) 8 concurrent dense parts is what
                # pushed the corpus-v4 prep over the edge (2026-07-12 OOM).
        _n_workers =min (int (_os .environ .get ("CP_PREP_WORKERS","8")),
        _os .cpu_count ()or 1 )
        logger .info ("pre-building %d kNN graphs (%d threads) ...",
        len (_to_build ),_n_workers )
        _t_build =time .time ()
        with _TPE (max_workers =_n_workers )as _pool :
            list (_pool .map (_build_one ,_to_build ))
        logger .info ("  kNN graphs ready in %.0fs",time .time ()-_t_build )

        # Pre-build feature tensors (normals/curvature/concavity) in parallel so the
        # first epoch doesn't stall ten serial per-batch CPU feature computation.
    if use_normals :
        _to_feat =[d for d in prepared if d .get ("featx")is None and d ["nbr"]is not None ]
        if _to_feat :
            def _build_feat (d ):
                d ["featx"]=_knn_feature_tensor (
                d ["verts_norm"],d ["nbr"],True ,"cpu",
                use_curvature ,use_concavity ,use_edge_dist )
            logger .info ("pre-building %d feature tensors (%d threads) ...",
            len (_to_feat ),_n_workers )
            _t_feat =time .time ()
            with _TPE (max_workers =_n_workers )as _pool :
                list (_pool .map (_build_feat ,_to_feat ))
            logger .info ("  feature tensors ready in %.0fs",time .time ()-_t_feat )

            # hierpoint: pre-build the pooling hierarchies (FPS + grouping/upsample
            # indices) in the same pool -- pure CPU numpy/scipy, disk-cached, so epoch 1
            # doesn't stall ten serial per-batch construction.
    if is_hier :
        _to_hier =[d for d in prepared 
        if d .get ("hier")is None and d ["nbr"]is not None ]
        if _to_hier :
            import os as _os2 
            from concurrent .futures import ProcessPoolExecutor as _PPE 

            # PROCESSES, not threads: _fps is a pure-Python loop, so the GIL caps a
            # ThreadPool at ~2 effective cores (measured: 6.2h for 7118 patches).
            # The build is deterministic, so this is a pure wall-clock win.
            _n_proc =min (int (_os2 .environ .get ("CP_PREP_WORKERS","0"))or 
            max ((_os2 .cpu_count ()or 2 )-1 ,1 ),8 )
            logger .info ("pre-building %d pooling hierarchies (%d processes%s) ...",
            len (_to_hier ),_n_proc ,
            ", disk-cached"if knn_cache_dir else 
            ", NOT cached -- pass --prep-cache-dir to reuse next run")
            _t_h =time .time ()
            _payload =[(d ["verts_norm"],_hier_kw ,knn_cache_dir )
            for d in _to_hier ]
            try :
                with _PPE (max_workers =_n_proc )as _pool :
                    for _d ,_h in zip (_to_hier ,_pool .map (_hier_worker ,_payload ,
                    chunksize =8 )):
                    # Lazy mode: the worker has just WRITTEN this hierarchy to the
                    # disk cache, which is the whole point of the prebuild here.
                    # Retaining it as well is what filled RAM -- drop it and let
                    # _get_hier read it back per batch.
                        _d ["hier"]=None if _lazy_hier else _h 
            except Exception as _exc :# noqa: BLE001 -- spawn/pickle issues
                logger .warning ("process-pool hierarchy prebuild failed (%s) -- "
                "falling back to threads",_exc )
                def _build_hier_one (d ):
                    try :
                    # _get_hier, not _ensure_hier: in lazy mode the latter is a
                    # no-op, so this fallback would leave the disk cache COLD and
                    # every batch would rebuild from scratch. _get_hier writes the
                    # cache either way and only retains in RAM when not lazy.
                        _get_hier (d )
                    except Exception as _e :# noqa: BLE001
                        logger .warning ("hierarchy build failed (part skipped): %s",_e )
                        d ["_skip"]=True 
                with _TPE (max_workers =_n_workers )as _pool :
                    list (_pool .map (_build_hier_one ,_to_hier ))
            logger .info ("  hierarchies ready in %.0fs",time .time ()-_t_h )

            # OPT-IN host-tensor pinning (t/m go straight into .to(device) every epoch, so
            # pinning lets non_blocking=True overlap the copy with compute). DEFAULT OFF:
            # ten this box's WDDM driver, pinning thousands of small tensors raises a CUDA
            # error ("resource already mapped") that CORRUPTS the CUDA context -- every
            # later kernel then spuriously OOMs and the process dies with an access
            # violation (0xC0000005). Catching the RuntimeError is not enough: the context
            # is already poisoned. The throughput win here is minor (the real wins are the
            # removed per-part syncs and fwd_verts, both pinning-independent), so it stays
            # off unless the user explicitly enables it ten a driver where it's safe.
            # Without pinning, non_blocking=True degrades harmlessly to a blocking copy
            # (== the proven pre-change behaviour).
    _pin =bool (pin_host_memory )# None (default) or False -> OFF
    if _pin and device =="cuda":
        try :
            for d in prepared :
                if not d .get ("_skip"):
                    d ["t"]=d ["t"].pin_memory ()
                    d ["m"]=d ["m"].pin_memory ()
        except RuntimeError as _exc :# page-locked RAM exhausted: stay pageable
            logger .warning ("pin_memory failed (%s) -- continuing unpinned",_exc )
            if device =="cuda":
                torch .cuda .empty_cache ()

                # Graph batching: group parts into accumulation WINDOWS so the loss
                # normalisation and optimizer-step cadence see several parts at before.
                # Parts <= max_gpu_verts are batched together (GPU); oversized parts (rare after
                # spatial-patch / uniform-cap) run solo through the CPU replica path.
    _BATCH_VERTS =max (4 *int (max_gpu_verts or 7000 ),16000 )
    # Per-FORWARD vertex budget. One EdgeConv forward above ~max_gpu_verts falls
    # off this card's memory cliff, and a fused _BATCH_VERTS-sized forward is 4x
    # past it. WDDM (Windows) drivers do not FAIL that allocation -- they satisfy
    # it silently from SYSTEM memory, so the OOM->CPU handler below never fires
    # and every kernel then reads over PCIe (observed: 10+ GB "shared GPU
    # memory", hours/epoch instead of minutes). Each window is therefore
    # EXECUTED in chunks of <= _FWD_VERTS vertices. With global_feat=False and
    # per-vertex LayerNorm the net never mixes information across disconnected
    # graphs, so chunked forwards + summed backwards give EXACTLY the gradients
    # of the fused forward -- only the peak GPU memory changes.
    # fwd_verts overrides the chunk budget independently of the patch cap: the
    # workload is launch-latency-bound (GPU sits at ~40-65% of VRAM with flat
    # batch times), so a memory-leaner backbone can fuse several patches per
    # forward and claw back the throughput the 7000-chunking cost.
    _FWD_VERTS =int (fwd_verts )if fwd_verts else int (max_gpu_verts or 7000 )

    def _batch_accumulate (batch_dicts ,run_device ,loss_div ):
        """Forward + backward for one accumulation window of parts. Each GPU
        forward runs a <=_FWD_VERTS chunk as one disconnected graph (kNN indices
        offset per part); losses are weighted and normalised over the WHOLE
        window, identically to a single fused forward."""
        total_w =sum (float (d .get ("graph_weight",1.0 ))for d in batch_dicts )
        chunks ,cur ,cur_v =[],[],0 
        for d in batch_dicts :
            n =d ["x"].shape [0 ]
            if cur and cur_v +n >_FWD_VERTS :
                chunks .append (cur );cur ,cur_v =[],0 
            cur .append (d );cur_v +=n 
        if cur :
            chunks .append (cur )
        tot_parts ={"total":0.0 ,"per_part":[]}
        for chunk in chunks :
            xs ,nbs ,offset =[],[],0 
            for d in chunk :
                if use_normals :
                    if d .get ("featx")is None :
                        d ["featx"]=_knn_feature_tensor (
                        d ["verts_norm"],d ["nbr"],True ,"cpu",
                        use_curvature ,use_concavity ,use_edge_dist )
                    xs .append (d ["featx"])
                else :
                    xs .append (d ["x"])
                if not is_hier :
                    nbs .append (d ["nbr"]+offset )
                    offset +=d ["x"].shape [0 ]
            X =torch .cat (xs ,dim =0 ).to (run_device )
            if is_hier :
            # collate offsets every level's indices per graph -- disconnected
            # graphs stay disconnected (segment-wise global pool), so the
            # chunked==fused gradient identity holds for hierpoint too
                NB =_hp .collate ([_get_hier (d )for d in chunk ],
                [d ["nbr"]for d in chunk ],device =run_device )
            else :
                NB =torch .cat (nbs ,dim =0 ).to (run_device )
            with torch .autocast (device_type ="cuda",dtype =torch .float16 ,
            enabled =(use_amp and run_device =="cuda")):
                out_all =model (X ,NB )
            chunk_loss =torch .zeros (1 ,device =run_device )
            seg =0 
            for d in chunk :
                nj =d ["x"].shape [0 ]
                out_j =out_all [seg :seg +nj ]
                # weight each graph's contribution by 1/n_patches-of-its-part
                # (graph_weight) so a patched (wscad/PXC) part's TOTAL gradient
                # share across its patches equals one non-patched part's share,
                # not patch_count-times more.
                # non_blocking: t/m are pinned at prep, so these copies overlap
                # the forward instead of serialising each launch (WDDM pageable
                # copies block the host).
                lj ,pj =_loss (out_j .float (),
                d ["t"].to (run_device ,non_blocking =True ),
                d ["m"].to (run_device ,non_blocking =True ),
                sign_inv =d .get ("sign_inv",False ),
                pw =d .get ("pw"),as_tensors =True )
                chunk_loss =chunk_loss +float (d .get ("graph_weight",1.0 ))*lj 
                # per-part loss, kept separate from the summed total so hard_mining
                # can weight EACH part by ITS OWN loss instead of the batch mean
                # (previously every part in a GPU batch got the same EMA update
                # regardless of its individual difficulty -- diluting the "mine the
                # parts the model keeps failing ten" signal into "mine hard BATCHES").
                # Both stay DETACHED 0-dim GPU tensors here; the epoch loop
                # materialises them in one sync per epoch (was: 4 float() syncs
                # per part, ~14k/epoch).
                tot_parts ["per_part"].append (pj ["total"])
                tot_parts ["total"]=tot_parts ["total"]+pj ["total"]
                seg +=nj 
                # backward per chunk, all scaled by the SAME window normalisation:
                # sum over chunks of chunk_loss/total_w/loss_div == the fused
                # window's total_loss/total_w/loss_div (gradients simply add up),
                # and each chunk's autograd graph is freed before the next forward.
            scaler .scale (chunk_loss /max (total_w ,1e-8 )/loss_div ).backward ()
            del X ,NB ,out_all ,chunk_loss 
        return tot_parts 

    def _make_epoch_batches (shuf_order ):
        """Group shuffled parts into (device, [indices]) pairs."""
        batches ,gpu_buf ,gpu_v =[],[],0 
        for i in shuf_order :
            d =prepared [i ]
            if d .get ("_skip")or d ["nbr"]is None :
                continue 
            n =d ["x"].shape [0 ]
            is_big =device !="cpu"and max_gpu_verts and n >max_gpu_verts 
            if is_big :
                if gpu_buf :
                    batches .append (("gpu",list (gpu_buf )));gpu_buf [:]=[];gpu_v =0 
                batches .append (("cpu",[i ]))
            else :
                if gpu_buf and gpu_v +n >_BATCH_VERTS :
                    batches .append (("gpu",list (gpu_buf )));gpu_buf [:]=[];gpu_v =0 
                gpu_buf .append (i );gpu_v +=n 
        if gpu_buf :
            batches .append (("gpu",list (gpu_buf )))
        return batches 

        # Hard-example mining: exponential moving average of per-part loss.
        # After epoch 0, parts with higher EMA loss are sampled more frequently.
    _loss_ema =np .ones (len (prepared ),dtype =np .float64 )
    _EMA_ALPHA =0.5 # decay: new = alpha * recent + (1-alpha) * old

    try :
        for ep in range (start_epoch ,epochs ):
            model .train ()
            if hard_mining and ep >start_epoch :
            # Sample proportional to EMA loss (with minimum floor = mean/4)
                w =hard_mining_probabilities (_loss_ema )
                order =list (np .random .default_rng (ep ).choice (
                len (prepared ),size =len (prepared ),replace =True ,p =w ))
            else :
                random .shuffle (order )
                # Lazy-build any graphs missed by the pre-builder (failed parts)
            for i in order :
                d =prepared [i ]
                if d ["nbr"]is None and not d .get ("_skip"):
                    try :
                        d ["nbr"]=_knn_graph (d ["verts_norm"],meta ["k"])
                    except Exception as exc :# noqa: BLE001
                        logger .warning ("kNN build failed (skip): %s",exc )
                        d ["_skip"]=True 
                if is_hier and not d .get ("_skip")and d .get ("hier")is None :
                    try :
                        _ensure_hier (d )
                    except Exception as exc :# noqa: BLE001
                        logger .warning ("hierarchy build failed (skip): %s",exc )
                        d ["_skip"]=True 
            ep_batches =_make_epoch_batches (order )
            opt .zero_grad (set_to_none =True )
            accum_since_step =0 
            tot =0.0 ;ep_t0 =time .time ()
            ep_ema_pairs =[]# (part_idx, loss) pairs, materialised before/epoch
            hb =max (1 ,len (ep_batches )//5 )
            for k ,(bdev ,bidx )in enumerate (ep_batches ,1 ):
                if k %hb ==0 or k ==len (ep_batches ):
                    n_parts_so_far =sum (len (b )for _ ,b in ep_batches [:k ])
                    # reserved MB makes a memory-cliff regression visible in the
                    # console instead of only as a mysterious slowdown
                    gpu_mb =(torch .cuda .memory_reserved ()//(1 <<20 )
                    if str (device ).startswith ("cuda")else 0 )
                    logger .info ("  epoch %d: %d/%d batches (%d parts, %.0fs, "
                    "gpu %dMB)",ep ,k ,len (ep_batches ),
                    n_parts_so_far ,time .time ()-ep_t0 ,gpu_mb )
                bd =[prepared [i ]for i in bidx ]
                # same normaliser the fused GPU path uses (_batch_accumulate), so a
                # CPU/OOM-fallback batch produces the SAME gradient magnitude as the
                # GPU batch it replaces
                bw =sum (float (d .get ("graph_weight",1.0 ))for d in bd )
                if bdev =="cpu":
                # Big-part CPU path: solo _accumulate (GPU→CPU grad transfer)
                    parts ={"total":0.0 ,"per_part":[]}
                    for d in bd :
                        p =_accumulate (d ,"cpu",device ,accum ,
                        sign_inv =d .get ("sign_inv",False ),
                        weight_div =bw )
                        parts ["total"]+=p .get ("total",0.0 )
                        parts ["per_part"].append (p .get ("total",0.0 ))
                else :
                    try :
                        parts =_batch_accumulate (bd ,device ,accum )
                    except (torch .cuda .OutOfMemoryError ,RuntimeError )as exc :
                        if "out of memory"not in str (exc ).lower ():
                            raise 
                        _guard_oom_fallback (accum_since_step )
                        torch .cuda .empty_cache ()
                        opt .zero_grad (set_to_none =True )
                        logger .warning ("CUDA OOM on %d-part batch -> CPU",len (bd ))
                        parts ={"total":0.0 ,"per_part":[]}
                        for d in bd :
                            p =_accumulate (d ,"cpu",device ,accum ,
                            sign_inv =d .get ("sign_inv",False ),
                            weight_div =bw )
                            parts ["total"]+=p .get ("total",0.0 )
                            parts ["per_part"].append (p .get ("total",0.0 ))
                accum_since_step +=1 
                if accum_since_step >=accum or k ==len (ep_batches ):
                    if grad_clip and grad_clip >0 :
                        scaler .unscale_ (opt )
                        torch .nn .utils .clip_grad_norm_ (model .parameters (),grad_clip )
                    scaler .step (opt )
                    scaler .update ()
                    opt .zero_grad (set_to_none =True )
                    _cpu ["dirty"]=True 
                    accum_since_step =0 
                    # parts["total"]/per_part are detached 0-dim GPU tensors ten the
                    # batched GPU path (floats ten the CPU path) -- accumulate without
                    # touching the host; materialised below, ONCE per epoch.
                tot =tot +parts ["total"]
                if hard_mining :
                    ep_ema_pairs .extend (zip (bidx ,parts ["per_part"]))
                    # ONE device sync per epoch for all bookkeeping numbers (was: 4
                    # float(cuda_tensor) syncs per part, ~14k/epoch ten v19 -- a measured
                    # throughput drain ten the launch-latency-bound T1200 path).
            if hard_mining and ep_ema_pairs :
                vals =[v for _ ,v in ep_ema_pairs ]
                t_pos =[j for j ,v in enumerate (vals )if torch .is_tensor (v )]
                if t_pos :
                    flat =torch .stack ([vals [j ]for j in t_pos ]).cpu ().tolist ()
                    for j ,fv in zip (t_pos ,flat ):
                        vals [j ]=float (fv )
                        # sequential update in batch order == the previous per-batch math
                        # (a part sampled twice decays twice, exactly as before)
                for (i ,_ ),pl in zip (ep_ema_pairs ,vals ):
                    _loss_ema [i ]=(_EMA_ALPHA *pl 
                    +(1.0 -_EMA_ALPHA )*_loss_ema [i ])
            tot =float (tot )
            # metric-driven schedulers (plateau) are stepped by the bookkeeper with
            # the val metric; per-epoch schedulers step here.
            if sched is not None and not sched_metric :
                sched .step ()
            stop =bk .after_epoch (ep ,tot /max (1 ,len (prepared )))
            if str (device ).startswith ("cuda"):
            # hand cached blocks back before per epoch so allocator
            # fragmentation can never ratchet the pool toward the cap
                torch .cuda .empty_cache ()
            if stop :
                logger .info ("early stopping at epoch %d",ep )
                break 
    except KeyboardInterrupt :
        bk .flush_history ()
        logger .warning ("training interrupted -- resume from the last checkpoint%s",
        f" ({last_path })"if last_path else "")
        raise 
    bk .finalize ()
    return model ,meta 


def infer_knngraph (model ,meta ,verts_norm ,device ="cpu",max_gpu_verts =60000 ,
offset_scale =1.0 ,graph_cache =None ,knn_cache_dir =None ,patch =False ,
part_nr =None ):
    """Run a trained kNN-graph regressor ten one part -> (N,7) numpy array ready
    for cp_targets.decode_predictions. Needs only normalised vertices (no faces).

    TWO vertex-selection modes (the model is one set of weights; the caller picks the
    mode that matches how the part was TRAINED -- see train_knngraph_regressor):

    patch=False (DEFAULT, the v8 path for ABB / general / unknown parts): a part with
    more than the model's cap of vertices is UNIFORMLY subsampled to the cap and the
    predictions scattered back (un-sampled verts stay heat 0). Safe default -- it does
    NOT flood big parts with detections.

    patch=True (for wscad-style parts with small cylindrical openings): the part is
    split into SPATIAL PATCHES each <= cap verts at FULL density (spatial_patches), so
    the small CP openings keep the geometry the model keys ten. Each block is run and
    scatter-averaged back. Use this only for parts trained WITH patching, or the big-
    part heatmap floods with false positives.

    The cap is read from meta['subsample'] (what training used); `max_gpu_verts` is
    only the fallback for older checkpoints. CUDA OOM ten any run falls back to CPU.

    part_nr : the part's identifier, needed to reconstruct the EXACT per-part
    subsample training picked for this part (part_subsample_seed(meta['seed'],
    part_nr)) when it's over the cap. Without it, this falls back to a fixed
    seed=0 -- a DIFFERENT vertex subset than what the model trained ten for this
    part (silently drops recall ten big ABB/general parts; this was an unreported
    train/inference mismatch, now fixed for every caller that passes part_nr).
    Only matters ten the non-patch (v8, uniform-subsample) path; patch=True already
    covers every vertex via spatial_patches, no subsample seed involved.

    graph_cache / knn_cache_dir : graph memoisation (in-process / ten-disk); used ten
    the single whole-part path. See the cache notes inline."""
    torch =_require_torch ()
    import copy 
    V =np .asarray (verts_norm )
    n =len (V )
    cap =int (meta .get ("subsample")or max_gpu_verts or 0 )
    use_normals =bool (meta .get ("normals",False ))
    use_curvature =bool (meta .get ("curvature",False ))
    use_concavity =bool (meta .get ("concavity",False ))
    use_edge_dist =bool (meta .get ("edge_dist",False ))
    # hierpoint checkpoints run through the SAME vertex-selection / patch /
    # vote-merge logic (it is backbone-independent); only the inner forward
    # differs -- it takes the collated pooling hierarchy instead of a kNN tensor.
    _is_hier =str (meta .get ("backbone","knngraph"))=="hierpoint"
    if _is_hier :
        import hierpoint as _hp 
        _hier_kw =_hp .hier_params_from_meta (meta )

    def _run (Vq ,nbr_cpu ):
        def __run (rd ):
            if _is_hier :
                hier =_hp .build_hierarchy (Vq ,cache_dir =knn_cache_dir ,
                **_hier_kw )
                nbr =_hp .collate ([hier ],[nbr_cpu ],device =rd )
            else :
                nbr =nbr_cpu .to (rd ).long ()# int32 host -> int64 ten device
            x =_knn_feature_tensor (Vq ,nbr_cpu ,use_normals ,rd ,
            use_curvature ,use_concavity ,use_edge_dist )
            mdl =model if rd ==device else copy .deepcopy (model ).to (rd )
            mdl .eval ()
            with torch .no_grad ():
                out =mdl (x ,nbr )
            return pred_to_array (out ,offset_scale =offset_scale )
        try :
            return __run (device )
        except (torch .cuda .OutOfMemoryError ,RuntimeError )as exc :
            if "out of memory"not in str (exc ).lower ():
                raise 
            torch .cuda .empty_cache ()
            logger .warning ("CUDA OOM during inference on %d-vertex run -> CPU",len (Vq ))
            return __run ("cpu")

    if not patch :
    # v8 path: single uniform subsample to cap, scatter back. Default for ABB /
    # general / unknown parts -- avoids the full-density big-part FP flood.
    # Reconstruct the SAME subsample training used for THIS part (part_nr must
    # be given); fall back to seed=0 only when the caller can't provide it
    # (older call sites / no identifier available) -- that fallback is a known
    # train/inference mismatch for any part over the cap, see docstring.
        if part_nr is not None :
            sub_seed =part_subsample_seed (meta .get ("seed",0 ),str (part_nr ))
        else :
            sub_seed =0 
            if cap and n >cap :
                logger .warning (
                "infer_knngraph: no part_nr given for a %d-vertex part (cap %d) "
                "-- using seed=0, which likely does NOT match the subsample this "
                "part was trained on. Pass part_nr to reproduce it exactly.",n ,cap )
        idx =uniform_subsample_idx (n ,cap ,seed =sub_seed )
        Vq =V if idx is None else V [idx ]
        if idx is not None :
            logger .info ("infer_knngraph: subsampled %d -> %d verts (%.0f%% kept)",
            n ,len (Vq ),100.0 *len (Vq )/n )
            # Key = (id, n_verts): id() alone is unsafe (Python may reuse a freed address);
            # pairing with n_verts removes that false-positive. uniform_subsample_idx is
            # deterministic (same sub_seed) so the same verts_norm -> the same Vq -> a
            # valid hit.
        cache_key =(id (verts_norm ),n )if graph_cache is not None else None 
        nbr_cpu =graph_cache .get (cache_key )if cache_key is not None else None 
        if nbr_cpu is None :
            nbr_cpu =(_knn_graph_disk (Vq ,meta ["k"],knn_cache_dir )if knn_cache_dir 
            else _knn_graph (Vq ,meta ["k"]))
            if cache_key is not None :
                graph_cache [cache_key ]=nbr_cpu 
        arr =_run (Vq ,nbr_cpu )
        if idx is None :
            return arr 
        full =np .zeros ((n ,N_CHANNELS ),dtype =arr .dtype )
        full [idx ]=arr # un-sampled verts stay heat 0 (no vote)
        return full 

        # patch=True: SPATIAL PATCHES (full-density blocks) for wscad-style small openings.
        #
        # CP_INFER_MAX_PATCHES raises spatial_patches' 8-patch ceiling AT INFERENCE ONLY.
        # That ceiling is a hard blind spot, not a cost knob: a part bigger than
        # max_patches*cap (8*14000 = 112,000 verts) is UNIFORMLY SUBSAMPLED to that budget
        # before patching -- i.e. it suffers exactly the density loss patching exists to
        # prevent. Measured ten v31's val (2026-07-14): 27 of 278 parts (10%) exceed the
        # budget and carry 390 GT CPs (18% of ALL GT); the worst keeps only 16.9% of its
        # geometry and the model finds 14 of its 42 CPs -- one part scores TP=0 ten 36 CPs.
        # Those CPs are UNREACHABLE regardless of model quality, which is why FN sat at
        # ~545 for ten straight epochs while train_loss kept falling.
        # Raising it here needs NO retraining: the model learned what a CP looks like
        # LOCALLY from patches, so showing it more patches of the same size at inference is
        # the standard train-ten-crops / infer-full-image pattern. Default 8 keeps the old
        # behaviour byte-for-byte.
    _max_patches =int (os .environ .get ("CP_INFER_MAX_PATCHES","8"))
    patches ,sel =spatial_patches (V ,cap ,margin_frac =PATCH_MARGIN_FRAC ,
    max_patches =_max_patches )
    if sel is None and len (patches )==1 and len (patches [0 ])==n :
        nbr_cpu =(_knn_graph_disk (V ,meta ["k"],knn_cache_dir )if knn_cache_dir 
        else _knn_graph (V ,meta ["k"]))
        return _run (V ,nbr_cpu )# part already <= cap: one whole-part run
    Vs =V if sel is None else V [sel ]# frame the patch indices point into
    ns =len (Vs )
    acc =np .zeros ((ns ,N_CHANNELS ),dtype =np .float64 )
    cnt =np .zeros (ns ,dtype =np .float64 )
    for pidx in patches :
        Vq =Vs [pidx ]
        nbr_cpu =_knn_graph (Vq ,meta ["k"])
        acc [pidx ]+=_run (Vq ,nbr_cpu )
        cnt [pidx ]+=1.0 
    cov =cnt >0 
    pred_s =np .zeros ((ns ,N_CHANNELS ),dtype =np .float32 )
    pred_s [cov ]=(acc [cov ]/cnt [cov ,None ]).astype (np .float32 )
    full =pred_s if sel is None else np .zeros ((n ,N_CHANNELS ),dtype =np .float32 )
    if sel is not None :
        full [sel ]=pred_s # un-sampled verts of a giant part stay heat=0
    nrm =np .linalg .norm (full [:,ct .DIRECTION ],axis =1 ,keepdims =True )
    full [:,ct .DIRECTION ]=full [:,ct .DIRECTION ]/np .where (nrm <1e-8 ,1.0 ,nrm )
    logger .info ("infer_knngraph: %d verts -> %d spatial patches (<=%d, %d kept), "
    "%.0f%% covered",n ,len (patches ),cap ,ns ,100.0 *cov .mean ())
    return full 


def infer_knngraph_tta (model ,meta ,verts ,device ="cpu",max_gpu_verts =60000 ,
n_aug =8 ,seed =0 ,patch =False ,part_nr =None ):
    """Test-time augmentation for the knngraph backbone -> (N,7) in the ORIGINAL frame.

    The raw-xyz knngraph is POSE-SENSITIVE, so its per-vertex heatmap is noisy under
    rotation. Run inference ten `n_aug` random rotations of the part, rotate each
    prediction BACK to the original frame, and average per vertex. A true connection
    point fires in (almost) every rotation -> stays high; a pose-dependent spurious
    peak fires in only some -> averages below threshold -> suppressed. Net: a large
    precision gain (measured +7 micro-F1 / FP cut ~70% ten the real corpus) with NO
    retraining. A rotation is an isometry, so the kNN graph is identical across views
    and only the input coords change.

    `verts` are RAW mm vertices (each view is re-normalised internally). Offset and
    direction are vectors in the rotated frame, so they are inverse-rotated (@R) back
    to the original frame before averaging; the heatmap is per-vertex and averaged
    directly. Use the TTA-tuned operating point (heatmap_thresh ~0.6) when decoding.
    """
    import augment as aug 
    V =np .asarray (verts ,dtype =np .float64 )
    rng =np .random .default_rng (seed )
    rots =[np .eye (3 )]+[aug ._random_rotation_matrix (rng )
    for _ in range (max (0 ,int (n_aug )-1 ))]
    c =V .mean (0 )
    acc =None 
    for R in rots :
        Vr =(V -c )@R .T +c 
        Vn ,_ ,scale =normalize_vertices (Vr )
        a =infer_knngraph (model ,meta ,Vn ,device =device ,
        max_gpu_verts =max_gpu_verts ,offset_scale =scale ,
        patch =patch ,part_nr =part_nr ).copy ()
        a [:,ct .OFFSET ]=a [:,ct .OFFSET ]@R # vectors back to original frame
        a [:,ct .DIRECTION ]=a [:,ct .DIRECTION ]@R 
        acc =a if acc is None else acc +a 
    arr =acc /len (rots )
    nrm =np .linalg .norm (arr [:,ct .DIRECTION ],axis =1 ,keepdims =True )
    arr [:,ct .DIRECTION ]=arr [:,ct .DIRECTION ]/np .where (nrm <1e-8 ,1.0 ,nrm )
    return arr 


def infer_knngraph_tta_batched (model ,meta ,verts ,device ="cpu",max_gpu_verts =60000 ,
n_aug =8 ,seed =0 ,patch =False ,part_nr =None ):
    """Batched TTA: all N rotation views in ONE GPU forward pass instead of N serial passes.

    Speed gain: for n_aug=8 and a 1000-vertex part the serial TTA does 8 forward passes;
    the batched version does ONE pass ten 8*1000 = 8000 vertices (a single kNN build +
    forward). On GPU the batch is ~N×faster than serial because GPU utilisation per small
    serial call is low. For large parts (>max_gpu_verts/n_aug) falls back automatically
    to the serial infer_knngraph_tta to stay within GPU memory.

    The graph construction also benefits: in serial TTA each rotated view rebuilds the
    kNN graph independently; here a SINGLE disconnected-graph construction covers all
    views at before (identical neighbourhood structure; only input coords differ).

    Returns (N,7) in the ORIGINAL frame, identical semantics to infer_knngraph_tta.
    """
    torch =_require_torch ()
    import augment as aug 

    # hierpoint's forward takes a collated pooling hierarchy, not a flat kNN
    # tensor, so the concatenated-views batched path below does not apply. The
    # serial TTA delegates to infer_knngraph (which IS backbone-aware), so route
    # hierpoint checkpoints there -- correct, just one forward per rotation.
    if str (meta .get ("backbone","knngraph"))=="hierpoint":
        return infer_knngraph_tta (model ,meta ,verts ,device =device ,
        max_gpu_verts =max_gpu_verts ,n_aug =n_aug ,
        seed =seed ,patch =patch ,part_nr =part_nr )

    V =np .asarray (verts ,dtype =np .float64 )
    n =len (V )
    cap =int (meta .get ("subsample")or max_gpu_verts or 0 )
    k =int (meta .get ("k",16 ))

    # Safety: if the batched tensor would exceed GPU memory, fall back to serial TTA.
    # Each view needs <=cap verts; if n > cap // n_aug, the batch would blow up GPU.
    batch_cap =cap //n_aug if cap else 0 
    if cap and n >batch_cap :
        return infer_knngraph_tta (model ,meta ,verts ,device =device ,
        max_gpu_verts =max_gpu_verts ,
        n_aug =n_aug ,seed =seed ,patch =patch ,
        part_nr =part_nr )

    use_normals =bool (meta .get ("normals",False ))
    use_curvature =bool (meta .get ("curvature",False ))
    use_concavity =bool (meta .get ("concavity",False ))
    use_edge_dist =bool (meta .get ("edge_dist",False ))

    rng =np .random .default_rng (seed )
    rots =[np .eye (3 )]+[aug ._random_rotation_matrix (rng )
    for _ in range (max (0 ,int (n_aug )-1 ))]
    c =V .mean (0 )

    # Build one disconnected graph covering all rotated views
    all_Vn =[]# normalised rotated vertices for each view
    all_R =[]# rotation matrices for back-rotation
    all_sc =[]# per-view scale factors
    for R in rots :
        Vr =(V -c )@R .T +c 
        Vn ,_ ,scale =normalize_vertices (Vr )
        all_Vn .append (Vn .astype (np .float32 ))
        all_R .append (R )
        all_sc .append (scale )

        # Concatenate all views into one large point cloud; kNN built ten each view
        # independently, then concatenated with index offsets (disconnected subgraphs)
    offset =0 
    cat_V_list =[]
    cat_nbr_list =[]
    for Vn in all_Vn :
    # int32 host graph; offsets stay well inside int32 (n_aug * n << 2^31)
        nbr_v =_knn_graph (Vn ,k )# (n, k) int32 tensor
        cat_V_list .append (torch .tensor (Vn ,dtype =torch .float32 ))
        cat_nbr_list .append (nbr_v +offset )
        offset +=len (Vn )

    cat_V =torch .cat (cat_V_list )# (n_aug*n, 3) or similar
    cat_nbr =torch .cat (cat_nbr_list )# (n_aug*n, k)

    # Feature tensor (curvature/concavity computed per-view ten the flat cat)
    # _pca_normals needs CPU numpy nbr, so pass the tensor's numpy view
    cat_nbr_np =cat_nbr .numpy ()
    x =_knn_feature_tensor (cat_V .numpy (),cat_nbr_np ,use_normals ,device ,
    use_curvature ,use_concavity ,use_edge_dist )

    model .eval ()
    try :
        with torch .no_grad ():
            batch_out =model (x ,cat_nbr .to (device ).long ())# (n_aug*n, C_out)
        batch_np =batch_out .cpu ().numpy ()
    except (torch .cuda .OutOfMemoryError ,RuntimeError )as exc :
        if "out of memory"not in str (exc ).lower ():
            raise 
        torch .cuda .empty_cache ()
        return infer_knngraph_tta (model ,meta ,verts ,device =device ,
        max_gpu_verts =max_gpu_verts ,
        n_aug =n_aug ,seed =seed ,patch =patch ,
        part_nr =part_nr )

        # Split, apply sigmoid + offset_scale in numpy, back-rotate vectors, accumulate
    def _post_np (raw ,scale ):
        """Apply sigmoid to heatmap + scale offsets ten a (N,7) raw numpy slice."""
        a =raw .astype (np .float64 )
        a [:,ct .HEATMAP ]=1.0 /(1.0 +np .exp (-a [:,ct .HEATMAP ]))# sigmoid
        a [:,ct .OFFSET ]=a [:,ct .OFFSET ]*float (scale )
        nd =np .linalg .norm (a [:,ct .DIRECTION ],axis =1 ,keepdims =True )
        a [:,ct .DIRECTION ]=a [:,ct .DIRECTION ]/np .where (nd <1e-8 ,1.0 ,nd )
        return a .astype (np .float32 )

    acc =None 
    for vi ,(R ,scale )in enumerate (zip (all_R ,all_sc )):
        a =_post_np (batch_np [vi *n :(vi +1 )*n ],scale )
        a [:,ct .OFFSET ]=a [:,ct .OFFSET ]@R 
        a [:,ct .DIRECTION ]=a [:,ct .DIRECTION ]@R 
        acc =a if acc is None else acc +a 

    arr =acc /len (rots )
    nrm =np .linalg .norm (arr [:,ct .DIRECTION ],axis =1 ,keepdims =True )
    arr [:,ct .DIRECTION ]=arr [:,ct .DIRECTION ]/np .where (nrm <1e-8 ,1.0 ,nrm )
    return arr 


def infer_knngraph_multicrop (model ,meta ,verts_norm ,device ="cpu",max_gpu_verts =60000 ,
offset_scale =1.0 ,n_crops =12 ,max_crops =24 ,graph_cache =None ):
    """Big-part recall fix WITHOUT retraining -> (N,7) numpy ready for decode.

    A part larger than the model's cap is seen only `cap` (e.g. 7000) verts at a
    time, so a SINGLE inference subsample drops most of a big mesh and misses the CPs
    in the dropped regions (measured: big parts F1 0.13 / recall 0.23 vs small parts
    F1 0.60). Here we run the model ten MULTIPLE independent uniform subsamples ("crops"),
    each at the trained density, accumulate per-vertex predictions over the crops each
    vertex lands in, and average -> a vertex covered by any crop gets a prediction, so
    CPs across the WHOLE big part are recoverable. Heat is averaged over the crops a
    vertex appeared in (a real CP fires whenever it is included -> stays high; a crop-
    specific spurious peak is diluted). Small parts (<= cap) fall through to the normal
    single-pass infer_knngraph unchanged.

    n_crops is the floor; it is raised toward `need` (crops to cover ~95% of verts,
    (1-cap/n)^K < 0.05) but capped at max_crops so the 471k-vertex part stays bounded
    (it just gets partial coverage, still far better than one crop). CUDA OOM ten a crop
    falls back to CPU for that crop.
    """
    torch =_require_torch ()
    import copy ,math 
    V =np .asarray (verts_norm )
    n =len (V )
    cap =int (meta .get ("subsample")or max_gpu_verts or 0 )
    if not cap or n <=cap :
        return infer_knngraph (model ,meta ,verts_norm ,device =device ,
        max_gpu_verts =max_gpu_verts ,offset_scale =offset_scale ,
        graph_cache =graph_cache )
    frac =cap /n 
    need =int (math .ceil (math .log (0.05 )/math .log (1.0 -frac )))if frac <1.0 else 1 
    K =max (int (n_crops ),min (need ,int (max_crops )))

    def _run (Vq ,nbr ,rd ):
        nb =nbr .to (rd ).long ()# int32 host -> int64 ten device
        x =torch .tensor (np .asarray (Vq ),dtype =torch .float32 ,device =rd )
        mdl =model if rd ==device else copy .deepcopy (model ).to (rd )
        mdl .eval ()
        with torch .no_grad ():
            out =mdl (x ,nb )
        return pred_to_array (out ,offset_scale =offset_scale )

    acc =np .zeros ((n ,N_CHANNELS ),dtype =np .float64 )
    cnt =np .zeros (n ,dtype =np .float64 )
    for k in range (K ):
        idx =uniform_subsample_idx (n ,cap ,seed =k )# a different crop each pass
        Vq =V [idx ]
        nbr_cpu =_knn_graph (Vq ,meta ["k"])
        try :
            arr_sub =_run (Vq ,nbr_cpu ,device )
        except (torch .cuda .OutOfMemoryError ,RuntimeError )as exc :
            if "out of memory"not in str (exc ).lower ():
                raise 
            torch .cuda .empty_cache ()
            arr_sub =_run (Vq ,nbr_cpu ,"cpu")
        acc [idx ]+=arr_sub 
        cnt [idx ]+=1.0 
    full =np .zeros ((n ,N_CHANNELS ),dtype =np .float32 )
    cov =cnt >0 
    full [cov ]=(acc [cov ]/cnt [cov ,None ]).astype (np .float32 )
    nrm =np .linalg .norm (full [:,ct .DIRECTION ],axis =1 ,keepdims =True )
    full [:,ct .DIRECTION ]=full [:,ct .DIRECTION ]/np .where (nrm <1e-8 ,1.0 ,nrm )
    logger .info ("infer_knngraph_multicrop: %d verts, %d crops -> %.0f%% covered",
    n ,K ,100.0 *cov .mean ())
    return full 


def _selftest ():
    """Overfit the MLP ten one synthetic plate; check it recovers the CPs."""
    torch =_require_torch ()
    torch .manual_seed (0 )
    np .random .seed (0 )
    import json_dataset as jd 
    # synthetic plate part
    xs ,ys =np .meshgrid (np .linspace (0 ,100 ,40 ),np .linspace (0 ,100 ,40 ))
    V =np .column_stack ([xs .ravel (),ys .ravel (),np .zeros (xs .size )])
    F =np .zeros ((0 ,3 ),dtype =np .int64 )
    cp_pts =np .array ([[25 ,25 ,0 ],[75 ,30 ,0 ],[50 ,80 ,0 ]],float )
    cp_dir =np .array ([[0 ,0 ,1.0 ]]*3 )
    part =jd .Part ("SYN",V ,F ,cp_pts ,cp_dir ,["a","b","c"])
    s =prepare_sample (part ,dedup =False )
    # 1200 epochs: the bce heatmap loss sharpens a lone-vertex peak slowly, so a
    # short overfit run lands the peak just under the 0.5 decode threshold (~0.48
    # at 400) and decodes nothing. An overfit smoke-test must run long enough to
    # actually overfit; centernet/focal converge faster but bce is the default.
    model =train_cpmlp ([s ],epochs =1200 ,lr =1e-3 ,log_every =0 )
    model .eval ()
    with torch .no_grad ():
        out =model (torch .tensor (s ["verts_norm"],dtype =torch .float32 ))
    arr =pred_to_array (out ,offset_scale =s ["scale"])
    got =ct .decode_predictions (V ,arr ,heatmap_thresh =0.5 ,nms_radius_mm =15.0 )
    print ("cp_regressor selftest: recovered",len (got ),"of 3 CPs")
    # each ground-truth CP should be matched by some prediction (recall=1)
    matched =0 
    for cp in cp_pts :
        d =min (np .linalg .norm (g ["point"]-cp )for g in got )if got else 1e9 
        if d <=10.0 :
            matched +=1 
    for g in got :
        d =np .linalg .norm (cp_pts -g ["point"],axis =1 ).min ()
        assert g ["direction"][2 ]>0.9 ,"direction not outward (+z)"
        print ("   point %s  locErr=%.2fmm  dir=%s"%
        (np .round (g ["point"],1 ),d ,np .round (g ["direction"],2 )))
    assert matched ==3 ,f"overfit MLP recovered only {matched }/3 GT CPs"
    print ("cp_regressor selftest OK")


if __name__ =="__main__":
    logging .basicConfig (level =logging .INFO )
    _selftest ()
