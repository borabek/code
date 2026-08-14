"""Assert-based unit tests for the connection-point encode/decode core.

These complement the module `_selftest()` smoke checks (which mostly verify code
RUNS) with focused correctness assertions ten the pure-numpy data path -- no torch
needed, so they run anywhere:

    pytest tests/            # or: python -m pytest tests/test_cp.py

The headline guarantee is the encode -> decode ROUND TRIP: if we encode the
ground-truth targets for a part and decode them back, we must recover the same
connection points (this is the ceiling the trained model is chasing, and it
catches any drift in the channel layout, sigma rule, vote, or NMS math).
"""
import numpy as np 
import pytest 

import json_dataset as jd 
import cp_targets as ct 


# ---------------------------------------------------------------------------
# fixtures: a simple plate mesh with a few connection points
# ---------------------------------------------------------------------------

def _plate (n =40 ,size =100.0 ):
    xs ,ys =np .meshgrid (np .linspace (0 ,size ,n ),np .linspace (0 ,size ,n ))
    return np .column_stack ([xs .ravel (),ys .ravel (),np .zeros (xs .size )])


@pytest .fixture 
def plate_with_cps ():
    V =_plate ()
    cp =np .array ([[25.0 ,25 ,0 ],[75.0 ,30 ,0 ],[50.0 ,80 ,0 ]])
    cd =np .array ([[0.0 ,0 ,1 ]]*3 )
    return V ,cp ,cd 


    # ---------------------------------------------------------------------------
    # encode -> decode round trip (the correctness ceiling)
    # ---------------------------------------------------------------------------

def test_encode_decode_round_trip_recovers_all_cps (plate_with_cps ):
    V ,cp ,cd =plate_with_cps 
    target ,_mask ,_sigma =ct .encode_targets (V ,cp ,cd )
    # decode_predictions expects sigmoid'd heat; the encoded target heat is
    # already in [0,1], so it is directly decodable.
    preds =ct .decode_predictions (V ,target ,heatmap_thresh =0.3 ,
    nms_radius_mm =5.0 ,min_votes =1 )
    assert len (preds )==len (cp )
    # every GT point is matched by some decoded point within a vertex spacing
    got =np .array ([p ["point"]for p in preds ])
    for c in cp :
        assert np .linalg .norm (got -c ,axis =1 ).min ()<5.0 


def test_decoded_directions_point_outward (plate_with_cps ):
    V ,cp ,cd =plate_with_cps 
    target ,_ ,_ =ct .encode_targets (V ,cp ,cd )
    preds =ct .decode_predictions (V ,target ,heatmap_thresh =0.3 ,min_votes =1 )
    for p in preds :
    # GT direction is +z; decoded direction must agree and insertion is -dir
        assert p ["direction"][2 ]>0.9 
        assert np .allclose (p ["insertion_axis"],-np .asarray (p ["direction"]))


def test_decode_empty_when_nothing_fires ():
    V =_plate ()
    pred =np .zeros ((len (V ),ct .N_CHANNELS ))# all heat 0
    assert ct .decode_predictions (V ,pred ,heatmap_thresh =0.3 )==[]


def test_min_votes_suppresses_lone_vertices ():
    V =_plate ()
    pred =np .zeros ((len (V ),ct .N_CHANNELS ))
    pred [100 ,ct .HEATMAP ]=0.9 # a single hot vertex
    pred [100 ,ct .DIRECTION ]=[0 ,0 ,1.0 ]
    assert len (ct .decode_predictions (V ,pred ,heatmap_thresh =0.3 ,min_votes =1 ))==1 
    assert ct .decode_predictions (V ,pred ,heatmap_thresh =0.3 ,min_votes =2 )==[]


    # ---------------------------------------------------------------------------
    # surface distance / snapping
    # ---------------------------------------------------------------------------

def test_surface_dist_reported_and_snap_moves_to_mesh ():
    V =_plate ()
    pred =np .zeros ((len (V ),ct .N_CHANNELS ))
    pred [200 ,ct .HEATMAP ]=0.9 
    pred [200 ,ct .OFFSET ]=[0 ,0 ,10.0 ]# vote 10mm off the z=0 plate
    pred [200 ,ct .DIRECTION ]=[0 ,0 ,1.0 ]
    d =ct .decode_predictions (V ,pred ,heatmap_thresh =0.3 ,min_votes =1 )
    assert d [0 ]["surface_dist"]>5.0 # floats off-surface
    snapped =ct .decode_predictions (V ,pred ,heatmap_thresh =0.3 ,min_votes =1 ,
    snap_to_surface =True )
    assert abs (snapped [0 ]["point"][2 ])<1e-9 # snapped back onto z=0


def test_cp_surface_distances_flags_floating_point ():
    V =_plate ()
    cp =np .array ([[25.0 ,25 ,0 ],# ten surface
    [50.0 ,50 ,40 ]])# 40mm above
    d =ct .cp_surface_distances (V ,cp )
    assert d [0 ]<5.0 and d [1 ]>30.0 


    # ---------------------------------------------------------------------------
    # sigma single source of truth
    # ---------------------------------------------------------------------------

def test_sigma_for_matches_encode ():
    V =_plate ()
    _ ,_ ,s_enc =ct .encode_targets (V ,np .array ([[10.0 ,10 ,0 ]]),
    np .array ([[0 ,0 ,1.0 ]]))
    assert abs (ct .sigma_for (V )-s_enc )<1e-9 


def test_sigma_scales_with_part_size ():
    small =ct .sigma_for (_plate (size =10.0 ))
    big =ct .sigma_for (_plate (size =1000.0 ))
    assert big >small *50 # ~100x bbox -> ~100x sigma


    # ---------------------------------------------------------------------------
    # dedup of coincident terminals -> one block
    # ---------------------------------------------------------------------------

def test_dedup_collapses_coincident_terminals ():
    V =_plate ()
    cp =np .array ([[10.0 ,10 ,0 ],[10.0 ,10 ,0 ],[10.0 ,10 ,0 ]])# one block
    cd =np .array ([[0 ,0 ,1.0 ]]*3 )
    part =jd .Part ("T",V ,np .zeros ((0 ,3 ),int ),cp ,cd ,["L1","L2","PE"])
    blocks ,bp ,bd =jd .dedup_connection_points (part )
    assert len (bp )==1 and len (blocks [0 ].names )==3 
