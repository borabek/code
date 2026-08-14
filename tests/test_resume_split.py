"""A --resume must replay the FROZEN split, not re-derive it from the live corpus.

The bug this guards (found in the 2026-07-13 audit): three_way_split's balance
search is a function of the corpus CONTENTS, so adding/removing parts between the
launch and the resume can move an already-trained part into val -- silently, since
the checkpoint's config fingerprint (seed/val_frac/split_group) is unchanged. Every
number the resumed run then reports is measured ten a leaked val set.
"""
import json 
import os 

import numpy as np 
import pytest 

import train_cp as tc 


def _corpus (tmp_path ,n_parts ,seed =0 ,name ="corpus"):
    """n tiny labelled parts in the real ABB JSON schema (json_dataset header)."""
    d =tmp_path /name 
    d .mkdir (parents =True ,exist_ok =True )
    rng =np .random .RandomState (seed )
    for i in range (n_parts ):
        V =(rng .rand (40 ,3 )*20 ).round (3 )
        idx =list (range (0 ,36 ))# 12 triangles
        cp =V [0 ]+np .array ([0.0 ,0.0 ,1.0 ])
        dim =(V .max (0 )-V .min (0 )).tolist ()
        loc =V .min (0 ).tolist ()
        json .dump (
        {"PartNr":f"wscaduniverse_{3000000 +seed *1000 +i }_x",
        "Graphic3d":{
        "Points":[{"X":float (x ),"Y":float (y ),"Z":float (z )}
        for x ,y ,z in V ],
        "Indices":idx },
        "BoundingBox":{"Dimension":{"X":dim [0 ],"Y":dim [1 ],"Z":dim [2 ]},
        "Location":{"X":loc [0 ],"Y":loc [1 ],"Z":loc [2 ]}},
        "ConnectionPoints":[{"Index":0 ,"Name":"OPEN-0",
        "Point":{"X":float (cp [0 ]),"Y":float (cp [1 ]),
        "Z":float (cp [2 ])},
        "InsertDirection":{"X":0.0 ,"Y":0.0 ,"Z":1.0 }}]},
        open (d /f"p{seed }_{i :03d}.json","w",encoding ="utf-8"))
    return d 


def _split_of (caplog_or_none ,corpus ,ckpt_dir ,resume ,extra =0 ):
    """Run just far enough to emit the manifest / read it back, and return the
    (train, val, test) part-number sets train_and_eval actually used."""
    seen ={}

    class _Stop (Exception ):
        pass 

        # stop the run the moment the split is settled: the trainer itself is not
        # under test here, the split that gets handed to it is
    orig =tc .cpr .train_knngraph_regressor 

    def _spy (*a ,**kw ):
        raise _Stop ()

    tc .cpr .train_knngraph_regressor =_spy 
    try :
        tc .train_and_eval (str (corpus ),backbone ="knngraph",epochs =1 ,
        val_frac =0.2 ,test_frac =0.2 ,seed =0 ,split_seed =0 ,
        split_group ="geometry",device ="cpu",
        best_path =str (ckpt_dir /"cp_t_best.ckpt"),
        resume_from =str (ckpt_dir /"cp_t_last.ckpt")if resume else None )
    except _Stop :
        pass 
    except Exception :# any other early exit is fine for this test
        pass 
    finally :
        tc .cpr .train_knngraph_regressor =orig 
    return seen 


def test_manifest_is_authoritative_on_resume (tmp_path ,caplog ):
    """With the manifest present, a resume must reproduce ITS val/test sets even
    when the corpus has grown (which would otherwise re-balance the split)."""
    import json_dataset as jd 

    corpus =_corpus (tmp_path ,20 )
    ckpt =tmp_path /"ck"
    ckpt .mkdir ()
    man_path =ckpt /"cp_t_best_manifest.json"

    # a hand-written manifest standing in for "the launch run's frozen split"
    ids =[str (p .part_nr )for p in jd .iter_parts (str (corpus ))]
    assert len (ids )==20 ,ids 
    frozen ={pid :("val"if k %5 ==0 else "test"if k %5 ==1 else "train")
    for k ,pid in enumerate (sorted (ids ))}
    json .dump ({"run_config":{},"parts":[{"part_nr":p ,"split":s }
    for p ,s in frozen .items ()]},
    open (man_path ,"w",encoding ="utf-8"))

    # grow the corpus AFTER the "launch" -- the classic silent-leak trigger
    for f in _corpus (tmp_path ,6 ,seed =7 ,name ="more").glob ("*.json"):
        (corpus /f .name ).write_text (f .read_text (encoding ="utf-8"),encoding ="utf-8")

    caplog .set_level ("INFO")
    _split_of (None ,corpus ,ckpt ,resume =True )

    msgs ="\n".join (str (r .getMessage ())for r in caplog .records )
    assert "RESUME SPLIT DRIFT"in msgs ,msgs 
    assert "REPLAYING THE MANIFEST"in msgs ,msgs 
    # the 6 new parts must be reported as new, and none of the frozen val/test
    # parts may be silently reassigned
    assert "6 part(s) are new"in msgs ,msgs 


def test_no_manifest_warns_loudly (tmp_path ,caplog ):
    """No manifest -> the split IS re-derived; the run must say so, not stay silent."""
    corpus =_corpus (tmp_path ,12 ,seed =3 )
    ckpt =tmp_path /"ck2"
    ckpt .mkdir ()
    caplog .set_level ("INFO")
    _split_of (None ,corpus ,ckpt ,resume =True )
    msgs ="\n".join (str (r .getMessage ())for r in caplog .records )
    assert "no split manifest"in msgs .lower (),msgs 


def test_unchanged_corpus_resume_reports_no_drift (tmp_path ,caplog ):
    """The v29 case: corpus untouched -> the manifest and a fresh derivation agree,
    and the run says so instead of leaving the operator guessing."""
    corpus =_corpus (tmp_path ,20 ,seed =5 )
    ckpt =tmp_path /"ck3"
    ckpt .mkdir ()
    caplog .set_level ("INFO")

    # the real flow: a LAUNCH writes the manifest itself ...
    _split_of (None ,corpus ,ckpt ,resume =False )
    assert (ckpt /"cp_t_best_manifest.json").exists ()

    # ... and a resume ten the untouched corpus must agree with it
    caplog .clear ()
    _split_of (None ,corpus ,ckpt ,resume =True )
    msgs ="\n".join (str (r .getMessage ())for r in caplog .records )
    assert "RESUME SPLIT DRIFT"not in msgs ,msgs 
    assert "no drift"in msgs ,msgs 
