"""Label the parts still missing from the corpus, one subprocess per part.

Why not just point step_openings at a directory: a single pathological STEP can
grind inside gmsh for 30+ minutes at 8GB RSS (measured 2026-07-13 ten three
2006-16xx parts), and because a shard walks its files sequentially, that one part
freezes the whole shard -- 13 shards finished 61 parts each while three sat ten
one. One part per subprocess with a hard wall-clock timeout means a bad mesh
costs `--timeout` seconds instead of the run.

Paths are handed to the child as argv (no shell), so Windows backslashes survive:
piping a glob list through xargs ate them as escapes and silently produced zero
parts with no error at all.
"""
import argparse 
import glob 
import os 
import subprocess 
import sys 
from concurrent .futures import ProcessPoolExecutor ,as_completed 

PY =os .path .join (".venv","Scripts","python.exe")


def label_one (path ,out_dir ,timeout ,deflection ,merge_tol ):
    cmd =[PY ,"step_openings.py",path ,"--label-corpus",out_dir ,
    "--auto","--deflection",str (deflection ),"--check-dirs",
    "--merge-tol",str (merge_tol )]
    part =os .path .splitext (os .path .basename (path ))[0 ]
    try :
        r =subprocess .run (cmd ,capture_output =True ,text =True ,timeout =timeout )
    except subprocess .TimeoutExpired :
        return part ,"TIMEOUT"
    if os .path .exists (os .path .join (out_dir ,part +".json")):
        return part ,"OK"
    tail =(r .stderr or r .stdout or "").strip ().splitlines ()
    return part ,"EMPTY: "+(tail [-1 ][:80 ]if tail else "no output")


def main ():
    ap =argparse .ArgumentParser (description =__doc__ )
    ap .add_argument ("--pool",default ="all_wscad_stp",help ="dir of .stp files")
    ap .add_argument ("--out",required =True ,help ="corpus dir to write into")
    ap .add_argument ("--done-dirs",nargs ="*",default =[],
    help ="corpus dirs whose parts are already labelled")
    ap .add_argument ("--skip-file",help ="newline-separated part_nrs to quarantine")
    ap .add_argument ("--timeout",type =int ,default =150 ,help ="seconds per part")
    ap .add_argument ("--jobs",type =int ,default =16 )
    ap .add_argument ("--deflection",type =float ,default =0.5 )
    ap .add_argument ("--merge-tol",type =float ,default =4.0 )
    a =ap .parse_args ()

    done =set ()
    for d in a .done_dirs :
        done |={os .path .basename (p )[:-5 ]for p in glob .glob (os .path .join (d ,"*.json"))}
    skip =set ()
    if a .skip_file and os .path .exists (a .skip_file ):
        skip ={l .strip ()for l in open (a .skip_file )if l .strip ()}

    todo =[p for p in sorted (glob .glob (os .path .join (a .pool ,"*.stp")))
    if os .path .splitext (os .path .basename (p ))[0 ]not in done |skip ]
    os .makedirs (a .out ,exist_ok =True )
    print (f"labelled already: {len (done )} | quarantined: {len (skip )} | TO LABEL: {len (todo )}",
    flush =True )
    if not todo :
        return 0 

    ok ,timed_out ,empty =0 ,[],[]
    with ProcessPoolExecutor (max_workers =a .jobs )as ex :
        futs ={ex .submit (label_one ,p ,a .out ,a .timeout ,a .deflection ,a .merge_tol ):p 
        for p in todo }
        for n ,f in enumerate (as_completed (futs ),1 ):
            part ,status =f .result ()
            if status =="OK":
                ok +=1 
            elif status =="TIMEOUT":
                timed_out .append (part )
                print (f"  TIMEOUT ({a .timeout }s): {part }",flush =True )
            else :
                empty .append (part )
                print (f"  no CPs: {part } -- {status }",flush =True )
            if n %25 ==0 or n ==len (todo ):
                print (f"  [{n }/{len (todo )}] ok={ok } timeout={len (timed_out )} "
                f"empty={len (empty )}",flush =True )

    print (f"\nDONE: {ok } labelled, {len (timed_out )} timed out, {len (empty )} produced no CPs")
    if timed_out :
        with open ("_timed_out_parts.txt","w",newline ="\n")as fh :
            fh .write ("\n".join (timed_out )+"\n")
        print ("pathological parts -> _timed_out_parts.txt")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
