# -*- coding: utf-8 -*-
"""P4 onkosulu: silindir DISI acikliklari part basina a times cikar, diske yaz."""
import argparse ,glob ,os ,pickle ,sys ,time 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import d6_record 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--cluster",default ="results/d6_sinav_kumesi.json")
    ap .add_argument ("--output",default ="results/_d6_acikliklar.pkl")
    ap .add_argument ("--vardiya",type =int ,default =0 )
    ap .add_argument ("--total",type =int ,default =1 )
    a =ap .parse_args ()
    import brep_opening 
    from corpus_identity import step_kimlik as SK 
    sv =d6_record .exam (a .cluster )
    S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
    yol =a .out_ if a .total_ ==1 else a .out_ .replace (".pkl",f"_{a .vardiya }.pkl")
    ob ={}
    if os .path .exists (yol ):
        with open (yol ,"rb")as f :
            ob =pickle .load (f )
    hedef =[p for p in sv ["pidler"]if p in S and p not in ob ]
    if a .total_ >1 :
        hedef =[p for i ,p in enumerate (hedef )if i %a .total_ ==a .vardiya ]
    print (f"cikarilacak {len (hedef )} part",flush =True )
    t0 =time .time ()
    for i ,p in enumerate (hedef ,1 ):
        try :
            ob [p ]=brep_opening .acikliklar (S [p ])
        except Exception :
            ob [p ]=[]
        if i %25 ==0 :
            print (f"  {i }/{len (hedef )}  {(time .time ()-t0 )/i :.1f}s/part",flush =True )
            with open (yol ,"wb")as f :
                pickle .dump (ob ,f )
    with open (yol ,"wb")as f :
        pickle .dump (ob ,f )
    n =sum (1 for v in ob .values ()if v )
    print (f"BITTI: {len (ob )} part | opening BULUNAN {n } | toplam opening "
    f"{sum (len (v )for v in ob .values ())}")


if __name__ =="__main__":
    main ()
