# -*- coding: utf-8 -*-
"""B-rep onbellekleri (silindir + opening) EGITIM korpusu for -- yeniden baslatilabilir.

`p3c_axis_selector.silindir_onbellek` same isi does but basarisiz parcayi
sessizce empty list yaziyor; here BASARISIZLIK SAYILIR and makbuza yazilir
(empty pool with cikarilamayan part AYNI SEY DEGIL).

Egitim and exam havuzlari AYNI bicimde kurulmali: D7 tarafinda hem silindir hem
opening present, that yuzden training tarafinda da IKISI de uretilir.
"""
import glob ,json ,os ,pickle ,sys ,time 
sys .path .insert (0 ,".")
os .environ .setdefault ("BA_ALLOW_SEEN","1")
import brep_snap ,brep_opening 
from corpus_identity import step_kimlik as SK 

KUME =os .environ .get ("BREP_KUME","results/brep_egitim_kumesi.json")
ON =os .environ .get ("BREP_ON","_brepegit")
ISLER =[("silindir",f"results/{ON }_silindirler.pkl",
lambda p :brep_snap .exact_cylinders (p )),
("opening",f"results/{ON }_acikliklar.pkl",
lambda p :brep_opening .acikliklar (p ))]
S ={SK (s ):s for s in glob .glob ("all_wscad_stp/*.stp")}
pidler =[str (p )for p in json .load (open (KUME ))["pidler"]]
print (f"cluster {len (pidler )} | STEP eslesen {sum (1 for p in pidler if p in S )}",flush =True )

for ad ,yol ,fn in ISLER :
    ob ,error ={},{}
    if os .path .exists (yol ):
        with open (yol ,"rb")as f :
            ob =pickle .load (f )
    eksik =[p for p in pidler if p not in ob and p in S ]
    print (f"\n{ad }: onbellekte {len (ob )} | cikarilacak {len (eksik )}",flush =True )
    t0 =time .time ()
    for i ,p in enumerate (eksik ,1 ):
        try :
            ob [p ]=fn (S [p ])
        except Exception as e :# yutulmaz: SAYILIR and raporlanir
            ob [p ]=[]
            error [p ]=f"{type (e ).__name__ }: {e }"[:200 ]
        if i %100 ==0 or i ==len (eksik ):
            with open (yol ,"wb")as f :
                pickle .dump (ob ,f )
            print (f"  {ad } {i }/{len (eksik )}  {(time .time ()-t0 )/i :.2f}s/part  "
            f"error {len (error )}",flush =True )
    with open (yol ,"wb")as f :
        pickle .dump (ob ,f )
    empty_ =sum (1 for p in ob if not ob [p ])
    print (f"{ad } BITTI: {len (ob )} part | BOS {empty_ } | HATA {len (error )}",flush =True )
    json .dump ({"n":len (ob ),"bos":empty_ ,"hata_sayisi":len (error ),
    "hatalar":dict (list (error .items ())[:40 ])},
    open (f"results/brep_onbellek_{ad }.json","w"),indent =1 )
