# -*- coding: utf-8 -*-
"""YOGUN PARCADA GATE ESIGI TARAMASI — bugunku sistemde.

RATIONALE: dagitilan dense-part esigi 0.35. 2026-07-29 taramasi 10 very-CP
parcasinda 0.25 -> F1 0.634, 0.35 -> 0.587 demis; i.e. DAHA IYISI
olculmus but dagitilmamis. O tarama ESKI sistemde (old turetme, old
gate, pose head YOKKEN) yapildi -> bugun YENIDEN olculur.

Kiyas ESLI: same parts, only threshold degisir.
"""
import io ,json ,os ,sys 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
import canonical_d7 as K 
import d6_record 
from sina_cluster import match_hungarian 

OLCUT =(("tespit",0.0 ,180.0 ,True ,False ),
("rob",2.0 ,10.0 ,False ,False ),
("rbi",2.0 ,10.0 ,False ,True ))
# CANLI PARAMETRE: dagitilan yolda GORELI_ESIK open, i.e. sabit threshold OLU.
# Gercek karar: skor >= ORAN * parca_maksimumu VE skor >= TABAN.
# Burada ORAN taranir (TABAN sabit 0.20).
ESIKLER =[float (x )for x in 
os .environ .get ("GE_ESIK","0.30,0.40,0.50,0.60").split (",")]
N =int (os .environ .get ("GE_N","18"))


def _b (v ):
    v =np .asarray (v ,float ).reshape (-1 ,3 )
    return v /np .maximum (np .linalg .norm (v ,axis =1 ,keepdims =True ),1e-12 )


def main ():
    import torch ,robot_cp 
    from infer_step_cp import load_any 
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cfg =json .load (io .open ("cp_config.json",encoding ="utf-8"))
    ca =float (cfg .get ("robot_conf_auto",0.5 ))
    mav =int (cfg .get ("robot_min_auto_votes",3 ))
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["robot_vote2_checkpoints"]]

    kay =K .yukle ()
    kay .update ({str (p ):r for p ,r in d6_record .yukle ().items ()
    if str (p )not in kay })
    STEP =K .step_map ()
    s3 =json .load (io .open ("results/split3.json",encoding ="utf-8"))
    val =[str (x )for x in s3 ["val"]["parts"]]
    # YOGUN parts (n_gt >= 8) -- regime esigi
    # REJIM SECILEBILIR: dense (n_gt>=8) ya da SEYREK (<8, katalogun %86'si).
    # Bir ayari only yogunda olcup dagitmak, katalogun cogunlugunu
    # olcmeden degistirmek olurdu.
    # BAGIMSIZ VERIFICATION (2026-08-14). Parametre VAL'de tarandiysa VAL'de
    # olculen kazanc SISIK may be -- this kampanyada tarama two times
    # yaniltti (vc0.35: taramada +0.01, full dagitimda -0.012).
    # `GE_KAYNAK=disari` -> VAL DISI parts (same parametre, BAGIMSIZ ornek).
    _rej =os .environ .get ("GE_REJIM","dense")
    if os .environ .get ("GE_KAYNAK")=="disari":
        lock ={str (x )for x in s3 ["locked"]["parts"]}
        vs =set (val )
        rng2 =np .random .default_rng (7 )
        pool =[p for p in kay if str (p )not in vs and str (p )not in lock 
        and str (p )in STEP and len (kay [p ]["G"])>=1 ]
        pool =[str (x )for x in pool ]
        rng2 .shuffle (pool )
        val =pool 
    if _rej =="sparse":
        dense =[p for p in val if p in kay and p in STEP 
        and 1 <=len (kay [p ]["G"])<8 ][:N ]
    else :
        dense =[p for p in val if p in kay and p in STEP 
        and len (kay [p ]["G"])>=8 ][:N ]
    print (f"{len (dense )} VAL parcasi (regime={_rej }) | GORELI_ORAN taramasi {ESIKLER }\n",
    flush =True )

    top ={e :{a :[0 ,0 ,0 ]for a ,*_ in OLCUT }for e in ESIKLER }
    part ={e :[]for e in ESIKLER }
    for i ,pid in enumerate (dense ,1 ):
        r =kay [pid ]
        G =np .asarray (r ["G"],float ).reshape (-1 ,3 )
        Gd =_b (r ["Gd"]);dg =float (r ["diag"])
        for e in ESIKLER :
            os .environ ["WG_GORELI_ORAN"]=str (e )
            import importlib ,wire_gate as _wg 
            importlib .reload (_wg );robot_cp .wire_gate =_wg 
            cps =robot_cp .extract (models ,STEP [pid ],dev ,ca ,mav )
            P =(np .asarray ([c ["point"]for c in cps ],float ).reshape (-1 ,3 )
            if cps else np .zeros ((0 ,3 )))
            D =_b ([c ["direction"]for c in cps ])if cps else np .zeros ((0 ,3 ))
            sat ={}
            for ad ,tol ,am ,pct ,isr in OLCUT :
                tp ,fp ,fn ,_ =match_hungarian (P ,D ,G ,Gd ,dg ,tol ,am ,pct ,
                signed =isr )
                top [e ][ad ][0 ]+=tp ;top [e ][ad ][1 ]+=fp ;top [e ][ad ][2 ]+=fn 
                sat [ad ]=(tp ,fp ,fn )
            part [e ].append (sat )
        print (f"  {i }/{len (dense )} {pid }",flush =True )
    os .environ .pop ("WG_GORELI_ORAN",None )

    f1 =lambda t :2 *t [0 ]/max (2 *t [0 ]+t [1 ]+t [2 ],1 )# noqa: E731
    print (f"\n{'threshold':>6s} {'tespit':>8s} {'robot':>8s} {'robot-ISR':>10s}")
    for e in ESIKLER :
        yz ="  <- DAGITILAN"if abs (e -0.50 )<1e-9 else ""
        print (f"{e :6.2f} {f1 (top [e ]['tespit']):8.4f} {f1 (top [e ]['rob']):8.4f} "
        f"{f1 (top [e ]['rbi']):10.4f}{yz }")

        # ESLI BOOTSTRAP: dagitilan 0.35'e according to
    def boot (a ,b ,ad ,n =4000 ):
        rng =np .random .default_rng (0 )
        A =np .asarray ([x [ad ]for x in a ],float )
        B =np .asarray ([x [ad ]for x in b ],float )
        d =[]
        for _ in range (n ):
            i =rng .integers (0 ,len (A ),len (A ))
            d .append (f1 (B [i ].sum (0 ))-f1 (A [i ].sum (0 )))
        d =np .asarray (d )
        return d .mean (),np .percentile (d ,2.5 ),np .percentile (d ,97.5 ),(d >0 ).mean ()

    if 0.50 in ESIKLER :
        print (f"\n--- ESLI BOOTSTRAP (0.35'e gore) ---")
        print (f"{'threshold':>6s} {'metrik':>8s} {'fark':>9s} {'%95 GA':>22s} {'poz%':>6s}")
        for e in ESIKLER :
            if abs (e -0.50 )<1e-9 :
                continue 
            for ad in ("tespit","rbi"):
                f ,lo ,hi ,pz =boot (part [0.50 ],part [e ],ad )
                yz =" *"if (lo >0 or hi <0 )else ""
                print (f"{e :6.2f} {ad :>8s} {f :+9.4f} [{lo :+.4f},{hi :+.4f}]{yz :>3s} {100 *pz :5.1f}")
    json .dump ({str (e ):{a :f1 (v )for a ,v in top [e ].items ()}for e in ESIKLER },
    io .open (f"results/gate_esik_{_rej }.json","w",encoding ="utf-8"),indent =1 )
    print ("\n-> results/gate_esik_yogun.json")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
