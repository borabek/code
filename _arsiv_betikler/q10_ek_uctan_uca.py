# -*- coding: utf-8 -*-
"""Q10: EK blok (iceri channel derinligi + renk) 18-sutunlu gate'in USTUNE a sey katiyor mu?

EK bloktaki 4 feature and neden buradalar:
  ic_derinlik   -- `bos_derinlik` yazilirken isin DISARI gonderilmisti; olcunce onun "disarisi
                   gercekten empty mu" oldugu anlasildi (TP %75.1 never carpmiyor, FP %35.6).
                   Kanal DERINLIGI never olculmemisti. Tel girisi ~5-15mm'de metalde biter.
  c_govde / kanalda_metal / metal_mesafe -- RENK. Olculdu (q8, 62 part): adayin on
                   durdugu silindir neredeyse never metal DEGIL (c_metal AUC 0.496 = OLU) but
                   metale YAKINLIK ayiriyor (metal_mesafe ters AUC 0.639; TP 42.5mm / FP 62.0mm).
                   Aday duzeyinde 18 -> 21: F1 +0.0097 (q9).

Renk parcalarin ~%65'inde cozuluyor; kalanda sutunlar NOTR. Yani kazanc seyreltilmis gelir.

RULE 2: two gate AYNI candidate havuzunda karsilastirilir -- candidates BIR KEZ turetilir, 22 column a
times is computed, 18-sutunlu arm same matrisin first 18 sutununu kullanir. Tek degisken: feature kumesi.

KILL (onceden yazili): DEV'de uctan uca tespit F1 artmazsa dagitilmaz. Artarsa VAL'de SINANIR;
VAL'de kaybederse yine dagitilmaz. Bu, this gece L2'yi olduren and fiziksel ozellikleri gecirer rule.
"""
import os ,sys ,json ,pickle 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_EK_FEATS"]="1"# test tarafinda 22 column uretilsin
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

# SUTUN SIRASI: [13 baseline][5 fiziksel][ic_derinlik, c_govde, kanalda_metal, metal_mesafe]
# Bu order thanks to "19 column" = 18 + YALNIZ ic_derinlik demek -> RENGIN katkisi BEDAVA ayrisir.
# Renk calisma aninda part basina a OCC STEP okumasi (0.24s) maliyeti getiriyor; katkisi
# otherwise that maliyet odenmemeli. Ayrica measured: kanalda_metal adaylarin %99'unda NOTR.
KAYNAK ={"fiz18":("results/gate_regrow_data_fiz.npz",18 ),
"ic19":("results/gate_regrow_data_ek.npz",19 ),
"ek22":("results/gate_regrow_data_ek.npz",22 )}


def main ():
    import cp_openings ,robot_cp ,wire_gate 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from sklearn .ensemble import RandomForestClassifier 
    from sina_cluster import esle ,f1w ,pr 

    assert len (wire_gate .FEAT_NAMES )==22 ,f"22 bekleniyordu, {len (wire_gate .FEAT_NAMES )}"
    cluster =(sys .argv [1 ]if len (sys .argv )>1 else "dev").lower ()
    cf =f"results/_probs_{cluster }.pkl"
    if not os .path .exists (cf )and cluster =="dev":
        cf ="results/_h_probs.pkl"
    cache =pickle .load (open (cf ,"rb"))
    for r in cache :
        r .setdefault ("pid",os .path .basename (r ["stp"]).split ("_")[1 ])
    cfg =json .load (open ("cp_config.json",encoding ="utf-8"))
    pp =cfg ["prediction_postproc"]
    MINV =int (pp ["min_vertices"]);VC =float (pp ["vertex_confidence_mask"]);CL =float (pp ["cluster_mm"])
    THR ={"dusuk":float (cfg ["robot_wire_gate_threshold"]),
    "cok":float (cfg ["robot_wire_gate_threshold_highcp"])}

    gk =json .load (open ("results/_strict_geometry_keys.json"))
    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in cache }
    CLF ={}
    for tag ,(f ,ncol )in KAYNAK .items ():
        d =np .load (f ,allow_pickle =True )
        pids =np .array ([str (x )for x in d ["pids"]])
        keep =~np .isin (np .array ([gk .get (p ,"yok:"+p )for p in pids ]),list (tg ))
        Xt =d ["X"][keep ][:,:ncol ]
        assert Xt .shape [1 ]==ncol ,f"{tag }: {d ['X'].shape [1 ]} sutunluk veri, {ncol } istendi"
        CLF [tag ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (Xt ,d ["y"][keep ])
        print (f"gate {tag }: {int (keep .sum ())} candidate x {ncol } sutun",flush =True )

    DER =[]
    for r in cache :
        V =np .ascontiguousarray (r ["V"],np .float64 )
        F =np .ascontiguousarray (r ["F"],np .int64 )
        plist =[np .asarray (pb ,np .float64 )for pb in r ["pbs"]]
        mk =lambda pr_ ,**kw :cp_openings .connection_points (
        V ,F ,pr_ .argmax (-1 ),min_v =MINV ,classes =(CE ,CT ),dedupe_mm =10.0 ,
        probs =pr_ ,vertex_conf =VC ,ct_depth_min_mm =1.0 ,cluster_mm =CL ,
        step_path =r ["stp"],**kw )
        merge =lambda L :robot_cp ._vote2 (L ,min_votes =1 )# URUN birlestiricisi
        base =merge ([mk (pb )for pb in plist ])
        is_hi =robot_cp ._highcp_router (r ["stp"],V ,len (base ))
        cps =merge ([mk (pb ,conn_promote =0.25 )for pb in plist ])if is_hi else base 
        Xc =(wire_gate .feats_for (V ,F ,sum (plist )/len (plist ),cps ,CE ,CT ,
        step_path =r ["stp"])if cps else None )
        DER .append (dict (X =Xc ,is_hi =is_hi ,
        P =np .array ([c ["point"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        Pd =np .array ([c ["direction"]for c in cps ],float )if cps else np .zeros ((0 ,3 )),
        G =r ["G"],Gd =r ["Gd"],n =r ["n"],diag =r ["diag"]))
    print (f"{len (DER )} part | {sum (len (r ['P'])for r in DER )} candidate (AYNI pool)",flush =True )
    # DENETIM DERSI (silent notrlesme): ozellikler error halinde NOTR returns. Notr donmek correct,
    # SESSIZ olmasi wrong. Kac adayda hangi path notre dustu -- prediction not, sayim.
    _fb =wire_gate .fallback_ozet ()
    _nad =sum (len (r ['P'])for r in DER )
    if _fb :
        print ('  NOTR-DONUS sayimi (candidate basina):')
        for k ,v in sorted (_fb .items (),key =lambda x :-x [1 ]):
            print (f'    {k :<28}{v :>6}  (%{100.0 *v /max (_nad ,1 ):.1f})')
    else :
        print ('  NOTR-DONUS absent: tum ozellikler each adayda hesaplandi')

    def kos (tag ):
        clf =CLF [tag ];nc =KAYNAK [tag ][1 ]
        det ,rob =[],[]
        for r in DER :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None :
                sc =clf .predict_proba (r ["X"][:,:nc ])[:,1 ]
                m =sc >=(THR ["cok"]if r ["is_hi"]else THR ["dusuk"])
                if m .any ():
                    P =r ["P"][m ];Pd =r ["Pd"][m ]
            k ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((k ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    print (f"\n{'gate':<26}{'tespit':>9}{'ROBOT':>9}{'kesin':>9}{'recall':>9}")
    R ={}
    for tag ,lab in (("fiz18","18 sutun (dagitilan)"),
    ("ic19","19 sutun (+ic_derinlik)"),
    ("ek22","22 sutun (+depth+renk)")):
        det ,rob =kos (tag )
        R [tag ]=(det ,rob )
        p_ ,r_ =pr (det )
        print (f"{lab :<26}{f1w (det ):>9.4f}{f1w (rob ):>9.4f}{p_ :>9.3f}{r_ :>9.3f}",flush =True )

    pickle .dump (R ,open (f"results/q10_parca_{cluster }.pkl","wb"))
    rng =np .random .RandomState (0 )
    n =len (R ["fiz18"][0 ]);IX =[rng .randint (0 ,n ,n )for _ in range (2000 )]
    print ("\nESLI BOOTSTRAP (ek22 - fiz18):")
    out ={}
    for mi ,mn in ((0 ,"tespit"),(1 ,"robot")):
        a ,b =R ["ek22"][mi ],R ["fiz18"][mi ]
        ds =np .array ([f1w ([a [i ]for i in ix ])-f1w ([b [i ]for i in ix ])for ix in IX ])
        lo_ ,hi_ =np .percentile (ds ,[2.5 ,97.5 ])
        print (f"  {mn :<8}{ds .mean ():>+9.4f}  [{lo_ :+.4f}, {hi_ :+.4f}]  "
        f"{'BELIRGIN'if lo_ >0 or hi_ <0 else 'noise'}",flush =True )
        out [mn ]=[float (ds .mean ()),float (lo_ ),float (hi_ )]

    dd =f1w (R ["ek22"][0 ])-f1w (R ["fiz18"][0 ])
    print (f"\nKILL: uctan uca tespit artmazsa DAGITILMAZ -> {dd :+.4f} => "
    f"{'DEVAM'if dd >0 else 'DAGITMA'}")
    json .dump ({"cluster":cluster ,
    "tespit":{k :float (f1w (v [0 ]))for k ,v in R .items ()},
    "robot":{k :float (f1w (v [1 ]))for k ,v in R .items ()},
    "precision":{k :float (pr (v [0 ])[0 ])for k ,v in R .items ()},
    "recall":{k :float (pr (v [0 ])[1 ])for k ,v in R .items ()},
    "bootstrap":out },open (f"results/q10_ek_{cluster }.json","w"),indent =1 )
    print (f"receipt -> results/q10_ek_{cluster }.json")


if __name__ =="__main__":
    main ()
