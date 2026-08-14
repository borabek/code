# -*- coding: utf-8 -*-
"""T2: REJIME KOSULLU KABUL KURALI -- olu koda donmus a kazanci geri al (ya da gomm).

FINDING (2026-08-04, kod incelemesi): `gate_goreli_esik: True` iken
`wire_gate.decision_mask` `threshold` parametresini TAMAMEN YOK SAYIYOR:

    if GORELI_ESIK:
        return (s >= GORELI_ORAN*max(s)) & (s >= GORELI_TABAN)    # threshold kullanilmaz
    return s >= threshold

Oysa `robot_cp` still rejime according to threshold HESAPLIYOR and `apply(threshold=_thr)` diye
geciriyor (`robot_wire_gate_threshold_highcp` = 0.35). O value URUNDE OLU.
[[pitstop-highcp-gate]] kaydinda rejime kosullu threshold OLCULMUS a kazancti
(high-CP'de candidate recall 0.798 iken dagitilan 0.485 idi). Goreli rule bunu ya
KAPSIYOR ya da sessizce GERI ALDI. Varsayilamaz -- olculur.

SORU: goreli kuralin PARAMETRELERI (ratio, baseline) rejime according to AYRILIRSA kazanc present mi?

REJIM YONLENDIRMESI GT'YE DOKUNMAZ (kritik): measurement tarafinda regime `r["n"]>=8` with,
i.e. GT'deki CP SAYISIYLA belirleniyor -- this a RAPORLAMA agirligidir, DECISION degiskeni
as kullanilamaz (leakage). Burada yonlendirme only ADAY SAYISIYLA is done; this
tamamen boru hattindan gelir, GT gormez. Yonlendirmenin GT rejimiyle uyumu da raporlanir.

PROTOKOL: AYAR = dev + atanmamis (94 part), VERDICT = val (100 part).
DEV single basina (54) rejime bolununce very small kalirdi; VAL'e HIC dokunulmaz.

KILL (onceden yazildi): VAL tespitinde +0.005 VE GA sifiri disliyor.
"""
import io ,json ,os ,sys 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import tezgah2 as T2 
import measure_set ,wire_gate 
from sina_cluster import esle ,f1w 

ORANLAR =(0.35 ,0.40 ,0.45 ,0.50 ,0.55 ,0.60 )
TABANLAR =(0.15 ,0.20 ,0.25 ,0.30 ,0.35 )
AYIRIM =25 # candidate count >= this whereas "very candidate" kolu (GT GORMEZ)


def main ():
    DER ,gate ,ek =T2 .yukle ()
    s3 =json .load (io .open ("results/split3.json",encoding ="utf-8"))
    val ={str (p )for p in s3 ["val"]["parts"]}
    HAZ =[]
    for r in DER :
        h ={"pid":r ["pid"],"geo":r ["geo"],"rj":"very"if r ["n"]>=8 else "dusuk",
        "G":np .asarray (r ["G"],float ),"Gd":np .asarray (r ["Gd"],float ),
        "diag":r ["diag"],"sk":None ,"na":0 }
        if r ["X"]is not None and r .get ("XR")is not None :
            X =np .hstack ([r ["X"],r ["XR"]])
            if X .shape [1 ]*2 ==gate ["n_feat"]:
                h ["sk"]=wire_gate .decision_score (gate ,X )
                h ["X"]=X ;h ["P"]=np .asarray (r ["P"],float )
                h ["Pd"]=np .asarray (r ["Pd"],float );h ["UYE"]=r .get ("UYE")
                h ["na"]=len (h ["sk"])
        h ["arm"]="cok_aday"if h ["na"]>=AYIRIM else "az_aday"
        HAZ .append (h )
        # yonlendirmenin GT rejimiyle uyumu (only RAPOR)
    a =np .array ([h ["arm"]=="cok_aday"for h in HAZ ])
    b =np .array ([h ["rj"]=="very"for h in HAZ ])
    print (f"yonlendirme (candidate>={AYIRIM }): cok_aday {int (a .sum ())} / az_aday {int ((~a ).sum ())}")
    print (f"  GT rejimiyle uyum: {float ((a ==b ).mean ()):.1%} "
    f"(yonlendirme GT GORMEZ, bu yalniz bilgi)")

    def puanla (h ,ratio ,baseline ):
        P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
        if h ["sk"]is not None and len (h ["sk"]):
            s =h ["sk"]
            k =(s >=ratio *max (float (s .max ()),1e-9 ))&(s >=baseline )
            if k .any ():
                P =h ["P"][k ].copy ();Pd =h ["Pd"][k ].copy ()
                c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                c =wire_gate .pose_correct (h ["X"][k ],c )
                c =wire_gate .angle_correct (h ["X"][k ],c )
                if h .get ("UYE"):
                    c =wire_gate .pick_member_direction (h ["X"][k ],c ,h ["UYE"])
                P =np .array ([x ["point"]for x in c ],float )
                Pd =np .array ([x ["direction"]for x in c ],float )
        return (esle (P ,Pd ,h ["G"],h ["Gd"],h ["diag"],0.0 ,180.0 ,True ),
        esle (P ,Pd ,h ["G"],h ["Gd"],h ["diag"],2.0 ,10.0 ,False ,signed =True ))

        # --- tum hucreleri BIR KEZ hesapla (part x ratio x baseline)
    print (f"\n{len (ORANLAR )*len (TABANLAR )} hucre x {len (HAZ )} part hesaplaniyor...",flush =True )
    HUC ={}
    for o in ORANLAR :
        for t in TABANLAR :
            HUC [(o ,t )]=[puanla (h ,o ,t )for h in HAZ ]
        print (f"  ratio {o :.2f} bitti",flush =True )

    ayar =[j for j ,h in enumerate (HAZ )if h ["pid"]not in val ]
    verdict =[j for j ,h in enumerate (HAZ )if h ["pid"]in val ]
    print (f"AYAR {len (ayar )} part | HUKUM(VAL) {len (verdict )} part")

    def f1_of (idx ,sec ):
        """sec: arm -> (ratio,baseline).  det, rob returns."""
        det =[(HAZ [j ]["rj"],)+HUC [sec [HAZ [j ]["arm"]]][j ][0 ]for j in idx ]
        rob =[(HAZ [j ]["rj"],)+HUC [sec [HAZ [j ]["arm"]]][j ][1 ]for j in idx ]
        return f1w (det ),f1w (rob ),det ,rob ,[HAZ [j ]["geo"]for j in idx ]

    MEV ={"cok_aday":(0.50 ,0.25 ),"az_aday":(0.50 ,0.25 )}
    # --- AYAR kumesinde arm basina most iyi (digeri mevcutta sabit tutulur)
    EN =dict (MEV )
    for arm in ("az_aday","cok_aday"):
        alt =[j for j in ayar if HAZ [j ]["arm"]==arm ]
        if len (alt )<10 :
            print (f"  {arm }: yalniz {len (alt )} part -- AYARLANMAZ, mevcut kalir")
            continue 
        en ,eb =None ,None 
        for o in ORANLAR :
            for t in TABANLAR :
                v =f1w ([(HAZ [j ]["rj"],)+HUC [(o ,t )][j ][0 ]for j in alt ])
                if en is None or v >en :
                    en ,eb =v ,(o ,t )
        mv =f1w ([(HAZ [j ]["rj"],)+HUC [MEV [arm ]][j ][0 ]for j in alt ])
        print (f"  {arm } ({len (alt )} part): mevcut {mv :.4f} -> en iyi {eb } {en :.4f} ({en -mv :+.4f})")
        EN [arm ]=eb 

    print (f"\nAYAR kumesinde secilen kural: {EN }")
    fn =lambda rows :f1w ([q for _ ,q in rows ])-f1w ([p for p ,_ in rows ])
    d0 ,r0 ,D0 ,R0 ,gg =f1_of (verdict ,MEV )
    d1 ,r1 ,D1 ,R1 ,_ =f1_of (verdict ,EN )
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (D0 ,D1 )),gg ,fn ,n =3000 )
    _ ,rlo ,rhi =measure_set .grup_bootstrap (list (zip (R0 ,R1 )),gg ,fn ,n =3000 )
    print (f"\n--- HUKUM: VAL ({len (verdict )} part) ---")
    print (f"{'rule':<26}{'tespit':>10}{'robot(FIZ)':>13}")
    print (f"{'mevcut (tek rule)':<26}{d0 :>10.4f}{r0 :>13.4f}")
    print (f"{'rejime kosullu':<26}{d1 :>10.4f}{r1 :>13.4f}")
    print (f"\nVAL tespit farki: {d1 -d0 :+.4f}  GA[{lo :+.4f},{hi :+.4f}] "
    f"{'GERCEK'if (lo >0 or hi <0 )else 'noise'}")
    print (f"VAL robot  farki: {r1 -r0 :+.4f}  GA[{rlo :+.4f},{rhi :+.4f}]")
    ha0 =f1_of (range (len (HAZ )),MEV );ha1 =f1_of (range (len (HAZ )),EN )
    print (f"\n[bilgi] HAVUZLANMIS: tespit {ha0 [0 ]:.4f} -> {ha1 [0 ]:.4f} | "
    f"robot {ha0 [1 ]:.4f} -> {ha1 [1 ]:.4f}   (HUKUM DEGIL: ayar kumesi havuzda)")
    gecti =(d1 -d0 )>=0.005 and lo >0 
    print (f"\nKILL: VAL tespit +0.005 VE GA>0 -> "
    f"{'GECTI'if gecti else 'GECMEDI -- tek rule KALIR'}")
    with io .open ("results/t2_regime_rule.json","w",encoding ="utf-8")as f :
        json .dump ({"ayirim_aday":AYIRIM ,"secilen":{k :list (v )for k ,v in EN .items ()},
        "val_mevcut":{"tespit":d0 ,"robot":r0 },
        "val_yeni":{"tespit":d1 ,"robot":r1 },
        "val_d_tespit":d1 -d0 ,"ga_tespit":[lo ,hi ],
        "val_d_robot":r1 -r0 ,"ga_robot":[rlo ,rhi ],
        "havuz_mevcut":{"tespit":ha0 [0 ],"robot":ha0 [1 ]},
        "havuz_yeni":{"tespit":ha1 [0 ],"robot":ha1 [1 ]},
        "gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/t2_regime_rule.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
