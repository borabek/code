# -*- coding: utf-8 -*-
"""F: "aile-disi" dedigimiz koruma GERCEKTE ne up to koruyor?

json_dataset.family_key this korpusta part numarasinin KENDISINI donduruyor (1542 part ->
1542 "aile"). Bu a error not, belgelenmis a boundary: part numarasi tamamen rakamsa soymak
that ureticinin TUM urunlerini single gruba cokertirdi. Docstring cozumu de yaziyor: `geometry_key`.

WHY IMPORTANT: kardes varyantlarin geometrisi BIREBIR same (measured: axis farki 0.000 derece,
hizalama residual 0.3mm -- PXC 3211813/14/19, SIE 3RV2011-1AA15/-4AA10). Yani "test aileleri
gate egitiminden cikarildi" derken gercekte only test PARCALARI cikarilmis; ikizleri egitimde
kalmis may be. Bu, tum "sizintisiz" iddialarimizin ne up to korumali oldugunu belirler.

OLCUM: same gate verisi, same protocol, TEK degisken = gruplama anahtari.
  arm A: family_key   (= part no; bugune kadarki "aile-disi")
  arm B: geometry_key (bbox boyutlari yuvarlanmis + kose/face count log2 kovalanmis)
Fark ne up to buyukse, old numbers that up to iyimserdi.
"""
import os ,sys ,json ,glob 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
W ={"dusuk":0.895 ,"cok":0.105 }
NPZ ="results/gate_regrow_data_rt2.npz"
KEYS ="results/_geometry_keys.json"


def main ():
    from sklearn .ensemble import RandomForestClassifier 
    from sklearn .model_selection import GroupKFold 
    from infer_step_cp import step_to_mesh 
    import thesis_remesh 

    d =np .load (NPZ ,allow_pickle =True )
    X ,y ,groups ,fams =d ["X"],d ["y"],d ["groups"],d ["fams"].astype (str )
    pids =np .array ([str (x )for x in d ["pids"]])
    ngt =dict (zip (d ["grp_ids"].tolist (),d ["ngt"].tolist ()))
    uniq =sorted (set (pids ))
    print (f"{len (y )} candidate / {len (uniq )} part",flush =True )

    # geometri anahtari: mesh onbelleginden (varsa), otherwise STEP'ten
    gk ={}
    if os .path .exists (KEYS ):
        gk =json .load (open (KEYS ))
        print (f"  {len (gk )} anahtar onbellekten",flush =True )
    step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
    miss =0 
    for k ,pid in enumerate (uniq ):
        if pid in gk :
            continue 
        try :
            mc =f"results/mesh_cache/{pid }.npz"
            if os .path .exists (mc ):
                m =np .load (mc );V =np .asarray (m ["V"],float );nf =int (m ["F"].shape [0 ])
            else :
                s_ =step .get (pid )
                if not s_ :
                    miss +=1 ;gk [pid ]="yok:"+pid ;continue 
                Vr ,Fr =step_to_mesh (s_ )
                V ,Fm =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
                V =np .asarray (V ,float );nf =int (np .asarray (Fm ).shape [0 ])
            dims =np .round (V .max (0 )-V .min (0 ),1 )
            gk [pid ]="g:%s|v%d|f%d"%(dims .tolist (),int (np .log2 (max (len (V ),1 ))),
            int (np .log2 (max (nf ,1 ))))
        except Exception :
            miss +=1 ;gk [pid ]="yok:"+pid 
        if (k +1 )%200 ==0 :
            print (f"  {k +1 }/{len (uniq )}",flush =True )
    json .dump (gk ,open (KEYS ,"w"))
    G_geo =np .array ([gk .get (p ,"yok:"+p )for p in pids ])
    ng_fam =len (set (fams ));ng_geo =len (set (G_geo ))
    print (f"\ngruplama: family_key {ng_fam } grup | geometry_key {ng_geo } grup "
    f"({miss } part okunamadi)")
    import collections 
    cc =collections .Counter (G_geo )
    multi =sum (1 for v in cc .values ()if v >1 )
    print (f"  geometry_key'in BIRLESTIRDIGI grup sayisi: {multi } "
    f"(yani bu kadar grupta >1 candidate havuzu var)")
    # kac PARCA a baskasiyla same geometri grubunda
    ppg =collections .defaultdict (set )
    for p ,g in zip (pids ,G_geo ):
        ppg [g ].add (p )
    shared =sum (len (v )for v in ppg .values ()if len (v )>1 )
    print (f"  IKIZI OLAN part sayisi: {shared }/{len (uniq )} = %{100 *shared /len (uniq ):.1f}"
    f"   <- eski bolmede bunlarin ikizi egitimde kalabiliyordu")

    reg =np .array ([("cok"if int (ngt .get (int (g ),0 ))>=8 else "dusuk")for g in groups ])
    tot ={"dusuk":0 ,"cok":0 }
    for g in {int (g )for g in groups }:
        n =int (ngt .get (g ,0 ))
        if n >0 :tot ["cok"if n >=8 else "dusuk"]+=n 

    def run (gkey ,lab ):
        o =np .zeros (len (y ))
        for tr ,te in GroupKFold (n_splits =5 ).split (X ,y ,gkey ):
            o [te ]=RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
            random_state =0 ).fit (X [tr ],y [tr ]).predict_proba (X [te ])[:,1 ]
        out ={}
        for k in ("dusuk","cok"):
            b =(0.0 ,0.40 )
            for t in np .arange (0.20 ,0.71 ,0.05 ):
                m =reg ==k ;s_ =(o >=t )&m 
                tp =int ((y [s_ ]==1 ).sum ());fp =int (s_ .sum ())-tp 
                p =tp /max (tp +fp ,1 );r =tp /max (tot [k ],1 )
                f =2 *p *r /max (p +r ,1e-9 )
                if f >b [0 ]:b =(f ,float (t ))
            out [k ]=b 
        w =sum (W [k ]*out [k ][0 ]for k in W )
        print (f"{lab :<34}{w :>9.4f}{out ['dusuk'][0 ]:>9.4f}{out ['cok'][0 ]:>9.4f}"
        f"{str ((out ['dusuk'][1 ],out ['cok'][1 ])):>16}",flush =True )
        return w 

    print (f"\n{'gruplama':<34}{'CP-F1':>9}{'dusuk':>9}{'cok':>9}{'esikler':>16}")
    a =run (fams ,"family_key (= part no, ESKI)")
    b =run (G_geo ,"geometry_key (GERCEK koruma)")
    print (f"\nSIZINTININ BEDELI: {b -a :+.4f}")
    print ("  negatifse eski sayilar o kadar IYIMSERDI ve dogru sayi geometri gruplamasindaki.")
    json .dump ({"family_key":a ,"geometry_key":b ,"fark":b -a ,
    "n_grup_family":ng_fam ,"n_grup_geometry":ng_geo ,
    "ikizi_olan_parca_orani":shared /max (len (uniq ),1 )},
    open ("results/f_geometri_bolme.json","w"),indent =1 )
    print ("receipt -> results/f_geometri_bolme.json")


if __name__ =="__main__":
    main ()
