# -*- coding: utf-8 -*-
"""R2: DIK-EKSEN KAFASI -- robotun kalan angle hatasinin %56'sini hedefler.

R1 BULGUSU: tespit edilen 873 ciftin %13.3'unde angle hatasi present; bunlarin 65'i (=%7.4)
>=60 DERECE and 45'i full 80-100 derece bandinda. Yani small a deviation not, SISTEMATIK
YANLIS EKSEN: manufacturer yonu, adayin silindir eksenine DIK.
Ve that sinif ozniteliklerden GRUP-CAPRAZ AUC 0.930 with prediction edilebiliyor.

MEVCUT `angle_correct` BUNLARI YAKALAMIYOR: R1 artigi already angle_correct UYGULANDIKTAN SONRA
measured. Sebebi muhtemelen esiginin muhafazakar olmasi (correct yonleri bozmamak for) --
that arm +0.0126 kazanmisti and "herkese uygulama" tuzagindan bilerek kaciniyordu.

R2 = AYRI and DAR a kafa:
    1. SINIFLANDIRICI: this adayin real yonu eksenine >=60 derece mi? (olculmus AUC 0.930)
    2. REGRESOR: only that sinifta, GT yonunun yerel cercevedeki dik bilesenlerini (a,b) prediction et
       -> direction = sqrt(1-a^2-b^2)*d + a*u + b*v
    3. Yalniz siniflandirici GUVENLI whereas uygula (threshold taranir, DURUST secimle).

VERI: results/pose_veri.npz -- Y[:,2:4] = (g.u, g.v), i.e. GT yonunun dik bilesenleri.
Olcum kumesinin geometri gruplari egitimden CIKARILIR.

KILL (onceden yazildi): havuzlanmis robot +0.01 VE grup bootstrap GA'si sifiri dislamali.
Tespit DUSMEMELI (yapisal as dusemez but olculur). Uretici-disi robot da dusmemeli.
"""
import io 
import json 

import numpy as np 

import gate_bench as T 

VERI ="results/pose_veri.npz"
DIK_ESIK =60.0 


def main ():
    import measure_set 
    import wire_gate 
    from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    tg ={r ["geo"]for r in D ["DER"]}

    pv =np .load (VERI ,allow_pickle =True )
    Xp =np .asarray (pv ["X"],float );Yp =np .asarray (pv ["Y"],float )
    gp =np .array ([str (x )for x in pv ["geo"]])
    egit =~np .isin (gp ,list (tg ))
    perp =np .linalg .norm (Yp [:,2 :],axis =1 )
    aci =np .degrees (np .arcsin (np .clip (perp ,0 ,1 )))
    dik =(aci >=DIK_ESIK ).astype (int )
    print (f"\npose verisi {len (Yp )} satir | egitimde {int (egit .sum ())} "
    f"({len (np .unique (gp [egit ]))} grup)")
    print (f"  DIK sinif (>= {DIK_ESIK :.0f} derece): {int (dik [egit ].sum ())} "
    f"({dik [egit ].mean ():.2%})")
    if dik [egit ].sum ()<40 :
        print ("KILL: egitimde DIK ornek sayisi cok az -> R2 kapanir")
        return 

    snf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
    random_state =0 ).fit (Xp [egit ],dik [egit ])
    m_ =egit &(dik ==1 )
    reg =RandomForestRegressor (n_estimators =400 ,min_samples_leaf =2 ,n_jobs =-1 ,
    random_state =0 ).fit (Xp [m_ ],Yp [m_ ,2 :4 ])
    print (f"  siniflandirici + regresor egitildi (regresor {int (m_ .sum ())} ornek)")

    def dik_duzelt (X ,cps ,threshold ):
        """Yalniz DIK denen adaylarda yonu yeniden kur."""
        if not len (cps ):
            return cps 
        X =np .asarray (X ,float )
        p =snf .predict_proba (X )[:,1 ]
        se =p >=threshold 
        if not se .any ():
            return cps 
        ab =reg .predict (X [se ])
        j =0 
        for i ,c in enumerate (cps ):
            if not se [i ]:
                continue 
            a ,b =float (ab [j ,0 ]),float (ab [j ,1 ]);j +=1 
            n2 =a *a +b *b 
            if n2 >1.0 :# dik bilesen unit kureyi asamaz
                a ,b =a /np .sqrt (n2 ),b /np .sqrt (n2 );n2 =1.0 
            d ,u ,v =wire_gate ._yerel_cerceve (c ["direction"])
            g =np .sqrt (max (1.0 -n2 ,0.0 ))*d +a *u +b *v 
            nn =np .linalg .norm (g )
            if nn >1e-6 :
                c ["direction"]=g /nn 
        return cps 

        # --- measurement: URUNUN zinciri + R2, threshold DURUST secilir (yarida sec / diger yaride olc)
    from sina_cluster import esle 

    def puanla (threshold ,alt ,gate ):
        det ,rob =[],[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            if r ["X"]is not None and r .get ("XR")is not None :
                Xr =np .hstack ([r ["X"],r ["XR"]])
                k =wire_gate .decision_mask (wire_gate .decision_score (gate ,Xr ))
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                    c =wire_gate .pose_correct (Xr [k ],c )
                    if cfg .get ("robot_aci_secici"):
                        c =wire_gate .angle_correct (Xr [k ],c )
                    if cfg .get ("robot_uye_secici")and r .get ("UYE"):
                        c =wire_gate .pick_member_direction (Xr [k ],c ,r ["UYE"])
                    if threshold is not None :
                        c =dik_duzelt (Xr [k ],c ,threshold )
                    P =np .array ([x ["point"]for x in c ],float )
                    Pd =np .array ([x ["direction"]for x in c ],float )
            rj ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
        return det ,rob 

    from sklearn .ensemble import RandomForestClassifier as RFC 
    X ,y ,pid ,mfg ,keep =D ["X"],D ["y"],D ["pid"],D ["mfg"],D ["keep"]
    Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
    for u_ in np .unique (pid ):
        i =np .where (pid ==u_ )[0 ]
        Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])

    def gate_kur (kp ):
        return {"clf":RFC (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (Z [kp ],y [kp ]),
        "n_feat":Z .shape [1 ],"donusum":D ["donusum"]}

    g_hav =gate_kur (keep )
    DER =D ["DER"]
    ESIKLER =[0.30 ,0.40 ,0.50 ,0.60 ,0.70 ,0.80 ]
    d0 ,r0 =puanla (None ,DER ,g_hav )
    print (f"\nTABAN  tespit {T .f1w (d0 ):.4f} | robot {T .f1w (r0 ):.4f}")
    print (f"{'threshold':<8}{'tespit':>10}{'robot':>10}{'d(robot)':>11}")
    for e in ESIKLER :
        d1 ,r1 =puanla (e ,DER ,g_hav )
        print (f"{e :<8.2f}{T .f1w (d1 ):>10.4f}{T .f1w (r1 ):>10.4f}{T .f1w (r1 )-T .f1w (r0 ):>+11.4f}")

        # --- DURUST threshold: yarida sec, diger yaride olc
    gruplar =sorted ({r ["geo"]for r in DER })
    rng =np .random .default_rng (0 );kar =list (gruplar );rng .shuffle (kar )
    A =set (kar [:len (kar )//2 ])
    altA =[r for r in DER if r ["geo"]in A ];altB =[r for r in DER if r ["geo"]not in A ]
    det_c ,rob_c ,det_t ,rob_t =[],[],[],[]
    for sec ,olc in ((altA ,altB ),(altB ,altA )):
        en ,ea =-1 ,ESIKLER [0 ]
        for e in ESIKLER :
            _ ,rr =puanla (e ,sec ,g_hav )
            if T .f1w (rr )>en :
                en ,ea =T .f1w (rr ),e 
        d1 ,r1 =puanla (ea ,olc ,g_hav );d2 ,r2 =puanla (None ,olc ,g_hav )
        det_c +=d1 ;rob_c +=r1 ;det_t +=d2 ;rob_t +=r2 
        print (f"  yarida secilen threshold {ea :.2f} -> diger yaride measured ({len (olc )} part)")
    print (f"\nDURUST (capraz threshold): robot {T .f1w (rob_t ):.4f} -> {T .f1w (rob_c ):.4f} "
    f"({T .f1w (rob_c )-T .f1w (rob_t ):+.4f}) | tespit {T .f1w (det_t ):.4f} -> {T .f1w (det_c ):.4f}")
    fn =lambda rows :T .f1w ([q for _ ,q in rows ])-T .f1w ([p for p ,_ in rows ])
    gg =[r ["geo"]for r in altB ]+[r ["geo"]for r in altA ]
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (rob_t ,rob_c )),gg ,fn ,n =3000 )

    # --- manufacturer-disi bedeli
    ud ={}
    for k ,mad in D ["kod"].items ():
        alt =[x for x in DER if x ["mfg"]==mad ]
        if len (alt )<10 :
            continue 
        gk_ =gate_kur (keep &(mfg !=k ))
        _ ,ra =puanla (None ,alt ,gk_ );_ ,rb =puanla (0.50 ,alt ,gk_ )
        ud [mad ]=(T .f1w (ra ),T .f1w (rb ))
        print (f"  {mad }-disi robot {T .f1w (ra ):.4f} -> {T .f1w (rb ):.4f} ({T .f1w (rb )-T .f1w (ra ):+.4f})")

    dr =T .f1w (rob_c )-T .f1w (rob_t )
    gecti =(dr >=0.01 )and lo >0 and all (b >=a -0.005 for a ,b in ud .values ())
    print (f"\nKILL: robot +0.01 VE GA>0 VE manufacturer-disi dusmesin")
    print (f"  robot {dr :+.4f} | GA[{lo :+.4f},{hi :+.4f}] -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/r2_dik_kafa.json","w",encoding ="utf-8")as f :
        json .dump ({"robot_taban":T .f1w (rob_t ),"robot_r2":T .f1w (rob_c ),"fark":dr ,
        "ga":[lo ,hi ],"tespit_taban":T .f1w (det_t ),"tespit_r2":T .f1w (det_c ),
        "manufacturer":{k :list (v )for k ,v in ud .items ()},
        "dik_egitim_n":int (dik [egit ].sum ()),"gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/r2_dik_kafa.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
