# -*- coding: utf-8 -*-
"""G5: ESIK POLITIKASI -- last ucuz arm, and DURUST olcumu.

DIGER KOLLARDAN FARKI: G1-G4 gate'in SIRALAMASINI degistirmeye calisti (all of them closed).
Esik siralamayi degistirmez, CALISMA NOKTASINI changes. Su anki rule:
    sec  <=>  skor >= max(0.5 x parcanin_en_yuksegi, 0.25)
Iki serbestlik: CARPAN (0.5) and TABAN (0.25). Bir de REJIM-KOSULLU olabilirler.

SISME TUZAGI: 194 parcada threshold tarayip most iyisini bildirmek OVERFIT'tir. Bu yuzden
CAPRAZ SECIM: geometri gruplarini ikiye bol, esigi A'da SEC, B'de OLC (and tersi).
Bildirilen number, "threshold ayarlamasi gercekte ne up to eder" sorusunun durust cevabidir.

KILL (onceden): capraz-secilmis threshold havuzlanmisi +0.005 artirmazsa VE GA sifiri
kapsiyorsa G5 kapanir. Ayrica manufacturer-disi ORT dusmemeli.
"""
import io 
import json 

import numpy as np 

import gate_bench as T 

CARPANLAR =[0.30 ,0.35 ,0.40 ,0.45 ,0.50 ,0.55 ,0.60 ,0.65 ,0.70 ]
TABANLAR =[0.10 ,0.15 ,0.20 ,0.25 ,0.30 ,0.35 ,0.40 ]


def main ():
    import measure_set 
    import wire_gate 
    from sina_cluster import esle 
    from sklearn .ensemble import RandomForestClassifier 

    D =T .yukle ()
    measure_set .rapor_bas (D ["rap"])
    cfg =D ["cfg"]
    X ,y ,pid ,mfg ,keep =D ["X"],D ["y"],D ["pid"],D ["mfg"],D ["keep"]

    # --- each split for gate egit, measurement kumesinde SKORLARI al (a times)
    def gate_kur (kp ):
        Z =np .zeros ((len (X ),X .shape [1 ]*2 ))
        for u in np .unique (pid ):
            i =np .where (pid ==u )[0 ]
            Z [i ]=wire_gate .within_part (X [i ],D ["donusum"])
        clf =RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (Z [kp ],y [kp ])
        return {"clf":clf ,"n_feat":Z .shape [1 ],"donusum":D ["donusum"]}

    bolmeler =[("havuzlanmis",None )]
    for k ,mad in D ["kod"].items ():
        if len ([x for x in D ["DER"]if x ["mfg"]==mad ])>=10 :
            bolmeler .append ((mad +"-disi",k ))

    SKOR ={}
    for b ,mk in bolmeler :
        kp =keep .copy ()
        if mk is not None :
            kp =kp &(mfg !=mk )
        g =gate_kur (kp )
        S ={}
        for r in D ["DER"]:
            if r ["X"]is None or r .get ("XR")is None :
                S [r ["pid"]]=None ;continue 
            Xr =np .hstack ([r ["X"],r ["XR"]])
            # POSE DUZELTMESI ADAY BASINA BAGIMSIZ (`m["model"].predict(X)`), i.e. esikten
            # ETKILENMEZ. Bir times TUM adaylara uygula, after threshold only ALTKUME alsin.
            # Ilk kosuda each threshold for yeniden cagriliyordu and tarama bitmiyordu.
            Pd_ =r ["Pd"];P_ =r ["P"]
            if cfg .get ("robot_pose_head"):
                cc =[{"point":P_ [i ],"direction":Pd_ [i ]}for i in range (len (P_ ))]
                cc =wire_gate .pose_correct (Xr ,cc )
                P_ =np .array ([x ["point"]for x in cc ],float )
                Pd_ =np .array ([x ["direction"]for x in cc ],float )
            S [r ["pid"]]=(wire_gate .decision_score (g ,Xr ),P_ ,Pd_ )
        SKOR [b ]=S 
        print (f"  skorlar hazir: {b }",flush =True )

    def puan (b ,alt ,carpan ,baseline ,carpan_cok =None ,taban_cok =None ):
        det =[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            v =SKOR [b ].get (r ["pid"])
            rj ="cok"if r ["n"]>=8 else "dusuk"
            c =carpan_cok if (rj =="cok"and carpan_cok is not None )else carpan 
            t =taban_cok if (rj =="cok"and taban_cok is not None )else baseline 
            if v is not None :
                s ,P_ ,Pd_ =v # P_/Pd_ pose duzeltmesi UYGULANMIS halde onbellekte
                k =(s >=c *max (float (s .max ()),1e-9 ))&(s >=t )
                if k .any ():
                # angle_correct / pick_member_direction SADECE YONU changes; TESPIT eslesmesi
                # 180 derece toleransla calistigi for sonucu DEGISTIREMEZLER -> atlanir.
                    P =P_ [k ];Pd =Pd_ [k ]
            det .append ((rj ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
        return det 

    DER =D ["DER"]
    # --- 1) TAM TARAMA (SISIK -- only tavani gormek for)
    en ,arg =-1 ,None 
    for c in CARPANLAR :
        for t in TABANLAR :
            f =T .f1w (puan ("havuzlanmis",DER ,c ,t ))
            if f >en :
                en ,arg =f ,(c ,t )
    su_an =T .f1w (puan ("havuzlanmis",DER ,0.5 ,0.25 ))
    print (f"\nsu anki (0.50, 0.25)   havuzlanmis {su_an :.4f}")
    print (f"TAM TARAMA en iyisi    {arg }  {en :.4f}   (+{en -su_an :.4f})  <- SISIK, ayni kumede secildi")

    # --- 2) CAPRAZ SECIM (DURUST)
    gruplar =sorted ({r ["geo"]for r in DER })
    rng =np .random .default_rng (0 )
    kar =list (gruplar );rng .shuffle (kar )
    A =set (kar [:len (kar )//2 ])
    altA =[r for r in DER if r ["geo"]in A ]
    altB =[r for r in DER if r ["geo"]not in A ]
    det_capraz =[]
    for sec_alt ,olc_alt in ((altA ,altB ),(altB ,altA )):
        e ,a =-1 ,(0.5 ,0.25 )
        for c in CARPANLAR :
            for t in TABANLAR :
                f =T .f1w (puan ("havuzlanmis",sec_alt ,c ,t ))
                if f >e :
                    e ,a =f ,(c ,t )
        det_capraz +=puan ("havuzlanmis",olc_alt ,a [0 ],a [1 ])
        print (f"  yarida secilen threshold {a } -> diger yaride measured ({len (olc_alt )} part)")
    f_capraz =T .f1w (det_capraz )
    print (f"CAPRAZ SECIM (DURUST)  {f_capraz :.4f}   ({f_capraz -su_an :+.4f})")

    # --- 3) REJIM-KOSULLU (capraz secimle)
    det_rej =[]
    for sec_alt ,olc_alt in ((altA ,altB ),(altB ,altA )):
    # REJIM-KOSULLU: before low-CP for most iyi (c,t), after SABIT tutup very-CP carpanini ara.
    # Tam capraz product 567 puanlama demekti; ardisik arama 63+9 = 72 with same tepeye ulasir.
        e ,a =-1 ,(0.5 ,0.25 ,0.5 ,0.25 )
        for c in CARPANLAR :
            for t in TABANLAR :
                f =T .f1w (puan ("havuzlanmis",sec_alt ,c ,t ))
                if f >e :
                    e ,a =f ,(c ,t ,c ,t )
        for cc_ in CARPANLAR :
            f =T .f1w (puan ("havuzlanmis",sec_alt ,a [0 ],a [1 ],cc_ ,a [1 ]))
            if f >e :
                e ,a =f ,(a [0 ],a [1 ],cc_ ,a [1 ])
        det_rej +=puan ("havuzlanmis",olc_alt ,a [0 ],a [1 ],a [2 ],a [3 ])
        print (f"  regime-kosullu threshold {a } secildi")
    f_rej =T .f1w (det_rej )
    print (f"REJIM-KOSULLU (DURUST) {f_rej :.4f}   ({f_rej -su_an :+.4f})")

    # --- 4) manufacturer-disi bedeli (full-tarama esigiyle)
    print (f"\nuretici-disi bedeli (tam-tarama esigi {arg }):")
    ud_su ,ud_ye =[],[]
    for b ,mk in bolmeler [1 :]:
        alt =[x for x in DER if x ["mfg"]==b .replace ("-disi","")]
        a_ =T .f1w (puan (b ,alt ,0.5 ,0.25 ));b_ =T .f1w (puan (b ,alt ,arg [0 ],arg [1 ]))
        ud_su .append (a_ );ud_ye .append (b_ )
        print (f"  {b :<12} {a_ :.4f} -> {b_ :.4f}  ({b_ -a_ :+.4f})")

    det_a =puan ("havuzlanmis",DER ,0.5 ,0.25 )
    lo ,hi =0.0 ,0.0 
    fn =lambda rows :T .f1w ([q for _ ,q in rows ])-T .f1w ([p for p ,_ in rows ])
    _ ,lo ,hi =measure_set .grup_bootstrap (list (zip (det_a ,det_capraz if len (det_capraz )==len (det_a )else det_a )),
    [r ["geo"]for r in DER ],fn ,n =2000 )
    gecti =(f_capraz -su_an >=0.005 )and lo >0 and np .mean (ud_ye )>=np .mean (ud_su )
    print (f"\nKILL: capraz-secim +0.005 VE GA>0 VE manufacturer-disi dusmesin")
    print (f"  capraz {f_capraz -su_an :+.4f} | GA[{lo :+.4f},{hi :+.4f}] | "
    f"manufacturer-ort {np .mean (ud_su ):.4f}->{np .mean (ud_ye ):.4f}")
    print (f"  -> {'GECTI'if gecti else 'GECMEDI'}")
    with io .open ("results/g5_esik.json","w",encoding ="utf-8")as f :
        json .dump ({"su_an":su_an ,"tam_tarama":en ,"tam_tarama_esik":list (arg ),
        "capraz_durust":f_capraz ,"rejim_kosullu":f_rej ,
        "uretici_su":float (np .mean (ud_su )),"uretici_yeni":float (np .mean (ud_ye )),
        "ga":[lo ,hi ],"gecti":bool (gecti )},f ,indent =1 )
    print ("receipt -> results/g5_esik.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
