# -*- coding: utf-8 -*-
"""KANONIK MANSET -- dagitilan urunun tum sayilarini TEK places produces and cp_config'e YAZAR.

Bu betik, gece along ortaya cikan OLCUM ZAAFIYETLERINI kapatmak for yazildi:

1. ELLE YAZILAN SAYI -> KARISIK KOL. cp_config manseti elle guncelleniyordu and 2026-08-01'de
   `tespit_F1` dagitilan koldan, `robot_hazir_F1`/`precision`/`recall` ONCEKI koldan kalmisti.
   Artik butun alanlar TEK kosudan gelir; elle yazilmaz.

2. OLCUM, URUNUN YOLUNU TAKLIT EDIYORDU. Betikler gate'i elde yeniden kuruyordu
   (`clf.predict_proba` + threshold). Urun whereas arada COKUS YONLENDIRMESI does. Artik ikisi de
   `wire_gate.decision_score` / `decision_mask` cagiriyor -- taklit not, AYNI KOD.

3. DEV and VAL BIRLESTIRILMISTI. Uc parcali split (DEV karar / VAL exam / LOCKED harcanmadi)
   kurulmustu but gece along "DEV+VAL havuzlanmis" single number raporlandi; i.e. exam da karar
   verirken kullanildi. Artik UCU DE AYRI basilir.

4. "EN KOTU URETICI" TEK OLCUMDU. Iki manufacturer = two split; urun karari that single sayiya dayaniyordu.
   Artik 7 URUN SERISI bolmesi de basilir -> 9 bolmelik DAGILIM.

5. GUVEN ARALIGI YOKTU. Bu sabah yazdigim headline ciplak sayilardan olusuyordu. Her number residual
   eslestirilmis bootstrap GA'si with gelir.

6. TEK REJIM ORTALAMASI. Korpus carpik (low-CP %89.5 / very-CP %10.5); duz mean yaniltir.
   Rejim kirilimi each bolmede basilir.

TEZ CIZGISI: here network, remesh, sinif tanimi ya da CP turetmesi YOK -- this a OLCUM betigi.

KULLANIM:
    python headline.py            # olc + tabloyu head + results/headline.json
    python headline.py --yaz      # also cp_config.current_product.headline_F1'i GUNCELLE
"""
import argparse 
import collections 
import io 
import json 
import os 
import pickle 
import sys 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1"
os .environ ["WG_TOPO"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
# OLCUM ONBELLEGI: dagitilan gate ZENGIN sutunlari kullaniyorsa zengin cache sart.
DER_CACHE =("results/_der_tam.pkl"if os .path .exists ("results/_der_tam.pkl")
else "results/_u4_der.pkl")
# EGITIM VERISI cp_config'ten okunur: dagitilan gate hangi veriyle egitildiyse measurement de
# ONUNLA yapilmalidir. Sabit file adi, 2026-08-02'de dagitilan parite verisi instead of ESKI
# veriyi olcturuyordu -- paritenin a baska yuzu.
def _egitim_verisi ():
    try :
        with io .open ("cp_config.json",encoding ="utf-8")as f :
            return json .load (f )["current_product"]["wire_gate"].get (
            "egitim_verisi","results/gate_regrow_data_topo.npz")
    except Exception :
        return "results/gate_regrow_data_topo.npz"


NPZ =_egitim_verisi ()
ONEK =2 
NBOOT =4000 


def _pose_acik ():
    """POSE HEAD bayragi. Urun uyguluyorsa OLCUM DE uygulamali -- aksi halde olculen sey
    dagitilan urun DEGILDIR (2026-08-02'de this sinifta two kusur output: headline dagitilan
    modelin donusumunu kurmuyordu and training npz'si sabit koduydu)."""
    try :
        import wire_gate as _w 
        with io .open ("cp_config.json",encoding ="utf-8")as f :
            if not json .load (f ).get ("robot_pose_head",False ):
                return False 
        return _w ._load (_w .POSE_PATH )is not None 
    except Exception :
        return False 


def _aci_acik ():
    """SECICI ACI DUZELTMESI bayragi -- urun uyguluyorsa measurement de uygulamali."""
    try :
        import wire_gate as _w 
        with io .open ("cp_config.json",encoding ="utf-8")as f :
            if not json .load (f ).get ("robot_aci_secici",False ):
                return False 
        return _w ._load (_w .ACI_PATH )is not None 
    except Exception :
        return False 


def _yon_secici_acik ():
    """AYRIK YON SECICI bayragi. Urun uyguluyorsa OLCUM DE uygulamali.

    Secici calisma aninda MESH ister; headline whereas onbellekten puanliyor. Bu yuzden
    measurement, sozlugu `results/_r4_sozluk.pkl` onbelleginden kurar and DAGITILAN modeli
    (`results/yon_secici.pkl`) uygular -- i.e. urunun own karar kodu, taklit not.
    Onbellek otherwise selector UYGULANMAZ and headline bunu basar (silent deviation olmaz).
    """
    try :
        import wire_gate as _w 
        with io .open ("cp_config.json",encoding ="utf-8")as f :
            if not json .load (f ).get ("robot_yon_secici",False ):
                return None 
        m =_w ._load ("results/yon_secici.pkl")
        if m is None or not os .path .exists ("results/_r4_sozluk.pkl"):
            return None 
        import pickle 
        with open ("results/_r4_sozluk.pkl","rb")as f :
            return (m ,pickle .load (f ))
    except Exception :
        return None 


def _uye_acik ():
    """UYE SECICI bayragi -- urun uyguluyorsa measurement de uygulamali."""
    try :
        import wire_gate as _w 
        with io .open ("cp_config.json",encoding ="utf-8")as f :
            if not json .load (f ).get ("robot_uye_secici",False ):
                return False 
        return _w ._load (_w .UYE_PATH )is not None 
    except Exception :
        return False 


def _boot (det ,gruplar ,n =NBOOT ):
    """GRUP duzeyinde bootstrap -> (F1, lower, upper).

    PARCA not GEOMETRI GRUBU ornekleniyor (2026-08-01 denetimi): 194 part 171 gruba dusuyor
    and gruplarin a kismi IKIZ parts iceriyor. Parca ornekleyen bootstrap ikizleri bagimsiz
    gozlem sayar and confidence araligini SAHTE DARALTIR."""
    from measure_set import grup_bootstrap 
    from sina_cluster import f1w 
    if not det :
        return 0.0 ,0.0 ,0.0 
    _ ,lo ,hi =grup_bootstrap (det ,gruplar ,f1w ,n =n )
    return float (f1w (det )),lo ,hi 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--yaz",action ="store_true",help ="cp_config mansetini guncelle")
    a =ap .parse_args ()

    import wire_gate 
    from big_arbiter import eligible 
    from sina_cluster import esle ,f1_rejim ,f1w 
    from sklearn .ensemble import RandomForestClassifier 

    with open ("cp_config.json",encoding ="utf-8")as f :
        cfg =json .load (f )
    mfg_of ={p :m for m ,p ,jf ,s in eligible ()}
    # OLCUM KUMESI TEK KAYNAKTAN (2026-08-01 denetimi): split3.json'dan DEV/VAL, PID dedup,
    # dogrudan kullanilan LOCKED parts CIKARILIR. Eskiden kumeler olasilik onbelleginden
    # cikariliyordu and "dev" aslinda old _h_probs.pkl idi -- atama %100 yanlisti.
    import measure_set 
    DER ,_kume_rap =measure_set .cluster (DER_CACHE )
    measure_set .rapor_bas (_kume_rap )
    for r in DER :
        r ["mfg"]=mfg_of .get (r ["pid"],"?")

    d =np .load (NPZ ,allow_pickle =True )
    with open ("results/_strict_geometry_keys.json",encoding ="utf-8")as f :
        gk =json .load (f )
    tr_pid =np .array ([str (x )for x in d ["pids"]])
    tr_grp =np .array ([gk .get (p ,"yok:"+p )for p in tr_pid ])
    tr_mfg =np .array ([str (x )for x in d ["mfg"]])
    tr_onek =np .array ([p [:ONEK ]for p in tr_pid ])
    # Egitim matrisi: zengin npz X22+XR carries, old npz single X.
    Xtr =(np .hstack ([np .asarray (d ["X22"],float ),np .asarray (d ["XR"],float )])
    if "X22"in d .files else np .asarray (d ["X"],float ))
    ytr =np .asarray (d ["y"])
    tg ={gk .get (r ["pid"],"yok:"+r ["pid"])for r in DER }
    kod ={k :collections .Counter (mfg_of .get (p ,"?")for p in tr_pid [tr_mfg ==k ]).most_common (1 )[0 ][0 ]
    for k in np .unique (tr_mfg )}

    # DAGITILAN MODELI OKU -- measurement onun yapisini TAKLIT ETMEZ, aynen kurar.
    dag =wire_gate ._load (wire_gate .MODEL_PATH )
    yonlendirmeli =dag .get ("clf_z")is not None 
    print (f"dagitilan gate: {dag ['n_feat']} sutun"
    +(f" + cokus yonlendirme (threshold {dag ['esik_cokus']:.4f}, "
    f"donusum {dag .get ('donusum_z')})"if yonlendirmeli else "")
    +f" | topo_r {dag .get ('topo_r')}")

    # Donusmus matris, model donusum VEYA yonlendirme tasidiginda is required.
    Ztr =None 
    if dag .get ("donusum")or yonlendirmeli :
        _dn =dag .get ("donusum")or dag .get ("donusum_z","zskor")
        Ztr =np .zeros ((len (Xtr ),Xtr .shape [1 ]*2 ))
        for u in np .unique (tr_pid ):
            i =np .where (tr_pid ==u )[0 ]
            Ztr [i ]=wire_gate .within_part (Xtr [i ],_dn )

    def kur (keep ):
        """Bolme for gate'i YENIDEN egit (sizintisiz) -- dagitilan modelin YAPISIYLA.

        PARITE: dagitilan model a DONUSUM tasiyorsa ana siniflandirici DONUSMUS matrisle
        egitilmelidir. Eskiden here always HAM matris is used and `n_feat`=22 with
        kirpiliyordu -- i.e. dagitilan 44-sutunlu model olculurken fiilen 22-sutunlu HAM arm
        olculuyordu (2026-08-01). Olcum, urunu TAKLIT bile edemiyordu."""
        rf =lambda M :RandomForestClassifier (n_estimators =400 ,min_samples_leaf =3 ,n_jobs =-1 ,
        random_state =0 ).fit (M [keep ],ytr [keep ])
        _don =dag .get ("donusum")
        _ana =Ztr if (_don and Ztr is not None )else Xtr 
        m ={"clf":rf (_ana ),"n_feat":_ana .shape [1 ],"donusum":_don ,
        "topo_r":dag .get ("topo_r")}
        if yonlendirmeli :
            m ["clf_z"]=rf (Ztr );m ["donusum_z"]=dag .get ("donusum_z","zskor")
            mx =[float (m ["clf"].predict_proba (Xtr [np .where ((tr_pid ==u )&keep )[0 ]])[:,1 ].max ())
            for u in np .unique (tr_pid [keep ])if ((tr_pid ==u )&keep ).any ()]
            m ["esik_cokus"]=float (np .quantile (mx ,dag .get ("yonlendirme_q",0.10 )))
        return m 

    _YS =_yon_secici_acik ()
    def puanla (m ,alt ):
        """URUNUN DECISION YOLU: wire_gate.decision_score + decision_mask (taklit not)."""
        det ,rob ,rbi =[],[],[]
        for r in alt :
            P =np .zeros ((0 ,3 ));Pd =np .zeros ((0 ,3 ))
            Xr =r ["X"]
            if Xr is not None and r .get ("XR")is not None :
                Xr =np .hstack ([Xr ,r ["XR"]])
            if Xr is not None and Xr .shape [1 ]*(2 if m .get ("donusum")else 1 )==m ["n_feat"]:
                s =wire_gate .decision_score (m ,Xr )
                k =wire_gate .decision_mask (s )
                if k .any ():
                    P =r ["P"][k ].copy ();Pd =r ["Pd"][k ].copy ()
                    # POSE HEAD: urun uyguluyorsa measurement de uygulamali (aksi halde olculen
                    # sey dagitilan urun DEGILDIR -- this gece two times that is why wrong olctum).
                    if _pose_acik ():
                        _c =[{"point":P [i ],"direction":Pd [i ]}for i in range (len (P ))]
                        _c =wire_gate .pose_correct (Xr [k ],_c )
                        if _aci_acik ():
                            _c =wire_gate .angle_correct (Xr [k ],_c )
                        if _uye_acik ()and r .get ("UYE"):
                            _c =wire_gate .pick_member_direction (Xr [k ],_c ,r ["UYE"])
                            # AYRIK YON SECICI: urun uyguluyorsa measurement de uygulamali.
                            # Secici calisma aninda MESH ister; headline onbellekten puanladigi
                            # for dictionary `c4_sozluk.pkl`den gelir but uygulanan model
                            # DAGITILAN modeldir (results/yon_secici.pkl) -- taklit not.
                        _ys =_YS 
                        if _ys is not None :
                            _m ,_sz =_ys 
                            _d =_sz .get (r ["pid"])
                            # KIMLIK KONTROLU (uzunluk DEGIL). Onbellek, tum measurement
                            # gruplari atilmis gate with kuruldu; headline whereas 9 bolmenin
                            # 8inde gate'i BASKA a keep with yeniden egitiyor ->
                            # different kabul kumesi. Uzunluk esitligi KIMLIK esitligi
                            # DEGILDIR: parcalarin yarisindan fazlasinda 1-3 candidate present,
                            # baska two candidate kabul edildiginde kontrol SESSIZCE geciyor
                            # and direction, residual kumede olmayan a adayin sozlugunden
                            # yaziliyordu. (2026-08-03 denetimi, bulgu 5.)
                            _ok =False 
                            if _d is not None and len (_d .get ("SY",_d .get ("SOZ",[])))==len (_c ):
                                _Pc =np .array ([x ["point"]for x in _c ],float )
                                _ok =(len (_d ["P"])==len (_Pc )and 
                                np .allclose (np .asarray (_d ["P"],float ),_Pc ,
                                atol =1e-6 ))
                            if _ok :
                                from d1_direction_distribute import satirla as _sat 
                                _Pd =np .array ([x ["direction"]for x in _c ],float )
                                _SY =_d .get ("SY",_d .get ("SOZ"))
                                _ax ,_ ,_rj =_sat (_d ["X"],_SY ,_d ["UZ"],_Pd ,None )
                                if _ax :
                                    _pr =_m ["clf"].predict_proba (np .array (_ax ,float ))[:,1 ]
                                    _sk ={}
                                    for _n ,(_i ,_gi )in enumerate (_rj ):
                                        _sk .setdefault (_i ,{})[_gi ]=_pr [_n ]
                                    _mj =float (_m .get ("marj",0.05 ))
                                    for _i ,_sc in _sk .items ():
                                        _g =max (_sc ,key =_sc .get )
                                        if _g !=0 and _sc [_g ]-_sc .get (0 ,0.0 )>=_mj :
                                            _c [_i ]["direction"]=_SY [_i ][_g ][1 ]
                        P =np .array ([x ["point"]for x in _c ],float )
                        Pd =np .array ([x ["direction"]for x in _c ],float )
            rej ="cok"if r ["n"]>=8 else "dusuk"
            det .append ((rej ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],0.0 ,180.0 ,True ))
            rob .append ((rej ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ))
            # ISARETLI (FIZIKSEL) robot metrigi: prediction ureticinin InsertDirection'iyla AYNI
            # yone bakmak zorunda. Eksen metrigi 180 derece tersi 0 sayiyordu; robot for
            # correct criterion budur. Olculdu: 0.5893 -> 0.5818 (loss 0.0075, sozlesme tutarli).
            rbi .append ((rej ,)+esle (P ,Pd ,r ["G"],r ["Gd"],r ["diag"],2.0 ,10.0 ,False ,
            signed =True ))
        return det ,rob ,rbi 

    SON ={}

    def kaydet (ad ,alt ,keep ,grup ):
        """Bir bolmeyi puanla and kaydet."""
        det ,rob ,rbi =puanla (kur (keep ),alt )
        _g =[x ["geo"]for x in alt ]
        t ,tlo ,thi =_boot (det ,_g )
        rb ,rlo ,rhi =_boot (rob ,_g )
        rbs ,_ ,_ =_boot (rbi ,_g )
        # REJIM KIRILIMI: f1w'yi lower kumeyle CAGIRMA (agirlikla carpar) -- see. sina_cluster.f1w
        rj =f1_rejim (det )
        SON [ad ]={"grup":grup ,"n_parca":len (alt ),
        "tespit_F1":round (t ,4 ),"tespit_GA":[round (tlo ,4 ),round (thi ,4 )],
        "robot_hazir_F1":round (rb ,4 ),"robot_GA":[round (rlo ,4 ),round (rhi ,4 )],
        "robot_ISARETLI_F1":round (rbs ,4 ),
        "precision":round (rj ["agirlikli_kesinlik"],3 ),
        "recall":round (rj ["agirlikli_recall"],3 ),
        "ham_kesinlik":round (rj ["ham_kesinlik"],3 ),
        "ham_recall":round (rj ["ham_recall"],3 ),
        "dusuk_CP_F1":(round (rj ["F1"]["dusuk"],4 )if rj ["F1"]["dusuk"]is not None 
        else None ),"n_dusuk":rj ["n"]["dusuk"],
        "cok_CP_F1":(round (rj ["F1"]["cok"],4 )if rj ["F1"]["cok"]is not None 
        else None ),"n_cok":rj ["n"]["cok"]}
        s =SON [ad ]
        nn =lambda v :v if v is not None else float ("nan")
        print (f"{ad :<22}{len (alt ):>5}{t :>9.4f}  [{tlo :.4f},{thi :.4f}]{rb :>9.4f}"
        f"{s ['precision']:>8.3f}{s ['recall']:>8.3f}"
        f"{nn (s ['dusuk_CP_F1']):>9.4f}{nn (s ['cok_CP_F1']):>9.4f}"
        f"{s ['robot_ISARETLI_F1']:>10.4f}",flush =True )

    hepsi =~np .isin (tr_grp ,list (tg ))
    print (f"\n{'split':<22}{'n':>5}{'tespit':>9}{'  %95 GA':>18}{'robot':>9}"
    f"{'kesin*':>8}{'recall*':>8}{'dusuk-CP':>9}{'cok-CP':>9}{'robot-ISRT':>10}")
    print ("-- 1. TANIDIK (manufacturer-karisik, geometri-ayrik) "+"-"*45 )
    for cluster in ("dev","val","atanmamis"):
        alt =[r for r in DER if r .get ("cluster")==cluster ]
        if alt :
            kaydet (f"  {cluster .upper ()}",alt ,hepsi ,"tanidik")
    kaydet ("  HAVUZLANMIS",DER ,hepsi ,"tanidik")

    print ("-- 2. GORULMEMIS URETICI (asil axis, n=2) "+"-"*49 )
    for k ,mad in kod .items ():
        alt =[x for x in DER if x ["mfg"]==mad ]
        if len (alt )>=10 :
            kaydet (f"  {mad } disarida",alt ,(tr_mfg !=k )&hepsi ,"uretici_disi")

    print ("-- 3. GORULMEMIS URUN SERISI (n=7, daha kolay axis) "+"-"*39 )
    te =collections .Counter (r ["pid"][:ONEK ]for r in DER )
    trs =collections .Counter (p [:ONEK ]for p in np .unique (tr_pid ))
    for s_ in sorted (k for k in te if te [k ]>=10 and trs [k ]>=30 ):
        kaydet (f"  seri {s_ }",[r for r in DER if r ["pid"][:ONEK ]==s_ ],
        (tr_onek !=s_ )&hepsi ,"seri_disi")

    ud =[v ["tespit_F1"]for v in SON .values ()if v ["grup"]=="uretici_disi"]
    sd =[v ["tespit_F1"]for v in SON .values ()if v ["grup"]=="seri_disi"]
    ozet ={"uretici_disi_ORT":round (float (np .mean (ud )),4 ),
    "uretici_disi_EN_KOTU":round (float (min (ud )),4 ),
    "uretici_disi_YAYILIM":round (float (max (ud )-min (ud )),4 ),
    "seri_disi_ORT":round (float (np .mean (sd )),4 ),
    "seri_disi_EN_KOTU":round (float (min (sd )),4 ),
    "seri_disi_MEDYAN":round (float (np .median (sd )),4 )}
    print ("\nOZET")
    for k ,v in ozet .items ():
        print (f"  {k :<22}{v :.4f}")

    cikti ={"olcum_tarihi":"2026-08-01","gate":{
    "n_feat":int (dag ["n_feat"]),"yonlendirmeli":bool (yonlendirmeli ),
    "esik_cokus":float (dag .get ("esik_cokus",0 ))if yonlendirmeli else None ,
    "topo_r":dag .get ("topo_r")},"bolmeler":SON ,"ozet":ozet ,
    "not":("Tum sayilar TEK kosudan; measurement urunun kendi karar yolunu (wire_gate.decision_score "
    "+ decision_mask) kullanir, taklit etmez. LOCKED harcanmadi.")}
    with io .open ("results/headline.json","w",encoding ="utf-8")as f :
        json .dump (cikti ,f ,indent =1 ,ensure_ascii =False )
    print ("\nmakbuz -> results/headline.json")

    if a .yaz :
        h =cfg ["current_product"]["headline_F1"]
        hav =SON ["  HAVUZLANMIS"]
        h ["tanidik"]={k :hav [k ]for k in ("tespit_F1","tespit_GA","robot_hazir_F1",
        "robot_GA","precision","recall","n_parca",
        "dusuk_CP_F1","cok_CP_F1")}
        h ["tanidik"]["DEV"]=SON ["  DEV"]["tespit_F1"]if "  DEV"in SON else None 
        h ["tanidik"]["VAL"]=SON ["  VAL"]["tespit_F1"]if "  VAL"in SON else None 
        for k ,v in SON .items ():
            if v ["grup"]=="uretici_disi":
                h ["gorulmemis_uretici"][k .strip ().split ()[0 ]+"_disarida"]={
                x :v [x ]for x in ("tespit_F1","tespit_GA","robot_hazir_F1","n_parca")}
        h ["gorulmemis_seri"]={k .strip ():{x :v [x ]for x in ("tespit_F1","n_parca")}
        for k ,v in SON .items ()if v ["grup"]=="seri_disi"}
        h ["ozet"]=ozet 
        # GATE TARIFI ARTEFAKTTAN URETILIR. Elle yazilan hali 2026-08-02'de BAYATLADI:
        # "22 column + cokus yonlendirme (44 column)" diyordu, oysa dagitilan gate 116 column and
        # yonlendirme GERI ALINMISTI. Kimlik residual dosyanin kendisinden okunur -> bayatlayamaz.
        import hashlib as _h 
        import wire_gate as _w 
        _p =getattr (_w ,"MODEL_PATH","results/wire_gate.pkl")
        h ["gate"]={
        "n_feat":int (dag ["n_feat"]),"topo_r":dag .get ("topo_r"),
        "donusum":dag .get ("donusum"),"yonlendirmeli":bool (yonlendirmeli ),
        "dosya":_p ,
        "md5":_h .md5 (open (_p ,"rb").read ()).hexdigest (),
        "egitim_verisi":NPZ ,
        "tarif":str (dag .get ("note",""))[:400 ],
        "URETIM":"headline.py --yaz ile ARTEFAKTTAN okunur; elle yazilmaz.",
        }
        h ["URETIM"]=("Bu blok headline.py --yaz ile URETILIR, ELLE YAZILMAZ. Elle guncelleme "
        "2026-08-01'de kollari karistirmisti (tespit yeni koldan, robot eskisinden).")
        with io .open ("cp_config.json","w",encoding ="ascii")as f :
            json .dump (cfg ,f ,indent =1 ,ensure_ascii =True )
        print ("cp_config.current_product.headline_F1 GUNCELLENDI (uretilmis, elle degil)")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
