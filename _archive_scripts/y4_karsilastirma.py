# -*- coding: utf-8 -*-
"""Y4: KARSILASTIRMALI RENDER -- "same mi?" sorusu, "opening mi?" sorusu DEGIL.

WHY DEGISTI: first galeri MUTLAK yargi soruyordu ("this real a opening mi?"). Kullanici
hakli as "ben bakarak anlayamiyorum" dedi. Bir klemensin depth haritasina bakip
"this tel girisi mi" demek uzmanlik ister; KARSILASTIRMA istemez.

YENI KURULUM: each FP for, AYNI PARCADA ureticinin LISTELEDIGI a girisi TIPATIP same
yontemle ciz and yan yana koy, RENK OLCEGI ORTAK. Soru residual:
    "Solda ureticinin onayladigi giris. Sagda robotun bulup listede olmayan point.
     Bu ikisi AYNI TUR sey mi?"
Hem insan for very more easy, hem de bilimsel as full sordugumuz sey: listelenmemis
detection, listelenmislerle AYNI CINSTEN mi?

REFERANS GT SECIMI: FP'ye most yakin GT DEGIL (bitisik becomes, ayirt edilemez); parcanin
GEOMETRIK ORTASINA most yakin GT -- i.e. TIPIK a giris. Rastgele not, tekrar uretilebilir.

Referansi olmayan parts (GT absent) AYRI isaretlenir and orana KATILMAZ.
"""
import io 
import json 
import os 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

import matplotlib # noqa: E402
matplotlib .use ("Agg")
import matplotlib .pyplot as plt # noqa: E402

from y2_fp_render import derinlik_haritasi ,YARICAP ,N # noqa: E402

DIZIN ="results/fp_karsilastirma"


def yuzeye_gore (last_ ):
    """Derinligi NOKTAYA according to not CEVREDEKI YUZEYE according to ifade et.

    WHY REQUIRED: manufacturer CP'si kanalin ICINDE tanimli (isinlar ona varmadan ~5mm before yuzeye
    carpar -> centre NEGATIF), robotun noktasi whereas agizda (tezin v_o'this = boundary noktalarinin
    ortasi -> isin kanala girer, centre POZITIF). Ayni fiziksel opening two KONVANSIYONDA
    zit gorunur and comparison yaniltir. Cevre yuzeyi sifir kabul edince ikisi de
    "centre cevresinden ne up to deep" der and AYNI dili konusur.
    """
    g =np .linspace (-YARICAP ,YARICAP ,N )
    yy ,xx =np .meshgrid (g ,g ,indexing ="ij")
    rr =np .sqrt (xx **2 +yy **2 )
    halka =last_ [(rr >=6.0 )&(rr <=9.0 )]
    halka =halka [np .isfinite (halka )]
    if not len (halka ):
        return last_ 
    return last_ -float (np .median (halka ))


def main ():
    import measure_set 
    import thesis_remesh 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 

    os .makedirs (DIZIN ,exist_ok =True )
    with io .open ("results/fp_denetim.json",encoding ="utf-8")as f :
        FD =json .load (f )
    FP =FD ["all of them"];sec =FD ["ornek_idx"]
    stp_of ={p :s for m ,p ,jf ,s in eligible ()}
    DER ,_ =measure_set .cluster ("results/_der_tam.pkl")
    GT ={r ["pid"]:(np .asarray (r ["G"],float ),np .asarray (r ["Gd"],float ))for r in DER }

    byp ={}
    for i in sec :
        byp .setdefault (FP [i ]["pid"],[]).append (i )
    print (f"{len (sec )} FP / {len (byp )} part",flush =True )

    KAYIT =[]
    t0 =time .time ()
    for k ,(pid ,idxs )in enumerate (sorted (byp .items ()),1 ):
        G ,Gd =GT .get (pid ,(np .zeros ((0 ,3 )),np .zeros ((0 ,3 ))))
        try :
            Vr ,Fr =step_to_mesh (stp_of [pid ])
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,float );F =np .ascontiguousarray (F ,np .int64 )
        except Exception as e :
            print (f"  {pid }: mesh HATA {type (e ).__name__ }");continue 

        ref =None 
        if len (G ):
            d_ =np .linalg .norm (G -G .mean (0 ),axis =1 )
            j =int (np .argsort (d_ )[len (d_ )//2 ])
            rs ,rolc ,_ =derinlik_haritasi (V ,F ,G [j ],Gd [j ])
            if rs is not None :
                ref =(yuzeye_gore (rs ),rolc )

        for i in idxs :
            f_ =FP [i ]
            p =np .array (f_ ["point"],float );d =np .array (f_ ["direction"],float )
            last_ ,olc ,_ =derinlik_haritasi (V ,F ,p ,d )
            if last_ is None :
                continue 
            last_ =yuzeye_gore (last_ )
            # ORTAK RENK OLCEGI: ayri olcekte two harita gorsel as KIYASLANAMAZ
            hep =[last_ ]+([ref [0 ]]if ref is not None else [])
            vv =np .concatenate ([x [np .isfinite (x )].ravel ()for x in hep 
            if np .isfinite (x ).any ()])
            lo ,hi =(float (np .percentile (vv ,2 )),float (np .percentile (vv ,98 )))if len (vv )>10 else (0.0 ,1.0 )
            if hi -lo <1e-6 :
                hi =lo +1.0 

            n =2 if ref is not None else 1 
            fig =plt .figure (figsize =(4.3 *n +0.7 ,3.9 ))
            if ref is not None :
                ax0 =fig .add_subplot (1 ,n ,1 )
                ax0 .imshow (ref [0 ],origin ="lower",vmin =lo ,vmax =hi ,cmap ="viridis",
                extent =[-YARICAP ,YARICAP ,-YARICAP ,YARICAP ])
                ax0 .plot (0 ,0 ,"w+",ms =15 ,mew =2.2 )
                ax0 .set_title ("ÜRETİCİNİN LİSTELEDİĞİ GİRİŞ\nkesin doğru örnek",
                fontsize =9.5 ,color ="#15703F",fontweight ="bold")
                ax0 .set_xticks ([]);ax0 .set_yticks ([])
            ax1 =fig .add_subplot (1 ,n ,n )
            im =ax1 .imshow (last_ ,origin ="lower",vmin =lo ,vmax =hi ,cmap ="viridis",
            extent =[-YARICAP ,YARICAP ,-YARICAP ,YARICAP ])
            ax1 .plot (0 ,0 ,"w+",ms =15 ,mew =2.2 )
            ax1 .set_title ("ROBOTUN BULDUĞU\nüretici listesinde YOK",
            fontsize =9.5 ,color ="#A8332A",fontweight ="bold")
            ax1 .set_xticks ([]);ax1 .set_yticks ([])
            cb =fig .colorbar (im ,ax =ax1 ,fraction =0.046 )
            cb .set_label ("cevre yuzeyine according to depth (mm)",fontsize =8 )
            fig .suptitle (f"{pid } · {f_ ['mfg']} · "
            +("aynı renk ölçeği · sıfır = çevredeki yüzey"if ref is not None else "REFERANS YOK"),
            fontsize =9 ,y =0.99 )
            fig .tight_layout (rect =[0 ,0 ,1 ,0.93 ])
            path =f"{DIZIN }/{pid }_{f_ ['aday_i']}.png"
            fig .savefig (path ,dpi =84 ,bbox_inches ="tight");plt .close (fig )
            KAYIT .append ({"idx":i ,"pid":pid ,"mfg":f_ ["mfg"],"regime":f_ ["regime"],
            "png":path ,"ref_var":ref is not None ,
            "gt_uzaklik":f_ ["gt_uzaklik"]})
        if k %10 ==0 :
            print (f"  {k }/{len (byp )}  {time .time ()-t0 :.0f}s",flush =True )

    with io .open ("results/fp_karsilastirma.json","w",encoding ="utf-8")as f :
        json .dump (KAYIT ,f ,indent =1 )
    nref =sum (1 for x in KAYIT if x ["ref_var"])
    print (f"\n{len (KAYIT )} kare | referansli {nref } | referanssiz {len (KAYIT )-nref }")
    print ("receipt -> results/fp_karsilastirma.json")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
