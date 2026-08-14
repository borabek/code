# -*- coding: utf-8 -*-
"""P6 OZNITELIK CIKARIMI: (konum x direction) option tablosu.

Dagitilan urun a adayi TEK yonle puanliyor. Bu betik each adayi
`direction_bank.options` with cogaltir and HER SECENEK for feature satiri produces.

OZNITELIK BLOKLARI (total 92 column):
  A  58  pool oznitelikleri (`_tam_oz`)      -- KONUM baglami, adayin own yonuyle
  B   9  mouth tanimlayicilari (`_tan_hizali`) -- KONUM baglami, adayin own yonuyle
  C  16  direction bankasi olculeri                 -- SECENEGE ozel (source, destek, hiza)
  D   9  mouth tanimlayicilari YENIDEN         -- SECENEK YONUYLE is computed

D blogu isin atislari gerektirir (`girme` / `erisim` / `girme_kenar`); asil maliyet
oradadir and a parcanin TUM secenekleri TEK cagrida toplu atilir.

WHY A and B option yonuyle YENIDEN hesaplanmiyor: 58 sutunun most konumsal
(segmentasyon olasiligi, topoloji, komsuluk) and yeniden uretimi part basina
saniyeler suruyor. A/B konum baglamini, C/D direction kararini carries. Bu ayrim same
zamanda modele "konum ne up to iyi" with "this direction correct mu" sorularini AYRI gives.

Kullanim:
    python run_p6_feature.py d6            # 468 part
    python run_p6_feature.py full           # 2583 training parcasi
    python run_p6_feature.py d7            # 835 (SINAV -- only kapida)

Devam edilebilir: output part basina a npz, VAR OLAN ATLANIR (pid anahtarli --
indeks anahtarli devam this projede more before sessizce wrong parcayi atlamisti).
"""
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")

OZ ="results/_tam_oz"
TAN ="results/_tan_hizali"
CIK =os .environ .get ("P6_CIK","results/_p6_oz")

# HAVUZ KAYNAKLARI. Onbellek three kaynagi da tasiyor:
#   0 segmentasyon (tezin `v_o`'this)   ~15 candidate/part
#   1 B-rep agzi                      ~85
#   2 MESH TEPESI                    ~354   <- ADAY_YOK kovasinin cevabi here
# Mesh tepeleri uctan uca UC times ZARAR vermisti, but that olcumlerde selector single
# yonlu and zayifti. Yon bankasi + ortak siralayici with yeniden acilir; kararı
# LOMO gives. `P6_KAYNAK=012` with acilir.
# `P6_MESH_MAX` mesh adaylarini SEGMENTASYON OLASILIGINA according to upper sinira indirir
# (0 = sinirsiz); so option count patlamaz.
_ks =os .environ .get ("P6_KAYNAK","01")
KAYNAKLAR =tuple (int (c )for c in _ks )
MESH_MAX =int (os .environ .get ("P6_MESH_MAX","250"))# baseline upper boundary
MESH_KAT =int (os .environ .get ("P6_MESH_KAT","4"))# kaynak0+1 sayisinin kati
MESH_R =float (os .environ .get ("P6_MESH_R","2.5"))# uzamsal seyreltme (mm)

# ten -> (silindir onbellegi, opening onbellegi, mesh onbellegi)
KUME ={
"tam":("results/_brepegit_silindirler.pkl","results/_brepegit_acikliklar.pkl",
"results/_p1_olasilik_brepegit"),
"d6":("results/_d6_silindirler.pkl","results/_d6_acikliklar.pkl",
"results/_p1_olasilik"),
"d7":("results/_d7_silindirler.pkl","results/_d7_acikliklar.pkl",
"results/_p1_olasilik_d7"),
}
# (KAYNAKLAR above cevre degiskeninden kuruluyor. Burada IKINCI a tanim
#  vardi and onu SESSIZCE eziyordu: `P6_KAYNAK=012` verilmesine despite mesh
#  adaylari havuza girmiyordu and no error cikmiyordu.)


def main ():
    on =sys .argv [1 ]if len (sys .argv )>1 else "d6"
    if on not in KUME :
        sys .exit (f"unknown cluster: {on }")
    sil_y ,ack_y ,ob =KUME [on ]
    import trimesh 

    import mouth_descriptor 
    import product_wide 
    import direction_bank as YB 

    os .makedirs (CIK ,exist_ok =True )
    cy =pickle .load (open (sil_y ,"rb"))
    ac =pickle .load (open (ack_y ,"rb"))
    dosyalar =sorted (f for f in os .listdir (OZ )
    if f .startswith (on +"_")and f .endswith (".npz"))
    # PARCALI KOSU: `P6_SHARD=i/n` -> only indeksi n'e bolumunden kalani i which is
    # dosyalar. Cikti part basina single file oldugu and VAR OLAN ATLANDIGI for
    # paylar birbirinin isini bozmaz; 16 cekirdegi kullanmanin most ucuz yolu.
    sh =os .environ .get ("P6_SHARD")
    if sh :
        i_ ,n_ =(int (x )for x in sh .split ("/"))
        dosyalar =[f for k ,f in enumerate (dosyalar )if k %n_ ==i_ ]
        print (f"  PAY {i_ }/{n_ }",flush =True )
    print (f"{on }: {len (dosyalar )} part | cikti {CIK }",flush =True )

    t0 =time .time ()
    written =skipped =empty_ =0 
    for i ,f in enumerate (dosyalar ,1 ):
        pid =f [len (on )+1 :-4 ]
        hedef =f"{CIK }/{on }_{pid }.npz"
        if os .path .exists (hedef ):
            skipped +=1 
            continue 
        z =np .load (f"{OZ }/{f }")
        kay =np .asarray (z ["source"],int )
        m =np .isin (kay ,KAYNAKLAR )
        if int (m .sum ())<2 :
            empty_ +=1 
            continue 
        T =np .asarray (np .load (f"{TAN }/{f }")["T"],float )
        if len (T )!=len (z ["X"]):
            print (f"  ! {pid }: HIZALAMA BOZUK, atlandi",flush =True )
            empty_ +=1 
            continue 
        mf =f"{ob }/{pid }.npz"
        if not os .path .exists (mf ):
            empty_ +=1 
            continue 
        zz =np .load (mf )
        V =np .ascontiguousarray (zz ["V"],np .float64 )
        Fc =np .ascontiguousarray (zz ["F"],np .int64 )
        if 2 in KAYNAKLAR and int ((kay ==2 ).sum ())>0 :
        # MESH ADAYLARINI SEYRELT. Kaynak 0/1 ASLA elenmez.
        #
        # MEASURED (D6, 468 part, YALNIZ KONUM recall'u / candidate-part):
        #   most high probability 60     0.6362 / 160
        #   most high probability 150    0.7163 / 214
        #   uzamsal 4.0mm, 2x120      0.8091 / 178
        #   uzamsal 2.5mm, 4x250      0.8713 / 251   <- SECILEN (diz)
        #   uzamsal 2.0mm, sinirsiz   0.9768 / 458
        # Yani KAPSAMA, GUVENI yeniyor: olasiligi most high vertices same
        # mouth cevresinde kumeleniyor and digerleri empty kaliyor. Seyreltme
        # high olasilikli tepeden baslar, MESH_R yaricapinda bastirir.
            import connector3d 

            import thin_pool 
            pb =np .mean ([np .asarray (q ,float )for q in zz ["pbs"]],axis =0 )
            pp =thin_pool .ppos (pb ,connector3d .CABLE_ENTRY ,
            connector3d .CONTACT )
            i2 =np .where (kay ==2 )[0 ]
            P2 =np .asarray (z ["P"],float )[i2 ]
            s2 =(pp [np .argmin (np .linalg .norm (
            P2 [:,None ,:]-V [None ,:,:],axis =-1 ),axis =1 )]
            if len (P2 )*len (V )<6e7 else np .zeros (len (P2 )))
            n01 =int (np .isin (kay ,[k for k in KAYNAKLAR if k !=2 ]).sum ())
            tut =np .zeros (len (i2 ),bool )
            tut [thin_pool .seyrelt (P2 ,s2 ,n01 ,MESH_R ,MESH_MAX ,
            MESH_KAT )]=True 
            m2 =m .copy ()
            m2 [i2 ]=tut 
            m =m2 
        A =np .asarray (z ["X"],float )[m ]
        B =T [m ]
        P =np .asarray (z ["P"],float )[m ]
        D =np .asarray (z ["D"],float )[m ]
        cyl =cy .get (str (pid ))
        # SIRA IMPORTANT: `mesh`/`diag` yelpaze for `options`e giriyor.
        # Once cagirip after tanimlamak first parcada NameError, sonrakilerde
        # BIR ONCEKI PARCANIN mesh'ini kullanmak demekti -- silent wrong.
        diag =float (np .linalg .norm (V .max (0 )-V .min (0 )))
        mesh =trimesh .Trimesh (V ,Fc ,process =False )
        # YELPAZE only MESH OLMAYAN adaylara: mesh tepesine 256 isin
        # atmak part basina face binlerce isin demek and that candidates already
        # own vertex normalini tasiyor.
        fmask =(kay [m ]!=2 )
        idx ,YD ,C =YB .options (P ,D ,cyl ,V ,mesh =mesh ,diag =diag ,
        fan_maske =fmask )
        if not len (idx ):
            empty_ +=1 
            continue 
            # D blogu: mouth tanimlayicilari SECENEK YONUYLE
        Dblok =product_wide .tanimlayici (P [idx ],YD ,cyl ,mesh ,diag )
        X =np .hstack ([A [idx ],B [idx ],C ,Dblok ]).astype (np .float32 )
        # `source` DE YAZILIR: so TEK cikarimdan hem (0,1) hem (0,1,2)
        # kolu egitilebilir and two kolu ayri ayri cikarmak gerekmez.
        # ATOMIK YAZIM: temp dosyaya yaz, after instead of tasi. Iki isci same
        # parcaya denk gelirse (yuk dengesizligi yuzunden yardimci isci
        # eklendiginde becomes) half yazilmis npz kalmaz.
        gec =f"{hedef }.{os .getpid ()}.tmp"
        np .savez_compressed (gec ,X =X ,idx =idx .astype (np .int32 ),
        YD =YD .astype (np .float32 ),P =P .astype (np .float32 ),
        D =D .astype (np .float32 ),
        src_ =kay [m ].astype (np .int8 ))
        os .replace (gec +".npz"if os .path .exists (gec +".npz")else gec ,hedef )
        written +=1 
        if i %25 ==0 :
            hz =(time .time ()-t0 )/max (written ,1 )
            print (f"  {i }/{len (dosyalar )}  written {written } skipped {skipped } "
            f"bos {empty_ }  {hz :.2f} s/part",flush =True )
    print (f"\nBITTI: written {written } | skipped {skipped } | bos {empty_ } | "
    f"{time .time ()-t0 :.0f} s",flush =True )
    print (f"sutun: 58 (A) + 9 (B) + {len (YB .OZ_AD )} (C) + "
    f"{len (mouth_descriptor .AD )} (D)")


if __name__ =="__main__":
    main ()
