# -*- coding: utf-8 -*-
"""K7.2 + K7.3 -- `dip_metal`: kanalin dibinde METAL present mi, and this TP/FP'yi ayiriyor mu?

K7.1 (cozuldu): STEP metninden surface->renk bagi (108/108), real boundary koseleri, and
`cad_eval.align_frames` with mesh cercevesine TAM hizalama (residual 0.000mm, %100 inside).

HIPOTEZ (tezden): tel girisi a KONTAKTA biter, alet agzi a arm/yayda. Kanalin dibinde
metal gorunmesi, geometrinin veremedigi FONKSIYONEL sinyaldir.
Dolayli evidence: `dip_CT` (segmentasyon tahmini) TP'de medyan 0.43, FP'de 0.22 -- direction correct but
girdisi gurultulu prediction. Renk TAHMIN DEGIL, CAD'in own verisi.

KAPI (olcumden before yazildi): AUC >= 0.70. `dip_CT` already 0.61 veriyor; renk bunu belirgin
gecmezse new bilgi getirmiyor demektir and K7 here durur.

ONBELLEK MESRU: this a GATE ozelligi (gate union sonrasi works), therefore
`single-derivation-proxy-invalid` kurali bunu kapsamaz -- that rule ADAY OLUSTURMA kararlari for.
"""
import os ,sys ,glob ,json ,pickle 
import numpy as np 
os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

CACHE ="results/pitstop2_cache"


def dip_features (mp_all ,metal_pts ,point ,direction ,axis_r =2.5 ,depth =14.0 ):
    """CP agzindan ICERIYE correct: most yakin metal ne up to derinde, and yakinda ne up to metal present.

    Doner: (dip_metal_dist, metal_in_channel, metal_frac_6mm)
      dip_metal_dist   : axis along iceride first metal noktasina uzaklik (otherwise depth)
      metal_in_channel : channel silindiri inside metal point VAR mi (0/1)
      metal_frac_6mm   : 6mm kure icindeki noktalarin metal orani (NaN = never point absent)
    """
    p =np .asarray (point ,float );d =np .asarray (direction ,float )
    d =d /(np .linalg .norm (d )+1e-9 )
    out =[depth ,0.0 ,np .nan ]
    if len (metal_pts ):
        rel =metal_pts -p 
        al =rel @(-d )# ICERIYE correct (mouth normali disari bakar)
        perp =np .linalg .norm (rel -al [:,None ]*(-d )[None ,:],axis =1 )
        inch =(al >0 )&(al <=depth )&(perp <=axis_r )
        if inch .any ():
            out [0 ]=float (al [inch ].min ());out [1 ]=1.0 
    if len (mp_all ):
        near =np .linalg .norm (mp_all [:,:3 ]-p ,axis =1 )<=6.0 
        if near .any ():
            out [2 ]=float (mp_all [near ,3 ].mean ())
    return out 


def auc (pos ,neg ):
    """Mann-Whitney U tabanli AUC -- BAGLARA and DENGESIZLIGE dayanikli.

    ONCEKI SURUM YANLISTI: `(ortalama_rank_farki)/n + 0.5` only siniflar DENGELI oldugunda
    correct sonuc gives. Dengesiz kumelerde sisiyor -- measured: TP orani 0.176 / FP orani 0.051
    which is ikili a ozellige 0.953 dedi (dogrusu 0.563). Ikili ozelliklerde AUC bagli siralari
    correct islemek zorunda, that is why 'average' rank is used."""
    pos =np .asarray (pos ,float );neg =np .asarray (neg ,float )
    n1 ,n0 =len (pos ),len (neg )
    if n1 ==0 or n0 ==0 :
        return float ("nan")
    v =np .concatenate ([neg ,pos ])
    order =np .argsort (v ,kind ="mergesort")
    ranks =np .empty (len (v ),float )
    sv =v [order ]
    i =0 
    while i <len (sv ):# BAGLI degerlere ORTALAMA rank
        j =i 
        while j +1 <len (sv )and sv [j +1 ]==sv [i ]:
            j +=1 
        ranks [order [i :j +1 ]]=0.5 *(i +j )+1.0 
        i =j +1 
    r_pos =ranks [n0 :].sum ()
    return float ((r_pos -n1 *(n1 +1 )/2.0 )/(n1 *n0 ))


def main ():
    import step_face_color_link as L 
    from cad_eval import align_frames 
    from big_arbiter import eligible 
    from infer_step_cp import step_to_mesh 

    stp_of ={p :s for _ ,p ,_ ,s in eligible ()}
    rows ={"dip_dist":[[],[]],"in_chan":[[],[]],"frac6":[[],[]]}
    n_ok =n_skip =0 
    for f in sorted (glob .glob (f"{CACHE }/*.npz")):
        pid =os .path .basename (f )[:-4 ]
        pk =f"{CACHE }/{pid }.cps.pkl"
        if pid not in stp_of or not os .path .exists (pk ):
            continue 
        try :
            d =np .load (f ,allow_pickle =True )
            cps =pickle .load (open (pk ,"rb"))
            if not cps :
                continue 
            faces =L .face_vertices_from_text (stp_of [pid ])
            if not faces :
                n_skip +=1 ;continue 
            Vr ,_ =step_to_mesh (stp_of [pid ])
            allp =np .vstack ([x ["pts"]for x in faces ])
            flag =np .concatenate ([np .full (len (x ["pts"]),1.0 if x ["is_metal"]else 0.0 )
            for x in faces ])
            R ,t ,res =align_frames (Vr ,allp )
            if res >1.0 :
                n_skip +=1 ;continue # hizalama guvenilir not -> ATLA (uydurma absent)
            allm =(allp -t )@R 
            mp_all =np .column_stack ([allm ,flag ])
            metal =allm [flag >0.5 ]

            G ,Gd ,tol =d ["G"],d ["Gd"],float (d ["tol"])
            P =np .array ([c ["point"]for c in cps ],float )
            D =np .array ([c .get ("direction",(0 ,0 ,1 ))for c in cps ],float )
            # TP/FP etiketi: big_arbiter konvansiyonu
            lab =np .zeros (len (P ),int )
            if len (G ):
                diff =P [:,None ,:]-G [None ,:,:]
                al =(diff *Gd [None ,:,:]).sum (-1 )
                pe =np .linalg .norm (diff -al [...,None ]*Gd [None ,:,:],axis =-1 )
                pe =np .where (np .abs (al )<=40.0 ,pe ,np .inf )
                used =set ();hit =np .zeros (len (G ),bool )
                for dd ,a ,b in sorted ((pe [a ,b ],a ,b )for a in range (len (P ))for b in range (len (G ))):
                    if dd >tol or a in used or hit [b ]:
                        continue 
                    used .add (a );hit [b ]=True ;lab [a ]=1 
            for i in range (len (P )):
                dd ,ic ,fr =dip_features (mp_all ,metal ,P [i ],D [i ])
                k =lab [i ]
                rows ["dip_dist"][k ].append (dd )
                rows ["in_chan"][k ].append (ic )
                if not np .isnan (fr ):
                    rows ["frac6"][k ].append (fr )
            n_ok +=1 
        except Exception :
            n_skip +=1 
            continue 
    print (f"{n_ok } part islendi, {n_skip } atlandi\n")
    print (f"{'feature':<18}{'TP medyan':>11}{'FP medyan':>11}{'AUC':>8}{'n(TP/FP)':>14}")
    res ={}
    for name in ("dip_dist","in_chan","frac6"):
        pos =np .array (rows [name ][1 ],float );neg =np .array (rows [name ][0 ],float )
        if len (pos )<10 or len (neg )<10 :
            print (f"{name :<18}  yeterli veri yok");continue 
        a =auc (pos ,neg )
        res [name ]={"auc":float (a ),"tp_med":float (np .median (pos )),
        "fp_med":float (np .median (neg )),"n_tp":len (pos ),"n_fp":len (neg )}
        print (f"{name :<18}{np .median (pos ):>11.3f}{np .median (neg ):>11.3f}{a :>8.3f}"
        f"{f'{len (pos )}/{len (neg )}':>14}")
    if res :
        best =max (abs (v ["auc"]-0.5 )for v in res .values ())+0.5 
        print (f"\nen guclu |AUC|: {best :.3f}   (dip_CT referansi 0.61)")
        print (f"KAPI (>= 0.70): {'GECTI'if best >=0.70 else 'OLU'}")
    json .dump (res ,open ("results/k7_dip_metal.json","w"),indent =1 )
    print ("receipt -> results/k7_dip_metal.json")


if __name__ =="__main__":
    main ()
