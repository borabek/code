# -*- coding: utf-8 -*-
"""N: GERCEK kilitli measurement kumesi kur (audit maddesi 4).

IKI DEFECT DUZELTILIYOR:
 1) `geometry_key` only BBOX olculeri + log2 kose/face kovalari kullaniyor. Ayni anahtardaki
    some parcalarin yuzeyi birkac mm different olabiliyor -- i.e. gruplama hem KABA hem de real
    mesh kimligi not.
 2) Mevcut 100 parcalik cluster residual holdout DEGIL: J and L2 kararlari onun ten secildi, and
    denetime according to ~40 parcasi SEGMENTASYON EGITIM geometrileriyle cakisiyor. Yani "tamamen
    unseen geometri" iddiasi kanitlanmis not.

KESKIN ANAHTAR: bbox olculeri (0.5mm) + B-rep imzasi (silindir count, YARICAP histogrami,
duzlem count) + kose/face kovalari. B-rep imzasi two parcanin same hole/surface takimina sahip
olup olmadigini soyler; bbox single basina soylemez.

KILITLI KUME KURALI: test grubu, SU UCUNUN HICBIRIYLE same geometri grubunu paylasmaz:
    - segmentasyon training + val (91 part)
    - gate training havuzu
    - calibration/karar alinan each sey
Cikti: results/locked_split_v2.json  (a more ten AYAR YAPILMAZ)
"""
import os ,sys ,json ,glob 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
OUT ="results/locked_split_v2.json"
KEYS ="results/_strict_geometry_keys.json"
INFLIGHT ="results/_strict_keys_inflight.txt"# zehirli-part korumasi
SKIPFILE ="results/_strict_keys_skip.txt"


def strict_key (pid ,stp ,mesh_cache ="results/mesh_cache"):
    """bbox (0.5mm) + B-rep imzasi (silindir/duzlem count + radius histogrami) + kovalar."""
    import brep_axes as B 
    mc =os .path .join (mesh_cache ,pid +".npz")
    if os .path .exists (mc ):
        m =np .load (mc )
        V =np .asarray (m ["V"],float );nf =int (m ["F"].shape [0 ])
    else :
        from infer_step_cp import step_to_mesh 
        import thesis_remesh 
        Vr ,Fr =step_to_mesh (stp )
        V ,Fm =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .asarray (V ,float );nf =int (np .asarray (Fm ).shape [0 ])
    dims =np .round (np .sort (V .max (0 )-V .min (0 )),1 )
    try :
        C ,A ,R =B .cylinders (stp )
        Pc ,Pn ,Pr =B .planes (stp )
        rad =np .sort (np .round (np .asarray (R ,float ),1 ))
        # radius histogrami: 0.5mm kovalar, first 12 bucket
        h =np .bincount (np .clip ((rad /0.5 ).astype (int ),0 ,11 ),minlength =12 )
        sig =f"c{len (C )}|p{len (Pc )}|h{'.'.join (map (str ,h .tolist ()))}"
    except Exception :
        sig ="brep_yok"
    return f"{dims .tolist ()}|v{int (np .log2 (max (len (V ),1 )))}|f{int (np .log2 (max (nf ,1 )))}|{sig }"


def main ():
    from big_arbiter import eligible 

    # 1) tum eligible parts
    parts =[]
    for m ,p ,jf ,s in eligible ():
        try :
            n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :
            continue 
        if n >0 :
            parts .append ((m ,p ,jf ,s ,n ))
    print (f"{len (parts )} eligible part",flush =True )

    # 2) segmentasyon training + val parcalari
    seg =set ()
    for split in ("train","val"):
        f =f"wscad_corpus_scheffler_exact/splits/{split }.txt"
        if os .path .exists (f ):
            seg |={x .strip ()for x in open (f )if x .strip ()}
    print (f"segmentasyon training+val: {len (seg )} part",flush =True )

    # 3) keskin anahtarlar
    keys =json .load (open (KEYS ))if os .path .exists (KEYS )else {}
    # ZEHIRLI PARCA: gmsh single a STEP'te asilabiliyor (this night two times yasandi).
    # Islenmeden ONCE name yazilir; yeniden baslatmada orada stopped part atlanir.
    skip =set ()
    if os .path .exists (SKIPFILE ):
        skip |={x .strip ()for x in open (SKIPFILE )if x .strip ()}
    if os .path .exists (INFLIGHT ):
        st =open (INFLIGHT ).read ().strip ()
        if st :
            skip .add (st )
            with open (SKIPFILE ,"a")as fh :
                fh .write (st +chr (10 ))
            print (f"  [zehirli part] {st } atlandi",flush =True )
        os .remove (INFLIGHT )
    step ={os .path .basename (s ).split ("_")[1 ]:s for s in glob .glob ("all_wscad_stp/*.stp")}
    todo =[(p [1 ],p [3 ])for p in parts if p [1 ]not in keys ]
    todo +=[(pid ,step [pid ])for pid in seg if pid not in keys and pid in step ]
    todo =[t for t in todo if t [0 ]not in skip ]
    for k ,(pid ,stp )in enumerate (todo ,1 ):
        with open (INFLIGHT ,"w")as fh :
            fh .write (pid )
        try :
            keys [pid ]=strict_key (pid ,stp )
        except Exception :
            keys [pid ]="error:"+pid 
        if os .path .exists (INFLIGHT ):
            os .remove (INFLIGHT )
        if k %100 ==0 :
            print (f"  anahtar {k }/{len (todo )}",flush =True )
            json .dump (keys ,open (KEYS ,"w"))
    json .dump (keys ,open (KEYS ,"w"))

    kaba =json .load (open ("results/_geometry_keys.json"))
    n_kaba =len ({kaba .get (p [1 ],p [1 ])for p in parts })
    n_kes =len ({keys .get (p [1 ],p [1 ])for p in parts })
    print (f"\ngruplama: KABA {n_kaba } grup -> KESKIN {n_kes } grup "
    f"({len (parts )} part)",flush =True )

    seg_groups ={keys [p ]for p in seg if p in keys }
    print (f"segmentasyon egitiminin kapladigi geometri grubu: {len (seg_groups )}")

    # 4) mevcut 100 parcalik cluster ne up to kirli?
    from json_dataset import family_key 
    LOCK =set (json .load (open ("results/split_lock.json"))["locked_parts"])
    pool =[p for p in parts if p [1 ]not in LOCK ]
    rng =np .random .RandomState (202 )
    lo =[x for x in pool if x [4 ]<8 ];hi =[x for x in pool if x [4 ]>=8 ]
    old_ =([lo [i ]for i in rng .choice (len (lo ),70 ,replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),30 ,replace =False )])
    kirli =sum (1 for p in old_ if keys .get (p [1 ])in seg_groups )
    print (f"\nMEVCUT 100 parcalik cluster: {kirli }/100 part segmentasyon training geometrisiyle "
    f"AYNI grupta  <- 'unseen geometri' iddiasi this upto zayif",flush =True )

    # 5) TEMIZ kilitli cluster: segmentasyon gruplarina DEGMEYEN parcalardan
    temiz =[p for p in pool if keys .get (p [1 ])not in seg_groups ]
    tlo =[x for x in temiz if x [4 ]<8 ];thi =[x for x in temiz if x [4 ]>=8 ]
    print (f"temiz pool: {len (temiz )} part ({len (tlo )} dusuk / {len (thi )} very-CP)")
    r2 =np .random .RandomState (31071 )# YENI seed: old secimle ortusmesin
    nlo ,nhi =min (70 ,len (tlo )),min (30 ,len (thi ))
    test =([tlo [i ]for i in r2 .choice (len (tlo ),nlo ,replace =False )]+
    [thi [i ]for i in r2 .choice (len (thi ),nhi ,replace =False )])
    test_groups =sorted ({keys [p [1 ]]for p in test })
    # gate egitimi this gruplarin HICBIRINI gormeyecek
    json .dump ({
    "created":"2026-07-31",
    "rule":("Bu cluster UZERINDE AYAR YAPILMAZ: threshold secimi, feature secimi, model secimi, "
    "none of them. Yalnizca TEK ATISLIK final measurement. Gate egitimi and calibration this "
    "geometri gruplarinin hicbirini gormemeli."),
    "anahtar":"keskin geometri (bbox 0.5mm + B-rep silindir/duzlem imzasi + kovalar)",
    "n_test":len (test ),
    "test_parts":[p [1 ]for p in test ],
    "test_groups":test_groups ,
    "seg_train_val_groups_excluded":len (seg_groups ),
    "eski_kume_kirliligi":f"{kirli }/100 part segmentasyon geometrisiyle same grupta",
    },open (OUT ,"w"),indent =1 )
    print (f"\nKILITLI KUME: {len (test )} part ({nlo } dusuk / {nhi } very-CP), "
    f"{len (test_groups )} geometri grubu")
    print (f"-> {OUT }")
    print ("BU KUME UZERINDE AYAR YAPILMAZ.")


if __name__ =="__main__":
    main ()
