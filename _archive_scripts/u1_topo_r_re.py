# -*- coding: utf-8 -*-
"""U1: gate training verisinin 4 TOPOLOJI sutununu BASKA BIR YARICAPLA yeniden uret.

WHY UCUZ: topoloji sutunlari YALNIZ mesh'ten and candidate noktasindan is computed. Aday noktalari
(`pts`/`dirs`) and part kimlikleri (`pids`) npz'de ZATEN sakli; mesh de diskte onbellekli
(results/mesh_cache/<pid>.npz = remesh SONRASI V,F). Yani network cikarimi, candidate turetme and etiketleme
never tekrarlanmaz -- only 4 column yeniden is computed. Tam gate_regrow ~80 dk, this ~10 dk.

KANIT SART (own kendini dogrulayan tasarim): before R=6.0 with yeniden uretip npz'deki MEVCUT
sutunlarla karsilastiriyorum. Eger yeniden uretim yolu (mesh kaynagi, edge yapisi, cagri order)
gate_regrow'unkiyle same degilse this test PATLAR. Ayni cikarsa R=8/12 sutunlari da guvenilir.
Bu kontrol olmadan "new R more iyi" demek, different a MESH yolunun etkisini R'ye yazmak olurdu
([[single-turetme-proxy-invalid]] dersinin same bicimi).

KULLANIM:
    python u1_topo_r_re.py --dogrula          # only R=6 tekrar-uretim testi
    python u1_topo_r_re.py --r 12.0           # new npz uret
"""
import argparse 
import os 
import sys 
import time 

import numpy as np 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
os .environ .setdefault ("BA_ALLOW_SEEN","1")

NPZ ="results/gate_regrow_data_topo.npz"
MESH_CACHE ="results/mesh_cache"
TOPO_SUT =slice (18 ,22 )# 13 baseline + 5 fiziksel = 18, sonraki 4 topoloji


def _mesh (pid ,stp =None ):
    """Remesh SONRASI V,F. Onbellekte varsa oradan; otherwise STEP'ten uretip onbellege yaz."""
    f =os .path .join (MESH_CACHE ,pid +".npz")
    if os .path .exists (f ):
        d =np .load (f )
        return np .ascontiguousarray (d ["V"],np .float64 ),np .ascontiguousarray (d ["F"],np .int64 )
    if not stp :
        return None ,None 
    import thesis_remesh 
    from infer_step_cp import step_to_mesh 
    Vr ,Fr =step_to_mesh (stp )
    V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
    V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
    os .makedirs (MESH_CACHE ,exist_ok =True )
    np .savez_compressed (f ,V =V .astype (np .float32 ),F =F .astype (np .int32 ))
    return V ,F 


def hesapla (R ,bound_ =0 ,ilerleme =True ):
    """Her candidate for 4 topoloji sutununu R yaricapiyla hesapla. (X_topo, islenen_pid_maskesi)"""
    import topo_feats 

    d =np .load (NPZ ,allow_pickle =True )
    pids =np .asarray (d ["pids"]).astype (str )
    pts =np .asarray (d ["pts"],float )
    dirs =np .asarray (d ["dirs"],float )
    stp_of ={}
    try :
        from big_arbiter import eligible 
        stp_of ={p :s for m ,p ,jf ,s in eligible ()}
    except Exception as e :
        print (f"  (uyari: STEP haritasi none -> {type (e ).__name__ }; only cache kullanilir)")

    benzersiz =list (dict .fromkeys (pids .tolist ()))
    if bound_ :
        benzersiz =benzersiz [:bound_ ]
    out =np .zeros ((len (pids ),4 ),float )
    tamam =np .zeros (len (pids ),bool )
    t0 =time .time ();skipped =0 
    for k ,pid in enumerate (benzersiz ,1 ):
        if ilerleme and k %100 ==0 :
            print (f"  {k }/{len (benzersiz )} part | {time .time ()-t0 :.0f}s | skipped {skipped }",
            flush =True )
        idx =np .where (pids ==pid )[0 ]
        try :
            V ,F =_mesh (pid ,stp_of .get (pid ))
            if V is None :
                skipped +=1 
                continue 
            onb =topo_feats .icbukey_kenarlar (V ,F )# R'den BAGIMSIZ, part basina 1 times
            for i in idx :
                out [i ]=topo_feats .topo_ozellik (V ,F ,pts [i ],dirs [i ],R =R ,cache =onb )
            tamam [idx ]=True 
        except Exception as e :
            skipped +=1 
            if skipped <=3 :
                print (f"  atlandi {pid }: {type (e ).__name__ }: {e }")
    print (f"  bitti: {len (benzersiz )-skipped }/{len (benzersiz )} part | {time .time ()-t0 :.0f}s")
    return out ,tamam 


def dogrula (bound_ =60 ):
    """R=6.0 with yeniden uret and npz'deki mevcut sutunlarla karsilastir."""
    d =np .load (NPZ ,allow_pickle =True )
    X =np .asarray (d ["X"],float )
    old_ =X [:,TOPO_SUT ]
    new_ ,tamam =hesapla (6.0 ,bound_ =bound_ )
    if not tamam .any ():
        print ("HIC part islenemedi -- yeniden uretim yolu KULLANILAMAZ");return False 
    e ,y =old_ [tamam ],new_ [tamam ]
    diff =np .abs (e -y )
    tam =float ((diff .max (1 )<1e-6 ).mean ())
    yakin =float ((diff .max (1 )<1e-3 ).mean ())
    print (f"\n  karsilastirilan satir: {tamam .sum ()} ({len (np .unique (np .asarray (d ['pids'])[tamam ]))} part)")
    print (f"  BIREBIR same (<1e-6): {tam :.4f}")
    print (f"  yakin      (<1e-3): {yakin :.4f}")
    for j ,nm in enumerate (["kon_oran","kon_sayi","kon_aci","kon_cevre"]):
        print (f"    {nm :10s} maks diff {diff [:,j ].max ():.6g} | ort {diff [:,j ].mean ():.6g}")
    ok =tam >0.99 
    print (f"\n  KARAR: {'YOL DOGRULANDI'if ok else 'YOL FARKLI -- R sonuclari GUVENILMEZ'}")
    return ok 


def main ():
    ap =argparse .ArgumentParser ()
    ap .add_argument ("--dogrula",action ="store_true")
    ap .add_argument ("--r",type =float ,default =0.0 )
    ap .add_argument ("--boundary",type =int ,default =0 )
    a =ap .parse_args ()
    if a .dogrula :
        dogrula (a .bound_ or 60 );return 
    assert a .r >0 ,"--r ver"
    X_topo ,tamam =hesapla (a .r ,bound_ =a .bound_ )
    d =np .load (NPZ ,allow_pickle =True )
    X =np .asarray (d ["X"],float ).copy ()
    # ISLENEMEYEN parcalarin sutunlarini ESKISIYLE birak: karsilastirmayi R farkina odaklar,
    # loss part etkisini karistirmaz.
    X [tamam ,TOPO_SUT ]=X_topo [tamam ]
    out =f"results/gate_regrow_data_topo_r{a .r :g}.npz".replace (".npz",".npz")
    rec_ ={k :d [k ]for k in d .files }
    rec_ ["X"]=X 
    np .savez_compressed (out ,**rec_ )
    print (f"\nyazildi -> {out } | guncellenen satir {int (tamam .sum ())}/{len (X )}")


if __name__ =="__main__":
    main ()
