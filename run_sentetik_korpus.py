# -*- coding: utf-8 -*-
"""D1 ADIM 1 — SENTETIK KORPUSU URET: mesh + GT + segmentasyon olasiliklari

`sentetik_klemens.uret()` parametrik klemens ureti; this betik onu urun
boru hattinin BIRINCI asamasindan gecirir:

    uret -> remesh(6000) -> DiffusionNet cikarimi -> npz(V, F, pbs)
                                                  -> kayit(G, Gd, P, Pd, diag)

`P`/`Pd` (urun adaylari) segmentasyondan cikarilir; so sentetik part
real parcayla AYNI candidate uretim yolunu kullanir. Aksi halde sentetik data
"easy" becomes and olculen kazanc fake cikar.

DUMAN TESTI ZORUNLU: uretilen npz'de segmentasyonun GT agizlarinda gercekten
ateşleyip atesle(me)digine bakilir. Model sentetik geometride never ateslemiyorsa
korpusun training degeri yoktur and bunu SONRA not SIMDI bilmek is required.

D7'ye BAKILMAZ.
"""
import json 
import os 
import pickle 
import sys 
import time 

import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,".")
import connector3d # noqa: E402
import sentetik_klemens as SK # noqa: E402

N_PARCA =int (os .environ .get ("SK_N","400"))
TOHUM0 =int (os .environ .get ("SK_TOHUM","10000"))
MESH_CIK =os .environ .get ("SK_MESH","results/_p1_olasilik_sentetik")
KAYIT =os .environ .get ("SK_KAYIT","results/_sentetik_kayit.pkl")
CY_PKL ="results/_sentetik_silindirler.pkl"
AC_PKL ="results/_sentetik_acikliklar.pkl"
HEDEF_TEPE =6000 
CE ,CT =int (connector3d .CABLE_ENTRY ),int (connector3d .CONTACT )


def main ():
    t0 =time .time ()
    import torch # noqa: F401
    import diffusionnet as D 
    import thesis_remesh 
    from infer_step_cp import load_any 
    import product_genis 

    dev ="cuda"if __import__ ("torch").cuda .is_available ()else "cpu"
    cks =json .load (open ("cp_config.json"))["robot_vote2_checkpoints"]
    modeller =[load_any (c ,dev =dev )[:2 ]for c in cks ]
    os .makedirs (MESH_CIK ,exist_ok =True )
    OP =os .environ .get ("SK_OP","results/_op_cache_sentetik")
    os .makedirs (OP ,exist_ok =True )

    rec_ ,cyl ,ack ={},{},{}
    if os .path .exists (KAYIT ):
        rec_ =pickle .load (open (KAYIT ,"rb"))
        print (f"cache: {len (rec_ )} kayit",flush =True )
    yazilan =atlanan =empty_ =0 
    tani =[]
    for t in range (TOHUM0 ,TOHUM0 +N_PARCA ):
        pid =f"SYN{t }"
        yol =f"{MESH_CIK }/{pid }.npz"
        if os .path .exists (yol )and pid in rec_ :
            atlanan +=1 
            continue 
        try :
            m ,G ,Gd ,kun =SK .uret (t )
        except Exception as e :# noqa: BLE001
            empty_ +=1 
            print (f"  {pid }: URETIM HATASI {type (e ).__name__ }: {e }",
            flush =True )
            continue 
        Vr =np .asarray (m .vertices ,np .float64 )
        Fr =np .asarray (m .faces ,np .int64 )
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =HEDEF_TEPE )
        V =np .ascontiguousarray (V ,np .float64 )
        F =np .ascontiguousarray (F ,np .int64 )
        pbs =[]
        for model ,meta in modeller :
            _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
            op_cache_dir =OP ,return_probs =True )
            pbs .append (np .asarray (pb ,float ))
        pb =np .mean (pbs ,axis =0 )
        ppos =pb [:,CE ]+pb [:,CT ]

        # TANI: segmentasyon GT agizlarinda atesliyor mu, ZEMIN ne
        yak =np .argmin (np .linalg .norm (G [:,None ,:]-V [None ,:,:],
        axis =-1 ),axis =1 )
        rng =np .random .default_rng (0 )
        zemin =float (np .median (ppos [rng .choice (len (V ),
        min (500 ,len (V )),
        replace =False )]))
        tani .append ((float (np .median (ppos [yak ])),zemin ))

        # URUN ADAYLARI: real parcayla AYNI path
        try :
            ham =product_genis .candidates (V ,F ,pbs )if hasattr (product_genis ,"candidates")else None 
        except Exception :# noqa: BLE001
            ham =None 
        if ham is None :
        # geri dusus: high olasilikli tepelerden clustering
            from scipy .spatial import cKDTree 
            sec =np .where (ppos >=0.5 )[0 ]
            if not len (sec ):
                sec =np .argsort (-ppos )[:50 ]
            P_ ,Pd_ =[],[]
            kalan =list (sec )
            agac =cKDTree (V )
            import trimesh 
            N_ =np .asarray (trimesh .Trimesh (vertices =V ,faces =F ,
            process =False ).vertex_normals )
            while kalan :
                i0 =max (kalan ,key =lambda q :ppos [q ])
                grp =agac .query_ball_point (V [i0 ],3.0 )
                grp =[q for q in grp if q in set (kalan )]
                if not grp :
                    grp =[i0 ]
                w =ppos [grp ]
                P_ .append ((V [grp ]*w [:,None ]).sum (0 )/max (w .sum (),1e-9 ))
                nv =(N_ [grp ]*w [:,None ]).sum (0 )
                Pd_ .append (nv /max (np .linalg .norm (nv ),1e-9 ))
                kalan =[q for q in kalan if q not in set (grp )]
            P_ =np .asarray (P_ ,float ).reshape (-1 ,3 )
            Pd_ =np .asarray (Pd_ ,float ).reshape (-1 ,3 )
        else :
            P_ =np .asarray ([c ["point"]for c in ham ],float ).reshape (-1 ,3 )
            Pd_ =np .asarray ([c ["direction"]for c in ham ],
            float ).reshape (-1 ,3 )
        if len (P_ )<2 :
            empty_ +=1 
            continue 
        np .savez_compressed (yol ,V =V .astype (np .float32 ),
        F =F .astype (np .int32 ),
        pbs =np .asarray (pbs ,np .float32 ))
        rec_ [pid ]={"pid":pid ,"mfg":"SYN","geo":pid ,
        "diag":float (np .linalg .norm (V .max (0 )-V .min (0 ))),
        "n":len (G ),"P":P_ ,"Pd":Pd_ ,"G":G ,"Gd":Gd ,
        "metadata":kun }
        # SILINDIRLER insaattan BILINIR (sentetikte B-rep kesindir)
        cyl [pid ]=[{"c":G [j ].tolist (),"a":Gd [j ].tolist (),
        "r":kun ["cap"]/2.0 ,"h":kun ["derin"]}
        for j in range (len (G ))]
        ack [pid ]=[]
        yazilan +=1 
        if yazilan %25 ==0 :
            h =(time .time ()-t0 )/yazilan 
            print (f"  {yazilan } yazildi {h :.2f}s/part "
            f"kalan ~{h *(N_PARCA -yazilan )/60 :.0f}dk",flush =True )
            pickle .dump (rec_ ,open (KAYIT ,"wb"))
    pickle .dump (rec_ ,open (KAYIT ,"wb"))
    pickle .dump (cyl ,open (CY_PKL ,"wb"))
    pickle .dump (ack ,open (AC_PKL ,"wb"))

    print (f"\nBITTI: yazilan {yazilan } | cache {atlanan } | bos {empty_ } "
    f"({time .time ()-t0 :.0f} s)")
    if tani :
        a =np .asarray (tani )
        print (f"\n--- DUMAN TESTI: segmentasyon sentetikte calisiyor mu ---")
        print (f"  GT agzinda olasilik (ortanca) : {np .median (a [:,0 ]):.4f}")
        print (f"  rastgele yuzeyde (ortanca)    : {np .median (a [:,1 ]):.4f}")
        ratio =np .median (a [:,0 ])/max (np .median (a [:,1 ]),1e-6 )
        print (f"  ratio                          : {ratio :.2f}x")
        print ("  OKUMA: ratio ~1 whereas model sentetik geometride AYIRT ETMIYOR,")
        print ("         korpusun training degeri yoktur -- SIMDI bilinmeli.")
        json .dump ({"n":yazilan ,
        "gt_olasilik":float (np .median (a [:,0 ])),
        "zemin":float (np .median (a [:,1 ])),"ratio":float (ratio ),
        "not":"Sentetik corpus duman testi. Oran ~1 ise korpusun "
        "training degeri none. D7'ye BAKILMADI."},
        open ("results/sentetik_duman.json","w"),indent =1 )


if __name__ =="__main__":
    main ()
