# -*- coding: utf-8 -*-
"""H4: very-CP ince-ayarli ckpt URUNU IYILESTIRIYOR mu? (regime ayrimli, fast teshis)

KILL (baslamadan yazildi):
  * very-CP CP-F1 katkisi < +0.02  -> OLU
  * VEYA low-CP'de -0.01'den extra loss -> REDDEDILIR
  (this night two kaldirac full this imzayla became: promote and adaptif yogunluk)

YONTEM: full corpus cikarimi pahali oldugu for before DOGRUDAN teshis --
segmentasyonun GT'lerde ATESLEME orani (T5'in ikinci adimi). Ince-setting whereas yaradiysa
very-CP'de atesleme %76'dan yukari cikmali. Cikmazsa full olcume gerek absent.
"""
import os ,sys ,json ,time 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))


def fire_rate (ckpt ,parts ,dev ,tag ):
    """Her GT agzinda network CE+CT >= 0.30 uretiyor mu (T5 olcutu)."""
    import torch ,trimesh 
    import thesis_remesh ,diffusionnet as D 
    from cad_eval import align_frames 
    from infer_step_cp import step_to_mesh ,load_any 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    from cp_geometry import mouths_for 
    from scipy .spatial import cKDTree 
    model ,meta =load_any (ckpt ,dev =dev )[:2 ]
    k =int (meta .get ("k_eig",64 ))
    agg ={"low":[0 ,0 ],"very":[0 ,0 ]}
    for mfg ,pid ,jf ,stp ,n in parts :
        try :
            Vr ,Fr =step_to_mesh (stp )
            V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
            V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
            _ ,pb =D .predict (model ,meta ,V ,F ,device =dev ,
            op_cache_dir =f"results/step_infer/ops_k{k }",return_probs =True )
            pb =np .asarray (pb ,float );conn =pb [:,CE ]+pb [:,CT ]
            j =json .load (open (jf ,encoding ="utf-8-sig"))
            Vj =np .array ([[q [c ]for c in "XYZ"]for q in j ["Graphic3d"]["Points"]],float )
            G =np .array ([[c ["Point"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =np .array ([[c ["InsertDirection"][q ]for q in "XYZ"]for c in j ["ConnectionPoints"]],float )
            Gd =Gd /(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
            R ,t ,_ =align_frames (Vr ,Vj );Gm =(G -t )@R ;Gdm =Gd @R 
            mth ,_ =mouths_for (trimesh .Trimesh (V ,F ,process =False ),Gm ,Gdm )
            tree =cKDTree (V )
            key ="very"if n >=8 else "low"
            for b in range (len (Gm )):
                idx =tree .query_ball_point (mth [b ],3.0 )
                agg [key ][0 ]+=1 
                agg [key ][1 ]+=int (bool (idx )and float (conn [idx ].max ())>=0.30 )
        except Exception :
            pass 
    out ={kk :(v [1 ]/max (v [0 ],1 ),v [0 ])for kk ,v in agg .items ()}
    print (f"{tag :<34}dusuk %{100 *out ['low'][0 ]:.1f} ({out ['low'][1 ]} GT) | "
    f"very %{100 *out ['very'][0 ]:.1f} ({out ['very'][1 ]} GT)",flush =True )
    return out 


def main ():
    import torch 
    from big_arbiter import eligible 
    lock =json .load (open ("results/split_lock.json"));LOCK =set (lock ["locked_parts"])
    # EGITIMDE KULLANILAN parts SKORLANMAZ (new ckpt onlari gordu)
    trained =set ()
    for f ,k in (("_label_targets_recall/trained_parts.json","parts"),):
        if os .path .exists (f ):trained |=set (json .load (open (f )).get (k ,[]))
    for d in ("_mfg_labels","_mfg_labels_highcp"):
        import glob 
        trained |={os .path .basename (os .path .normpath (p ))for p in glob .glob (f"{d }/train/*/")}
    parts =[]
    for m ,p ,jf ,s in eligible ():
        if p in LOCK or p in trained :continue 
        try :n =len (json .load (open (jf ,encoding ="utf-8-sig"))["ConnectionPoints"])
        except Exception :continue 
        parts .append ((m ,p ,jf ,s ,n ))
    rng =np .random .RandomState (3 )
    lo =[x for x in parts if x [4 ]<8 ];hi =[x for x in parts if x [4 ]>=8 ]
    sel =([lo [i ]for i in rng .choice (len (lo ),min (20 ,len (lo )),replace =False )]+
    [hi [i ]for i in rng .choice (len (hi ),min (20 ,len (hi )),replace =False )])
    print (f"skorlanan: {len (sel )} part (egitimde kullanilan {len (trained )} part HARIC)\n",flush =True )
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    cfg =json .load (open ("cp_config.json"))
    base =cfg ["current_product"]["checkpoints"][1 ]# keig96 urun ckpt'i (same k_eig)
    new ="results/seg_extra/highcp_ft_s0.pt"
    a =fire_rate (base ,sel ,dev ,"MEVCUT urun (keig96)")
    if os .path .exists (new ):
        b =fire_rate (new ,sel ,dev ,"YENI very-CP ince-ayarli")
        print (f"\n{'FARK':<34}dusuk {100 *(b ['low'][0 ]-a ['low'][0 ]):+.1f} puan | "
        f"very {100 *(b ['very'][0 ]-a ['very'][0 ]):+.1f} puan")
        print ("\nKAPI: very-CP atesleme ARTMALI, low-CP DUSMEMELI ->",
        "GECTI"if (b ['very'][0 ]>a ['very'][0 ]and b ['low'][0 ]>=a ['low'][0 ]-0.02 )else "SUPHELI/OLU")
        json .dump ({"base":{k :v [0 ]for k ,v in a .items ()},
        "finetuned":{k :v [0 ]for k ,v in b .items ()}},
        open ("results/h4_fire_rate.json","w"),indent =1 )
    else :
        print (f"\n{new } henuz none -- training bitmedi")


if __name__ =="__main__":
    main ()
