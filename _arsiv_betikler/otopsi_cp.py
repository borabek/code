# -*- coding: utf-8 -*-
"""OTOPSI: GLB'de "wrong" gorunen CP'lerin FIZIKSEL olcumu (FAZ 1 -- KANIT, DUZELTME YOK).

BELIRTI (2026-08-04, gozle): uretilen GLB'lerde isaretciler body ucundan tasiyor, some oklar
seridin UZUN EKSENI along, dense parcada isaretciler goruntudeki yariklarla hizali not.
CELISKI: same parcalardan biri (2669020000) metrikte **F1 1.000** aliyor.

ILK IPUCU (eslesme log'u): perp 0.4-2.1mm but **AXIAL +12..+15mm**. `sina_cluster.esle` axial
toleransi **40mm** -- i.e. metrik eksende 40mm uzaktaki noktayi "correct" sayar.

HIPOTEZ SIRALAMASI (Faz 3'te teker teker sinanacak, before KANIT toplanir):
  H1  TANIM FARKI: tez CP'yi ACIKLIGIN AGZINDA (v_o), manufacturer KONTAKTA (seat) tanimlar.
      Eksenel 12-15mm this farkin ta kendisidir; error degildir. -> Sinamasi: bizim noktamizi
      manufacturer seat'inin KENDI AGZIYLA (`cp_geometry.seat_to_mouth`) karsilastir. Fark
      COKERSE H1 dogrulanir.
  H2  YON HATASI: direction serbest uzayi gostermiyor (kanala girmiyor). -> Sinamasi: +direction and
      -direction along serbest distance. Takma yonu OPEN must be, ters direction MALZEMEYE carpmali.
  H3  YERLESIM HATASI: point a agizda not, duz yuzeyde/body inside. -> Sinamasi:
      `is_inside` and `mouth_width`.
  H4  CIZIM HATASI: point and direction correct but GLB wrong ciziyor. -> Sinamasi: this betigin
      olcumleri with GLB makbuzundaki degerleri karsilastir (ayri kosuda).

Bu betik HICBIR SEYI DUZELTMEZ. results/otopsi_cp.json produces.
"""
import io ,json ,os ,sys 
import numpy as np 

os .environ .setdefault ("BA_ALLOW_SEEN","1")
os .environ ["WG_FIZ_FEATS"]="1";os .environ ["WG_TOPO"]="1";os .environ ["WG_ZENGIN"]="1"
sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))

PARCALAR =["2669020000","2464750000","2506380000"]


def serbest (mesh ,p ,d ,max_mm =60.0 ):
    """p'den d yonunde ILK yuzeye uzaklik (otherwise inf). 'Bu direction open mi' sorusunun olcusu."""
    from cp_geometry import ray_hits 
    h =ray_hits (mesh ,np .asarray (p ,float ),np .asarray (d ,float ),max_mm )
    return float (min (h ))if len (h )else float ("inf")


def main ():
    import torch ,trimesh 
    import thesis_remesh ,robot_cp as RC ,wire_gate 
    import cad_eval 
    from cp_geometry import is_inside ,mouth_width ,seat_to_mouth 
    from infer_step_cp import load_any ,step_to_mesh 
    from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
    import diffusionnet as D_ 
    import glob as _g 

    cfg =json .load (io .open ("cp_config.json",encoding ="utf-8"))
    dev ="cuda"if torch .cuda .is_available ()else "cpu"
    models =[load_any (c ,dev =dev )[:2 ]for c in cfg ["current_product"]["checkpoints"]]
    RAPOR ={}

    for pid in PARCALAR :
        stp =_g .glob (f"all_wscad_stp/*_{pid }_*.stp")
        jf =(_g .glob (rf"C:\Users\DE00024082\Desktop\JSON\*.{pid }.json")
        or _g .glob (f"_ds1/DataSet/*.{pid }_*.json"))
        if not stp or not jf :
            print (f"{pid }: dosya yok");continue 
        stp ,jf =stp [0 ],jf [0 ]
        Vr ,Fr =step_to_mesh (stp )
        V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        mesh =trimesh .Trimesh (vertices =V ,faces =F ,process =False )
        boyut =V .max (0 )-V .min (0 )

        pbs =[]
        for model ,meta in models :
            _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
            op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",
            return_probs =True )
            pbs .append (np .asarray (pb ,float ))
        cps ,probs ,is_hi ,uyeler =RC .derive_candidates (V ,F ,pbs ,stp ,cfg =cfg )
        X =wire_gate .feats_for (V ,F ,probs ,cps ,CE ,CT ,step_path =stp )
        gate =wire_gate ._load ()
        k =wire_gate .decision_mask (wire_gate .decision_score (gate ,X ))
        c =[{"point":np .asarray (cps [i ]["point"],float ),
        "direction":np .asarray (cps [i ]["direction"],float )}for i in np .where (k )[0 ]]
        Xk =X [k ]
        c =wire_gate .pose_correct (Xk ,c )
        if cfg .get ("robot_aci_secici"):
            c =wire_gate .angle_correct (Xk ,c )
        if cfg .get ("robot_uye_secici")and uyeler :
            c =wire_gate .pick_member_direction (Xk ,c ,uyeler )
        if cfg .get ("robot_yon_secici"):
            c =wire_gate .pick_direction_from_dictionary (Xk ,c ,V ,step_path =stp ,uyeler =uyeler )
        P =np .array ([x ["point"]for x in c ],float )
        Dd =np .array ([x ["direction"]for x in c ],float )

        j =json .load (io .open (jf ,encoding ="utf-8-sig"))
        g =j .get ("ConnectionPoints")or []
        G =np .array ([[q ["Point"][a ]for a in "XYZ"]for q in g ],float )
        Gd =np .array ([[q ["InsertDirection"][a ]for a in "XYZ"]for q in g ],float )
        Gd /=(np .linalg .norm (Gd ,axis =1 ,keepdims =True )+1e-9 )
        Vj =np .array ([[q ["X"],q ["Y"],q ["Z"]]for q in j ["Graphic3d"]["Points"]],float )
        R ,t_ ,_ =cad_eval .align_frames (Vr ,Vj )
        G =(G -t_ )@R ;Gd =Gd @R 

        # --- H1: URETICI SEAT'inin KENDI AGZI (tezle same tanim)
        Gm =[]
        for i in range (len (G )):
            try :
                m_ ,off =seat_to_mouth (mesh ,G [i ],Gd [i ])
                Gm .append ((np .asarray (m_ ,float ),float (off )))
            except Exception :
                Gm .append ((G [i ],0.0 ))
        Gmouth =np .array ([m for m ,_ in Gm ],float )
        seat_off =np .array ([o for _ ,o in Gm ],float )

        line_ =[]
        for i in range (len (P )):
            p ,d =P [i ],Dd [i ]
            # most yakin GT (seat) -- metrigin kullandigi criterion
            diff =p -G ;al =(diff *Gd ).sum (1 )
            perp =np .linalg .norm (diff -al [:,None ]*Gd ,axis =1 )
            b =int (np .argmin (np .where (np .abs (al )<=40 ,perp ,np .inf )))if len (G )else -1 
            # same GT'nin AGZINA uzaklik -- H1'in sinavi
            db =p -Gmouth [b ]
            al_m =float (db @Gd [b ]);perp_m =float (np .linalg .norm (db -al_m *Gd [b ]))
            line_ .append ({
            "cp":i +1 ,
            "perp_seat":float (perp [b ]),"axial_seat":float (al [b ]),
            "perp_agiz":perp_m ,"axial_agiz":al_m ,
            "seat_offset":float (seat_off [b ]),
            "aci_isaretli":float (np .degrees (np .arccos (np .clip (float (d @Gd [b ]),-1 ,1 )))),
            "iceride_mi":bool (is_inside (mesh ,p )),
            "serbest_ileri":serbest (mesh ,p +1e-3 *d ,d ),
            "serbest_geri":serbest (mesh ,p -1e-3 *d ,-d ),
            # mouth_width -> (ic_cap, ort_cap). Nokta govdenin DISINDAYSA (0.0, 0.0) returns:
            # "olculemedi" demektir, "width sifir" DEMEZ. Ikisi ayri okunur.
            "agiz_ic_cap":float (mouth_width (mesh ,p ,d )[0 ]),
            "agiz_ort_cap":float (mouth_width (mesh ,p ,d )[1 ]),
            "en_yakin_tepe_mm":float (np .linalg .norm (V -p ,axis =1 ).min ()),
            })
        R_ ={"part":pid ,"boyut_mm":[round (float (x ),1 )for x in boyut ],
        "gt":len (G ),"tahmin":len (P ),
        "seat_offset_medyan":float (np .median (np .abs (seat_off ))),
        "cp":line_ }
        RAPOR [pid ]=R_ 

        print (f"\n{'='*96 }\n{pid }  boyut {boyut .round (1 )}  GT {len (G )}  tahmin {len (P )}")
        print (f"manufacturer seat'inin KENDI agzina uzakligi (medyan): {R_ ['seat_offset_medyan']:.1f}mm")
        print (f"{'CP':>3}{'perp_seat':>10}{'ax_seat':>9}{'perp_AGIZ':>11}{'ax_AGIZ':>9}"
        f"{'aci':>7}{'ici?':>6}{'ileri':>8}{'geri':>8}{'ic_cap':>9}{'ort_cap':>9}")
        for s in line_ :
            print (f"{s ['cp']:>3}{s ['perp_seat']:>10.2f}{s ['axial_seat']:>9.2f}"
            f"{s ['perp_agiz']:>11.2f}{s ['axial_agiz']:>9.2f}{s ['aci_isaretli']:>7.1f}"
            f"{str (s ['iceride_mi']):>6}{s ['serbest_ileri']:>8.1f}{s ['serbest_geri']:>8.1f}"
            f"{s ['agiz_ic_cap']:>9.2f}{s ['agiz_ort_cap']:>9.2f}")

    with io .open ("results/otopsi_cp.json","w",encoding ="utf-8")as f :
        json .dump (RAPOR ,f ,indent =1 )
    print ("\nmakbuz -> results/otopsi_cp.json")
    print ("\nOKUMA KILAVUZU:")
    print ("  perp_AGIZ ~ perp_seat AMA ax_AGIZ ~ 0  -> H1 DOGRU (tanim farki, error not)")
    print ("  ax_AGIZ still large                     -> H1 YANLIS, real yerlesim hatasi")
    print ("  serbest_ileri small / geri large       -> direction TERS (H2)")
    print ("  iceride_mi True                        -> point body inside (H3)")


if __name__ =="__main__":
    import traceback 
    try :
        main ()
    except BaseException :
        traceback .print_exc ();raise 
