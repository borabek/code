# -*- coding: utf-8 -*-
"""R1b: spread EKSENEL mi YANAL mi? Cozumu this belirler.

YANAL whereas  -> cluster_mm buyutmek is required AMA komsu kutuplari merges (step 3.5-6mm) = tehlikeli
EKSENEL whereas-> cozum EKSEN-FARKINDALIKLI oy havuzlamasi: dik distance kucukse same opening say,
              depth farkini YOK SAY. (big_arbiter puanlayicisi ZATEN boyle calisiyor;
              _vote2 whereas duz Oklid kullaniyor -- urunun own inside tutarsizlik.)
"""
import io ,json ,os ,sys ,time 
import numpy as np ,torch 
os .environ .setdefault ("BA_ALLOW_SEEN","1");sys .path .insert (0 ,".")
import cp_openings ,diffusionnet as D_ ,measure_set ,thesis_remesh 
from big_arbiter import eligible 
from connector_constants import CABLE_ENTRY as CE ,CONTACT as CT 
from infer_step_cp import load_any ,step_to_mesh 
cfg =json .load (io .open ("cp_config.json",encoding ="utf-8"))
pp =cfg .get ("prediction_postproc",{})
mv ,vc ,cl =int (pp .get ("min_vertices",4 )),float (pp .get ("vertex_confidence_mask",0.3 )),float (pp .get ("cluster_mm",3.0 ))
stp ={p :s for m ,p ,jf ,s in eligible ()}
DER ,rap =measure_set .cluster ("results/_der_tam.pkl");measure_set .rapor_bas (rap )
rng =np .random .default_rng (0 )
sec =[DER [i ]for i in rng .permutation (len (DER ))[:45 ]if len (DER [i ]["G"])]
cks =cfg ["current_product"].get ("checkpoints")or cfg ["robot_vote2_checkpoints"]
dev ="cuda"if torch .cuda .is_available ()else "cpu"
models =[load_any (c ,dev =dev )[:2 ]for c in cks ]
YAN ,EKS =[],[]
t0 =time .time ()
for k ,r in enumerate (sec ,1 ):
    if k %15 ==0 :print (f"  {k }/{len (sec )} {time .time ()-t0 :.0f}s",flush =True )
    try :
        Vr ,Fr =step_to_mesh (stp [r ["pid"]]);V ,F =thesis_remesh .remesh_uniform (Vr ,Fr ,target =6000 )
        V =np .ascontiguousarray (V ,np .float64 );F =np .ascontiguousarray (F ,np .int64 )
        UYE =[]
        for model ,meta in models :
            _ ,pb =D_ .predict (model ,meta ,V ,F ,device =dev ,
            op_cache_dir =f"results/step_infer/ops_k{int (meta .get ('k_eig',64 ))}",return_probs =True )
            q =np .asarray (pb ,float )
            lst =cp_openings .connection_points (V ,F ,q .argmax (-1 ),min_v =mv ,classes =(CE ,CT ),
            dedupe_mm =10.0 ,probs =q ,vertex_conf =vc ,ct_depth_min_mm =1.0 ,cluster_mm =cl ,
            step_path =stp [r ["pid"]])
            UYE .append (np .array ([c ["point"]for c in lst ],float )if lst else np .zeros ((0 ,3 )))
    except Exception :continue 
    G =np .asarray (r ["G"],float );Gd =np .asarray (r ["Gd"],float )
    for b in range (len (G )):
        yak =[]
        for P in UYE :
            if not len (P ):continue 
            v =P -G [b ];al =v @Gd [b ]
            pe =np .linalg .norm (v -al [:,None ]*Gd [b ],axis =1 );pe =np .where (np .abs (al )>40 ,np .inf ,pe )
            i =int (np .argmin (pe ))
            if np .isfinite (pe [i ])and pe [i ]<=max (3.0 ,0.06 *float (r ["diag"])):yak .append (P [i ])
        if len (yak )<2 :continue 
        Y =np .array (yak );d =Gd [b ]
        for a in range (len (Y )):
            for c in range (a +1 ,len (Y )):
                w =Y [a ]-Y [c ]
                EKS .append (abs (float (w @d )))# axis along
                YAN .append (float (np .linalg .norm (w -(w @d )*d )))# eksene DIK
YAN =np .array (YAN );EKS =np .array (EKS )
print (f"\n{len (YAN )} uye cifti (same GT'yi bulan)")
print (f"{'olcu':<22}{'medyan':>9}{'%75':>9}{'%90':>9}{'>5mm':>8}")
print (f"{'EKSENEL (depth)':<22}{np .median (EKS ):>9.2f}{np .percentile (EKS ,75 ):>9.2f}{np .percentile (EKS ,90 ):>9.2f}{float ((EKS >5 ).mean ()):>8.1%}")
print (f"{'YANAL (dik)':<22}{np .median (YAN ):>9.2f}{np .percentile (YAN ,75 ):>9.2f}{np .percentile (YAN ,90 ):>9.2f}{float ((YAN >5 ).mean ()):>8.1%}")
print ()
print (f"EKSEN-FARKINDALIKLI cluster (lateral<=3mm, depth SERBEST) with birlesecek cift: {float ((YAN <=3.0 ).mean ()):.1%}")
print (f"DUZ OKLID 5mm with birlesen cift                                            : {float ((np .sqrt (YAN **2 +EKS **2 )<=5.0 ).mean ()):.1%}")
json .dump ({"n":len (YAN ),"eksenel_p50":float (np .median (EKS )),"eksenel_p90":float (np .percentile (EKS ,90 )),
"yanal_p50":float (np .median (YAN )),"yanal_p90":float (np .percentile (YAN ,90 )),
"eksen_farkindalikli_birlesme":float ((YAN <=3.0 ).mean ()),
"duz_oklid_birlesme":float ((np .sqrt (YAN **2 +EKS **2 )<=5.0 ).mean ())},
io .open ("results/r1b_eksenel_yanal.json","w"),indent =1 )
print ("receipt -> results/r1b_eksenel_yanal.json")
