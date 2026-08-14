# -*- coding: utf-8 -*-
"""URUNUN TAM POZ ZINCIRI -- measurement betikleri bunu cagirir.

BULUNAN DEFECT (2026-08-07): butun D6/D7 olcumlerim `derive_candidates` + gate with duruyordu.
Oysa urunun real yolu (`robot_cp.extract`) gate'ten SONRA two step more kosuyor:

    pose_correct      lateral deviation duzeltmesi   -> robot +0.0148 (D6'da measured)
    pick_direction_from_dictionary   fiziksel direction SOZLUGU     -> robot +0.0218 (ustune)

Yani olctugum 0.2103, urunun real 0.2469'unun altindaydi. Bu modul that farki kapatir:
each measurement betigi same fonksiyonu cagirir, so "measured_path sey urunun YAPTIGI sey" becomes
([[measurement-zaafiyetleri-kapatildi]] with same ilke).

TEZ DEGISMEZ: two step da SON ISLEM; `v_o` turetmesi, 5 sinif and ~6000 remesh aynen kalir.
Ham `v_o` istenirse `duzelt=False` with alinir and yan yana raporlanabilir.
"""
import numpy as np 


def tam_poz (V ,F ,avg_probs ,P ,D ,step_path =None ,uyeler =None ,cfg =None ,
pose_head =True ,yon_sozluk =True ):
    """Gate'ten gecmis (P, D) ciftlerine urunun poz zincirini uygula.

    Doner: (P2, D2). Herhangi a step duserse GIRDI aynen returns -- uydurma absent.
    """
    import connector3d 
    import wire_gate as _wg 
    CE =int (connector3d .CABLE_ENTRY );CT =int (connector3d .CONTACT )
    if not len (P ):
        return P ,D 
    cps =[{"point":np .asarray (P [i ],float ).tolist (),
    "direction":np .asarray (D [i ],float ).tolist (),
    "confidence":0.8 ,"cls":CE ,"_votes":2 }for i in range (len (P ))]
    try :
        Xp =_wg .feats_for (V ,F ,avg_probs ,cps ,CE ,CT ,step_path =step_path )
    except Exception :
        return P ,D 
    if pose_head :
        try :
            cps =_wg .pose_correct (Xp ,cps )
        except Exception :
            pass 
    if yon_sozluk :
        try :
            cps =_wg .pick_direction_from_dictionary (Xp ,cps ,V ,step_path =step_path ,uyeler =uyeler )
        except Exception :
            pass 
    P2 =np .asarray ([c ["point"]for c in cps ],float )
    D2 =np .asarray ([c ["direction"]for c in cps ],float )
    return P2 ,D2 
