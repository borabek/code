# -*- coding: utf-8 -*-
"""CIZIM SOZLESMESI -- ciziciyle denetcinin PAYLASTIGI single real kaynagi.

robot_viz.py (cizen) and glb_audit.py (denetleyen) this dosyadan reads. Renkleri two places ELLE
yazmak, sozlesmeyi anlamsiz kilar: cizici wrong renk kullansa denetci de same yanlisi bekler.

See. CIZIM_SOZLESMESI.md -- 9 madde and ihlal politikasi.
"""
import numpy as np 

# --- M6'nin referansi: renk = eslesme durumu -----------------------------------------------------
VIZ_COLORS ={
"body":(185 ,185 ,190 ),# parcanin kendisi
"gt_hit":(0 ,210 ,0 ),# YESIL  manufacturer CP, robot buldu
"gt_miss":(255 ,210 ,0 ),# SARI   manufacturer CP, robot kacirdi (FN)
"pred_ok":(220 ,0 ,0 ),# KIRMIZI robot tahmini, correct (TP)
"pred_fp":(255 ,0 ,200 ),# MOR    robot tahmini, fazladan (FP)
"link":(0 ,120 ,255 ),# MAVI   eslesme cizgisi
}
RGBA ={k :list (v )+[255 ]for k ,v in VIZ_COLORS .items ()}

# hangi renk hangi kaynaktan -- denetci M1/M2/M6 for kullanir
GT_ROLES =("gt_hit","gt_miss")
PRED_ROLES =("pred_ok","pred_fp")

# --- IKI MOD ------------------------------------------------------------------------------------
# "robot_only" (VARSAYILAN, urun gorunumu): SADECE robotun koydugu CP'ler cizilir, all of them same
#   kirmizi. Uretici isaretcisi and eslesme cizgisi YOKTUR. Gercek kullanimda (gorulmemis WSCAD
#   parcasi) manufacturer CP'si already yoktur; this mod GT olmadan da works.
# "compare" (teshis): yesil/sari/kirmizi/mor + mavi line -- only bizim error analizimiz for.
MODES =("robot_only","compare")
ROBOT_ROLE ="pred_ok"# robot_only modunda TUM robot CP'leri this renkte (kirmizi)

# Bir isaretcinin kac BAGLI BILESENDEN olustugu. Cizici and denetci same sayiyi bilmek
# ZORUNDA: denetci bilesenleri sayip isaretci adedini bundan turetiyor.
#   kure (CP noktasinin kendisi) + silindir body + konik three = 3
# 2026-07-29: ok ucundaki kure koniyle degistirilince CP noktasi unsigned kalmisti; kure
# geri kondu, this times OKUN UCUNA not CP'NIN AGZINA -- robotun gidecegi yer orasi.
COMPONENTS_PER_MARKER =3 

# Bir isaretcinin tabani, ait oldugu CP'den EN FAZLA this up to uzaga tasinabilir.
#
# DEGER FIZIKTEN GELIYOR, tahminden not: manufacturer CP'si kontak YUVASINDA oturur and yuzeydeki
# agizdan 5-25 mm iceridedir (cp_geometry.seat_to_mouth, olculmus aralik). Yuva->mouth hareketi
# that is why 25 mm'ye up to MESRUDUR. Ustune emniyet payi: 30 mm.
#
# Once 8.0 was tried and YANLISTI: mesru derin CP'lerin agza tasinmasini da engelledi, oklar
# govdenin inside kaldi (measured: 4/19, 6/40, 6/17 three iceride). Sinir, gozle secilen a number
# not, parcanin fiziginden okunan a number must be.
#
# Cizici bunu SINIR, denetci same sayiyi M9 TOLERANSI as kullanir -- single source, aksi halde
# ikisi ayrisir and cizici denetcinin reddedecegi dosyayi uretmeye devam eder.
MAX_MARKER_OFFSET_MM =30.0 

RECEIPT_VERSION =2 


def classify_colors (colors ):
    """Nx4 (or Nx3) vertex rengi -> each vertex for rol adi ('body','gt_hit',... or None).

    Tam esitlik aranir; GLB'de renkler uint8 as korunur. Eslesmeyen vertex None takes and this
    denetimde 'bilinmeyen renk' ihlali sayilir.
    """
    c =np .asarray (colors )[:,:3 ].astype (int )
    out =np .full (len (c ),None ,dtype =object )
    for role ,rgb in VIZ_COLORS .items ():
        m =(c [:,0 ]==rgb [0 ])&(c [:,1 ]==rgb [1 ])&(c [:,2 ]==rgb [2 ])
        out [m ]=role 
    return out 


def receipt_path (glb_path ):
    return str (glb_path ).rsplit (".glb",1 )[0 ]+".receipt.json"
