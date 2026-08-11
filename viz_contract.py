# -*- coding: utf-8 -*-
"""CIZIM SOZLESMESI -- ciziciyle denetcinin PAYLASTIGI tek gercek kaynagi.

robot_viz.py (cizen) ve glb_audit.py (denetleyen) bu dosyadan okur. Renkleri iki yerde ELLE
yazmak, sozlesmeyi anlamsiz kilar: cizici yanlis renk kullansa denetci de ayni yanlisi bekler.

Bkz. CIZIM_SOZLESMESI.md -- 9 madde ve ihlal politikasi.
"""
import numpy as np

# --- M6'nin referansi: renk = eslesme durumu -----------------------------------------------------
VIZ_COLORS = {
    "body":     (185, 185, 190),   # parcanin kendisi
    "gt_hit":   (0, 210, 0),       # YESIL  uretici CP, robot buldu
    "gt_miss":  (255, 210, 0),     # SARI   uretici CP, robot kacirdi (FN)
    "pred_ok":  (220, 0, 0),       # KIRMIZI robot tahmini, dogru (TP)
    "pred_fp":  (255, 0, 200),     # MOR    robot tahmini, fazladan (FP)
    "link":     (0, 120, 255),     # MAVI   eslesme cizgisi
}
RGBA = {k: list(v) + [255] for k, v in VIZ_COLORS.items()}

# hangi renk hangi kaynaktan -- denetci M1/M2/M6 icin kullanir
GT_ROLES = ("gt_hit", "gt_miss")
PRED_ROLES = ("pred_ok", "pred_fp")

# --- IKI MOD ------------------------------------------------------------------------------------
# "robot_only" (VARSAYILAN, urun gorunumu): SADECE robotun koydugu CP'ler cizilir, hepsi ayni
#   kirmizi. Uretici isaretcisi ve eslesme cizgisi YOKTUR. Gercek kullanimda (gorulmemis WSCAD
#   parcasi) uretici CP'si zaten yoktur; bu mod GT olmadan da calisir.
# "compare" (teshis): yesil/sari/kirmizi/mor + mavi cizgi -- yalniz bizim hata analizimiz icin.
MODES = ("robot_only", "compare")
ROBOT_ROLE = "pred_ok"          # robot_only modunda TUM robot CP'leri bu renkte (kirmizi)

# Bir isaretcinin kac BAGLI BILESENDEN olustugu. Cizici ve denetci ayni sayiyi bilmek
# ZORUNDA: denetci bilesenleri sayip isaretci adedini bundan turetiyor.
#   kure (CP noktasinin kendisi) + silindir govde + konik uc = 3
# 2026-07-29: ok ucundaki kure koniyle degistirilince CP noktasi isaretsiz kalmisti; kure
# geri kondu, bu kez OKUN UCUNA degil CP'NIN AGZINA -- robotun gidecegi yer orasi.
COMPONENTS_PER_MARKER = 3

# Bir isaretcinin tabani, ait oldugu CP'den EN FAZLA bu kadar uzaga tasinabilir.
#
# DEGER FIZIKTEN GELIYOR, tahminden degil: uretici CP'si kontak YUVASINDA oturur ve yuzeydeki
# agizdan 5-25 mm iceridedir (cp_geometry.seat_to_mouth, olculmus aralik). Yuva->agiz hareketi
# bu yuzden 25 mm'ye kadar MESRUDUR. Ustune emniyet payi: 30 mm.
#
# Once 8.0 denendi ve YANLISTI: mesru derin CP'lerin agza tasinmasini da engelledi, oklar
# govdenin icinde kaldi (olculdu: 4/19, 6/40, 6/17 uc iceride). Sinir, gozle secilen bir sayi
# degil, parcanin fiziginden okunan bir sayi olmali.
#
# Cizici bunu SINIR, denetci ayni sayiyi M9 TOLERANSI olarak kullanir -- tek kaynak, aksi halde
# ikisi ayrisir ve cizici denetcinin reddedecegi dosyayi uretmeye devam eder.
MAX_MARKER_OFFSET_MM = 30.0

RECEIPT_VERSION = 2


def classify_colors(colors):
    """Nx4 (veya Nx3) vertex rengi -> her vertex icin rol adi ('body','gt_hit',... veya None).

    Tam esitlik aranir; GLB'de renkler uint8 olarak korunur. Eslesmeyen vertex None alir ve bu
    denetimde 'bilinmeyen renk' ihlali sayilir.
    """
    c = np.asarray(colors)[:, :3].astype(int)
    out = np.full(len(c), None, dtype=object)
    for role, rgb in VIZ_COLORS.items():
        m = (c[:, 0] == rgb[0]) & (c[:, 1] == rgb[1]) & (c[:, 2] == rgb[2])
        out[m] = role
    return out


def receipt_path(glb_path):
    return str(glb_path).rsplit(".glb", 1)[0] + ".receipt.json"
