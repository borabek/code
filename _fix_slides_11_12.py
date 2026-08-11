# -*- coding: utf-8 -*-
"""Slayt 11 ve 12'yi MEVCUT dosya uzerinde duzelt -- yeniden uretme YOK.

NEDEN DOSYA UZERINDE: slayt 12'deki iki resim kaynak koda (_deck_content.py) hic girmemis,
PowerPoint'te elle eklenmis. Kaynaktan yeniden uretmek onlari SILERDI.

OLCULEN KUSURLAR:
  Slayt 11 -- hucrelerde fazladan BOS PARAGRAFLAR var; "Low CP density" baslik hucresi 4 satira
    tasip 1.19in yer kapliyor, oysa satir araligi 0.36in. Sonuc: baslik altindaki UC satirin
    uzerine biniyor ("high/low CP yazilari alta kaymis").
  Slayt 12 -- (a) metin bloklari resimlerle CAKISIYOR (sol metin 1.95-3.61, resim 3.28'de
    basliyor), (b) resimler kucuk (3.38x1.33in), (c) ikinci resim ORANI BOZULARAK sikistirilmis
    (dogal 2.544, slaytta 1.991).
"""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt

SRC = "ConnectionPointDetector_Bora_Bayrakci.pptx"
IN = Inches


def strip_blank_tail(sh):
    """Hucre sonundaki bos paragraflari at -- yukseklik tasmasinin kaynagi."""
    if not sh.has_text_frame:
        return 0
    ps = sh.text_frame.paragraphs
    n = 0
    while len(ps) > 1 and not ps[-1].text.strip():
        ps[-1]._p.getparent().remove(ps[-1]._p)
        ps = sh.text_frame.paragraphs
        n += 1
    return n


def set_font(sh, size=None):
    if not sh.has_text_frame or size is None:
        return
    for p in sh.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(size)


def main():
    prs = Presentation(SRC)

    # ---------------- SLAYT 11 ----------------
    s = prs.slides[10]
    removed = sum(strip_blank_tail(sh) for sh in s.shapes)

    # Baslik satirina IKI satirlik gercek yer ver, veri satirlarini asagi kaydirip esit araliga otur.
    HEAD_T, HEAD_H = 1.58, 0.66
    ROW_T0, ROW_DY, ROW_H = 2.36, 0.42, 0.40
    for sh in s.shapes:
        t = round(sh.top / 914400, 2)
        if abs(t - 1.60) < 0.05:                      # baslik hucreleri
            sh.top, sh.height = IN(HEAD_T), IN(HEAD_H)
            set_font(sh, 13)
        elif abs(t - 1.92) < 0.05:                    # baslik alti ayrac cizgisi
            sh.top = IN(HEAD_T + HEAD_H + 0.06)
        else:
            for k, old in enumerate((2.02, 2.38, 2.74, 3.10, 3.46)):
                if abs(t - old) < 0.05:
                    sh.top = IN(ROW_T0 + k * ROW_DY)
                    sh.height = IN(ROW_H)
                    set_font(sh, 13)
                    break
    # Tablo 4.34'te bitiyor; madde blogunu nefes payiyla ayir.
    for sh in s.shapes:
        if abs(round(sh.top / 914400, 2) - 4.95) < 0.05:
            sh.top = IN(4.72)
            sh.height = IN(2.05)

    # ---------------- SLAYT 12 ----------------
    s = prs.slides[11]
    for sh in s.shapes:
        strip_blank_tail(sh)

    pics = [sh for sh in s.shapes if sh.shape_type == 13]
    pics.sort(key=lambda sh: sh.left)

    COL_L, COL_R, COL_W = 0.50, 6.78, 6.05
    HEAD_T = 1.34
    PIC_T = 1.74
    RATIO = 2.545                                    # her iki resmin DOGAL orani (2.547 / 2.544)
    PIC_W, PIC_H = COL_W, COL_W / RATIO              # 6.05 x 2.38 -- eskisinin ~3.2 kati alan
    BUL_T = PIC_T + PIC_H + 0.16                     # 4.28
    BUL_H = 1.30
    FOOT_T = BUL_T + BUL_H + 0.14                    # 5.72

    for sh in s.shapes:
        t = round(sh.top / 914400, 2)
        l = round(sh.left / 914400, 2)
        if sh.shape_type == 13:
            continue
        if abs(t - 1.50) < 0.05:                     # sutun basliklari
            sh.top, sh.height = IN(HEAD_T), IN(0.36)
            sh.left = IN(COL_L if l < 3 else COL_R)
            sh.width = IN(COL_W)
        elif abs(t - 1.95) < 0.05:                   # sutun maddeleri -> RESIMLERIN ALTINA
            sh.top, sh.height = IN(BUL_T), IN(BUL_H)
            sh.left = IN(COL_L if l < 3 else COL_R)
            sh.width = IN(COL_W)
            set_font(sh, 13)
        elif abs(t - 4.55) < 0.05:                   # "I show the failure on purpose" basligi
            sh.top, sh.height = IN(FOOT_T), IN(0.32)
            sh.left, sh.width = IN(COL_L), IN(12.33)
        elif abs(t - 4.95) < 0.05:                   # alt maddeler
            sh.top, sh.height = IN(FOOT_T + 0.36), IN(0.88)
            sh.left, sh.width = IN(COL_L), IN(12.33)
            set_font(sh, 12.5)

    for i, sh in enumerate(pics):                    # ORAN KORUNARAK buyut
        sh.left = IN(COL_L if i == 0 else COL_R)
        sh.top = IN(PIC_T)
        sh.width = IN(PIC_W)
        sh.height = IN(PIC_H)

    prs.save(SRC)
    print(f"kaydedildi: {SRC}")
    print(f"  slayt 11: {removed} bos paragraf temizlendi, satirlar {ROW_T0}in'den {ROW_DY}in araliga oturdu")
    print(f"  slayt 12: resimler {PIC_W:.2f}x{PIC_H:.2f}in (eski 3.38x1.33 / 2.64x1.33), oran {RATIO} KORUNDU")


if __name__ == "__main__":
    main()
