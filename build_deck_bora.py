# -*- coding: utf-8 -*-
"""Build a NEW 12-slide deck from scratch (English, percentage metrics, light text).

Branding is reused from the existing deck: the bottom banner (five segment tabs +
FRIEDHELM LOH GROUP + date/author + page number) is copied shape-for-shape, so the new
deck keeps the corporate look without re-creating it.

Every metric ten these slides comes from a measurement receipt in results/, not from the
previous deck. See NUMBERS at the bottom of this file for the source of each figure.
"""
import copy ,os 
from pptx import Presentation 
from pptx .util import Inches ,Pt 
from pptx .dml .color import RGBColor 
from pptx .enum .text import PP_ALIGN 

SRC ="ConnectionPointDetector_Presentation_v3.pptx"
OUT =os .environ .get ("DECK_OUT","ConnectionPointDetector_Bora_Bayrakci.pptx")
AUTHOR ="Bora Bayrakci"
FOOTER =f"WiringRobot.ConnectionPointDetector  |  {AUTHOR }  |  R&D RAS  |  29.07.2026"

NAVY =RGBColor (0x1F ,0x4E ,0x79 )
CYAN =RGBColor (0x00 ,0x90 ,0xD0 )
GREY =RGBColor (0x40 ,0x40 ,0x40 )
DARK =RGBColor (0x2D ,0x2D ,0x2D )
LIGHT =RGBColor (0x90 ,0x90 ,0x90 )
GREEN =RGBColor (0x1E ,0x88 ,0x3C )
RED =RGBColor (0xC0 ,0x39 ,0x2B )
WHITE =RGBColor (0xFF ,0xFF ,0xFF )


def txt (slide ,l ,t ,w ,h ,text ,size ,color =GREY ,bold =False ,align =PP_ALIGN .LEFT ):
    b =slide .shapes .add_textbox (Inches (l ),Inches (t ),Inches (w ),Inches (h ))
    tf =b .text_frame 
    tf .word_wrap =True 
    p =tf .paragraphs [0 ]
    p .alignment =align 
    r =p .add_run ();r .text =text 
    r .font .size =Pt (size );r .font .bold =bold ;r .font .color .rgb =color 
    r .font .name ="Calibri"
    return b 


def bullets (slide ,l ,t ,w ,h ,items ,size =15 ,gap =6 ):
    """items: list of str, or (marker, text) or (marker, text, color)."""
    b =slide .shapes .add_textbox (Inches (l ),Inches (t ),Inches (w ),Inches (h ))
    tf =b .text_frame 
    tf .word_wrap =True 
    first =True 
    for it in items :
        if isinstance (it ,tuple ):
            lead ,body =it [0 ],it [1 ]
            lcol =it [2 ]if len (it )>2 else DARK 
        else :
            lead ,body ,lcol ="",it ,DARK 
        p =tf .paragraphs [0 ]if first else tf .add_paragraph ()
        first =False 
        p .space_after =Pt (gap )
        # HER madde gorunur a im with baslar. Onceki surumde kalin giris ifadesi imin YERINE
        # geciyordu, therefore that maddeler duz paragraf like duruyordu.
        rb =p .add_run ();rb .text ="•  "
        rb .font .size =Pt (size );rb .font .bold =True ;rb .font .color .rgb =CYAN 
        rb .font .name ="Calibri"
        if lead .strip ():
            rl =p .add_run ();rl .text =lead 
            rl .font .size =Pt (size );rl .font .bold =True ;rl .font .color .rgb =lcol 
            rl .font .name ="Calibri"
        r2 =p .add_run ();r2 .text =body 
        r2 .font .size =Pt (size );r2 .font .color .rgb =GREY ;r2 .font .name ="Calibri"
    return b 


def table (slide ,l ,t ,w ,rows ,col_w ,head_size =13 ,size =14 ,row_h =0.36 ):
    """Light table: header row + data rows, no gridlines -- just aligned text + rules."""
    y =t 
    for i ,row in enumerate (rows ):
        x =l 
        for j ,cell in enumerate (row ):
            is_head =(i ==0 )
            col =NAVY if is_head else GREY 
            bold =is_head 
            if not is_head and j >0 and isinstance (cell ,str )and cell .endswith ("%"):
                bold =True 
                col =DARK 
            txt (slide ,x ,y ,col_w [j ],row_h ,str (cell ),
            head_size if is_head else size ,col ,bold )
            x +=col_w [j ]
        if i ==0 :
            ln =slide .shapes .add_shape (1 ,Inches (l ),Inches (y +row_h -0.04 ),
            Inches (w ),Inches (0.02 ))
            ln .fill .solid ();ln .fill .fore_color .rgb =NAVY 
            ln .line .fill .background ();ln .shadow .inherit =False 
            y +=0.06 
        y +=row_h 
    return y 


    # ------------------------------------------------------------------------------------------------
src =Presentation (SRC )
banner_xml =[copy .deepcopy (sh ._element )for sh in src .slides [1 ].shapes 
if sh .name in ("Rectangle 14","Up Arrow Callout 15","Up Arrow Callout 16",
"Up Arrow Callout 17","Up Arrow Callout 18","Up Arrow Callout 19",
"TextBox 20")]
title_bg =[copy .deepcopy (sh ._element )for sh in src .slides [0 ].shapes 
if sh .name in ("Rectangle 1",)]

prs =Presentation (SRC )
# Bos desteden basla but tema/master'i koru. NOTE: only sldIdLst'ten silmek yetmez --
# slayt PARCALARI pakette kalir and kayit during 'Duplicate name: slide1.xml' produces; PowerPoint
# boyle a dosyayi bozuk sayabilir. Iliskiyi de dusurmek is required.
for sid in list (prs .slides ._sldIdLst ):
    prs .part .drop_rel (sid .rId )
    prs .slides ._sldIdLst .remove (sid )
BLANK =prs .slide_masters [0 ].slide_layouts [6 ]

_page =[0 ]


def new_slide (section =None ,headline =None ,dark_bg =False ):
    s =prs .slides .add_slide (BLANK )
    _page [0 ]+=1 
    if dark_bg :
        for el in title_bg :
            s .shapes ._spTree .append (copy .deepcopy (el ))
    for el in banner_xml :
        s .shapes ._spTree .append (copy .deepcopy (el ))
        # ALTBILGI: bant RITTAL KIRMIZISI (C8102E). Uzerine open gri yazmak metni absent ediyordu --
        # orijinal deste beyaz kullaniyor. Ad SOYAD kalin and large harf, so pasif kalmiyor.
    fb =s .shapes .add_textbox (Inches (3.05 ),Inches (7.31 ),Inches (7.70 ),Inches (0.17 ))
    ftf =fb .text_frame ;ftf .word_wrap =False 
    fp =ftf .paragraphs [0 ]
    for piece ,is_bold in (("WiringRobot.ConnectionPointDetector",False ),("   |   ",False ),
    (AUTHOR .upper (),True ),("   |   ",False ),
    ("R&D RAS  ·  29.07.2026",False )):
        r =fp .add_run ();r .text =piece 
        r .font .size =Pt (9 if is_bold else 8 );r .font .bold =is_bold 
        r .font .color .rgb =WHITE ;r .font .name ="Calibri"
    txt (s ,12.85 ,7.32 ,0.40 ,0.16 ,str (_page [0 ]),8 ,WHITE )
    if section :
        txt (s ,0.50 ,0.25 ,9.00 ,0.35 ,section ,12 ,DARK ,True )
    if headline :
        txt (s ,0.50 ,0.68 ,12.30 ,0.60 ,headline ,26 ,NAVY ,True )
    return s 



    # ---- slide copy lives in _deck_content.py (sales voice) -----------------------------------------
import _deck_content 
_deck_content .build ({
"new_slide":new_slide ,"txt":txt ,"bullets":bullets ,"table":table ,
"NAVY":NAVY ,"CYAN":CYAN ,"GREY":GREY ,"DARK":DARK ,"LIGHT":LIGHT ,
"GREEN":GREEN ,"RED":RED ,"AUTHOR":AUTHOR ,
})

prs .save (OUT )
print (f"{OUT } written -- {len (prs .slides )} slides")
