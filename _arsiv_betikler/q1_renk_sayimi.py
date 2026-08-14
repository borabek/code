# -*- coding: utf-8 -*-
"""Q1: STEP face-basina RENK kanali BOS mu?

BAGLAM: kayitli bulgu, AP214 renginin face basina ayristirilabildigi (OVER_RIDING_STYLED_ITEM,
108/108 face) but that parcada TUM yuzlerin same gumusi rengi tasidigi. Entegrasyon da bloke:
gmsh rengi atiyor, order-eslemesi null testte cakti, XCAF 0 lower-shape etiketi veriyor.

SIRALAMA MANTIGI: "yuzleri geometriyle eslestiren a matcher yaz" pahali a istir. Ondan ONCE
kanalda BILGI olup olmadigi olculur -- eslestirmeye gerek YOK, because a parcadaki TUM yuzler
same renkteyse hangi yuzun hangi renk oldugu already onemsizdir.

KILL (onceden yazildi): parcalarin %10'undan azinda >=2 ayrik renk varsa channel BOS ilan edilir
and matcher isi ACILMAZ. >=2 renk yayginsa, renk metal kontagi plastik govdeden ayiriyor may be
-- that zaman matcher mesru a yatirimdir.
"""
import os ,sys ,re ,json ,glob ,collections 

sys .path .insert (0 ,os .path .dirname (os .path .abspath (__file__ )))
# AP214 renk tanimi: COLOUR_RGB('',r,g,b)
RGB =re .compile (rb"COLOUR_RGB\s*\(\s*'[^']*'\s*,\s*([-0-9.E+]+)\s*,\s*([-0-9.E+]+)\s*,"
rb"\s*([-0-9.E+]+)\s*\)",re .I )
STYLED =re .compile (rb"STYLED_ITEM|OVER_RIDING_STYLED_ITEM",re .I )


def renkler (path ):
    with open (path ,"rb")as fh :
        raw =fh .read ()
    out =[]
    for m in RGB .finditer (raw ):
        try :
            out .append (tuple (round (float (x ),3 )for x in m .groups ()))
        except ValueError :
            continue 
    return out ,len (STYLED .findall (raw ))


def main ():
    n =int (sys .argv [1 ])if len (sys .argv )>1 else 300 
    files =sorted (glob .glob ("all_wscad_stp/*.stp"))[:n ]
    print (f"{len (files )} STEP taraniyor...",flush =True )
    cok ,hic ,tek =0 ,0 ,0 
    hist =collections .Counter ()
    styled_yok =0 
    ornek =[]
    for f in files :
        try :
            cs ,ns =renkler (f )
        except Exception :
            continue 
        u =sorted (set (cs ))
        hist [len (u )]+=1 
        if ns ==0 :
            styled_yok +=1 
        if len (u )==0 :
            hic +=1 
        elif len (u )==1 :
            tek +=1 
        else :
            cok +=1 
            if len (ornek )<6 :
                ornek .append ((os .path .basename (f ),u [:4 ]))
    tot =max (hic +tek +cok ,1 )
    print (f"\n{'renk count':<16}{'part':>8}{'ratio':>9}")
    for k in sorted (hist ):
        print (f"{k :<16}{hist [k ]:>8}{hist [k ]/tot :>9.3f}")
    print (f"\nhic renk yok      : {hic :>5} ({hic /tot :.3f})")
    print (f"tek renk          : {tek :>5} ({tek /tot :.3f})")
    print (f">=2 ayrik renk    : {cok :>5} ({cok /tot :.3f})   <- KANALIN BILGISI BURADA")
    print (f"STYLED_ITEM yok   : {styled_yok :>5}")
    if ornek :
        print ("\ncok renkli ornekler:")
        for a ,b in ornek :
            print (f"  {a }: {b }")
    ratio =cok /tot 
    print (f"\nKILL (onceden yazili): >=2 renkli part orani <0.10 ise kanal BOS -> "
    f"{ratio :.3f} => {'KANAL CANLI, matcher mesru'if ratio >=0.10 else 'KANAL BOS, ACMA'}")
    json .dump ({"n":tot ,"never":hic ,"single":tek ,"very":cok ,"oran_cok":ratio ,
    "styled_yok":styled_yok ,
    "karar":"CANLI"if ratio >=0.10 else "BOS"},
    open ("results/q1_renk_sayimi.json","w"),indent =1 )
    print ("receipt -> results/q1_renk_sayimi.json")


if __name__ =="__main__":
    main ()
