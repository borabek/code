# -*- coding: utf-8 -*-
"""TR -> EN toplu cevirisi. Kod + config + betik + dokuman AYNI ANDA.

WHY AYNI ANDA: `gate_goreli_oran` like anahtarlar hem koddan hem
`cp_config.json`'dan okunuyor. Birini cevirip digerini birakmak urunu
SESSIZCE breaks -- bugun full this turden four bayatlama was found.

SAFETY:
  * kelime siniri (\b) with changes
  * UZUN anahtar ONCE (match_hungarian, esle'den before)
  * each file before ayristirilir (.py) -- bozulursa geri alinir
"""
import ast ,io ,os ,re ,subprocess ,sys 
import importlib .util as _u 
_s =_u .spec_from_file_location ("sz","_ceviri_dict.py")
_m =_u .module_from_spec (_s );_s .loader .exec_module (_m )
SOZLUK =_m .SOZLUK 

UZANTI =(".py",".json",".sh",".md",".txt",".yaml",".yml",".ini")
anahtarlar =sorted (SOZLUK ,key =len ,reverse =True )
DESEN =re .compile (r"\b("+"|".join (re .escape (k )for k in anahtarlar )+r")\b")


def cevir (metin ):
    return DESEN .sub (lambda m :SOZLUK [m .group (1 )],metin )


def main ():
    uygula ="--uygula"in sys .argv 
    dosyalar =[f for f in subprocess .run (["git","ls-files"],
    capture_output =True ,text =True ).stdout .split ("\n")
    if f .endswith (UZANTI )and os .path .exists (f )]
    deg ,bozuk ,top =0 ,[],0 
    for f in dosyalar :
        try :
            src =io .open (f ,encoding ="utf-8",errors ="ignore").read ()
        except Exception :
            continue 
        yeni =cevir (src )
        if yeni ==src :
            continue 
        n =len (DESEN .findall (src ))
        if f .endswith (".py"):
            try :
                ast .parse (yeni )
            except SyntaxError as e :
                bozuk .append ((f ,str (e )[:50 ]))
                continue 
        if f .endswith (".json"):
            import json 
            try :
                json .loads (yeni )
            except Exception as e :
                bozuk .append ((f ,"json: "+str (e )[:40 ]))
                continue 
        deg +=1 ;top +=n 
        if uygula :
            io .open (f ,"w",encoding ="utf-8").write (yeni )
    print (f"dosya: {len (dosyalar )} tarandi | {deg } degisecek | {top } degisim")
    if bozuk :
        print (f"ATLANDI ({len (bozuk )}) -- ayristirma bozulurdu:")
        for f ,e in bozuk [:6 ]:
            print (f"  {f }: {e }")
    if not uygula :
        print ("\n(kuru kosum -- --uygula ile yazar)")
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
