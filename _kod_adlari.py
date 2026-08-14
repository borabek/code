# -*- coding: utf-8 -*-
"""Kod icindeki TURKCE DEGISKEN adlari -> EN. tokenize ile YALNIZ NAME.

GUVENLIK:
  * hedef ad PYTHON ANAHTAR KELIMESI ya da builtin olamaz
  * NAME belirteci disinda hicbir seye dokunulmaz (string/comment ayri turda islendi)
  * her dosya ayristirilir; bozulursa yazilmaz
"""
import ast ,builtins ,io ,keyword ,subprocess ,sys ,tokenize 

AD ={
"var":"present","yok":"missing","kaynak":"source","kural":"rule",
"deger":"value","degerler":"values","sayi":"count","sayilar":"counts",
"ayni":"same","farkli":"different","once":"before","sonra":"after",
"cok":"many","az":"few","hic":"none_","hepsi":"all_",
"bos":"empty","dolu":"full","eski":"old","yeni":"new",
"ilk":"first","son":"last","orta":"mid","ust":"upper","alt":"lower",
"sol":"left_","sag":"right_","tek":"single","cift":"double",
"toplam":"total","ortalama":"mean_","ortanca":"median_",
"yuzde":"pct","oran":"ratio","adet":"count_","birim":"unit_",
"genislik":"width","yukseklik":"height","derinlik":"depth_",
"uzunluk":"length","mesafe":"dist","aci":"angle_","yaricap":"radius_",
"merkez":"center","sinir":"bound","bolge":"region_",
"dosya":"file_","dizin":"dir_","yol":"path_","ad":"name_",
"satir":"line_","sutun":"col","dizi":"arr","liste":"lst",
"sozluk":"dct","anahtar":"key_","girdi":"inp","cikti":"out_",
"sonuc":"res","hata":"err","uyari":"warn_","mesaj":"msg",
"durum":"state_","adim":"step_","asama":"stage_","dongu":"loop_",
"kosul":"cond","secim":"sel","sirali":"sorted_","gecerli":"valid_",
"gecersiz":"invalid_","dogru":"correct","yanlis":"wrong",
"kayit":"rec_","kayitlar":"recs","veri":"data_","etiket":"label_",
"etiketler":"labels_","model":"model_","modeller":"models_",
"tahmin":"pred","gercek":"true_","kahin":"oracle",
"esik":"thr","agirlik":"wgt","puan":"score_","skor":"score_",
"sira":"rank_","kume":"clust","kumeler":"clusters",
"havuz":"pool_","aday":"cand","adaylar":"cands",
"nokta":"pt","noktalar":"pts","tepe":"vtx","tepeler":"verts",
"yuz":"face_","yuzler":"faces_","kenar":"edge_",
"yon":"dirn","eksen":"axis_","isaret":"sign_",
"parca":"part_","parcalar":"parts_","kol":"arm_",
"olcum":"meas","olcut":"crit","kat":"fold_","tohum":"seed_",
"bolme":"split_","sinav":"test_set","gerekce":"why_",
"makbuz":"receipt_","sebep":"cause","neden":"why2",
}
YASAK =set (keyword .kwlist )|set (dir (builtins ))
AD ={k :v for k ,v in AD .items ()if v not in YASAK }


def isle (yol ):
    try :
        src =io .open (yol ,encoding ="utf-8").read ()
        toks =list (tokenize .generate_tokens (io .StringIO (src ).readline ))
    except Exception :
        return None ,0 
    new_ ,n =[],0 
    for t in toks :
        v =t .string 
        if t .type ==tokenize .NAME and v in AD :
            v =AD [v ];n +=1 
        new_ .append ((t .type ,v ))
    if not n :
        return None ,0 
    try :
        out =tokenize .untokenize (new_ )
        ast .parse (out )
    except Exception :
        return None ,0 
    return out ,n 


def main ():
    uygula ="--uygula"in sys .argv 
    d =t =0 
    for f in subprocess .run (["git","ls-files","*.py"],capture_output =True ,
    text =True ).stdout .split ():
        out ,n =isle (f )
        if out is None :
            continue 
        d +=1 ;t +=n 
        if uygula :
            io .open (f ,"w",encoding ="utf-8").write (out )
    print (f"{d } dosya | {t } degisken adi cevrildi")
    if not uygula :
        print ("(kuru kosum)")


main ()
