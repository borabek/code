# -*- coding: utf-8 -*-
"""FB-2b: train_seg_extra.py'ye TEL/ALET yardimci kafasini adds.

TEZ IHLALI YOK: ana 5-sinif kafasi, softmax'i and kaybi BIT DUZEYINDE does not change.
Yardimci kafa paylasilan govdeden beslenir and AYRI a kayipla egitilir.
"""
import io 

P ="train_seg_extra.py"
N ="\r\n"
s =io .open (P ,encoding ="utf-8",newline ="").read ()


def yama (old ,new ,s ):
    assert old in s ,"KALIP YOK: "+repr (old [:70 ])
    return s .replace (old ,new ,1 )


    # --- 1) load_extra: yanindaki .aux.txt dosyasini oku
o1 ='            rec = {"part_id": pid, "verts": V, "faces": F, "labels": L}'+N 
n1 =(o1 
+'            # FB-2: TEL/ALET yardimci etiketi (varsa). -1 maskeli / 0 tel / 1 alet.'+N 
+'            # Tezin 5-sinif etiketini DEGISTIRMEZ: ayri file, ayri kafa, ayri loss.'+N 
+'            _ax = os.path.join(d, pid + ".aux.txt")'+N 
+'            if os.path.exists(_ax):'+N 
+'                try:'+N 
+'                    A = np.array([int(x) for x in open(_ax).read().split()], np.int64)'+N 
+'                    if len(A) == len(V):'+N 
+'                        rec["aux"] = A'+N 
+'                except Exception:'+N 
+'                    pass'+N )
s =yama (o1 ,n1 ,s )

# --- 2) prep: tensore cevir
o2 ='               "partial_ce": bool(s.get("partial_ce", False))}'+N 
n2 =(o2 
+'        if s.get("aux") is not None:'+N 
+'            rec["aux"] = torch.tensor(np.asarray(s["aux"]), dtype=torch.long)'+N )
s =yama (o2 ,n2 ,s )

# --- 3) bayraklar
o3 ='    ap.add_argument("--partial-pos-weight", type=float, default=20.0,'
n3 =(
'    ap.add_argument("--aux-wire", action="store_true",'+N 
+'                    help="FB-2 TEL/ALET yardimci supervizyonu. Paylasilan govdeye IKINCI "'+N 
+'                         "a kafa eklenir; ana 5-sinif kafasi and kaybi BIT DUZEYINDE "'+N 
+'                         "does not change (tez ihlali YOK). Gerekce: tezin Contact sinifi "'+N 
+'                         "tasarimi geregi Kontaktierung bzw. Werkzeugeinschub -- tel and "'+N 
+'                         "alet TEK sinif, i.e. backbone ayrimi SILMEK so as to egitildi.")'+N 
+'    ap.add_argument("--aux-w", type=float, default=0.5, help="yardimci kaybin agirligi")'+N 
+'    ap.add_argument("--aux-pos-weight", type=float, default=2.0,'+N 
+'                    help="ALET pozitif agirligi (measured: alet/tel vertex orani ~0.46)")'+N 
+o3 )
s =yama (o3 ,n3 ,s )

io .open (P ,"w",encoding ="utf-8",newline ="").write (s )
print ("1-3 yamalandi: label yukleme + prep + bayraklar")
