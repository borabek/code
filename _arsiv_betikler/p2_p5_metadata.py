# -*- coding: utf-8 -*-
"""FAZ 2 / P5 -- METADATA KAYNAK VERIFICATION (kullanici kurali: source GT'den BAGIMSIZ must be).
Soru: metadata-assisted mod 'CP count biliniyor' varsayar. Olcumlerimizde this number
len(GT ConnectionPoints) = GT dosyasinin KENDISI. GT'den BAGIMSIZ a source present mi?
Aday bagimsiz source: STEP PRODUCT adi (ornek 'USK_4_FSR_4-2.8-0.8-select') -- STEP dosyasinda,
GT JSON'da not. Icindeki numbers CP sayisini ONGORUYOR mu?
Karar: ongoruyorsa metadata modu MESRU (bagimsiz source present). Ongormuyorsa metadata sonucu
'varsayimsal' as isaretlenir and BAZ (geometri-only) birincil number becomes."""
import json ,re 
import numpy as np 

lock =json .load (open ("results/split_lock.json"))
P =lock ["parts"]
rows =[(pid ,r )for pid ,r in P .items ()if r .get ("product")]
print (f"{len (rows )} parcada STEP PRODUCT adi var (toplam {len (P )})")

# PRODUCT adindaki number token'lari
def toks (s ):return [float (t )for t in re .findall (r"\d+(?:\.\d+)?",s or "")]
n_with =sum (1 for _ ,r in rows if toks (r ["product"]))
print (f"  sayi token'i iceren: {n_with }")

best =None 
for k in range (4 ):# k'inci number token'i CP sayisini veriyor mu
    xs ,ys =[],[]
    for pid ,r in rows :
        t =toks (r ["product"])
        if len (t )>k :
            xs .append (t [k ]);ys .append (r ["n_cps"])
    if len (xs )<30 :continue 
    xs ,ys =np .array (xs ),np .array (ys )
    exact =float (np .mean (xs ==ys ))
    corr =float (np .corrcoef (xs ,ys )[0 ,1 ])if xs .std ()>0 else 0.0 
    print (f"  token[{k }]: n={len (xs )} | CP sayisina TAM esitlik %{100 *exact :.1f} | korelasyon {corr :+.3f}")
    if best is None or exact >best [1 ]:best =(k ,exact ,corr )

    # alternatif: PRODUCT adindaki 'pole' benzeri desenler
pat_hits =0 
for pid ,r in rows :
    if re .search (r"(?i)(\d+)\s*(?:pol|pole|way|leiter|L)\b",r ["product"]or ""):
        pat_hits +=1 
print (f"  acik 'pol/pole/way/Leiter' deseni: {pat_hits } part")

print ("\n=== KARAR (P5) ===")
if best and best [1 ]>=0.60 :
    print (f"  STEP PRODUCT token[{best [0 ]}] CP sayisini %{100 *best [1 ]:.0f} tam veriyor -> BAGIMSIZ KAYNAK VAR")
    print ("  -> metadata-assisted mod MESRU sayilabilir (yine de baz ayri raporlanir).")
else :
    e =best [1 ]if best else 0.0 
    print (f"  STEP PRODUCT adindan CP sayisi TURETILEMIYOR (en iyi tam-esitlik %{100 *e :.0f}).")
    print ("  -> Olcumlerdeki CP-count GT-TUREVLI. Metadata-assisted numbers 'VARSAYIMSAL' etiketlenir;")
    print ("     BIRINCIL number = BAZ (geometri-only). Robot hucresi katalogdan kutup count saglayabilirse")
    print ("     mod gecerli becomes but this DOGRULANMADI.")
print ("  YON-PRIOR: InsertDirection only GT JSON'inda -> bagimsiz source YOK -> KULLANILMAYACAK (rule).")
