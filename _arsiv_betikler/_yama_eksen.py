# -*- coding: utf-8 -*-
"""_vote2'yi EKSEN-FARKINDALIKLI hale getirir (cp_config bayragiyla).

NEDEN: `_vote2` iki adayi duz OKLID mesafesiyle "ayni aciklik" sayiyor. Ama urunun
puanlayicisi `big_arbiter.greedy` EKSEN-FARKINDALIKLI: dik mesafeye bakip derinligi
serbest birakiyor, cunku tez CP'yi AGIZDA (v_o), uretici ise KONTAKTA tanimliyor.
Urun kendi icinde tutarsizdi.

OLCULDU (2026-08-03, 604 uye cifti): ayni GT'yi bulan uyelerin nokta farki
   EKSENEL (derinlik): medyan 1.34mm, %90 16.03mm
   YANAL (dik)       : medyan 1.57mm, %90  5.55mm
Birlesme orani: duz Oklid 5mm ile %67.7, eksen-farkindalikli (yanal<=3mm) ile %70.7.

Yanal 3mm siniri KOMSU KUTUPLARI KORUR (klemens adimi 3.5-6mm) -- duz 5mm'lik Oklid
kuresi bu acidan daha risklidir.

UC CAGRI NOKTASI VAR; bayragi fonksiyonun ICINE koyuyoruz ki hepsi tutarli olsun ve
egitim/olcum turetmeleri arasinda PARITE bozulmasin.
"""
import io

P = "robot_cp.py"
N = "\r\n"
s = io.open(P, encoding="utf-8", newline="").read()

anchor = ("    allc = [dict(c, _mid=i) for i, lst in enumerate(cp_lists) for c in lst]" + N
          + "    allc.sort(key=lambda c: -float(c.get(\"confidence\", 0.0)))" + N
          + "    kept = []" + N
          + "    for c in allc:" + N
          + "        p = np.asarray(c[\"point\"], float)" + N
          + "        hit = next((k for k in kept if np.linalg.norm(p - np.asarray(k[\"point\"], float)) <= cluster_mm), None)" + N)

yeni = ("    allc = [dict(c, _mid=i) for i, lst in enumerate(cp_lists) for c in lst]" + N
        + "    allc.sort(key=lambda c: -float(c.get(\"confidence\", 0.0)))" + N
        + "    # EKSEN-FARKINDALIKLI HAVUZLAMA (cp_config.robot_eksen_havuz): iki aday ayni" + N
        + "    # aciklktan sayilir -> DIK mesafe <= yanal_mm VE eksenler hizali (<=20 derece);" + N
        + "    # DERINLIK farki serbest. Gerekce: tez CP'yi AGIZDA (v_o), uretici KONTAKTA" + N
        + "    # tanimlar; uyeler ayni kanali farkli derinlikte isaretleyebilir. Puanlayici" + N
        + "    # (big_arbiter.greedy) zaten boyle calisiyor -- bu, urunun kendi icindeki" + N
        + "    # tutarsizligi kapatir. Olculdu: birlesme %67.7 -> %70.7; ayrica yanal 3mm" + N
        + "    # siniri KOMSU KUTUPLARI (adim 3.5-6mm) duz 5mm kureden daha iyi korur." + N
        + "    _eks = bool(_cfg.get(\"robot_eksen_havuz\", False))" + N
        + "    _yan = float(_cfg.get(\"robot_eksen_havuz_yanal_mm\", 3.0))" + N
        + "    kept = []" + N
        + "    for c in allc:" + N
        + "        p = np.asarray(c[\"point\"], float)" + N
        + "        if _eks:" + N
        + "            d = np.asarray(c[\"direction\"], float)" + N
        + "            d = d / (np.linalg.norm(d) + 1e-9)" + N
        + "            hit = None" + N
        + "            for k in kept:" + N
        + "                e = np.asarray(k[\"direction\"], float)" + N
        + "                e = e / (np.linalg.norm(e) + 1e-9)" + N
        + "                if abs(float(d @ e)) < 0.9397:      # cos(20 derece)" + N
        + "                    continue" + N
        + "                v = p - np.asarray(k[\"point\"], float)" + N
        + "                if float(np.linalg.norm(v - (v @ e) * e)) <= _yan:" + N
        + "                    hit = k; break" + N
        + "        else:" + N
        + "            hit = next((k for k in kept if np.linalg.norm(p - np.asarray(k[\"point\"], float)) <= cluster_mm), None)" + N)

assert anchor in s, "KALIP YOK"
s = s.replace(anchor, yeni, 1)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("robot_cp._vote2 eksen-farkindalikli hale getirildi (bayrakla)")
