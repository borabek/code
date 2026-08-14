# -*- coding: utf-8 -*-
"""R4: NOKTAYI analitik silindir eksenine IZDUSUR + askida kalan P2 kapi taramasi.

IKI AYRI SEY, tek olcum:

  (1) YANAL. `pe` = noktanin GT eksen DOGRUSUNA dik mesafesi. Nokta gercek aciklik ekseninin
      UZERINDE olsaydi bu mesafe yapisal olarak ~0 olurdu. Hafizadaki "B-rep snap OLU" kaydi
      YONU yuvarlamayi olcmustu (1.31 -> 1.18mm ortalama); NOKTAYI eksene izdusurmek AYRI bir
      islem ve hic olculmedi.
  (2) ACI. Ayni silindirin analitik ekseni dogrudan yon olarak kullanilirsa aci ne olur?

  (3) P2 KAPI TARAMASI (kodda ASILI is): `axis_at`in kapilari (max_off_mm, r_range ust siniri)
      yaricap YAY HATASI varken ayarlanmisti -- yaricaplar 3.5 kat KUCUK cikiyordu, yani
      r_range ust siniri 12mm fiilen ~42mm demekti ve eksen NOKTASI eksenden ~r kayikti,
      max_off_mm=5.0 o kaymayi tolere ediyordu. Hata 2026-07-31'de duzeltildi, kapilar
      TARANMADI. Burada taraniyor.

DURUSTLUK: silindir secimi `axis_at` ile AYNI kurala gore yapilir (GT'ye BAKMAZ). Olculen sey
"kahin" degil, gercekten uygulanabilir bir islem.
"""
import json
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("BA_ALLOW_SEEN", "1")
os.environ["WG_FIZ_FEATS"] = "1"
os.environ["WG_TOPO"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KAPILAR = [(3.0, 12.0), (3.0, 20.0), (5.0, 12.0), (5.0, 20.0), (8.0, 20.0), (2.0, 12.0)]


def sec(p, s, C, A, R, max_off, r_max, max_turn=40.0):
    """axis_at ile AYNI secim kurali -> silindir indeksi (yoksa None)."""
    rel = p - C
    al = (rel * A).sum(1)
    off = np.linalg.norm(rel - al[:, None] * A, axis=1)
    ok = (off <= max_off) & (R >= 0.5) & (R <= r_max)
    if not ok.any():
        return None
    cand = np.where(ok)[0]
    j = cand[int(np.argmax(np.abs(A[cand] @ s)))]
    if np.degrees(np.arccos(min(1.0, float(abs(A[j] @ s))))) > max_turn:
        return None
    return j


def main():
    import brep_axes
    import wire_gate
    from big_arbiter import eligible
    from sklearn.ensemble import RandomForestClassifier

    stp_of = {p: s for m, p, jf, s in eligible()}
    with open("results/_u4_der.pkl", "rb") as f:
        DER = pickle.load(f)
    d = np.load("results/gate_regrow_data_topo.npz", allow_pickle=True)
    with open("results/_strict_geometry_keys.json", encoding="utf-8") as f:
        gk = json.load(f)
    tr_pid = np.array([str(x) for x in d["pids"]])
    tr_grp = np.array([gk.get(p, "yok:" + p) for p in tr_pid])
    Xtr = np.asarray(d["X"], float); ytr = np.asarray(d["y"])
    keep = ~np.isin(tr_grp, list({gk.get(r["pid"], "yok:" + r["pid"]) for r in DER}))
    dag = wire_gate._load(wire_gate.MODEL_PATH)
    rf = lambda M: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                          random_state=0).fit(M[keep], ytr[keep])
    Ztr = np.zeros((len(Xtr), Xtr.shape[1] * 2))
    for u in np.unique(tr_pid):
        i = np.where(tr_pid == u)[0]
        Ztr[i] = wire_gate.parca_ici(Xtr[i], dag.get("donusum_z", "zskor"))
    m = {"clf": rf(Xtr), "clf_z": rf(Ztr), "n_feat": Xtr.shape[1],
         "donusum": dag.get("donusum"), "donusum_z": dag.get("donusum_z", "zskor")}
    mx = [float(m["clf"].predict_proba(Xtr[np.where((tr_pid == u) & keep)[0]])[:, 1].max())
          for u in np.unique(tr_pid[keep]) if ((tr_pid == u) & keep).any()]
    m["esik_cokus"] = float(np.quantile(mx, dag.get("yonlendirme_q", 0.10)))

    SAT = []
    for i_, r in enumerate(DER, 1):
        if i_ % 50 == 0:
            print(f"  {i_}/{len(DER)}", flush=True)
        if r["X"] is None:
            continue
        s_ = wire_gate.karar_skoru(m, r["X"])
        k = wire_gate.karar_maskesi(s_)
        if not k.any():
            continue
        P = r["P"][k]; Pd = r["Pd"][k]
        G = np.asarray(r["G"], float); Gd = np.asarray(r["Gd"], float)
        if not len(G):
            continue
        try:
            C, A, R = brep_axes.cylinders(stp_of.get(r["pid"]))
        except Exception:
            C = A = R = np.zeros((0, 3))
        diff = P[:, None, :] - G[None, :, :]
        al = (diff * Gd[None, :, :]).sum(-1)
        pe = np.linalg.norm(diff - al[..., None] * Gd[None, :, :], axis=-1)
        an = np.degrees(np.arccos(np.clip(np.abs(Pd @ Gd.T), 0, 1)))
        pe_t = np.where(np.abs(al) > 40, np.inf, pe)
        tt = max(3.0, 0.06 * float(r["diag"]))
        used, hit = set(), set()
        for dd, a_, b_ in sorted((pe_t[a, b], a, b)
                                 for a in range(len(P)) for b in range(len(G))):
            if dd > tt or a_ in used or b_ in hit:
                continue
            used.add(a_); hit.add(b_)
            kay = {"yanal": float(pe[a_, b_]), "aci": float(an[a_, b_]),
                   "rejim": "cok" if r["n"] >= 8 else "dusuk", "pid": r["pid"]}
            for mo, rx in KAPILAR:
                yeni_y, yeni_a = kay["yanal"], kay["aci"]
                if len(C):
                    j = sec(P[a_], Pd[a_], C, A, R, mo, rx)
                    if j is not None:
                        q = C[j] + float((P[a_] - C[j]) @ A[j]) * A[j]      # IZDUSUM
                        dq = q - G[b_]
                        aq = float(dq @ Gd[b_])
                        yeni_y = float(np.linalg.norm(dq - aq * Gd[b_]))
                        yeni_a = float(np.degrees(np.arccos(
                            min(1.0, abs(float(A[j] @ Gd[b_]))))))
                kay[f"y_{mo}_{rx}"] = yeni_y
                kay[f"a_{mo}_{rx}"] = yeni_a
            SAT.append(kay)

    ya = np.array([x["yanal"] for x in SAT]); ac = np.array([x["aci"] for x in SAT])
    su = float(((ya <= 2) & (ac <= 10)).mean())
    print(f"\n{len(SAT)} eslesen nokta | SU AN: yanal<=2 {(ya<=2).mean():.1%} | "
          f"aci<=10 {(ac<=10).mean():.1%} | IKISI {su:.1%}")
    print(f"\n{'kapi (off,rmax)':<18}{'yanal med':>11}{'yanal<=2':>10}{'aci<=10':>9}"
          f"{'IKISI':>8}{'fark':>9}")
    en_iyi, en_iyi_ad = su, "IZDUSUMSUZ"
    OUT = {}
    for mo, rx in KAPILAR:
        y = np.array([x[f"y_{mo}_{rx}"] for x in SAT])
        a = np.array([x[f"a_{mo}_{rx}"] for x in SAT])
        i2 = float(((y <= 2) & (a <= 10)).mean())
        print(f"{str((mo, rx)):<18}{np.median(y):>11.2f}{(y<=2).mean():>10.1%}"
              f"{(a<=10).mean():>9.1%}{i2:>8.1%}{i2-su:>+9.1%}")
        OUT[f"{mo}_{rx}"] = {"yanal_ok": float((y <= 2).mean()),
                             "aci_ok": float((a <= 10).mean()), "ikisi": i2}
        if i2 > en_iyi:
            en_iyi, en_iyi_ad = i2, str((mo, rx))
    print(f"\nEN IYI: {en_iyi_ad} -> {en_iyi:.1%} (su an {su:.1%})")
    print(f"robot-hazir 0.65 icin gereken oran: %87.4")
    with open("results/r4_izdusum.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(SAT), "su_an": su, "kapilar": OUT, "en_iyi": en_iyi_ad,
                   "en_iyi_oran": en_iyi}, f, indent=1)
    with open("results/r4_satir.pkl", "wb") as f:
        pickle.dump(SAT, f)
    print("makbuz -> results/r4_izdusum.json")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except BaseException:
        traceback.print_exc(); raise
