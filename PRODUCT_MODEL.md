# PRODUCT MODEL — SINGLE SOURCE OF TRUTH

> ⚠️ Everything BELOW this top block is HISTORICAL PROVENANCE (earlier products that were later
> superseded — 3-seed ensemble, human77c, recall_s2, recall_hard_s2, etc.). It is kept for the audit
> trail. The CURRENT PRODUCT is defined ONLY here and in `cp_config.json > current_product`.

## ✅ CURRENT PRODUCT (2026-08-02) = `vote1_union + 116-sütunlu gate + pose head + açı seçici`

**Pipeline:** STEP → `thesis_remesh(6000)` → 4-model segmentasyon →
**`robot_cp.derive_candidates`** (tek source; gate eğitimi de aynı fonksiyonu çağırır) →
**wire-gate** → **göreli eşik** → **POSE HEAD** (lateral düzeltme) → **SEÇİCİ AÇI DÜZELTMESİ** →
iki katman.

**Gate:** `results/wire_gate.pkl` — **58 ham × 2 (parça-içi z-score) = 116 sütun**
 (13 baseline + 5 B-rep fiziksel + 4 içbükey topoloji + **36 zengin**: konum9 + çok-yarıçap24 + normal-std3)
**Pose head:** `results/pose_head.pkl` — gate kararından after lateral düzeltme, ≤3mm, axial depth fixed
**Açı seçici:** `results/aci_secici.pkl` — önce "this açı yanlış mı", yalnız öyleyse düzelt
Hepsi MD5 damgalı ve `tests/test_artifact.py` with doğrulanıyor.

### 📊 MANŞET — `headline.py --yaz` ÜRETİR (194 parça / 171 grup, **grup** bootstrap)

| bölme | n | detection F1 | %95 GA | robot-hazır | düşük-CP | çok-CP |
|---|---|---|---|---|---|---|
| DEV *(kararlar burada)* | 54 | 0.8012 | [0.707, 0.878] | **0.6336** | 0.8108 | 0.7196 |
| **VAL** *(sınav)* | 100 | **0.7503** | [0.701, 0.799] | 0.5441 | 0.7696 | 0.5862 |
| havuzlanmış | 194 | **0.7538** | [0.710, 0.796] | **0.5801** | 0.7656 | 0.6530 |
| PXC dışarıda | 90 | 0.6750 | [0.603, 0.745] | 0.5562 | 0.6873 | 0.5702 |
| **WEI dışarıda** | 103 | **0.5968** | [0.533, 0.654] | 0.3746 | 0.6100 | 0.4847 |

**2026-08-02 gecesi:** robot-hazır **0.4391 → 0.5801 (+0.141)**, detection 0.7384 → 0.7538.
**95 grup-temiz LOCKED parça harcanmadı.**

### 🔧 GECENİN ÜÇ KOLU

| arm | etki | kanıt |
|---|---|---|
| **zengin bloklar** (+36 sütun) | detection **+0.0233** | GA(WEI) [+0.0005, +0.0697] · `decision_criterion` 5/5 şart |
| **pose head** (lateral) | robot **+0.0428** | GA [+0.0220, +0.0680] · detection +0.0001 |
| **seçici açı** | robot **+0.0126** | GA [+0.0002, +0.0325] · detection bedeli **yapısal sıfır** |
| **üye yön seçici** | robot **+0.0278** | GA [+0.0105, +0.0490] · kâhin +0.0536'nın %52'si

**Pose head'in dersi:** aynı fikir 1068 satırla **kanıtsızdı** (+0.0222, GA sıfırı içeriyor),
6330 satırla **kanıtlı** (+0.0428). Veri, ölçüm belirsizliğini kapattı.

**Neden öğrenme, rule değil:** 2026-08-01'de rule tabanlı dört konum/yön arm denendi, dördü
de öldü (izdüşüm −0.045, axis uzlaşısı −110 nokta, yarık yönü −187, B-rep kapısı 0.000).
Ortak kusur: düzeltme herkese aynı uygulanıyordu.

### 🔑 PARÇA-İÇİ Z-SKOR BİR **KURTARMA**, GENEL İYİLEŞTİRME DEĞİL

9 bölmede (7 seri-dışı + 2 üretici-dışı) desen tekdüze — **baseline zayıfken kazanıyor,
sağlıklıyken kaybediyor**:

| bölme | baseline | each parçaya z-skorun farkı |
|---|---|---|
| WEI dışarıda | 0.4832 (çöküş) | **+0.0869** |
| seri 25 | 0.5654 | **+0.0639** |
| seri 10 | 0.5980 | +0.0086 |
| seri 17 / 30 / 15 / 32 / 16 | 0.66–0.78 | −0.060 … −0.008 |
| PXC dışarıda | 0.7203 (sağlıklı) | **−0.0375** |

Bu yüzden **each parçaya uygulanmıyor.** Yönlendirme, gate'in kendi score dağılımından çöküşü
sezer (ölçülmüş teşhis: çöküşte model adayların %10.3'üne pozitif diyor, gerçek %24.1) ve
kazancın **%98'ini** korurken PXC vergisinin **yarısını**, seri vergisinin **%60'ını** geri alır.
Metadata gerektirmez.

> **Dürüst not (kendi çubuğum):** *"yönlendirmeli arm each eksende en az z-score up to iyi olmalı"*
> testim WEI'de kıl payı düştü — diff −0.0020, %95 GA [−0.0066, **+0.0000**], üst uç tam sıfır.
> Ürün kararını a `>0`/`>=0` sınır artefaktına bırakmadım: yönlendirme WEI'de 0.0020 verip
> PXC'de 0.0201 ve tanıdıkta 0.0049 alıyor (10'a 1) ve üretici ortalamasında en iyisi.
> Gerekçe budur, artefakt gizlenmedi.

### 🚪 KAPANAN DAL: topoloji yarıçapı

Aday düzeyi R=12'yi kazanan gösterdi (+0.0100). **Uçtan uca çürüdü:** R=12 tanıdıkta −0.0109;
R=8'in kazancı **gürültü** (WEI +0.018, GA [−0.007, +0.044]) but bedeli **gerçek**
(PXC −0.024, GA [−0.044, −0.006]). **R=6.0 kalır.** Aday düzeyi this araştırmada **dördüncü kez**
yanılttı — eleme kapısıdır, karar kapısı değildir.

**Ölçüm protokolü:** bölme **keskin geometri anahtarıyla** (bbox 0.5 mm + B-rep silindir/düzlem
imzası); `family_key` this korpusta parça numarasını döndürüyordu ve **parçaların %80'inin ikizi present**.
DEV (karar) / VAL (sınav) / **LOCKED (harcanmadı)**.

**Kazançların kaynağı (sırayla):** 5 B-rep fiziksel özellik (`brep_r` with `size` korelasyonu
−0.009 = yeni bilgi) → 4 içbükey topoloji sütunu (`kon_cevre` AUC 0.709; gerçek açıklıklarda
medyan 1.000 = tam tur halka) → göreli eşik (calibration) → parça-içi z-score (sıralama).

**🎓 TEZ-NATIVE METRİK (Scheffler benchmark'ı = SEGMENTASYON, mean Jaccard 0.514):**
`best_full.pt` val 20-parça: **mean-IoU 0.681** · Dice 0.790 · acc 0.864.
Dağıtılan ürün (`recall_hard_s2`): IoU 0.633 / Dice 0.746 / acc 0.831 — o da tezi geçiyor.
`seg_thesis_eval.py` · `results/seg_thesis_val.json`

**📌 0.80 ÖZETİ (hangi sayı ≥0.80, dürüst):** tez-native segmentasyon acc 0.86/Dice 0.79 ✅ · PXC-tipik CP-F1 0.823 ✅ ·
count-assisted full-stack 0.807 ✅. **ALL CP-nokta-F1 0.756 etiketsiz 0.80 OLMAZ** (top-N ceiling 0.775, 5 yolla kanıtlı).

**Primary metric (user decision 2026-07-24): AUTO+REVIEW F1** vs manufacturer ConnectionPoints,
leakage-free GroupKFold-OOF. **AUTO precision = separate safety metric.**

**CURRENT (RF gate, leakage-free OOF — canon = `results/product_f1_receipt.json`, 2026-07-26).**
GB-era 0.693/0.742 RETIRED (kept below under HISTORICAL).

| set | base (geometry-only) | + CP-count (metadata) |
|---|---|---|
| **ALL** | **0.756** | 0.775 |
| **PXC typical** | **0.823** | 0.837 (ceiling) |
| typical overall (non-high-CP) | 0.797 | 0.816 (ceiling) |
| WEI typical | 0.696 | 0.723 (ceiling) |
| full-stack high-CP (≥11 CP) | — | **0.807** part-out / **0.799** family-out |

**Key facts (proven, receipt):** PXC-typical base already ≥0.80 (0.823). ALL 0.756 and WEI 0.696 CANNOT
reach 0.80 pre-labeling — top-N ceilings (0.775 / 0.723) are below 0.80. WEI aggressive candidate oracle
lifts to 0.969 but the selector still caps ~0.70 (coverage ≠ discrimination). 0.80 for those needs human
labels; 0.85 needs the WEI + high-CP recall-painting round.

**Reporting rule:** quote from the stratified receipt; never a single conflated number; never a non-OOF
(train=test) gate number. `wire_gate.pkl` is full-trained for DEPLOY; scoring uses the OOF gate.

---

### HISTORICAL (GB-era, RETIRED 2026-07-25 — superseded by the RF table above)
| set | P | R | **F1** |
|---|---|---|---|
| ALL | 0.716 | 0.672 | 0.693 |
| WEI | 0.627 | 0.655 | 0.641 |
| PXC | 0.735 | 0.675 | 0.703 |

GB metadata-assisted: ALL 0.742. Old receipt: `results/product_f1_2026_07_24.json`.

**Failure taxonomy (2837 mfg CPs):** kept-TP 1906 · gate dropped 1740 tool_fp (cost: 386 real wires, 4.5:1) ·
kept-FP 753 (unlisted-wire/residual) · FN 931. Recall gap = WEI + complex multi-CP; precision residue = unlisted wires.

**Honest corrections (this replaces earlier inflated claims):**
- Old "precision ~0.98" / "robot-effective ~0.80" were INFLATED — they counted tool/actuator openings as
  correct CPs. The wire-gate exposed and removed this. Honest F1 = **~0.69**.
- 0.90 raw manufacturer F1 is not realistic now; needs full human labels (recall) + wire/tool sub-labels (precision).
- `results/hc_final.json` (human held-out 0.708) is STALE — from `human77c_s2` WITHOUT the gate; re-measure before quoting.

---

# Product model + metric discipline (2026-07-20, CORRECTED after FBI)

## 🏆🏆🏆 PRODUCT (2026-07-21 final) = 3-SEED ENSEMBLE of the 77-human-label models
Ensembling was a measured negative on the OLD pre-human models but had **never been tried on the
human-trained ones**. It is a real, twice-confirmed win — and building a bigger measurement at the
same time corrected an over-optimistic number.

| metric | single seed (old product) | **3-seed ensemble** |
|---|---|---|
| **HELD-OUT: 26 unseen parts / 82 human CPs** (batch-4, never trained on) | F1 **0.539** (P .644 R .463) | **F1 0.623** (P .768 R .524) |
| 18-CP manufacturer arbiter | mean 0.698 | **0.778** (TP14 FP4 FN4) |
| direction | 0.0° median | 0.0° median |

Both precision AND recall improve, on **two independent measurements** → not the seed lottery.

### ⚠️ HONESTY CORRECTION: the 18-CP arbiter is OPTIMISTIC
The new 82-CP held-out set (batch-4) says the product is **~0.62**, not ~0.78. The 9 arbiter parts are
simple 2-CP terminals; real unseen terminals are harder. **Quote 0.62 for real-world expectation.**

### Rules learned
- **Only ensemble seeds trained on the SAME data.** Mixing the 77- and 103-label models scored **0.500**
  (their predictions disagree; averaging blurs them). Seed-ensemble only.
- **Post-processing is at its ceiling.** The ensemble's 4 remaining FPs are *indistinguishable* from its
  14 TPs on confidence (.965 vs .944), size, area and outwardness (.800 vs .744) — every all-TP-preserving
  threshold drops 0/4. They may even be real openings the manufacturer didn't list. The outward gate does
  nothing (0.778 → 0.778). **Precision cannot be post-processed further; only recall is left.**
- **103 labels did NOT beat 77** (arbiter 0.667/0.650 vs 0.698 mean) → the label learning curve is FLAT
  from ~77. More labels are no longer the lever.

Deployed: `cp_config.json checkpoint` (list), `infer_step_cp.CKPT` (list, softmax-averaged).


## 🏆🏆 PRODUCT (2026-07-21, updated) = `human77c_s1.pt` — 77 human labels, CP F1 0.520→0.70
More labels, more gain, seed-confirmed. Batch-3 added 29 catalog terminals (BT/BTO screw-only-no-cable
types excluded per the user's domain call) → 77 total human labels (20 batch-1 + 28 batch-2 + 29
batch-3). Arbiter CP F1 (frozen v3.1), 3 seeds:

| labels | arbiter CP F1 seeds | mean |
|---|---|---|
| 0 (product selftrain_120) | — | **0.520** |
| 48 (human48c) | 0.595 / 0.605 / 0.700 | 0.633 |
| **77 (human77c)** | **0.778 / 0.667 / 0.650** | **0.698** |

Monotonic with label count: **0.520 → 0.633 → 0.698**. Deployed = seed 1 (chosen by best held-out val
0.6712 → its arbiter 0.667); the family choice 77>48 rests on the robust arbiter MEAN, the seed choice
on val (NOT cherry-picking seed 0's 0.778). All 3 seeds clear both 0.605 (48) and 0.520 (no-human).
Note: human77c val Conn IoU (mean 0.651) is slightly BELOW human48c (0.672) — val and the arbiter
diverge (val = corpus-convention proxy, arbiter = real manufacturer CPs); the arbiter is the product
target, so 77 wins. The +pseudo lever is under test on top of the 77 (C2 warned pseudo can tank the
arbiter — adopt only if it beats 0.698).

---

## PRIOR product (2026-07-21) = `human48c_s1.pt` — 48 human labels, SEED-CONFIRMED CP F1 0.520→0.63
The lever finally paid off. 48 human partial labels (20 batch-1 Weidmüller + 28 batch-2 Phoenix/Wago
catalog terminals, ingested from the annotator via `label_tool.html`) trained on 71 corpus with the
**connection-channel masked loss** (`--partial-target connection`) beat the previous product on BOTH
metrics, and — unlike the 20-label attempt — the arbiter gain SURVIVES seeds:

| model | val Conn IoU | arbiter CP F1 (v3.1) |
|---|---|---|
| 71-only baseline | 0.622 | — |
| 20 human labels (human20c), 3 seeds | 0.63 | 0.533 / 0.419 / 0.489 → mean **0.480** ❌ (didn't survive) |
| previous product selftrain_120 | 0.676 | **0.520** |
| **48 human (human48c) seed 0** | 0.6501 | **0.595** |
| **48 human seed 1 (= PRODUCT)** | **0.6933** | **0.605** |
| **48 human seed 2** | 0.6729 | **0.700** |
| **48 human MEAN** | 0.672 | **0.633** |

All three seeds clear the old 0.520 (0.595 / 0.605 / 0.700) → the pre-registered ≥2-seed rule is MET.
The deployed checkpoint is **seed 1**, chosen by best held-out **val** Conn IoU (0.6933) — selection
is on val, NOT the arbiter test set, so 0.605 is an honest held-out number (mean 0.633).
- **First time the segmentation gain AND the CP metric moved together** (earlier every model sat at
  CP F1 0.520 regardless of val IoU; the connection-channel labels broke that).
- The gain is precision: v3.1 false positives 19 (product) → 8–12 (human48c), recall held at 0.72.
- **The outward gate is now REDUNDANT** — human labels already fixed the FP pattern it targeted
  (0.605 with gate off == 0.605 with gate on). Keep `outward_min=0`.
- Why 48 worked where 20 didn't: 20 was within seed noise (mean 0.480); doubling to 48 (batch-2 added
  28 clean typed terminals) pushed the mean decisively above 0.520. Consistent with the learning
  curve saying ~40–60 labels for a step change.

Deployed: `cp_config.json checkpoint`, `infer_step_cp.CKPT`. Follow-ups: (a) +pseudo on top of the 48
(C2-style, may help or hurt — measure); (b) label the ~10 skipped BT/BTO screw terminals for +labels;
(c) more labels toward 60–80.



## 🟢 NEW PRECISION LEVER (2026-07-20): the "outward" gate — first autonomous F1-mover in weeks
The arbiter loss is precision, not recall (product v3.1: P 0.406 / R 0.722, 19 FP). Characterising
those 19 FP vs the 13 TP (`fp_characterize.py`, product `selftrain_120`) found ONE feature that
separates them cleanly and physically: **outward** = how far a CP's entry point sits along its own
body→point ray relative to the hull, `(p-bc)·û / max_v (V-bc)·û`.

| | mean outward |
|---|---|
| TP (real manufacturer CPs) | **0.767** (on the outer housing, where wires enter) |
| FP (spurious) | **0.539** (further inside) |

A **round-prior** gate (not fitted to the weakest TP) keeps ALL 13 TP across the whole band and
drops FPs — end-to-end arbiter, verified through `cp_openings.connection_points(outward_min=…)`:

| outward_min | TP | FP | FN | F1 |
|---|---|---|---|---|
| 0.00 (product) | 13 | 19 | 5 | **0.520** |
| 0.50 | 13 | 12 | 5 | **0.605** |
| 0.60 | 13 | 12 | 5 | 0.605 |
| 0.70 | 13 | 8 | 5 | 0.667 |

Why this is more than the last false alarms: recall is untouched (all TP kept for ANY threshold in
0.50–0.70, since the weakest TP is at 0.730), it is a round prior not a fitted edge, and it is
physically grounded (connection openings are on the accessible outer housing; 17/19 FP are also
Contact-sourced interior blobs). **NOT adopted yet** — measured on n=18 CPs / 9 parts; it must be
confirmed HELD-OUT on the batch-2 human labels (`_label_targets_2`) before flipping on, and a
recessed real CP (deep gland) is a possible failure. Implemented OFF-by-default
(`cp_openings.py outward_min`, `cp_config.json outward_min_CANDIDATE`). If it holds on batch 2, it
raises the product CP F1 from 0.520 to ~0.60 with **no retraining**.

## ⚠️⚠️ CORRECTION 2 (2026-07-20, cp-v3): "CP = round hole only" was a FALLACY

The user caught it: CPs are NOT only round wire holes. Some parts have square/clamp entries as the
real connection point. cp-v2 (CableEntry-only, frozen 2026-07-18) baked in that fallacy and its
"manufacturer-validated" claim rested on only 2/102 parts.

**Arbiter test** (`measure_cp_defs.py`): raced 4 candidate CP definitions against the ONLY
part-matched manufacturer ground truth available — 9 in-scope PXC terminal blocks
(`_cad_eval_pxc/`) with their real `Desktop\JSON` `ConnectionPoints` (18 CPs), frame-aligned via
`cad_eval.align_frames` (residual 0.27–1.78mm):

| definition | P | R | F1 |
|---|---|---|---|
| **v2 (CableEntry-only, the old default)** | 0.000 | 0.000 | **0.000** — finds NONE of the 18 |
| v1 (CE+Contact, dedupe, no depth gate) | 0.213 | 0.722 | 0.329 |
| **v3a (CE + Contact gated by insertion depth ≥1mm) — WINNER** | 0.265 | 0.722 | **0.388** |
| v3b (CE + Contact, no depth gate, dedupe only) | 0.255 | 0.722 | 0.377 |

cp-v2 scored **zero** because the PXC clamp-terminal parts have their real CPs labelled **Contact**,
not CableEntry — exactly the shape the user flagged. Geometric census independently confirms it:
of 18 manufacturer CPs, 13 sit on a round hole, **5 (28%) sit on no round/rect opening at all**
(square/clamp entries the round-hole detector is blind to).

**New default (cp-v3-thesis-connection-classes, `cp_openings.py`, `cp_config.json`)**: a CP is
every **CableEntry** component, PLUS every **Contact** component with thesis insertion depth
`|v_o - v_s| ≥ 1.0mm` (drops flat screw-heads/pads, keeps real recessed openings — often square).
This also matches the thesis itself (line 1606: v_o is derived for "Kabeleinführung ODER
Kontaktierung" — both classes were always connection openings; `TERMINAL_TYPES` in
`connector_constants.py` was always `{CONTACT, CABLE_ENTRY}`). cp-v2 stays reachable via
`classes=(CABLE_ENTRY,)` for provenance. **n=18 CPs is small — the only part-matched manufacturer
sample we have; not over-fit (4 simple candidates raced, no hand-tuned rules — RESULTS-11s lesson).**

**What changed / what didn't:**
- Product-model DECISION unchanged: refit91 still ranks #1 under the new merged "Connection IoU"
  metric (CableEntry+Contact merged, `measure_val_cableentry.py`) — **0.672**, vs all sweep variants
  ≤0.636 and rec_augcew 0.486. The metric-discipline conclusions (val≠generalisation, heuristic≠
  quality, data ceiling ~0.64, learning-curve ROI) all STAND — this correction is about CP
  *definition*, not model selection.
- `infer_step_cp.py` now derives cp-v3 CPs by default (green=CableEntry, orange diamond=Contact-CP);
  earlier "wrong location" visual-QA verdicts on 826-159/0294380000/etc. should be treated as
  PROVISIONAL pending re-render under cp-v3 (a "wrong" CP on a flat Housing patch is still wrong;
  one that lands on a real depth-gated Contact opening may now be correct — re-check before reusing
  those verdicts).
- Labelling guidance corrected: annotators paint by what a feature physically IS (CableEntry = cable
  gland/conduit entry; Contact = clamp/screw terminal, often square, IS a CP if it has real depth) —
  not "paint everything CableEntry". See `_label_targets/README.md` and `label_tool.html`.

### cp-v3.1 (2026-07-20): F1 0.388 → **0.520** by fixing PRECISION (recall untouched)
Recall (0.722) is segmentation-bound → needs labels. Precision was the lever: on a 2-terminal part we
emitted 4–10 CPs, because one manufacturer terminal has several connection features (clamp, screw, wire
slot) each segmented separately. Fix = **one CP per terminal** + noise suppression:

| definition | P | R | F1 | FP |
|---|---|---|---|---|
| cp-v2 (CableEntry only) | 0.000 | 0.000 | 0.000 | — |
| cp-v1 (CE+CT, no gate) | 0.224 | 0.722 | 0.342 | 45 |
| cp-v3 (CE + depth-gated CT) | 0.265 | 0.722 | 0.388 | 36 |
| **cp-v3.1 (+conf 0.7, min_v 60, cluster 10mm)** | **0.406** | **0.722** | **0.520** | **19** |

Adopted recipe (`cp_config.json.prediction_postproc`, wired into `infer_step_cp.py`): `min_v=60`,
`vertex_conf=0.7`, `cluster_mm=10`, `ct_depth_min_mm=1.0`. Chosen to **preserve recall — all 13 TPs
kept** — while halving FP. `cp_openings._cluster_terminals` does the single-linkage per-terminal merge
(PREDICTION-side only; GT is never clustered, `cluster_mm=0` default). Sweep: `improve_cp_precision.py`
→ `results/cp_precision_sweep.json`.

**Two honest notes.** (1) `vertex_confidence_mask` moved 0.9 → **0.7**: 0.9 was too aggressive and
killed true positives (recall 0.722 → 0.500). (2) **A first version of the sweep was WRONG and I caught
it by cross-checking**: it hand-rolled the forward pass with `meta.get("n_eig", 48)` while refit91 uses
**n_eig=64**, silently building the wrong eigenbasis and reporting flattering numbers (F1 0.60/0.70).
The arbiter and the sweep disagreed (0.439 vs 0.596); a per-part diff proved the clustering was
identical and the *predictions* differed. Fixed to `D.predict(return_probs=True)`; sweep and arbiter now
agree exactly. **Never report a number that two independent paths do not reproduce.**

**Re-grade of yesterday's visual QA under refit91 + cp-v3** (`infer_step_cp.py`, green o=CableEntry,
orange ◆=Contact):
| part | old verdict (rec_augcew, cp-v2) | new (refit91, cp-v3) | reversed? |
|---|---|---|---|
| 0294380000 | 3 CP at the foot / abstain — WRONG | 3 Contact CP on the clamp terminals — roughly right | **YES** |
| 826-159 | 1 CP in centre of a flat face — WRONG | 1 Contact CP at an edge; spurious central blob GONE | partly (was a rec_augcew artefact) |
| 1231700 | 2 CP double-ball — WRONG | 7 Contact CP scattered — still WRONG (over-count) | no — needs labels |
| 1SNA511003R0500 | 3 CP scattered — WRONG | 0 CP (refit91 abstains) — unusual geometry | no — needs labels |
| 0321019 (control) | 2 CableEntry — good | 2 CableEntry + 3 Contact = 5 | **over-count risk** |
| 0270018 (control) | 5 CableEntry | 2 CableEntry + 4 Contact = 6 | over-count risk |

Two honest lessons: (a) several of yesterday's "wrong location" verdicts were measuring **rec_augcew**
(the over-eager model I wrongly picked), not the product refit91 — they conflated two separate bugs.
(b) **cp-v3 has a real precision tradeoff, and the arbiter can't see it:** the 9-part arbiter set is ALL
PXC clamp-terminals (where cp-v2 scored 0), so it maximally favours cp-v3; but on **feed-through**
terminals with a Contact clamp band (0321019, 0270018) cp-v3 ADDS Contact CPs beyond the manufacturer's
per-side count = likely over-count. The `ct_depth_min_mm=1.0` gate is tuned on 9 clamp parts only.
**So cp-v3 is the evidence-based CORRECTION of a proven-wrong definition (Contact clamps ARE CPs), NOT a
finished definition** — refining the Contact gate needs manufacturer-matched GT across BOTH terminal
types (feed-through + clamp), which we do not yet have. Do not present cp-v3 counts as exact.

Full evidence: `results/cp_def_eval.json`, `cp_openings.py` module docstring, `cp_config.json`.

## ⚠️ CORRECTION (2026-07-20): the "recipe-tuned rec_augcew" was NOT the win it looked like

Last night's headline was "rec_augcew_s0: clean 34/40, no_cableentry 0 = best generalisation."
An FBI review measured the ONE thing that actually drives CP placement — **CableEntry
segmentation IoU on HUMAN-GT val (20 parts, locked-11 untouched)** — and it REVERSED the call:

| model | **CableEntry IoU (human-GT val)** | acc | mIoU | failure mode |
|---|---|---|---|---|
| **refit91** (in-domain) | **0.649** | 0.837 | 0.641 | conservative — ABSTAINS when unsure (0 CP) |
| rec_augcew_s0 ("product") | **0.424** | 0.638 | 0.412 | over-eager — paints CableEntry anywhere |

**rec_augcew segments CableEntry 35% WORSE on human GT.** The `clean 34 / no_ce 0` heuristic
rewarded the model for painting CableEntry *somewhere* (dropping "complete misses") while its real
segmentation quality collapsed. Visual A/B confirms the mechanism: on 0294380000 and 826-159
(where rec_augcew put CPs at the foot / centre of a flat face) **refit91 outputs 0 CP — it
abstains instead of mis-placing.** For a robot product a wrong CP is worse than a flagged miss, so
the conservative, higher-quality model is the better product.

## ✅ THE PRODUCT MODEL (frozen 2026-07-20) = `results/seg_extra/selftrain_120.pt`
| | |
|---|---|
| checkpoint | `results/seg_extra/selftrain_120.pt` |
| training | 71 human-labelled + 122 confident model pseudo-labels (`make_pseudo_labels.py`) |
| **val (human-GT, 20 parts)** | **Connection IoU 0.676 · CableEntry 0.705 · mIoU 0.664 · acc 0.844** |
| **vs the thesis** | thesis: CableEntry IoU **0.472**, mean Jaccard **0.514**, Dice 0.656 → **we beat it substantially** |
| thesis-compliance | **endorsed**: the thesis conclusion explicitly recommends "die Rückführung der Modellvorhersagen in das Labelingtool, um eine Feedbackschleife zu erstellen" (feed model predictions back as a labelling feedback loop) as the recommended future work. Architecture, task, augmentation (rotation/noise) and CP derivation are unchanged from the thesis. **Caveat: the thesis pictures a human correcting in the tool; we used the confident predictions directly (no human step).** |
| CP-level arbiter | F1 **0.520** — identical to refit91: the segmentation gain does NOT move the CP metric |
| deploy | `infer_step_cp.py` (now defaults to this checkpoint) |

### Previous product (kept for provenance) = `results/scheffler_semantic/refit91.pt`
| | |
|---|---|
| checkpoint | `results/scheffler_semantic/refit91.pt` |
| **CableEntry IoU (human-GT)** | **val 0.649**, locked-11 **0.648 pooled / 0.741 macro** (leakage-safe, the real number) |
| overall (locked-11 human-GT) | acc **0.858** / mIoU **0.680** / Dice 0.805 — beats thesis 0.51 |
| failure mode | conservative: abstains (0 CP) or under-counts multi-pole when unsure — SAFE for a robot |
| deploy | `infer_step_cp.py --ckpt results/scheffler_semantic/refit91.pt --device cpu` |

### Honest tradeoff (it is precision/recall, not a slam-dunk)
- **refit91**: higher measured quality, safe failure mode, but UNDER-counts on multi-pole
  (0270018: 2 CP, misses the right-side slots) and misses more truly-unseen parts entirely.
- **rec_augcew**: fires on more unseen parts (lower no_ce on gen_dev) but places WRONG when it
  fires (0.424 IoU, mis-placed). Its "win" was on an UNLABELLED heuristic that rewards the wrong
  thing. Keep only as the documented cautionary example.
- Both miss/under-count on unusual geometry → the real fix is human labels, not model choice.

## FBI findings this round (all MEASURED, negatives kept)
1. **Seed robustness (3 seeds, gen_dev_40): clean 34/33/29, no_ce 0/3/2.** `no_ce=0` was seed-luck;
   robust no_ce ≈ 2. Never advertise "finds CableEntry on every part."
2. **Geometry×semantics fusion: DEAD (killed by the doability gate before any code).** frames are
   identical (offset 0.0mm) but `step_openings` CAD cylinders (screw/mounting bores) do NOT coincide
   with the semantic CableEntry regions — even on the GOOD control parts (0270018 4/5 seg CPs 11-18mm
   from any opening; 0321019 3/3 15-17mm away). Gating by geometry would DESTROY the good placements
   = the RESULTS-11s failure replayed. Re-confirms the project's root truth: CAD geometry ≠ human
   semantics. Gate scripts: scratchpad `step0_fusiongate.py` / `step0_framecheck.py`.
3. **3-seed ensemble: clean 31 / no_ce 2.** Regresses to the seed mean, does NOT beat s0. Confirms
   the true no_ce≈2. Not a win. Tool: `measure_gen_ensemble.py`.
4. **The heuristic (`clean`/no_ce on unlabelled gen_dev) is anti-correlated with human-GT CableEntry
   quality** — the same "val mIoU lies" trap, in the other direction. Only human-GT (val/locked)
   gives a trustworthy CP-placement number: `measure_val_cableentry.py`.

## Metric discipline (the rule this project runs on)
1. **Select by a HUMAN-GT number, not a heuristic.** `clean`/no_ce on unlabelled parts is a
   heuristic and can reward the WRONG behaviour (over-painting). Use `measure_val_cableentry.py`
   (CableEntry IoU on human-GT val) as the selector; locked-11 is the one-shot final benchmark.
2. **A wrong CP is worse than an abstention** for a robot — prefer the conservative model.
3. **gen_dev_40 is DEV (model selection), NOT a benchmark.** `final_holdout_40.json` is the untouched
   final set (unlabelled → score only before a human labels it).
4. **Axis-snapped CP direction is a MANUFACTURER-CONVENTION output**, not a fit to a human angle.
5. **Don't over-train / don't chase heuristics.** More diverse real human-labelled DATA is the lever.
6. `train_seg_extra.py` requires `--checkpoint-out` (never clobber a model implicitly).

## Recipe sweep (2026-07-20) — CONFIRMS the training ceiling on the HONEST metric
Ran a 4-variant sweep with `train_seg_extra.py` (71 train, held-out val, **selected by val CableEntry
IoU** — the honest CP driver, not the heuristic). Ranked val CableEntry IoU:

| model | CableEntry IoU | note |
|---|---|---|
| refit91 | **0.649** | current product (val slightly optimistic — refit on val after freeze) |
| sw_v1_base (no augment) | 0.639 | held-out & FAIR — **tied with refit91 within 20-part noise** |
| sw_v3_notversky (tversky 0) | 0.639 | tversky does NOT affect CableEntry |
| sw_v4_ce15 (CE-weight ×1.5) | 0.633 | even a MILD CE boost slightly hurts |
| sw_v5_thesisw (refit91's thesis weights, held-out) | 0.605 | thesis weights HURT held-out — refit91's 0.649 is the val-refit, not the weights |
| sw_v2_mildaug (augment 0.35) | 0.589 | augment hurts quality even when mild |
| rec_augcew_s0 (heuristic-tuned) | **0.424** | the celebrated "win" is by far the WORST |

**Resolved:** refit91's 0.649 is NOT reproducible via its thesis class weights held-out (those give
only 0.605); the edge is its val-refit (optimistic). **inv-freq is the best held-out recipe (v1_base
0.639).** So the honest held-out ceiling on 71 parts is ~0.64, and refit91 stays the product only
because it carries the leakage-clean locked-11 number (0.648/0.741); v1_base was not run on locked-11.

**Findings:** (1) No tested recipe beats refit91; the plateau at **~0.64** across diverse recipes is a
DATA ceiling on 71 parts, not a recipe ceiling. (2) Augment and CE-weight both HURT on the honest
metric — reconfirming the rec_augcew "win" optimised a misleading heuristic. (3) Honest caveat: val is
only 20 parts, so refit91 vs v1_base is a statistical tie — do NOT claim refit91 strictly superior;
claim only that nothing beat it and the ceiling is ~0.64. Selector: `measure_val_cableentry.py`.
Label_assist confirms it: refit91 predicts 0 CableEntry on **17/22** queue parts (abstains on the hard
families) → those need human labels, full stop. `label_assist.py` renders a bias-safe VERIFY reference
per queue part (neutral template untouched — no automation bias).

## Learning curve (2026-07-20) — how many more labels? (measures the labelling ROI)
Trained on seed-fixed TRAIN subsets (val kept full 20), `train_seg_extra.py --train-limit N`, val
CableEntry IoU: **15→0.418, 25→0.553, 45→0.572, 71→0.639.** Per-part gains: 15→25 +0.135, 25→45
+0.019 (noisy flat), 45→71 +0.067. **The curve is STILL RISING at 71 with diminishing returns.**
Rough extrapolation of the 45→71 slope: ~**+40–60 more labelled parts for a meaningful gain**
(e.g. toward 0.75), consistent with the earlier "100–150 total" estimate. Honest caveats: single seed
per point, 20-part val is noisy (hence the 25→45 flat spot). Plot: `results/learning_curve_cableentry.png`.
**Takeaway: human labels have GOOD measured ROI — they are not just the only lever, they demonstrably
work. This is the concrete answer to "is labelling worth it": yes, ~+40–60 parts.**

## EEC extra data RE-TESTED (2026-07-20) — the earlier "didn't help" verdict was from a BROKEN run
**Bug found:** `train_seg_extra.load_extra()` used `d.rstrip("/")`, but Windows `glob` returns a
trailing **backslash** → `basename()` returned `""` → **every one of the 132 EEC parts was silently
skipped** while the run reported success. So the 2026-07-19 ablation that concluded "+132 EEC didn't
help" never actually loaded the data. Fixed to `os.path.normpath` (verified: 132 load, 19 terminal).

First VALID measurement (val Connection IoU, human-GT; training-time numbers independently
re-verified by `measure_val_cableentry.py` — they match exactly):

| training set | parts | Connection IoU | CableEntry | Contact |
|---|---|---|---|---|
| 71-only baseline | 71 | 0.622 | 0.639 | 0.614 |
| + 19 terminal-only EEC | 90 | 0.633 | 0.513 | 0.699 |
| + all 132 EEC | 203 | **0.637** | 0.572 | 0.666 |
| refit91 (product) | 71(+val refit) | **0.672** | 0.649 | 0.680 |

**Findings.** (1) The extra expert data DOES help, but only **+0.015** Connection IoU — real yet flat.
(2) It shifts the model toward **Contact** and away from CableEntry, because EEC labels push-in
terminals' CPs as Contact (verified directly: 2002-1208 has CableEntry **0%**, its CPs are Contact) —
independent confirmation of the cp-v3 correction. (3) **Per-part value is wildly uneven:** the 19
terminal parts bought +0.011, the other 113 non-terminal parts only +0.004 → a terminal-block part is
worth roughly **16x** a non-terminal one. (4) Product decision UNCHANGED: refit91 (0.672) still leads;
the recipe gap (~0.05) is larger than the data gain (+0.015), so more EEC data cannot close it.
**Conclusion: the free public data is now genuinely squeezed (+0.015 total). Same-distribution WSCAD
expert labels remain worth several times more per part (our own curve: +26 parts → +0.067).**

## SELF-TRAINING (2026-07-20) — works, saturates fast, does NOT move the CP metric
Human labelling is blocked (user has no electrical domain knowledge; my own blind/datasheet-informed
attempts scored Connection IoU 0.408 / 0.276 vs the model's 0.672 → my labels would REGRESS it).
Remaining autonomous lever: pseudo-label the **3715 unlabelled WSCAD parts** with refit91, keep the
most confident, retrain. `make_pseudo_labels.py` → `_pseudo_extra/`, `train_seg_extra --pseudo-dir`.

| model | train set | Connection | CableEntry | mIoU | acc |
|---|---|---|---|---|---|
| 71-only (same recipe) | 71 | 0.622 | 0.639 | 0.622 | 0.824 |
| refit91 (previous product) | 71 (+val refit) | 0.672 | 0.649 | 0.641 | 0.837 |
| **selftrain_120** | 71 + 122 pseudo | **0.676** | **0.705** | **0.664** | **0.844** |
| selftrain_450 | 71 + 450 pseudo | 0.675 | 0.664 | 0.632 | 0.830 |

**Honest reading.** (1) Real gain vs the SAME-recipe baseline: **+0.054** — self-training closed the
whole gap to refit91 without refitting on val (so it is the fairer number of the two). (2) Gain vs
refit91 is **+0.004 = noise on a 20-part val**; the meaningful difference is CableEntry **0.705 vs
0.649** (a genuine record) and it wins all four metrics. (3) **Scaling SATURATES:** 120 → 450
pseudo-labels changed nothing (0.676 → 0.675). Expected — pseudo-labels carry the model's own
knowledge, so the first batch acts as a regulariser and more adds no new information. (4) **CRITICAL:
the CP-level arbiter F1 is UNCHANGED at 0.520** (identical tp13/fp19/fn5 under the cp-v3.1 recipe;
the model really did change — the other definitions moved, e.g. v3a 0.388 → 0.361). So the
segmentation gain does NOT translate into better CP placement on the manufacturer set.
**Verdict: a real but modest win. Self-training is now also squeezed. Do not oversell it.**

## CP DIRECTION validated + recall ceiling DIAGNOSED (2026-07-20)
Two gaps that were never checked, both now measured against the manufacturer JSON.

**1. Direction — the unmeasured half of the product.** Everything so far scored CP POSITION; the
robot also needs the approach vector, and the manufacturer JSON carries `InsertDirection` for every
CP. `measure_cp_direction.py` → **13/13 matched CPs at exactly 0.0° error (100% within 15°, none
inverted)**. So the axis-snap convention, the outward orientation and the frame alignment are all
correct. Honest caveat: both sides are axis-aligned (6 possible values), so the test is "easy" —
but picking the right one of 6 on 13/13 (random ≈17%) does validate the convention.
Receipt: `results/cp_direction_eval.json`.

**2. Why recall is pinned at 0.722 — it is NOT thresholds and NOT the point definition.**
- The model DOES see the 5 "missed" CPs: at 3048357 it predicts **28/28 vertices CableEntry at 0.97
  confidence** right at the manufacturer CP; 3031238 and 3212140 show 81–91 Contact vertices there.
- But **no post-processing setting recovers them**: swept min_v 5→60, vertex_conf 0→0.7, depth 0→1.0
  — recall stays *exactly* 0.722 (tp13) across the whole grid.
- **Standoff offset rejected**: shifting the CP outward along its direction (the old "manufacturer
  point is ~4.7mm off-surface" note) makes it strictly worse (F1 0.520 → 0.120 at 4.7mm).
- **Point definition: the thesis is right.** v_o (opening midpoint) **F1 0.400** vs vertex centroid
  0.218 vs v_s 0.259 — the thesis's v_o choice is independently validated.
→ **Diagnosis: the segmented region's SHAPE/EXTENT is wrong on those geometries, so its opening
midpoint lands out of tolerance even though the class is right.** That is a segmentation-quality
problem on unfamiliar geometry — precisely what more labelled examples of those families fix, and
nothing in post-processing can. The "human labels" conclusion now rests on a verified mechanism
rather than an assumption.

## The real lever (unchanged, now the ONLY one left): human labels
Every autonomous lever is exhausted and measured negative (TTA, remesh density, synthetic, +132
non-terminal EEC, recipe past CE-weight, geometry fusion, ensemble). The families where BOTH models
fail (multi-pole under-count, unusual geometry) need human segmentation labels.
- Queue READY: `_label_targets/` (22 parts: remeshed OBJ + template + README), prioritised in
  `labeling_queue.json` (8 no_cableentry first, then 9 noisy).
- Human labels are needed to TRAIN the missed families AND to MEASURE placement (currently only
  20 val + 11 locked parts have human GT for CableEntry).

### First 20 human labels ARRIVED (2026-07-20) — and they HURT. Root cause: CONVENTION, not quality
The annotator labelled 20 queue parts (CableEntry only, via `label_tool.html`). Trained with a
masked BCE (`--partial-dir`) so the unmarked classes are not taught as Housing.

| run | val Connection IoU | arbiter CP F1 |
|---|---|---|
| **selftrain_120 (product)** | **0.676** | **0.520** |
| B = 71 + 450 pseudo + 20 human | 0.630 | 0.351 |
| A = 71 + 20 human | 0.586 | 0.308 |

The labels are GOOD: exact vertex counts, only classes 0/3, 2051 CableEntry vertices, visually
symmetric and sitting on the wire entries. The problem is **extent**: the human marks cover
**1.45 %** of vertices, the 71 expert corpus parts cover **4.99 %** — the same openings described
in two different languages, so the model is torn between two definitions of the class boundary.

**MY PROCESS ERROR, not the annotator's**: I wrote the labelling guide and never showed how WIDE
the expert regions are. Any future labelling brief MUST state the target extent (~5 % of vertices)
and show calibrated expert examples.

**Fix tested → the convention hypothesis is DISPROVEN.** `dilate_partial_labels.py` ring-expands
each marked region over mesh edges to the corpus convention (measured 1.49 % → 5.48 %, visually
still on the openings, no housing spill — `results/_dilated_check.png`). Retrained (`run_dilated.sh`):

| run | val Connection IoU |
|---|---|
| A  = 71 + 20 human, tight 1.49 % | 0.5858 |
| **A2 = 71 + 20 human, dilated 5.48 %** | **0.5837** |

A2 ≈ A ⇒ label EXTENT was never the cause. B2 (with pseudo) was interrupted by a planned shutdown.

### ROOT CAUSE FOUND: a CLASS-ASSIGNMENT conflict, and it is the cp-v3 issue again
Found while building `make_hybrid_labels.py`: the product model overlapped the human CableEntry
marks on **0 % of vertices on 18/20 parts**. Diagnosing that (`class_conflict.py`) gave the answer —
over every human-marked vertex, the corpus-trained model predicts:

| model says | share |
|---|---|
| **Contact** | **49.8 %** |
| Housing | 32.0 % |
| LabelSurface | 13.6 % |
| CableEntry | 4.5 % |

The annotator marked **square/clamp entries**, which the corpus ontology calls **Contact**. So the
human label ("CableEntry here") and the corpus label ("Contact here") contradicted each other on
about half the marks and cancelled out. This is the SAME cp-v3 fallacy the user caught on the
derivation side, resurfacing on the training side.

**Fix (no new annotation work): supervise the CONNECTION channel.** The masked BCE now targets
`p_CableEntry + p_Contact` (`--partial-target connection`, the new default). The annotator's claim
is "a wire goes in here" = a connection region; whether the corpus calls it CE or CT is an internal
convention the model can keep deciding. Thesis-faithful: cp-v3 defines a CP as CableEntry **or**
Contact, and both product metrics score the union.

**RESULT — the human labels flip from harmful to net-positive.**

| run | val Connection IoU |
|---|---|
| A = 71 + 20 human, CableEntry channel | 0.5858 |
| A2 = 71 + 20 human, dilated | 0.5837 |
| 71-only baseline | 0.622 |
| **C1 = `human20c.pt`, connection channel** | **0.6325** |

Arbiter, same code, every candidate definition (`measure_cp_defs.py`):

| CP definition | product `selftrain_120` | **C1 `human20c`** |
|---|---|---|
| v2_ce_only | 0.000 | 0.100 |
| **v1_ce_ct_dedupe** | 0.328 | **0.619** |
| v3a_ce_ct_depth | 0.361 | 0.585 |
| v3b_ce_ct_nodepth | 0.344 | 0.578 |
| v3_1_precision (frozen product recipe) | **0.520** | 0.533 |

The gain is PRECISION: under v1 the product emits 38 false positives, C1 emits 11. The human labels
taught the model where a connection is *not*.

**Honest caveats — read before quoting 0.619.**
1. Under the FROZEN recipe (v3.1) the gain is 0.520 → 0.533, thin for n=18 manufacturer CPs.
2. Reaching 0.619 means switching the recipe to v1, and that recipe was chosen on the SAME 9 parts
   it is scored on — recipe selection on the test set. Treat 0.619 as an upper bound, not a claim.
3. What is convincing is not any single number but that C1 ≥ product under **all five** definitions.
4. Product val Connection IoU (0.676) is still above C1 (0.6325); those two metrics disagree because
   val is scored against corpus-convention labels while the arbiter is scored against real
   manufacturer CPs. The arbiter is the product target.

**SEED CONFIRMATION KILLED THE ARBITER "WIN". Product stays `selftrain_120`.** The seed-0 numbers
above (0.533 on v3.1, 0.619 on v1, FP 19→11) did NOT replicate. On the frozen v3.1 recipe:

| model | arbiter CP F1 (v3.1) |
|---|---|
| product `selftrain_120` | **0.520** |
| C1 `human20c` seed 0 | 0.533 |
| C1 `human20c_s1` seed 1 | 0.419 |
| C1 `human20c_s2` seed 2 | 0.489 |
| **C1 seed mean** | **0.480 ± 0.05** |

Only 1 of 3 seeds clears 0.520; the mean (0.480) is below the product. My pre-registered rule (≥2
seeds must clear 0.520) is NOT met, so C1 is NOT adopted. **The equal-recall FP win (19→11) was also
a seed-0 artefact**: v1 FP counts across seeds were 11 / 33 / 29 — seed 0 was the lucky draw, the
other two are worse. This is the rec_augcew lesson repeating exactly; the seeds are why I ran them.

**C2 (`human20c_pseudo`, connection channel + 450 pseudo) is also REJECTED.** It tops val Connection
IoU (0.6994, the best of any model) but is the WORST on the arbiter (v3.1 F1 **0.464** < 0.520). This
is the second time val Connection IoU and the arbiter point opposite ways — select on the arbiter,
never on val Connection IoU alone.

### What SURVIVED the seeds (the real, keepable result)
1. **The class-conflict diagnosis** (a measurement, seed-independent): over every human-marked vertex
   the corpus model says Contact 49.8 % / CableEntry 4.5 %. The annotator marked square/clamp entries
   the corpus calls Contact. This is why the CableEntry-channel runs (A/A2) were stuck at ~0.585.
2. **Supervising the CONNECTION channel is the correct way to use partial human labels**, and this
   IS seed-robust: val Connection IoU 0.6325 / 0.6574 / 0.6415 (mean 0.644) vs the CableEntry-channel
   0.586 — +0.05 on all three seeds, and back up to the 71-only baseline (0.622).

### Honest verdict on the 20 human labels
They are correctly placed and the connection-channel fix makes them non-harmful, but **20 labels do
not reliably move the real CP metric** — the apparent gain was within seed noise. Consistent with the
learning-curve ROI: a step change needs ~40–60 labels, not 20. The labelling pipeline and the
loss-side fix are now correct and ready; the lever is more labels, not a cleverer use of these 20.
