# Strategy FBI — cp-v2 vs the thesis, the JSON corpus, and our new direction (2026-07-18)

Interrogating whether our approach is faithful to the three sources of truth, and where it has
contradictions / weaknesses / bugs. Findings ordered by severity, each with what I did.

## F1 (BIGGEST) — the JSON corpus was sidelined; metric #3 was declared "unmeasurable" while we sit on 11,927 real CPs
`Desktop\JSON` = 479 watertight meshes (~5500 verts, the thesis's remesh target) each carrying real
manufacturer ConnectionPoints (Point + InsertDirection + Name), 11,927 total, 0 zero-CP parts. This
is the most direct CP ground truth in the whole project, and we had it labelled "metric #3: NOT
MEASURED, no human labels exist." That was wrong: the JSON IS a manufacturer point+direction CP
benchmark.
**Done:** (a) validated cp-v2's CONVENTIONS against it (`json_cp_convention.py`): InsertDirection
100% axis-aligned, 98.3% outward — exactly cp-v2's direction model; (b) built the in-domain
benchmark harness (`json_cp_benchmark.py`: dataset accessor + SHA-deduped 341/74/64 split +
point-and-direction scorer at 5mm/15deg). Metric #3 is now MEASURABLE.
**Honest limit:** JSON families (A-B etc.) are OUT-of-domain vs the WSCAD-STEP terminal-block
target, and the JSON->STEP domain gap is proven ([[domain-gap-is-the-blocker]]). So this is an
in-domain CAPABILITY benchmark, not the target metric. A learned detector is known feasible
in-domain (M0 overfit F1~1.0); a clean held-out number needs training (next step).

## F2 — cp-v2's CP POINT differs from the manufacturer's by ~4.7 mm (a standoff offset)
The manufacturer CP Point sits ~4.7 mm OFF the mesh surface (only 25.5% within 2 mm) — it is a
standoff / insertion point, not the on-surface opening midpoint cp-v2 emits (v_o, ~1-2 mm on the
region). For a robot target and for scoring against manufacturer CPs, these differ by ~one opening
radius. **Action (flagged):** the metric-#3 scorer must target the manufacturer point; cp-v2 should
emit BOTH the opening midpoint AND a standoff point (connector3d already has `standoff_point` =
entry_point + approach_vector*distance) — wire the standoff into the CP record before comparing.

## F3 — cp-v2 vs the thesis: the thesis localises BOTH Contact and CableEntry; cp-v2 counts CableEntry only
Thesis §5.3.6/Abb.44 computes position+normal for **every Kontaktierung AND Kabeleinfuehrung**
feature (for cabinet assembly + wiring). cp-v2 counts CableEntry only, treating Contact as
auxiliary. This is NOT unfaithful — the user's frozen task is "physical CABLE-ENTRY CP point +
approach direction", so CableEntry-only is the right target, and it matches the manufacturer's
1-CP-per-terminal count (validated 0444048/0446017 2=2). **But note:** if the robot must also
actuate the screw/clamp, the Contact feature is a SEPARATE required output — cp-v2 does not drop
it from the segmentation, only from the CP count. Keep Contact localisation available.

## F4 — cp-v2 does NOT scale; the full WSCAD-STEP target still needs the JSON+remesh bridge
cp-v2 derives CPs from the human CableEntry SEGMENTATION, which exists for only 102 Scheffler parts.
It cannot label the 3535 untouched WSCAD STEP parts (`benchmark_candidates.json`) without more human
segmentation. The scalable path to the user's real target is the thesis plan: train a CP/segmentation
model on the JSON GT (11,927 CPs) and bridge to WSCAD STEP via UNIFORM REMESHING
([[thesis-remesh-solution]]) — which is currently DORMANT. Strategic fork the user must pick:
(a) hand-label more WSCAD terminal blocks (Scheffler-style), or (b) revive JSON-train + remesh-transfer.

## F5 — metric-vs-thesis comparison hygiene
The thesis reports 0.51 Jaccard (segmentation IoU). Our fair comparison is semantic IoU 0.680 >
0.51 (both segmentation) — correct in the report. The CP-F1 numbers are a DIFFERENT (downstream)
metric and must NEVER be compared to 0.51. Verified the report keeps them separate.

## F6 — GT labelling inconsistency (from round 2): 20/102 parts label connections as Contact, not CableEntry
Already surfaced; cp-v2 scores 82/102 and flags the rest gt_incomplete. This is the top GT-quality
issue and needs re-labelling. Under the JSON benchmark (F1) this problem disappears (manufacturer
CPs are explicit), which is another reason the JSON path is strategically stronger long-term.

## Bug sweep (correctness)
- gt_min_v=1 is safe: smallest CableEntry component is 45 verts (median 135); zero tiny/noise components.
- vertex_conf=0.9 re-validated UNDER cp-v2 on val (0.8->0.967, 0.9->0.989, 0.95->0.923) — still optimal.
- Added a test-peek guard: `cp_evaluate.py` refuses test_locked without `--allow-locked`.
- Full test suite green (7 CP + 66 total).

## Net
cp-v2 is faithful to the USER's cable-entry task and to the manufacturer CONVENTION (validated on
JSON). Its weaknesses are scale (F4) and the sidelined direct-CP corpus (F1), both now surfaced with
built infrastructure (harness + candidate set). The one concrete reconciliation to wire in next is
the standoff point (F2).
