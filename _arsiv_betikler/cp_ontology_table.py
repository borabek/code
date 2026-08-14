# -*- coding: utf-8 -*-
"""Reconcile the CP ontology against manufacturer connection counts, per the audit's P0.

For every part: cableentry_cp (cp-v2 count), contact_components (auxiliary), the manufacturer
count where known, a status, and needs_human. Status:
  match          : cableentry_cp == manufacturer  (cp-v2 confirmed)
  over_v1_only   : cableentry_cp == manufacturer but Contact+CableEntry would over-count
  gt_incomplete  : manufacturer > 0 but cableentry_cp == 0 (human labels miss the connections)
  mismatch       : manufacturer known and != cableentry_cp (and not gt_incomplete)
  unknown        : manufacturer count not available offline (needs catalog lookup)

Manufacturer counts are the 4 the user supplied from product pages; the other ~98 are external
catalog research (needs_human/needs_lookup). This builds the framework + fills what we have.

Output: results/cp_ontology_table.json  (+ printed summary).
"""
import json, os
import numpy as np
import scheffler_dataset as dataset
import connector3d, cp_openings

CE = int(connector3d.CABLE_ENTRY); CT = int(connector3d.CONTACT)
# user-supplied manufacturer connection counts (from product pages / Phoenix Contact)
MFR = {"0271017": 8, "0444048": 2, "0446017": 2, "3209635": 3}
GT_MIN = 1


def status(pid, ce, ct, mfr):
    if mfr is None:
        return "unknown", True
    if ce == mfr:
        return "match", False
    if mfr > 0 and ce == 0:
        return "gt_incomplete", True
    return "mismatch", True


def main():
    rows = []
    for sp in ("train", "val", "test_locked"):
        for s in dataset.load_split("wscad_corpus_scheffler_exact", sp, allow_locked=True, verify_hashes=False):
            pid = s["part_id"]; V = np.asarray(s["verts"], float); F = np.asarray(s["faces"], int); L = np.asarray(s["labels"])
            ce = len(cp_openings.connection_points(V, F, L, min_v=GT_MIN, classes=(CE,)))
            frags = connector3d.build_fragments(V, F, L, min_vertices=GT_MIN)
            ct = sum(1 for f in frags if int(f.label) == CT)
            mfr = MFR.get(pid)
            st, need = status(pid, ce, ct, mfr)
            rows.append({"part_id": pid, "split": sp, "cableentry_cp": ce, "contact_components": ct,
                         "manufacturer_count": mfr, "status": st, "needs_human": need,
                         "note": ""})
    # notes for the priority cases
    notes = {"0271017": "mfr 8 but 0 CableEntry AND 0 Contact -> human segmentation misses all connections",
             "3209635": "mfr 3 but 0 CableEntry (2 Contact) -> CableEntry label missing; do NOT count Contact",
             "0444048": "cp-v2 (2) matches mfr; Contact+CableEntry would give 5 -> over-count",
             "0446017": "cp-v2 (2) matches mfr (Phoenix Contact page); Contact+CableEntry gives 5 -> over-count"}
    for r in rows:
        r["note"] = notes.get(r["part_id"], "")
    os.makedirs("results", exist_ok=True)
    known = [r for r in rows if r["manufacturer_count"] is not None]
    summary = {"n_parts": len(rows), "n_manufacturer_known": len(known),
               "n_match": sum(1 for r in known if r["status"] == "match"),
               "n_gt_incomplete": sum(1 for r in rows if r["status"] == "gt_incomplete"),
               "n_needs_human": sum(1 for r in rows if r["needs_human"]),
               "total_cableentry_cp": sum(r["cableentry_cp"] for r in rows),
               "total_contact_components": sum(r["contact_components"] for r in rows),
               "hypothesis": "CableEntry = physical CP (matches mfr on cleanly-labelled parts); Contact = auxiliary, not counted",
               "rows": rows}
    json.dump(summary, open("results/cp_ontology_table.json", "w"), indent=1)
    print(f"{len(rows)} parts | mfr-known {len(known)} | match {summary['n_match']}/{len(known)} | "
          f"gt_incomplete {summary['n_gt_incomplete']} | needs_human {summary['n_needs_human']}")
    print(f"{'part':<10}{'split':<12}{'CE_cp':>6}{'Contact':>8}{'mfr':>5}  status")
    for r in rows:
        if r["manufacturer_count"] is not None or r["status"] == "gt_incomplete":
            print(f"{r['part_id']:<10}{r['split']:<12}{r['cableentry_cp']:>6}{r['contact_components']:>8}"
                  f"{str(r['manufacturer_count']):>5}  {r['status']}")
    print("-> results/cp_ontology_table.json (all 102, needs_human flags the catalog lookups)")


if __name__ == "__main__":
    main()
