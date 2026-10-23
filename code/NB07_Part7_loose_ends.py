# =============================================================================================
# NB07 — Part 7: Loose ends before manuscript writing (reasoning doc §14)
# =============================================================================================
# Paste each "CELL L#" block into its own Colab cell, AFTER the notebook's Parts 1-6 have been
# run in this session (definitions only are needed, plus Cell 18's DATE_PAIRS / run_trend_test).
# Every input is a Drive cache hit for these dates: no new Earth Engine tiling.
#
# Pre-declared reporting rule (applies to every cell below): each result is saved and reported
# as it comes out, whichever direction it points. None of these cells is allowed to change
# Table 13.1, the 500 m rule, or the own-threshold definition; they are sensitivity/provenance
# checks ABOUT those choices.
#
# Rough runtime: L1 ~15-25 min, L2 ~2 min, L3 ~20-30 min, L4 ~5-10 min (one ~90 s CBERS window
# read per scene not already cached), L5 seconds.
# =============================================================================================


# %% ------------------------------------------------------------------------------------------
# CELL L0 — shared helpers for Part 7
# ---------------------------------------------------------------------------------------------
import os, json, gc
import numpy as np

P7_DIR = os.path.join(RESULTS_DIR, "phase3_loose_ends")
os.makedirs(P7_DIR, exist_ok=True)

AUG = {  # S2 date -> augmented sample file already on Drive
    "2024-09-09": EXPORT_DIR + "/mvp2_bands_2024-09-09_cbers2024-09-09_aug.npz",
    "2024-09-14": EXPORT_DIR + "/mvp2_bands_2024-09-14_cbers2024-09-14_aug.npz",
    "2024-09-12": EXPORT_DIR + "/mvp2_bands_2024-09-12_cbers2024-09-14_aug.npz",
    "2024-08-08": EXPORT_DIR + "/mvp2_bands_2024-08-08_cbers2024-08-09_aug.npz",
    "2024-05-07": EXPORT_DIR + "/mvp2_bands_2024-05-07_cbers2024-05-08_aug.npz",
    "2024-02-05": EXPORT_DIR + "/mvp2_bands_2024-02-05_cbers2024-02-05_aug.npz",
}


def _save(name, obj):
    path = os.path.join(P7_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("saved:", path)
    return path


def _item_for(cbers_date, item_id):
    return next(it for it in find_cbers_items(cbers_date) if it.id == item_id)


def _with_weight(w, fn, *args, **kw):
    """Run fn with the global FDI_W_OURS temporarily set to w (own_threshold_open_water_auc reads
    that global). Always restored, even on error."""
    g = globals(); old = g["FDI_W_OURS"]; g["FDI_W_OURS"] = w
    try:
        return fn(*args, **kw)
    finally:
        g["FDI_W_OURS"] = old
        assert abs(globals()["FDI_W_OURS"] - (833.0 - 740.0) / (1610.0 - 740.0)) < 1e-12


def _fmt(r):
    if r is None or r.get("auc") is None:
        return f"not evaluable ({(r or {}).get('note', 'no result')})"
    return f"{r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  cells {r['n_cells']}  n {r['n_cand']}"

# ---- Version guard: Part 7 must run on the SAME definitions that produced Table 13.1 ----------
import inspect
_src = inspect.getsource(own_threshold_open_water_auc)
assert "own_thr=float(own_thr)" in _src and "cbers_item" in _src, (
    "own_threshold_open_water_auc in this session is NOT the NB07_..._final.ipynb version (Cell 17). "
    "Re-run Cell 17 (and Cells 12-15, 16b) from the _final notebook, then re-run L0.")
assert "spearman_exact_p_onesided" in inspect.getsource(run_trend_test), (
    "run_trend_test is not the _final Cell 18 version. Re-run Cell 18's definition cell, then L0.")
print("[L0] version guard passed: session definitions match NB07_..._final.ipynb")

print("Part 7 ready | FDI_W_OURS =", round(FDI_W_OURS, 4), "| FDI_W_PAPER =", round(FDI_W_PAPER, 4),
      "| output dir:", P7_DIR)


# %% ------------------------------------------------------------------------------------------
# CELL L1 — FDI interpolation-weight sensitivity (§14.1)
# ---------------------------------------------------------------------------------------------
# (a) Whole-scene: top-1% candidate-set overlap and FDI correlation, 740 nm weight vs the printed
#     lambda_RED (665 nm) weight, for the 5 trend-test dates + 09-12 (replaces the two
#     inconsistent sensitivity statements in §13.6 / Cell 85 with one saved record).
# (b) Headline sensitivity: own-threshold open-water AUC and the trend test recomputed with the
#     printed weight. Same background, same 500 m rule, same cells, same bootstrap.
def weight_overlap(R):
    w = water_mask(R)
    fo = fdi_from_dn(R["B6"][w], R["B8"][w], R["B11"][w], FDI_W_OURS)
    fp = fdi_from_dn(R["B6"][w], R["B8"][w], R["B11"][w], FDI_W_PAPER)
    to, tp = np.percentile(fo, 99), np.percentile(fp, 99)
    a, b = fo >= to, fp >= tp
    return dict(n_water=int(w.sum()), thr_ours=float(to), thr_paper=float(tp),
                n_top_ours=int(a.sum()), n_top_paper=int(b.sum()),
                overlap_of_ours=float((a & b).sum() / a.sum()),
                jaccard=float((a & b).sum() / (a | b).sum()),
                pearson_fdi=float(np.corrcoef(fo[::50], fp[::50])[0, 1]))

_prev_overlap = globals().get("L1", {}).get("overlap", {}) if isinstance(globals().get("L1"), dict) else {}
L1 = dict(definition="top-1% of SCL-water FDI per date; ours w=(833-740)/(1610-740), "
                     "paper w=(833-665)/(1610-665); both x10, '+' sign per Author Correction",
          overlap=dict(_prev_overlap), auc_paper_weight=[], trend_paper_weight=None)

for s2_date in ["2024-09-09", "2024-09-14", "2024-09-12", "2024-08-08", "2024-05-07", "2024-02-05"]:
    if s2_date in L1["overlap"]:
        o = L1["overlap"][s2_date]
        print(f"[L1a] {s2_date}: already computed this session -- reused")
        continue
    R = get_s2_raster(s2_date)
    L1["overlap"][s2_date] = o = weight_overlap(R)
    print(f"[L1a] {s2_date}: overlap {o['overlap_of_ours']:.3f}  jaccard {o['jaccard']:.3f}  "
          f"r {o['pearson_fdi']:.3f}  thr ours {o['thr_ours']:.4f} / paper {o['thr_paper']:.4f}")
    del R; gc.collect()
_save("L1a_fdi_weight_overlap.json", dict(definition=L1["definition"], overlap=L1["overlap"]))

for s2_date, cbers_date, item_id, gap in DATE_PAIRS:
    r = _with_weight(FDI_W_PAPER, own_threshold_open_water_auc, s2_date, cbers_date,
                     _item_for(cbers_date, item_id))
    print(f"[L1b] {s2_date} (paper weight): {_fmt(r)}")
    L1["auc_paper_weight"].append(dict(s2_date=s2_date, cbers_date=cbers_date, gap_days=gap,
                                       own_thr=r["own_thr"], auc=r.get("auc"),
                                       ci=list(r["auc_ci"]) if r.get("auc") is not None else None,
                                       n_cand=r["n_cand"], n_cells=r.get("n_cells")))
    gc.collect()

if all(x["auc"] is not None for x in L1["auc_paper_weight"]):
    print("[L1b] trend test under the printed weight:")
    L1["trend_paper_weight"] = run_trend_test(L1["auc_paper_weight"])
else:
    print("[L1b] trend test not run: at least one date not evaluable under the printed weight (reported as-is).")
_save("L1_fdi_weight_sensitivity.json", L1)


# %% ------------------------------------------------------------------------------------------
# CELL L2 — Land-adjacency decay curve, per distance bin (Figure 5 data, §14.4)
# ---------------------------------------------------------------------------------------------
# Definition (declared here, before seeing the output): strong-band (band2, FDI >= 0.0309)
# candidates vs AOI background, BOTH restricted to the same distance-to-land bin, then matched
# on the same 1 km cell (_cell_table), cell-weighted AUC (_agg), cell bootstrap CI (_boot_one).
# Bins are disjoint: [0,200), [200,500), [500,1000), [1000, inf) m.
# Primary dates: 09-09, 09-14, 09-12 (the three where band2 ~ each date's own top 1% or stronger).
# Supplementary: 08-08, 05-07, 02-05 — band2 is a far-tail selection on those dates (05-07's own
# p99 is 0.0040), so they are saved but flagged, not used for the figure's main curves.
# NOTE: §13.4's "0.84-0.91 within 0-200 m" may correspond to the land-CLASS AUC in classauc_*.json
# (09-14 strong/land = 0.842) rather than a distance bin. Whatever this cell gives is the number
# of record for Figure 5; §13.4's wording is updated to match, not the other way round.
DIST_BINS_M = [(0, 200), (200, 500), (500, 1000), (1000, np.inf)]


def binned_land_auc(S, band_key="band2"):
    kb, kn = cbers_k_factors(float(S["abscal_blue"]), float(S["abscal_nir"]), float(S["sun_elev"]),
                             int(S["doy"]), float(S["esun_blue"]), float(S["esun_nir"]))
    def prep(pre, lo, hi):
        ll = S[f"{pre}_lonlat"]; d = np.asarray(S[f"{pre}_dland"])
        m = np.asarray(S[f"{pre}_valid"], bool) & (d >= lo) & (d < hi)
        con = kn * np.asarray(S[f"{pre}_nir"]) - kb * np.asarray(S[f"{pre}_blue"])
        return dict(con=con[m], cell=cell_ids(ll[:, 0], ll[:, 1])[m])
    out = []
    for lo, hi in DIST_BINS_M:
        cd, bg = prep(band_key, lo, hi), prep("bgL", lo, hi)
        t = _cell_table(cd, bg)
        row = dict(bin_m=[lo, None if np.isinf(hi) else hi], n_cand=int(len(cd["con"])),
                   n_bg=int(len(bg["con"])), n_cells=int(len(t["auc"])))
        if len(t["auc"]) >= 5:
            row.update(auc=_agg(t), auc_ci=_boot_one(t))
        else:
            row.update(auc=None, note="fewer than 5 eligible cells")
        out.append(row)
    return out

L2 = dict(definition="band2 (FDI>=0.0309) vs bgL, same dland bin, same 1 km cell; MIN_C=MIN_B=5; "
                     "cell bootstrap B=CELL_BOOT_B", bins_m=[[a, None if np.isinf(b) else b] for a, b in DIST_BINS_M],
          primary={}, supplementary={})
for s2_date, path in AUG.items():
    key = "primary" if s2_date in ("2024-09-09", "2024-09-14", "2024-09-12") else "supplementary"
    L2[key][s2_date] = rows = binned_land_auc(load_npz(path))
    print(f"[L2] {s2_date} ({key})")
    for r in rows:
        print(f"      {str(r['bin_m']):>14}: {_fmt(r)}  (bg {r['n_bg']})")
_save("L2_land_adjacency_decay.json", L2)


# %% ------------------------------------------------------------------------------------------
# CELL L3 — Distance-cut sensitivity for the headline result (§14.4)
# ---------------------------------------------------------------------------------------------
# Same own-threshold open-water AUC as Table 13.1, at the two other pre-declared cuts in
# DIST_CUTS_M (200 m and 1000 m). The 500 m values are loaded from the saved Table 13.1 JSON,
# not recomputed. Trend test rerun at each cut where all five dates are evaluable.
with open(os.path.join(RESULTS_DIR, "own_threshold_summary_table13_1.json")) as f:
    t131 = json.load(f)

L3 = dict(definition="own_threshold_open_water_auc(min_dist=cut); everything else identical to Table 13.1",
          by_cut={"500": dict(rows=t131, trend="see trend_test_result.json")})
for cut in (200.0, 1000.0):
    rows = []
    for s2_date, cbers_date, item_id, gap in DATE_PAIRS:
        r = own_threshold_open_water_auc(s2_date, cbers_date, _item_for(cbers_date, item_id), min_dist=cut)
        print(f"[L3] {s2_date} @ >= {cut:.0f} m: {_fmt(r)}")
        rows.append(dict(s2_date=s2_date, cbers_date=cbers_date, gap_days=gap, own_thr=r["own_thr"],
                         auc=r.get("auc"), ci=list(r["auc_ci"]) if r.get("auc") is not None else None,
                         n_cand=r["n_cand"], n_cells=r.get("n_cells")))
        gc.collect()
    trend = None
    if all(x["auc"] is not None for x in rows):
        print(f"[L3] trend test @ >= {cut:.0f} m:")
        trend = run_trend_test(rows)
    else:
        print(f"[L3] trend test @ >= {cut:.0f} m not run: a date is not evaluable (reported as-is).")
    L3["by_cut"][str(int(cut))] = dict(rows=rows, trend=trend)
_save("L3_distance_cut_sensitivity.json", L3)


# %% ------------------------------------------------------------------------------------------
# CELL L4 — CBERS scene coverage for every scene in the record (Table 3 / Figure 1, §14.3)
# ---------------------------------------------------------------------------------------------
# One fixed 25x25 AOI grid per scene (cbers_grid_coverage), plus the scene's catalog-envelope
# overlap with the AOI, so the table can show envelope vs real footprint side by side.
# Covers the 5 used scenes and every dead scene found (01-10, 02-05 row 143, 02-10).
from shapely.geometry import shape, box

AOI_BOX = box(AOI["min_lon"], AOI["min_lat"], AOI["max_lon"], AOI["max_lat"])
SCENE_DATES = ["2024-09-09", "2024-09-14", "2024-08-09", "2024-05-08", "2024-02-05",
               "2024-01-10", "2024-02-10"]
used_ids = {p[2] for p in DATE_PAIRS}

L4 = []
for d in SCENE_DATES:
    for it in find_cbers_items(d):
        env = shape(it.geometry)
        row = dict(scene=it.id, date=d, used_in_study=it.id in used_ids,
                   envelope_frac_of_aoi=float(env.intersection(AOI_BOX).area / AOI_BOX.area))
        try:
            cov = cbers_grid_coverage(it, d)
            row.update(real_overall=float(cov["overall"]), real_west_half=float(cov["west_half"]),
                       n_grid_pts=int(cov["n_pts"]))
            msg = f"real overall {row['real_overall']:.2f}  real west-half {row['real_west_half']:.2f}"
        except Exception as e:           # a scene whose raster doesn't reach the AOI at all
            row.update(real_overall=0.0, real_west_half=0.0, error=repr(e)[:200])
            msg = f"window read failed -> recorded as 0 coverage ({repr(e)[:80]})"
        L4.append(row)
        print(f"[L4] {it.id}  used={row['used_in_study']!s:5}  envelope {row['envelope_frac_of_aoi']:.2f}  {msg}")
_save("L4_cbers_scene_coverage.json", L4)


# %% ------------------------------------------------------------------------------------------
# CELL L5 — Explain the identical background counts (08-08 vs 02-05, §14.6)
# ---------------------------------------------------------------------------------------------
L5 = {}
for d in ["2024-08-08", "2024-02-05", "2024-05-07"]:
    S = load_npz(AUG[d])
    v = np.asarray(S["bgL_valid"], bool); dl = np.asarray(S["bgL_dland"])
    L5[d] = dict(n_bg=int(len(v)), n_valid=int(v.sum()), n_valid_ge500=int((v & (dl >= 500)).sum()),
                 n_ge500_any=int((dl >= 500).sum()), n_nan_dland=int(np.isnan(dl).sum()))
    print(f"[L5] {d}: {L5[d]}")
_save("L5_background_counts.json", L5)
