"""nb08_confirm — confirmatory dates + blind candidate characterization.
Implements MVP3_Phase3_frozen_protocol.md exactly. All analysis steps call the frozen nb07_core;
this module only selects dates, orchestrates, guards memory, logs, and runs the pre-declared tests.
Terminology: outputs describe floating marine-debris/macroplastic aggregation candidates; no ground truth.
"""
import os, json, gc, shutil, hashlib, datetime as dt
from itertools import permutations
import numpy as np
from scipy import stats
import psutil
import nb07_core as core

# ---------------------------------------------------------------- frozen protocol parameters (§3-§5)
BASE          = "/content/drive/MyDrive/Marine_Plastic_Detection"
OUT_DIR       = f"{BASE}/results/phase3_confirmatory"
LOG_PATH      = f"{OUT_DIR}/confirm_log.json"
WINDOW        = ("2023-01-01", "2025-12-31")
EXCLUDED_S2   = {"2024-09-09", "2024-09-12", "2024-09-14", "2024-08-08", "2024-05-07", "2024-02-05", "2024-01-11"}
MAX_GAP_DAYS  = 1            # E1
CLOUD_MAX     = 30.0         # E2: mean CLOUDY_PIXEL_PERCENTAGE over granules intersecting the AOI that day
MIN_WATER_PX  = 11_000_000   # E3: unique SCL-water pixels in the AOI raster
MIN_CBERS_OK  = 0.50         # E4: share of own-threshold candidates CBERS-valid on the chosen scene
TARGET_N, MIN_N = 8, 6       # stopping rule / minimum for a confirmatory claim
N_CAND, CAND_SEED = 30_000, 11                     # verbatim from NB07 Cell 17
EXACT_MAX_N, MC_B, MC_SEED = 9, 200_000, 0         # permutation tests
CUTS_SENS     = (200.0, 1000.0)
MEM_LIMIT_PCT = 75.0
CLEAN_TILES   = True         # delete per-tile EE files once the date's raster .npz and aug .npz are saved
CHIP_SEED, CHIP_N_PER_GROUP, CHIP_M, CHIP_PX = 2026, 20, 500, 256
SGLI_DIR      = f"{BASE}/sgli_confirmatory"   # place GCOM-C/SGLI L2 NWLR 1 km (NWLRK) .h5 files here
SGLI_PCT      = 90                            # per-scene BSI threshold percentile (as exploratory)
CLASSES = ["1 linear/patchy floating feature", "2 foam/whitecap", "3 vessel/wake/fixed structure (incl. aquaculture)",
           "4 glint/haze artifact", "5 cloud or cloud-shadow edge", "6 nothing visible/uniform water", "7 uncertain"]


def hb(msg): core.hb(msg)


# ---------------------------------------------------------------- persistence, provenance, memory
def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)                     # atomic: a crash never leaves a half-written log


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return dict(protocol="MVP3_Phase3_frozen_protocol.md", dates={}, scout=None, hashes=None)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def record_hashes(paths):
    log = load_log()
    hs = {os.path.basename(p): sha256(p) for p in paths}
    if log.get("hashes") and log["hashes"] != hs:
        raise RuntimeError(f"Frozen files changed since the first run: {log['hashes']} vs {hs}. Log a deviation (protocol §6).")
    log["hashes"] = hs
    _save_json(LOG_PATH, log)
    for k, v in hs.items():
        print(f"  {k}: {v}")
    return hs


def mem_guard(step):
    gc.collect()
    vm = psutil.virtual_memory()
    if vm.percent > MEM_LIMIT_PCT:
        raise MemoryError(f"[mem] {vm.percent:.0f}% RAM used before '{step}' (limit {MEM_LIMIT_PCT:.0f}%). "
                          "Runtime > Restart session, re-run the setup cells, then re-run this cell: "
                          "every finished step is cached on Drive and the log resumes where it stopped.")
    hb(f"[mem] {vm.percent:.0f}% RAM, {vm.available / 1e9:.1f} GB free — {step}")


def _clear_core_state():
    for k in ("s2_lon", "s2_lat", "s2_fdi", "cbers_items"):
        setattr(core, k, None)
    gc.collect()


# ---------------------------------------------------------------- scouting: E1 + E2 (metadata only, no pixels)
def pair_dates(s2_days, cbers_days, excluded=EXCLUDED_S2, max_gap=MAX_GAP_DAYS, cloud_max=CLOUD_MAX):
    """Pure function (unit-testable). s2_days: {date: (mean_cloud, n_granules)}; cbers_days: {date: [item ids]}.
    Returns chronological list of dicts with status 'queued' or 'excluded' and the reason."""
    out = []
    for d in sorted(s2_days):
        cloud, ng = s2_days[d]
        d0 = dt.date.fromisoformat(d)
        near = sorted(c for c in cbers_days if abs((dt.date.fromisoformat(c) - d0).days) <= max_gap)
        rec = dict(s2_date=d, mean_cloud_pct=round(cloud, 2), n_granules=ng, cbers_dates=near)
        if d in excluded:
            rec.update(status="excluded", reason="used in exploratory phase")
        elif not near:
            rec.update(status="excluded", reason=f"E1: no CBERS-4A/WPM scene within {max_gap} d")
        elif cloud > cloud_max:
            rec.update(status="excluded", reason=f"E2: mean CLOUDY_PIXEL_PERCENTAGE {cloud:.1f} > {cloud_max:.0f}")
        else:
            rec.update(status="queued")
        out.append(rec)
    return out


def _cbers_catalog():
    import pystac_client
    cat = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")
    return cat, next(c.id for c in cat.get_collections() if "WPM" in c.id.upper())


def _aoi_geojson():
    A = core.AOI
    return {"type": "Polygon", "coordinates": [[[A["min_lon"], A["min_lat"]], [A["max_lon"], A["min_lat"]],
            [A["max_lon"], A["max_lat"]], [A["min_lon"], A["max_lat"]], [A["min_lon"], A["min_lat"]]]]}


def scout():
    import ee
    log = load_log()
    if log.get("scout"):
        hb("[scout] already done — using logged result (delete confirm_log.json only via a logged deviation)")
        return _summary(log)
    end = (dt.date.fromisoformat(WINDOW[1]) + dt.timedelta(days=1)).isoformat()
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(core.aoi_ee_geometry()).filterDate(WINDOW[0], end))
    rows = coll.reduceColumns(ee.Reducer.toList(2), ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]).get("list").getInfo()
    acc = {}
    for ms, cloud in rows:
        d = dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat()
        acc.setdefault(d, []).append(float(cloud))
    s2_days = {d: (float(np.mean(v)), len(v)) for d, v in acc.items()}
    hb(f"[scout] S2: {len(rows)} granules on {len(s2_days)} dates in {WINDOW}")
    cat, wpm = _cbers_catalog()
    items = cat.search(collections=[wpm], intersects=_aoi_geojson(),
                       datetime=f"{WINDOW[0]}T00:00:00Z/{WINDOW[1]}T23:59:59Z").items()
    cbers_days = {}
    for it in items:
        t = it.datetime or dt.datetime.fromisoformat(it.properties["start_datetime"].replace("Z", "+00:00"))
        cbers_days.setdefault(t.date().isoformat(), []).append(it.id)
    hb(f"[scout] CBERS-4A/WPM: {sum(map(len, cbers_days.values()))} scenes on {len(cbers_days)} dates")
    recs = pair_dates(s2_days, cbers_days)
    log["scout"] = dict(run_utc=dt.datetime.utcnow().isoformat(), n_s2_dates=len(s2_days), n_cbers_dates=len(cbers_days))
    for r in recs:
        log["dates"][r["s2_date"]] = r
    _save_json(LOG_PATH, log)
    return _summary(log)


def independent_dates(log):
    """E6 (independence): one S2 date per CBERS scene. Among confirmed dates sharing a scene, keep the
    smallest S2-CBERS gap; tie -> earlier S2 date. Returns (kept, dropped) lists of S2 dates."""
    best = {}
    for d, r in sorted(log["dates"].items()):
        if r["status"] == "confirmed":
            k = (r["gap_days"], d)
            if r["cbers_scene"] not in best or k < best[r["cbers_scene"]]:
                best[r["cbers_scene"]] = k
    kept = sorted(v[1] for v in best.values())
    dropped = sorted(d for d, r in log["dates"].items() if r["status"] == "confirmed" and d not in kept)
    return kept, dropped


def _summary(log):
    st = {}
    for r in log["dates"].values():
        st[r["status"]] = st.get(r["status"], 0) + 1
    kept, dropped = independent_dates(log)
    print("[status]", st, "| independent confirmed:", kept, "| shared-scene (sensitivity only):", dropped)
    return st


# ---------------------------------------------------------------- per-date processing: E3, E4, pipeline, E5
def own_threshold_candidates(R):
    """Verbatim candidate generation from NB07 Cell 17 (own top-1%, 30,000 subsample, seed 11)."""
    w_mask = core.water_mask(R)
    fdi_full = core.fdi_from_dn(R["B6"], R["B8"], R["B11"], core.FDI_W_OURS)
    own_thr = np.percentile(fdi_full[w_mask], 99)
    rows, cols = np.where(w_mask)
    strong = fdi_full[w_mask] >= own_thr
    rr, cc = rows[strong], cols[strong]
    del fdi_full, rows, cols, strong
    x = R["X0"] + R["phx"] + cc * R["res"]
    y = R["Y0"] + R["phy"] + rr * R["res"]
    from pyproj import Transformer
    lon, lat = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True).transform(x, y)
    pts = np.column_stack([lon, lat])
    rng = np.random.default_rng(CAND_SEED)
    if len(pts) > N_CAND:
        pts = pts[rng.choice(len(pts), N_CAND, replace=False)]
    return pts, float(own_thr), int(w_mask.sum())


def _item(cbers_date, item_id):
    return next(it for it in core.find_cbers_items(cbers_date) if it.id == item_id)


def _auc_row(r):
    ok = r.get("auc") is not None
    return dict(auc=r.get("auc"), ci=list(r["auc_ci"]) if ok else None, n_cand=r.get("n_cand"),
                n_cells=r.get("n_cells"), own_thr=r.get("own_thr"), note=r.get("note"))


def process_date(d):
    log = load_log(); rec = log["dates"][d]
    upd = lambda **k: (rec.update(k), log["dates"].__setitem__(d, rec), _save_json(LOG_PATH, log))

    # E3 + E4: one raster pass (EE ~12-15 min, resumable per tile; cache hit afterwards)
    if "cbers_scene" not in rec:
        mem_guard(f"{d} raster")
        R = core.get_s2_raster(d)
        pts, own_thr, n_water = own_threshold_candidates(R)
        del R; gc.collect()
        if n_water < MIN_WATER_PX:
            return upd(status="excluded", reason=f"E3: {n_water:,} SCL-water px < {MIN_WATER_PX:,}", n_water_px=n_water)
        best, tested = None, []
        for cd in rec["cbers_dates"]:
            try:
                its = core.find_cbers_items(cd)
            except RuntimeError:
                continue
            for it in its:
                mem_guard(f"{d} CBERS window {it.id}")
                wnd = core._get_cbers_window(it, cd)
                frac = float(core.cbers_dn_at_points(wnd, pts)["valid"].mean())
                del wnd; gc.collect()
                tested.append(dict(scene=it.id, cbers_date=cd, cand_valid_frac=round(frac, 4)))
                hb(f"[E4] {d} x {it.id}: {frac:.1%} of own-threshold candidates CBERS-valid")
                if best is None or frac > best["cand_valid_frac"]:
                    best = tested[-1]
        del pts; gc.collect()
        if best is None or best["cand_valid_frac"] < MIN_CBERS_OK:
            return upd(status="excluded", reason=f"E4: best CBERS candidate validity "
                       f"{(best or {}).get('cand_valid_frac', 0):.1%} < {MIN_CBERS_OK:.0%}",
                       n_water_px=n_water, own_thr=own_thr, cbers_tested=tested)
        gap = abs((dt.date.fromisoformat(best["cbers_date"]) - dt.date.fromisoformat(d)).days)
        upd(status="in_progress", n_water_px=n_water, own_thr=own_thr, cbers_tested=tested,
            cbers_scene=best["scene"], cbers_date=best["cbers_date"], gap_days=gap)

    cd, sid, gap = rec["cbers_date"], rec["cbers_scene"], rec["gap_days"]
    item = _item(cd, sid)
    tag = f"{d}_cbers{cd}"
    band_npz = os.path.join(core.EXPORT_DIR, f"mvp2_bands_{tag}.npz")
    aug_npz = os.path.join(core.EXPORT_DIR, f"mvp2_bands_{tag}_aug.npz")

    # Frozen NB07 pipeline: bgL background, distance augmentation, class breakdown
    if not os.path.exists(aug_npz):
        mem_guard(f"{d} S2 FDI points")
        core.S2_DATE, core.CBERS_DATE, core.cbers_items = d, cd, [item]
        lon, lat, fdi = core.get_s2_fdi_array(d)          # EE ~8-9 min, checkpointed; cache hit afterwards
        core.s2_lon, core.s2_lat, core.s2_fdi = np.asarray(lon), np.asarray(lat), np.asarray(fdi)
        del lon, lat, fdi
        if not os.path.exists(band_npz):
            mem_guard(f"{d} strength bands")
            core.record_strength_bands()
        mem_guard(f"{d} augmentation")
        core.augment_saved_samples(d, cd)
        _clear_core_state()
    if not os.path.exists(os.path.join(core.RESULTS_DIR, f"classauc_{d}_cbers{cd}.json")):
        mem_guard(f"{d} class analysis")
        core.record_class_analysis(aug_npz, d, cd, gap)
    if CLEAN_TILES:
        tiles_dir = core._raster_paths(d)[1]
        if os.path.isdir(tiles_dir) and os.path.exists(core._raster_paths(d)[0]) and os.path.exists(aug_npz):
            shutil.rmtree(tiles_dir, ignore_errors=True)

    # E5 + primary metric + pre-declared sensitivities (same function as Table 13.1)
    mem_guard(f"{d} primary AUC")
    prim = _auc_row(core.own_threshold_open_water_auc(d, cd, item))
    if prim["auc"] is None:
        return upd(status="excluded", reason=f"E5: {prim['note']}", primary=prim)
    sens = {}
    for cut in CUTS_SENS:
        mem_guard(f"{d} sensitivity {cut:.0f} m")
        sens[f"cut{int(cut)}"] = _auc_row(core.own_threshold_open_water_auc(d, cd, item, min_dist=cut))
    mem_guard(f"{d} sensitivity printed weight")
    w0 = core.FDI_W_OURS
    try:
        core.FDI_W_OURS = core.FDI_W_PAPER
        sens["printed_weight"] = _auc_row(core.own_threshold_open_water_auc(d, cd, item))
    finally:
        core.FDI_W_OURS = w0
    assert core.FDI_W_OURS == w0
    upd(status="confirmed", primary=prim, sensitivity=sens)
    hb(f"[confirmed] {d} x {cd}: AUC {prim['auc']:.3f} [{prim['ci'][0]:.3f}, {prim['ci'][1]:.3f}], "
       f"cells {prim['n_cells']}, own p99 {prim['own_thr']:.4f}")


def run(max_dates=None):
    """Processes queued/in-progress dates chronologically until TARGET_N confirmed or the queue ends.
    Safe to re-run after any crash or restart."""
    done = 0
    while True:
        log = load_log()
        n_conf = len(independent_dates(log)[0])
        if n_conf >= TARGET_N:
            hb(f"[stop] stopping rule reached: {n_conf} confirmed dates"); break
        todo = sorted(d for d, r in log["dates"].items() if r["status"] in ("queued", "in_progress"))
        if not todo:
            hb(f"[stop] window exhausted: {n_conf} confirmed dates"); break
        if max_dates is not None and done >= max_dates:
            break
        d = todo[0]
        hb(f"[run] {d} ({n_conf} confirmed so far)")
        try:
            process_date(d)
        finally:
            _clear_core_state()
        done += 1
    return _summary(load_log())


# ---------------------------------------------------------------- SGLI leg, secondary (§4b)
def sgli_files(d):
    import glob
    return sorted(glob.glob(os.path.join(SGLI_DIR, f"GC1SG1_{d.replace('-', '')}*NWLRK*.h5")))


def sgli_needed():
    """Lists confirmed dates and whether their SGLI file is present in SGLI_DIR."""
    os.makedirs(SGLI_DIR, exist_ok=True)
    log = load_log()
    for d, r in sorted(log["dates"].items()):
        if r["status"] == "confirmed":
            f = sgli_files(d)
            print(f"  {d}: {'OK ' + os.path.basename(f[0]) if f else 'MISSING  -> GC1SG1_' + d.replace('-', '') + '*_L2SG_NWLRK_*.h5'}"
                  f"{'  (+%d more)' % (len(f) - 1) if len(f) > 1 else ''}")


def _sgli_clusters(lat, lon, bsi, thr, pts):
    """Per distinct SGLI pixel: number of matched points and positive flag. Matching verbatim from
    core.analyze_sgli (same KDTree metric, same tolerance), so pooled estimates reduce to it for one date."""
    import math
    from scipy.spatial import cKDTree
    lat = np.asarray(lat); lon = np.asarray(lon)
    m_lat = 111320; m_lon = 111320 * math.cos(math.radians(lat.mean()))
    tree = cKDTree(np.column_stack([lon * m_lon, lat * m_lat]))
    pts = np.asarray(pts)
    dd, ii = tree.query(np.column_stack([pts[:, 0] * m_lon, pts[:, 1] * m_lat]), distance_upper_bound=core.SGLI_MATCH_TOLERANCE_M)
    cnt = np.bincount(ii[np.isfinite(dd)], minlength=len(bsi)); u = np.where(cnt > 0)[0]
    return cnt[u], (np.asarray(bsi) > thr)[u]


def cluster_boot(cc, cp, bc, bp, B=None, seed=None):
    """Point estimate + cluster bootstrap (resampling SGLI pixels within each group), as core.analyze_sgli."""
    B = core.SGLI_BOOT_B if B is None else B
    rng = np.random.default_rng(core.BOOT_SEED if seed is None else seed)
    cc, cp, bc, bp = map(np.asarray, (cc, cp, bc, bp))
    rate = lambda c, p: (c * p).sum() / c.sum()
    pn, pb = rate(cc, cp), rate(bc, bp)
    pu, bu = np.arange(len(cc)), np.arange(len(bc)); out = []
    for _ in range(B):
        a = rng.choice(pu, len(pu)); b = rng.choice(bu, len(bu))
        rn_, rb_ = rate(cc[a], cp[a]), rate(bc[b], bp[b])
        out.append((rn_ - rb_, rn_ / rb_ if rb_ > 0 else np.nan))
    out = np.array(out)
    lr_ci = [float(v) for v in np.nanpercentile(out[:, 1], [2.5, 97.5])]
    return dict(p_cand=float(pn), p_bg=float(pb), gap=float(pn - pb), lr=float(pn / pb) if pb > 0 else None,
                gap_ci=[float(v) for v in np.nanpercentile(out[:, 0], [2.5, 97.5])], lr_ci=lr_ci,
                distinct_px_cand=int(len(cc)), distinct_px_bg=int(len(bc)),
                verdict="corroborates" if lr_ci[0] > 1 else "contradicts" if lr_ci[1] < 1 else "inconclusive")


def run_sgli():
    """For every confirmed date: pick the NWLRK file with the most QA-valid AOI pixels, apply the frozen
    exploratory SGLI leg (all top-1% FDI candidates, 20,000 background, own p90 BSI threshold, 300 m).
    Re-run after adding files: dates without a file are re-checked every time."""
    log = load_log()
    for d in sorted(k for k, r in log["dates"].items() if r["status"] == "confirmed"):
        rec = log["dates"][d]
        if rec.get("sgli", {}).get("status") in ("done", "no_valid_pixels"):
            continue
        files = sgli_files(d)
        if not files:
            rec["sgli"] = dict(status="no_file"); _save_json(LOG_PATH, log); hb(f"[SGLI] {d}: no file"); continue
        best, tried = None, []
        for f in files:
            mem_guard(f"SGLI {os.path.basename(f)}")
            try:
                g = core.load_sgli_bsi_grid(f)
            except AssertionError:
                tried.append(dict(file=os.path.basename(f), n_valid_px=0)); continue
            tried.append(dict(file=os.path.basename(f), n_valid_px=int(len(g[2]))))
            if best is None or len(g[2]) > len(best[1][2]):
                best = (os.path.basename(f), g)
            del g; gc.collect()
        if best is None:
            rec["sgli"] = dict(status="no_valid_pixels", tried=tried); _save_json(LOG_PATH, log); continue
        fname, (lat, lon, bsi) = best
        thr = float(np.percentile(bsi, SGLI_PCT))
        mem_guard(f"SGLI {d} S2 points")
        slon, slat, sfdi = core.get_s2_fdi_array(d)
        near, bg = core.build_near_threshold_and_background(np.asarray(slon), np.asarray(slat), np.asarray(sfdi))
        del slon, slat, sfdi; gc.collect()
        res = core.analyze_sgli(lat, lon, bsi, thr, near[:, :2], bg[:, :2], core.SGLI_MATCH_TOLERANCE_M)
        cc, cp = _sgli_clusters(lat, lon, bsi, thr, near[:, :2]); bc, bp = _sgli_clusters(lat, lon, bsi, thr, bg[:, :2])
        del near, bg, lat, lon, bsi; gc.collect()
        rec["sgli"] = dict(status="done", file=fname, tried=tried, threshold_p90=thr, result=res,
                           clusters=dict(cand_n=cc.tolist(), cand_pos=cp.astype(int).tolist(),
                                         bg_n=bc.tolist(), bg_pos=bp.astype(int).tolist()))
        _save_json(LOG_PATH, log)
        cb = res.get("cluster_bootstrap", {})
        hb(f"[SGLI] {d}: {fname} | distinct px cand {res.get('distinct_px_cand')}, bg {res.get('distinct_px_bg')} | "
           f"LR+ {res.get('lr')} CI {cb.get('lr_ci')} | evaluable={res.get('evaluable')}")
    sgli_needed()


def sgli_summary(log):
    per, C = {}, dict(cc=[], cp=[], bc=[], bp=[])
    for d, r in sorted(log["dates"].items()):
        if r["status"] != "confirmed":
            continue
        sg = r.get("sgli", {"status": "not run"})
        per[d] = dict(status=sg["status"], file=sg.get("file"), result=sg.get("result"))
        cl = sg.get("clusters")
        if sg["status"] == "done" and cl and cl["cand_n"] and cl["bg_n"]:
            C["cc"] += cl["cand_n"]; C["cp"] += cl["cand_pos"]; C["bc"] += cl["bg_n"]; C["bp"] += cl["bg_pos"]
    n_used = sum(v["status"] == "done" for v in per.values())
    pooled = cluster_boot(C["cc"], C["cp"], C["bc"], C["bp"]) if C["cc"] and C["bc"] else None
    return dict(per_date=per, n_dates_with_sgli=n_used, pooled_H_S1=pooled,
                verdict=(pooled["verdict"] if pooled else "not assessable: no confirmed date with usable SGLI data"))


# ---------------------------------------------------------------- pre-declared tests (§4)
def _perm_p(x, y, rank):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if rank:
        x, y = stats.rankdata(x), stats.rankdata(y)
    n = len(x)
    zx = (x - x.mean()) / x.std(); zy = (y - y.mean()) / y.std()
    obs = float(zx @ zy / n)
    if n <= EXACT_MAX_N:
        P = np.array(list(permutations(range(n))), dtype=np.int8)
        count = int(((zy[P] @ zx) / n >= obs - 1e-12).sum())
        return obs, count / len(P), f"exact ({len(P)} permutations)"
    rng, count, left = np.random.default_rng(MC_SEED), 0, MC_B
    while left:
        b = min(20_000, left)
        P = rng.permuted(np.tile(np.arange(n), (b, 1)), axis=1)
        count += int(((zy[P] @ zx) / n >= obs - 1e-12).sum()); left -= b
    return obs, (1 + count) / (1 + MC_B), f"Monte Carlo ({MC_B} permutations, seed {MC_SEED})"


def trend_tests(rows):
    fdi = np.array([r["own_thr"] for r in rows], float)
    auc = np.array([r["auc"] for r in rows], float)
    se = np.array([(r["ci"][1] - r["ci"][0]) / (2 * 1.96) for r in rows])
    rho, p_s, how = _perm_p(fdi, auc, rank=True)
    r_p, p_p, _ = _perm_p(fdi, auc, rank=False)
    n = len(rows); X = np.column_stack([np.ones(n), fdi]); W = np.diag(1 / se ** 2)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ auc); resid = auc - X @ beta
    s2 = (resid @ W @ resid) / (n - 2); se_b = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ W @ X)))
    p_w = float(2 * (1 - stats.t.cdf(abs(beta[1] / se_b[1]), n - 2)))
    return dict(n=n, spearman_rho=rho, spearman_p_onesided=p_s, pearson_r=r_p, pearson_p_onesided=p_p,
                wls_slope=float(beta[1]), wls_slope_se=float(se_b[1]), wls_p_twosided=p_w, permutation=how)


def _verdict(t, n):
    if n < MIN_N:
        return "underpowered: descriptive only (fewer than 6 confirmatory dates)"
    if t["spearman_p_onesided"] < 0.05:
        return "trend replicated on independent dates under frozen rules"
    if t["spearman_rho"] > 0:
        return "direction consistent, not confirmed; trend remains a hypothesis"
    return "exploratory trend did not replicate"


def final_tests():
    log = load_log()
    kept, dropped = independent_dates(log)
    mk = lambda d: dict(s2_date=d, own_thr=log["dates"][d]["primary"]["own_thr"], auc=log["dates"][d]["primary"]["auc"],
                        ci=log["dates"][d]["primary"]["ci"], sens=log["dates"][d]["sensitivity"],
                        cbers_scene=log["dates"][d]["cbers_scene"])
    new, shared = [mk(d) for d in kept], [mk(d) for d in dropped]
    with open(os.path.join(core.RESULTS_DIR, "own_threshold_summary_table13_1.json")) as f:
        expl = json.load(f)
    le = os.path.join(core.RESULTS_DIR, "phase3_loose_ends")
    with open(os.path.join(le, "L3_distance_cut_sensitivity.json")) as f:
        l3 = json.load(f)["by_cut"]
    with open(os.path.join(le, "L1_fdi_weight_sensitivity.json")) as f:
        l1 = json.load(f)["auc_paper_weight"]
    expl_s = {"cut200": l3["200"]["rows"], "cut1000": l3["1000"]["rows"], "printed_weight": l1}
    out = dict(protocol=log["protocol"], hashes=log["hashes"], n_new=len(new), n_exploratory=len(expl))
    if len(new) >= 3:
        out["H_C1_primary_new_only"] = t = trend_tests(new)
        out["verdict"] = _verdict(t, len(new))
    out["H_C2_pooled_descriptive"] = trend_tests(expl + new) if new else None
    out["shared_scene_dates"] = [dict(s2_date=r["s2_date"], cbers_scene=r["cbers_scene"]) for r in shared]
    out["sensitivity_including_shared_scene_dates"] = trend_tests(new + shared) if shared and len(new + shared) >= 3 else None
    out["sensitivity"] = {}
    for k in ("cut200", "cut1000", "printed_weight"):
        nr = [dict(own_thr=r["sens"][k]["own_thr"], auc=r["sens"][k]["auc"], ci=r["sens"][k]["ci"])
              for r in new if r["sens"][k]["auc"] is not None]
        pr = [r for r in expl_s[k] if r.get("auc") is not None] + nr
        out["sensitivity"][k] = dict(new_only=trend_tests(nr) if len(nr) >= 3 else None,
                                     pooled=trend_tests(pr) if len(pr) >= 3 else None)
    out["per_date_new"] = new
    out["sgli_secondary"] = sgli_summary(log)
    out["exclusion_log"] = {d: r.get("reason") for d, r in sorted(log["dates"].items()) if r["status"] == "excluded"}
    _save_json(os.path.join(OUT_DIR, "confirmatory_results.json"), out)
    if "H_C1_primary_new_only" in out:
        t = out["H_C1_primary_new_only"]
        print(f"H-C1 (new dates only, n={t['n']}): rho = {t['spearman_rho']:.3f}, p (one-sided) = "
              f"{t['spearman_p_onesided']:.4f} [{t['permutation']}] -> {out['verdict']}")
    sg = out["sgli_secondary"]
    print(f"H-S1 (SGLI, pooled over {sg['n_dates_with_sgli']} dates): {sg['verdict']}"
          + (f" | LR+ {sg['pooled_H_S1']['lr']:.2f} CI {[round(x, 2) for x in sg['pooled_H_S1']['lr_ci']]}, "
             f"distinct px cand {sg['pooled_H_S1']['distinct_px_cand']}" if sg["pooled_H_S1"] else ""))
    return out


# ---------------------------------------------------------------- blind visual characterization (§5)
def _chip_points(s2_date, cbers_date, item_id, rng, dist_m=500.0):
    """Open-water candidates (as in Cell 17) and open-water bgL background for one exploratory date."""
    R = core.get_s2_raster(s2_date)
    pts, _, _ = own_threshold_candidates(R)
    land = ~core.water_mask(R) & np.isin(R["SCL"], core.SCL_LAND)
    dl = core._dist(land); del land
    ix, iy, _ = core.to_grid_index(R, pts[:, 0], pts[:, 1])
    dland = dl[iy, ix]; del dl, R; gc.collect()
    wnd = core._get_cbers_window(_item(cbers_date, item_id), cbers_date)
    valid = core.cbers_dn_at_points(wnd, pts)["valid"]; del wnd
    cand = pts[valid & (dland >= dist_m)]
    S = core.load_npz(os.path.join(core.EXPORT_DIR, f"mvp2_bands_{s2_date}_cbers{cbers_date}_aug.npz"))
    bg = S["bgL_lonlat"][np.asarray(S["bgL_valid"], bool) & (np.asarray(S["bgL_dland"]) >= dist_m)]
    del S; gc.collect()
    pick = lambda a: a[rng.choice(len(a), CHIP_N_PER_GROUP, replace=False)]
    return pick(cand), pick(bg)


def _thumb(img, lon, lat, bands, vmax, path):
    import ee, requests, io
    from PIL import Image, ImageDraw
    region = ee.Geometry.Point([lon, lat]).buffer(CHIP_M / 2).bounds()
    url = img.getThumbURL(dict(region=region, dimensions=CHIP_PX, bands=bands, min=0, max=vmax,
                               gamma=1.3, format="png", crs="EPSG:32723"))
    for attempt in range(3):
        try:
            im = Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("RGB"); break
        except Exception:
            if attempt == 2: raise
    c, r = CHIP_PX // 2, CHIP_PX // 10
    ImageDraw.Draw(im).ellipse([c - r, c - r, c + r, c + r], outline=(255, 0, 255), width=2)   # ring; centre left visible
    im.save(path)


def export_chips():
    import ee, csv
    cdir = os.path.join(OUT_DIR, "chips"); kdir = os.path.join(OUT_DIR, "KEY_do_not_share_with_labelers")
    os.makedirs(cdir, exist_ok=True); os.makedirs(kdir, exist_ok=True)
    key_path = os.path.join(kdir, "chip_key.csv")
    if not os.path.exists(key_path):
        rng, rows = np.random.default_rng(CHIP_SEED), []
        for s2d, cbd, iid, _ in core.DATE_PAIRS:
            mem_guard(f"chips {s2d}")
            cand, bg = _chip_points(s2d, cbd, iid, rng)
            rows += [(s2d, float(lo), float(la), "candidate") for lo, la in cand]
            rows += [(s2d, float(lo), float(la), "background") for lo, la in bg]
            hb(f"[chips] {s2d}: {len(cand)} candidates + {len(bg)} background sampled")
        rows = [rows[i] for i in rng.permutation(len(rows))]
        with open(key_path, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["chip_id", "s2_date", "lon", "lat", "group"])
            for k, r in enumerate(rows, 1):
                wr.writerow([f"P{k:03d}", *r])
    with open(key_path) as f:
        key = list(csv.DictReader(f))
    mosaics = {}
    for r in key:
        tc, fc = (os.path.join(cdir, f"{r['chip_id']}_{s}.png") for s in ("truecolor", "falsecolor"))
        if os.path.exists(tc) and os.path.exists(fc):
            continue
        d = r["s2_date"]
        if d not in mosaics:
            end = (dt.date.fromisoformat(d) + dt.timedelta(days=1)).isoformat()
            mosaics[d] = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                          .filterBounds(core.aoi_ee_geometry()).filterDate(d, end).mosaic())
        lon, lat = float(r["lon"]), float(r["lat"])
        _thumb(mosaics[d], lon, lat, ["B4", "B3", "B2"], 2000, tc)
        _thumb(mosaics[d], lon, lat, ["B8", "B4", "B3"], 3000, fc)
    for who in ("A", "B"):
        p = os.path.join(OUT_DIR, f"labels_labeler_{who}.csv")
        if not os.path.exists(p):
            with open(p, "w", newline="") as f:
                wr = csv.writer(f); wr.writerow(["chip_id", "class_1_to_7", "notes"])
                for r in sorted(key, key=lambda r: r["chip_id"]):
                    wr.writerow([r["chip_id"], "", ""])
    with open(os.path.join(OUT_DIR, "labeling_instructions.txt"), "w") as f:
        f.write("Label each chip_id with ONE class (1-7). Use both the truecolor and falsecolor chip. The magenta ring\n"
                "marks the point; judge what lies at its centre. Work independently; do not discuss until both sheets\n"
                "are complete. VHR imagery may be used only to identify persistent fixed structures.\n\n" + "\n".join(CLASSES))
    hb(f"[chips] {len(key)} points, 2 chips each, in {cdir}; label sheets and instructions in {OUT_DIR}")


def score_labels(consensus_csv=None):
    import csv
    rd = lambda p: {r["chip_id"]: int(r["class_1_to_7"]) for r in csv.DictReader(open(p)) if r["class_1_to_7"].strip()}
    A, B = rd(os.path.join(OUT_DIR, "labels_labeler_A.csv")), rd(os.path.join(OUT_DIR, "labels_labeler_B.csv"))
    ids = sorted(set(A) & set(B)); a, b = np.array([A[i] for i in ids]), np.array([B[i] for i in ids])
    po = float((a == b).mean()); pe = float(sum((a == k).mean() * (b == k).mean() for k in range(1, 8)))
    kappa = (po - pe) / (1 - pe)
    final = rd(consensus_csv) if consensus_csv else A
    key = {r["chip_id"]: r["group"] for r in csv.DictReader(open(os.path.join(
        OUT_DIR, "KEY_do_not_share_with_labelers", "chip_key.csv")))}
    tab = np.zeros((2, 7), int)
    for cid, k in final.items():
        tab[0 if key[cid] == "candidate" else 1, k - 1] += 1
    prop = {}
    for g, row in zip(("candidate", "background"), tab):
        n = int(row.sum())
        prop[g] = {CLASSES[k]: dict(n=int(row[k]), p=row[k] / n, ci=list(core.wilson(row[k] / n, n))) for k in range(7)}
    keep = tab.sum(0) > 0; t = tab[:, keep]
    chi2 = float(stats.chi2_contingency(t)[0])
    rng, grp = np.random.default_rng(CHIP_SEED), np.repeat([0, 1], t.sum(1))
    lab = np.concatenate([np.repeat(np.arange(t.shape[1]), t[i]) for i in range(2)])
    def chi_of(g):
        m = np.array([[np.sum((g == i) & (lab == j)) for j in range(t.shape[1])] for i in range(2)])
        e = m.sum(1, keepdims=True) * m.sum(0, keepdims=True) / m.sum()
        return float(((m - e) ** 2 / np.where(e > 0, e, 1)).sum())
    perm = np.array([chi_of(rng.permutation(grp)) for _ in range(20_000)])
    out = dict(n_double_labeled=len(ids), percent_agreement=po, cohen_kappa=kappa, table=tab.tolist(),
               proportions=prop, chi2=chi2, chi2_perm_p=float((1 + (perm >= chi2 - 1e-9).sum()) / (1 + len(perm))),
               labels_used="consensus" if consensus_csv else "labeler A (no consensus file given)")
    _save_json(os.path.join(OUT_DIR, "chip_label_results.json"), out)
    print(f"kappa {kappa:.3f} (agreement {po:.1%}, n={len(ids)}) | chi2 {chi2:.2f}, permutation p {out['chi2_perm_p']:.4f}")
    return out
