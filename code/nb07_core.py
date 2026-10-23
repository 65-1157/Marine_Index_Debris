"""nb07_core — frozen definitions extracted verbatim from NB07_MVP2_Phase2_workflow_final.ipynb.
Load-time executions (drive mount, pip, per-date runs, prints, self-test) are stripped; nothing else is changed.
DO NOT EDIT: the SHA-256 of this file is recorded in the frozen protocol."""

# ===== NB07 cell index 8 =====
REFERENCE_POINTS = {
    "Angra dos Reis": (-44.3136, -23.0117),
    "Ilha Grande": (-44.2330, -23.1500),
    "Paraty": (-44.7150, -23.2178),
    "Mangaratiba": (-44.0400, -22.9600),
}

# ===== NB07 cell index 10 =====
"""
Gate B — Composite floating-debris/macroplastic aggregation index
MVP-2: Baia da Ilha Grande / Mosaico Bocaina, RJ, Brazil
Date triple: S2 2024-09-09 / CBERS 2024-09-09 / SGLI 2024-09-09 (same-day triple)

Terminology reminder (project rule, carried forward):
  Use "floating marine-debris/macroplastic aggregation candidates" in every
  print/plot/report. NEVER "plastic identification" — sensor agreement on
  an anomaly does not confirm it is plastic vs. foam, glint, or wave state.
  No ground truth exists at any AOI so far.

Pipeline
--------
1. S2 baseline: NIR-Red contrast, top-1% / 97th-99th percentile threshold
   -> near-threshold candidate set + random background-water control set.
2. CBERS corroboration: NIR-Blue contrast, radiometrically calibrated
   (DN -> TOA radiance -> TOA reflectance), background-rate control test.
3. SGLI corroboration: NWLR_670 - NWLR_443 contrast (visible-only proxy,
   no NIR band available), QA-masked, KDTree-matched to S2 points within
   ~300m, background-rate control test.
4. Report: near-threshold rate vs. background rate, gap, sample size (n),
   for each corroboration layer — state plainly if n is too small to trust.

Run in Colab. Requires: earthengine-api, pystac-client, h5py, scipy,
rasterio, pyproj, requests.
"""
import ee
import numpy as np
import requests
import xml.etree.ElementTree as ET
import h5py
from scipy.ndimage import zoom
from scipy.spatial import cKDTree
import rasterio
from rasterio.warp import transform as warp_transform
import pystac_client
import math
import datetime as dt
import time
import gc
def hb(msg):
    """Heartbeat log with timestamp — per the project's own engineering
    lesson: always print progress with timestamps rather than letting a
    long cell run silently, so a hang or silent bug is visible immediately
    rather than three cells downstream."""
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
class ProgressTimer:
    """Tracks elapsed time and estimated-time-remaining for an iterative
    loop with a known total (e.g. the S2 tile loop) — the only part of
    this pipeline where per-unit cost is roughly constant, so the ETA is
    actually meaningful rather than a guess."""
    def __init__(self, total_units, label):
        self.total = total_units
        self.label = label
        self.start = time.time()
        self.n_done = 0

    def update(self, n_done):
        self.n_done = n_done
        elapsed = time.time() - self.start
        rate = elapsed / self.n_done if self.n_done else None
        remaining = (self.total - self.n_done) * rate if rate else None
        eta_str = f"{remaining:.0f}s (~{remaining/60:.1f} min)" if remaining is not None else "n/a"
        pct = 100 * self.n_done / self.total if self.total else 0
        hb(f"[{self.label}] {self.n_done}/{self.total} ({pct:.0f}%) — "
           f"elapsed {elapsed:.0f}s, estimated remaining {eta_str}")
class PipelineTimer:
    """Stage-level elapsed time + a rough total-pipeline ETA. The ETA is
    a simple avg-stage-time x remaining-stages estimate — approximate
    only, since S2/CBERS/SGLI have very different costs, but it directly
    answers 'how much longer is this whole thing going to take.'"""
    def __init__(self, n_stages):
        self.start = time.time()
        self.n_stages = n_stages
        self.stage_start = self.start
        self.stage_num = 0
        self.stage_durations = []

    def stage_begin(self, name):
        self.stage_num += 1
        self.stage_start = time.time()
        elapsed = self.stage_start - self.start
        hb(f"=== Stage {self.stage_num}/{self.n_stages}: {name} — starting "
           f"(total elapsed so far: {elapsed:.0f}s) ===")

    def stage_end(self):
        dur = time.time() - self.stage_start
        self.stage_durations.append(dur)
        avg = sum(self.stage_durations) / len(self.stage_durations)
        remaining_stages = self.n_stages - self.stage_num
        rough_eta = avg * remaining_stages
        total_elapsed = time.time() - self.start
        hb(f"[timer] stage {self.stage_num} took {dur:.0f}s. "
           f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min). "
           f"Rough ETA to finish (~{rough_eta:.0f}s, {remaining_stages} stage(s) "
           f"left) — approximate only, stages differ a lot in cost.")

    def finish(self):
        total = time.time() - self.start
        hb(f"=== Pipeline complete — total time: {total:.0f}s ({total/60:.1f} min) ===")
EE_PROJECT = "marplastdetection"
AOI = {"min_lon": -44.65, "max_lon": -44.05, "min_lat": -23.35, "max_lat": -22.90}
S2_DATE = "2024-09-09"
CBERS_DATE = "2024-09-09"
SGLI_FILE = "/content/drive/MyDrive/Marine_Plastic_Detection/notebooks/Japan_check_MVP2_16set26/GC1SG1_202409091246C27414_L2SG_NWLRK_3002.h5"
S2_TOP_PCT = 99
S2_LOWER_PCT_FOR_BG = 50
N_BACKGROUND_SAMPLES = 20000
SGLI_MATCH_TOLERANCE_M = 300
ESUN = {"Blue": 1984.65, "NIR": 981.91}

# ===== NB07 cell index 15 =====
def init_ee():
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)
    print(f"[EE] Initialized on project: {EE_PROJECT}")
def aoi_ee_geometry():
    return ee.Geometry.Rectangle(
        [AOI["min_lon"], AOI["min_lat"], AOI["max_lon"], AOI["max_lat"]]
    )
def get_s2_contrast_array(date):
    hb(f"[S2] fetching scene(s) for {date}")
    aoi = aoi_ee_geometry()
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date, (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat())
    )
    n_images = coll.size().getInfo()
    hb(f"[S2] {n_images} S2 granule(s) intersect the AOI on {date} — mosaicking "
       f"so the full AOI is covered even if it spans a granule boundary")
    assert n_images > 0, f"[S2] No S2 images found for {date}"
    img = coll.mosaic()  # merge granules — .first() left southern/eastern parts
                          # of this AOI outside a single granule's footprint

    nodata_mask = img.select("B4").neq(0).And(img.select("B8").neq(0))

    # Water mask — use Sentinel-2's own Scene Classification Layer (SCL)
    # rather than a hand-picked NDWI threshold (SCL class 6 = water, ESA's
    # own multi-band classifier, robust to turbidity in a way a single
    # index threshold isn't).
    #
    # CRITICAL: force this explicitly onto the S2 10m projection with
    # .reproject(). Without this, combining SCL (native 20m) with the 10m
    # nodata_mask via .And() can silently fall back to Earth Engine's
    # default coarse projection for the composite — collapsing what should
    # be a real ~50km2 bay's worth of water pixels down to a tiny handful.
    # This is the same class of bug as the earlier pixelLonLat() default-
    # projection issue that caused the IndexError a few fixes ago — two
    # different mask formulas (NDWI, now SCL) both landing on the exact
    # same suspiciously-small count (117) is the signature of this bug,
    # not a coincidence of two real results agreeing.
    TARGET_CRS = "EPSG:32723"
    TARGET_SCALE = 10
    s2_proj = ee.Projection(TARGET_CRS)
    water_mask = img.select("SCL").eq(6).reproject(crs=TARGET_CRS, scale=TARGET_SCALE)
    combined_mask = nodata_mask.And(water_mask).reproject(crs=TARGET_CRS, scale=TARGET_SCALE)
    img = img.updateMask(combined_mask)
    hb(f"[S2] applied SCL water mask (class 6), reprojected onto canonical "
       f"{TARGET_CRS} at {TARGET_SCALE}m — on top of nodata mask")

    # Quick sanity check BEFORE launching the full 143-tile loop — a coarse
    # server-side count at 30m, not the full 10m tiling pass, so a bad mask
    # is caught in seconds rather than after another multi-minute run.
    aoi_geom = aoi_ee_geometry()
    check_count = (
        combined_mask.selfMask()
        .reduceRegion(reducer=ee.Reducer.count(), geometry=aoi_geom, scale=30,
                      maxPixels=1e9, bestEffort=True)
        .getInfo()
    )
    n_check = list(check_count.values())[0] if check_count else 0
    hb(f"[S2] quick mask sanity check (30m, coarse): ~{n_check} water pixels in AOI "
       f"before full-resolution tiling — if this looks implausibly small "
       f"(a ~50km2 bay should show thousands+), stop and investigate before "
       f"running the full tile loop")

    contrast = img.normalizedDifference(["B8", "B4"]).rename("contrast")
    contrast_filled = contrast.unmask(-9999)

    # Force the coordinate bands onto the SAME projection/scale as the S2 data
    # band before sampling. pixelLonLat() defaults to its own projection,
    # which differs from S2's native 10m UTM grid — sampling mismatched-
    # projection bands with sampleRectangle silently returns DIFFERENT array
    # shapes per band. That mismatch is what caused the final IndexError.

    # Combine bands FIRST, then reproject the whole stack together in one
    # call. Reprojecting pixelLonLat() separately before combining (the
    # previous attempt) doesn't reliably propagate scale — it silently
    # returned 1x1 coordinate arrays against a correctly-sized contrast
    # array. Reprojecting the combined image forces every band onto the
    # identical output grid.
    stack = contrast_filled.addBands(ee.Image.pixelLonLat()).reproject(crs=TARGET_CRS, scale=TARGET_SCALE)

    # --- Tiling: sampleRectangle has a hard 512x512 (262,144) pixel limit
    # per call. At S2's native 10m resolution, this AOI (~61km x 50km) is
    # ~30M pixels — far too large for one call. Split into sub-rectangles
    # that each stay safely under the limit, sample each, and concatenate.
    # This preserves full native resolution rather than downsampling.
    S2_SCALE_M = 10
    TILE_PX = 480  # safety margin under the 512 hard limit
    lat0 = (AOI["min_lat"] + AOI["max_lat"]) / 2
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(lat0))
    tile_deg_lat = TILE_PX * S2_SCALE_M / m_per_deg_lat
    tile_deg_lon = TILE_PX * S2_SCALE_M / m_per_deg_lon

    lon_edges = np.arange(AOI["min_lon"], AOI["max_lon"], tile_deg_lon)
    lat_edges = np.arange(AOI["min_lat"], AOI["max_lat"], tile_deg_lat)
    n_tiles = len(lon_edges) * len(lat_edges)
    hb(f"[S2] tiling AOI into {len(lon_edges)}x{len(lat_edges)} = {n_tiles} tiles "
       f"(~{TILE_PX}px each, {S2_SCALE_M}m native resolution) to respect the "
       f"sampleRectangle 512x512 pixel limit")

    all_lon, all_lat, all_val = [], [], []
    done = 0
    tile_timer = ProgressTimer(n_tiles, "S2 tiling")
    for lo in lon_edges:
        for la in lat_edges:
            tile_region = ee.Geometry.Rectangle([
                lo, la,
                min(lo + tile_deg_lon, AOI["max_lon"]),
                min(la + tile_deg_lat, AOI["max_lat"]),
            ])
            try:
                sampled = stack.sampleRectangle(region=tile_region)
                info = sampled.getInfo()  # ONE round trip instead of 3 —
                                           # cuts network calls per tile by 3x
                props = info["properties"]
                t_lon = np.array(props["longitude"], dtype=np.float32)
                t_lat = np.array(props["latitude"], dtype=np.float32)
                t_val = np.array(props["contrast"], dtype=np.float32)
                if not (t_lon.shape == t_lat.shape == t_val.shape):
                    print(f"  [WARN] tile at ({lo:.4f},{la:.4f}) shape mismatch — "
                          f"lon{t_lon.shape} lat{t_lat.shape} val{t_val.shape} — skipping tile")
                else:
                    all_lon.append(t_lon.ravel())
                    all_lat.append(t_lat.ravel())
                    all_val.append(t_val.ravel())
            except Exception as e:
                print(f"  [WARN] tile at ({lo:.4f},{la:.4f}) failed: {e}")
            done += 1
            if done % 10 == 0 or done == n_tiles:
                tile_timer.update(done)

    lon = np.concatenate(all_lon) if all_lon else np.array([])
    lat = np.concatenate(all_lat) if all_lat else np.array([])
    val = np.concatenate(all_val) if all_val else np.array([])

    valid = val != -9999
    n_valid = int(valid.sum())
    hb(f"[S2] {n_valid} valid contrast pixels pulled for {date} across {n_tiles} tiles")
    assert n_valid > 0, f"[S2] Zero valid pixels for {date} — check nodata masking and date."
    return lon[valid], lat[valid], val[valid]
def build_near_threshold_and_background(lon, lat, val):
    p_top = np.percentile(val, S2_TOP_PCT)
    p_bg_upper = np.percentile(val, S2_LOWER_PCT_FOR_BG)

    near_mask = val >= p_top
    bg_pool_mask = val <= p_bg_upper

    near_pts = np.column_stack([lon[near_mask], lat[near_mask]])
    bg_pool_pts = np.column_stack([lon[bg_pool_mask], lat[bg_pool_mask]])

    rng = np.random.default_rng(42)
    n_bg = min(N_BACKGROUND_SAMPLES, len(bg_pool_pts))
    bg_idx = rng.choice(len(bg_pool_pts), size=n_bg, replace=False)
    bg_pts = bg_pool_pts[bg_idx]

    print(f"[S2] near-threshold (top {100 - S2_TOP_PCT}%): {len(near_pts)} points")
    print(f"[S2] background-water control: {len(bg_pts)} points (random, <= p{S2_LOWER_PCT_FOR_BG})")
    assert len(near_pts) > 0, "[S2] Zero near-threshold points — check S2_TOP_PCT / input data."
    assert len(bg_pts) > 0, "[S2] Zero background-control points — check pool size."
    return near_pts, bg_pts

# ===== NB07 cell index 17 =====
def find_cbers_items(date):
    hb(f"[CBERS] searching STAC for {date}")
    catalog = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")
    collections = [c.id for c in catalog.get_collections()]
    wpm_id = next(c for c in collections if "WPM" in c.upper())
    aoi_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [AOI["min_lon"], AOI["min_lat"]], [AOI["max_lon"], AOI["min_lat"]],
            [AOI["max_lon"], AOI["max_lat"]], [AOI["min_lon"], AOI["max_lat"]],
            [AOI["min_lon"], AOI["min_lat"]],
        ]],
    }
    search = catalog.search(
        collections=[wpm_id],
        intersects=aoi_geojson,
        datetime=f"{date}T00:00:00Z/{date}T23:59:59Z",
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"No CBERS item found for {date}")
    hb(f"[CBERS] found {len(items)} item(s) intersecting AOI: {[it.id for it in items]} — "
       f"a single scene may not fully cover this AOI, same issue as S2 granule boundaries")
    return items
def strip_namespace(elem):
    """This CBERS XML declares a default namespace, which makes
    ElementTree internally rename every tag to '{namespace}tag' — so a
    bare search for 'band' matches nothing (hence the empty [] seen
    earlier), not a wrong-name-format issue. Strip it so plain tag
    names work as written in the project's own documentation."""
    for el in elem.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return elem
def parse_calib_xml(xml_url, band_n):
    resp = requests.get(xml_url)
    root = ET.fromstring(resp.content)
    root = strip_namespace(root)

    # Scope the search to THIS specific section — the document also has
    # <band> elements under <availableBands> and <gain> (hex gain codes,
    # not calibration values); searching the whole doc risked matching
    # the wrong one.
    cal_section = root.find(".//absoluteCalibrationCoefficient")
    if cal_section is None:
        top_level_tags = [child.tag for child in root]
        raise RuntimeError(
            f"[CBERS calib] No <absoluteCalibrationCoefficient> section found "
            f"in {xml_url}. Top-level tags present: {top_level_tags}."
        )

    tried_names = [str(band_n), f"{band_n:02d}", f"BAND{band_n}", f"B{band_n}"]
    abscal = None
    for band in cal_section.findall("band"):
        if band.attrib.get("name") in tried_names:
            abscal = float(band.text)
            break

    if abscal is None:
        found_names = [b.attrib.get("name") for b in cal_section.findall("band")]
        raise RuntimeError(
            f"[CBERS calib] No coefficient for band {band_n}. Names found "
            f"inside <absoluteCalibrationCoefficient>: {found_names}."
        )

    elev_el = root.find(".//sunPosition/elevation")
    if elev_el is None:
        raise RuntimeError(f"[CBERS calib] No sunPosition/elevation found in {xml_url}")
    sun_elev = float(elev_el.text)

    return abscal, sun_elev
def dn_to_reflectance(dn, abscal, sun_elev, esun, doy):
    L = dn * abscal  # TOA radiance
    theta = math.radians(90 - sun_elev)
    d2 = 1 / (1 + 0.033 * math.cos(2 * math.pi * doy / 365))
    R = (math.pi * L * d2) / (esun * math.cos(theta))
    return R
def sample_cbers_contrast_at_points(items, points_lonlat, date):
    """Handles multiple CBERS items — a single scene may not cover the
    whole AOI (same class of bug as the S2 granule-boundary issue). Each
    point is assigned to the first item whose raster footprint actually
    contains it; batched per-item rather than per-point for speed."""
    hb(f"[CBERS] calibrating and sampling {len(points_lonlat)} points "
       f"across {len(items)} item(s) for {date}")
    t_func_start = time.time()
    doy = dt.date.fromisoformat(date).timetuple().tm_yday

    n = len(points_lonlat)
    results = [None] * n
    assigned = np.zeros(n, dtype=bool)
    lons = np.array([p[0] for p in points_lonlat])
    lats = np.array([p[1] for p in points_lonlat])

    item_timer = ProgressTimer(len(items), "CBERS items")

    for item_num, item in enumerate(items, start=1):
        if assigned.all():
            hb(f"[CBERS] all {n} points already assigned — skipping remaining items")
            break

        t0 = time.time()
        blue_href = item.assets["BAND1"].href
        nir_href = item.assets["BAND4"].href
        blue_xml = item.assets["BAND1_xml"].href
        nir_xml = item.assets["BAND4_xml"].href
        abscal_blue, sun_elev = parse_calib_xml(blue_xml, 1)
        abscal_nir, _ = parse_calib_xml(nir_xml, 4)

        with rasterio.open(blue_href) as blue_ds, rasterio.open(nir_href) as nir_ds:
            xs, ys = warp_transform("EPSG:4326", blue_ds.crs, lons.tolist(), lats.tolist())
            xs, ys = np.array(xs), np.array(ys)
            b = blue_ds.bounds
            in_bounds = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
            todo_idx = np.where(in_bounds & (~assigned))[0]

            if len(todo_idx) == 0:
                hb(f"[CBERS] item {item.id}: 0 new points in bounds — skipping "
                   f"({time.time() - t0:.1f}s)")
                item_timer.update(item_num)
                continue

            pts = list(zip(xs[todo_idx], ys[todo_idx]))
            blue_dn = list(blue_ds.sample(pts))
            nir_dn = list(nir_ds.sample(pts))

            n_valid_here = 0
            for local_i, idx in enumerate(todo_idx):
                b_dn, n_dn = blue_dn[local_i][0], nir_dn[local_i][0]
                assigned[idx] = True  # in-bounds = assigned to this item, even if nodata
                if b_dn == 0 or n_dn == 0:
                    continue
                r_blue = dn_to_reflectance(b_dn, abscal_blue, sun_elev, ESUN["Blue"], doy)
                r_nir = dn_to_reflectance(n_dn, abscal_nir, sun_elev, ESUN["NIR"], doy)
                results[idx] = r_nir - r_blue
                n_valid_here += 1

        hb(f"[CBERS] item {item.id}: {len(todo_idx)} points in bounds, "
           f"{n_valid_here} valid ({time.time() - t0:.1f}s)")
        item_timer.update(item_num)

    n_valid = sum(r is not None for r in results)
    n_unmatched = n - int(assigned.sum())
    hb(f"[CBERS] sampling complete in {time.time() - t_func_start:.1f}s — "
       f"{n_valid}/{n} valid, {n_unmatched} points fell outside ALL "
       f"{len(items)} item(s)' footprints")
    return results

# ===== NB07 cell index 19 =====
def load_sgli_contrast_grid(filepath):
    """Memory-optimized: windows down to the AOI (+ padding) at the
    TIE-POINT stage — which is cheap to load fully — then only reads and
    interpolates that small window from the full-resolution Image_data
    arrays, instead of loading/interpolating the entire global-swath
    scene (which can be a large fraction of a Colab session's RAM)."""
    hb(f"[SGLI] opening {filepath}")
    with h5py.File(filepath, "r") as f:
        lat_tie = f["Geometry_data/Latitude"][:].astype(np.float32)
        lon_tie = f["Geometry_data/Longitude"][:].astype(np.float32)
        hb(f"[SGLI] tie-point grid loaded: shape {lat_tie.shape}")

        red_ds = f["Image_data/NWLR_670"]
        blue_ds = f["Image_data/NWLR_443"]
        qa_ds = f["Image_data/QA_flag"]
        target_shape = red_ds.shape  # metadata only — cheap, no data read yet

        zoom_row = target_shape[0] / lat_tie.shape[0]
        zoom_col = target_shape[1] / lat_tie.shape[1]

        # Find tie-point indices whose lat/lon fall near the AOI (with a
        # padding buffer so the interpolation window has margin on all sides).
        pad_deg = 0.05
        tie_in_aoi = (
            (lon_tie >= AOI["min_lon"] - pad_deg) & (lon_tie <= AOI["max_lon"] + pad_deg) &
            (lat_tie >= AOI["min_lat"] - pad_deg) & (lat_tie <= AOI["max_lat"] + pad_deg)
        )
        rows, cols = np.where(tie_in_aoi)
        assert len(rows) > 0, (
            "[SGLI] No tie-points fall near the AOI even with padding — "
            "this file likely does not cover this AOI at all. Stop here "
            "rather than proceeding with a meaningless empty window."
        )
        tie_row_min, tie_row_max = rows.min(), rows.max()
        tie_col_min, tie_col_max = cols.min(), cols.max()
        hb(f"[SGLI] AOI-relevant tie-point window: rows {tie_row_min}-{tie_row_max}, "
           f"cols {tie_col_min}-{tie_col_max} (of {lat_tie.shape})")

        # Convert to full-resolution index window, with one extra tie-cell
        # of padding on each side to avoid edge artifacts in interpolation.
        tie_pad = 1
        r0 = max(tie_row_min - tie_pad, 0)
        r1 = min(tie_row_max + tie_pad + 1, lat_tie.shape[0])
        c0 = max(tie_col_min - tie_pad, 0)
        c1 = min(tie_col_max + tie_pad + 1, lat_tie.shape[1])

        full_r0 = int(r0 * zoom_row)
        full_r1 = min(int(r1 * zoom_row), target_shape[0])
        full_c0 = int(c0 * zoom_col)
        full_c1 = min(int(c1 * zoom_col), target_shape[1])
        hb(f"[SGLI] windowed full-res read: rows {full_r0}:{full_r1}, "
           f"cols {full_c0}:{full_c1} (full scene is {target_shape}) — "
           f"reading a small window, not the whole scene")

        # Windowed HDF5 slicing — only this subset is actually read from disk.
        red = red_ds[full_r0:full_r1, full_c0:full_c1].astype(np.float32)
        blue = blue_ds[full_r0:full_r1, full_c0:full_c1].astype(np.float32)
        qa = qa_ds[full_r0:full_r1, full_c0:full_c1]
        hb(f"[SGLI] windowed data read: {red.shape} — "
           f"~{red.nbytes / 1e6:.1f} MB per band, vs full scene "
           f"~{np.prod(target_shape) * 4 / 1e6:.0f} MB if unwindowed")

        red_slope = red_ds.attrs.get("Slope", 1.0)
        red_offset = red_ds.attrs.get("Offset", 0.0)
        blue_slope = blue_ds.attrs.get("Slope", 1.0)
        blue_offset = blue_ds.attrs.get("Offset", 0.0)

        # Interpolate only the cropped tie sub-grid, targeted to this window's exact shape.
        lat_tie_crop = lat_tie[r0:r1, c0:c1]
        lon_tie_crop = lon_tie[r0:r1, c0:c1]
        win_shape = red.shape
        zr = win_shape[0] / lat_tie_crop.shape[0]
        zc = win_shape[1] / lat_tie_crop.shape[1]
        lat_full = zoom(lat_tie_crop, (zr, zc), order=1)[:win_shape[0], :win_shape[1]].astype(np.float32)
        lon_full = zoom(lon_tie_crop, (zr, zc), order=1)[:win_shape[0], :win_shape[1]].astype(np.float32)

    # Free the tie-point arrays and any references before computing on the window.
    del lat_tie, lon_tie
    gc.collect()

    red = red * red_slope + red_offset
    blue = blue * blue_slope + blue_offset
    contrast = (red - blue).astype(np.float32)
    good = (qa == 0)

    n_good = int(good.sum())
    hb(f"[SGLI] windowed QA-valid pixels: {n_good} of {good.size} "
       f"({100 * n_good / good.size:.1f}%)")
    assert n_good > 0, (
        "[SGLI] Zero QA-valid pixels in the windowed AOI region — matches "
        "the 0.0% valid_frac already seen for some dates in this batch. "
        "This is a real null result, not a bug, but downstream matching "
        "will have nothing to match against."
    )

    result = lat_full[good], lon_full[good], contrast[good]

    # Drop the large windowed grids now that only the small masked point
    # arrays (result) are needed going forward.
    del red, blue, qa, lat_full, lon_full, contrast, good
    gc.collect()

    return result
def match_sgli_to_points(sgli_lat, sgli_lon, sgli_contrast, points_lonlat, tol_m=SGLI_MATCH_TOLERANCE_M):
    hb(f"[SGLI] KDTree-matching {len(points_lonlat)} points against "
       f"{len(sgli_contrast)} valid SGLI pixels (tol={tol_m}m)")
    # Rough meters-per-degree at this latitude for a simple KDTree in degree-space,
    # then filter by a converted tolerance (good enough at AOI scale; not a full geodesic).
    lat0 = np.mean(sgli_lat)
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(lat0))

    tree_pts = np.column_stack([sgli_lon * m_per_deg_lon, sgli_lat * m_per_deg_lat])
    tree = cKDTree(tree_pts)

    query_pts = np.column_stack([
        np.array([p[0] for p in points_lonlat]) * m_per_deg_lon,
        np.array([p[1] for p in points_lonlat]) * m_per_deg_lat,
    ])
    dist, idx = tree.query(query_pts, distance_upper_bound=tol_m)

    matched = []
    for d, i in zip(dist, idx):
        if np.isfinite(d) and i < len(sgli_contrast):
            matched.append(sgli_contrast[i])
        else:
            matched.append(None)
    n_matched = sum(m is not None for m in matched)
    hb(f"[SGLI] matched {n_matched}/{len(matched)} points within tolerance")
    return matched

# ===== NB07 cell index 21 =====
def background_rate_control(near_values, bg_values, label, threshold=0.0):
    near_valid = [v for v in near_values if v is not None]
    bg_valid = [v for v in bg_values if v is not None]

    near_pos_rate = 100 * np.mean([v > threshold for v in near_valid]) if near_valid else float("nan")
    bg_pos_rate = 100 * np.mean([v > threshold for v in bg_valid]) if bg_valid else float("nan")
    gap = near_pos_rate - bg_pos_rate if near_valid and bg_valid else float("nan")

    print(f"\n[{label}] threshold used: {threshold:.5f}")
    print(f"[{label}] near-threshold positive rate: {near_pos_rate:.1f}% (n={len(near_valid)})")
    print(f"[{label}] background-water positive rate: {bg_pos_rate:.1f}% (n={len(bg_valid)})")
    print(f"[{label}] gap: {gap:.1f} points")
    if len(near_valid) < 30 or len(bg_valid) < 30:
        print(f"[{label}] WARNING: sample size below ~30 — treat as INCONCLUSIVE, "
              f"not negative, per the project's own evidentiary standard.")
    return near_pos_rate, bg_pos_rate, gap

# ===== NB07 cell index 24 =====
import os
import pickle
CACHE_DIR = "/content/drive/MyDrive/Marine_Plastic_Detection/cache"
CHECKPOINT_EVERY = 20
def _s2_fdi_cache_path(date):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"s2_fdi_{date}.pkl")
def get_s2_fdi_array(date, checkpoint_every=CHECKPOINT_EVERY):
    cache_path = _s2_fdi_cache_path(date)

    # --- Fast path: complete cache for this date -> zero EE calls. ---
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("date") == date and cached.get("complete"):
            hb(f"[S2-FDI] complete cache hit ({len(cached['fdi'])} px) -- skipping EE entirely")
            return cached["lon"], cached["lat"], cached["fdi"]

    hb(f"[S2-FDI] fetching scene(s) for {date}")
    aoi = aoi_ee_geometry()
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date, (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat())
    )
    n_images = coll.size().getInfo()
    hb(f"[S2-FDI] {n_images} S2 granule(s) intersect the AOI on {date} -- mosaicking")
    assert n_images > 0, f"[S2-FDI] No S2 images found for {date}"
    img = coll.mosaic()

    nodata_mask = (img.select("B8").neq(0)
                   .And(img.select("B6").neq(0))
                   .And(img.select("B11").neq(0)))

    TARGET_CRS, TARGET_SCALE = "EPSG:32723", 10
    water_mask = img.select("SCL").eq(6).reproject(crs=TARGET_CRS, scale=TARGET_SCALE)
    combined_mask = nodata_mask.And(water_mask).reproject(crs=TARGET_CRS, scale=TARGET_SCALE)
    img = img.updateMask(combined_mask)
    hb(f"[S2-FDI] SCL water mask + B6/B11 nodata mask applied, reprojected onto {TARGET_CRS}")

    # FDI = R_NIR - [R_RE2 + (R_SWIR1-R_RE2) * (833-740)/(1610-740) * 10]
    r_nir = img.select("B8").divide(10000.0)
    r_re2 = img.select("B6").divide(10000.0)
    r_sw1 = img.select("B11").divide(10000.0)
    baseline = r_re2.add(
        r_sw1.subtract(r_re2).multiply((833.0 - 740.0) / (1610.0 - 740.0)).multiply(10)
    )
    fdi_filled = r_nir.subtract(baseline).rename("fdi").unmask(-9999)
    stack = fdi_filled.addBands(ee.Image.pixelLonLat()).reproject(crs=TARGET_CRS, scale=TARGET_SCALE)

    TILE_PX = 480
    lat0 = (AOI["min_lat"] + AOI["max_lat"]) / 2
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(lat0))
    tile_deg_lat = TILE_PX * TARGET_SCALE / m_per_deg_lat
    tile_deg_lon = TILE_PX * TARGET_SCALE / m_per_deg_lon
    lon_edges = np.arange(AOI["min_lon"], AOI["max_lon"], tile_deg_lon)
    lat_edges = np.arange(AOI["min_lat"], AOI["max_lat"], tile_deg_lat)
    tile_coords = [(lo, la) for lo in lon_edges for la in lat_edges]
    n_tiles = len(tile_coords)
    hb(f"[S2-FDI] tiling into {len(lon_edges)}x{len(lat_edges)} = {n_tiles} tiles")

    # --- Resume from a partial checkpoint for this date, if one exists. ---
    start_idx = 0
    all_lon, all_lat, all_val = [], [], []
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("date") == date and not cached.get("complete"):
            start_idx = cached["done_idx"]
            all_lon = cached["lon_parts"]
            all_lat = cached["lat_parts"]
            all_val = cached["val_parts"]
            hb(f"[S2-FDI] resuming from checkpoint: {start_idx}/{n_tiles} tiles already done")

    tile_timer = ProgressTimer(n_tiles, "S2-FDI tiling")
    if start_idx:
        tile_timer.update(start_idx)

    for i in range(start_idx, n_tiles):
        lo, la = tile_coords[i]
        tile_region = ee.Geometry.Rectangle([
            lo, la,
            min(lo + tile_deg_lon, AOI["max_lon"]),
            min(la + tile_deg_lat, AOI["max_lat"]),
        ])
        try:
            info = stack.sampleRectangle(region=tile_region).getInfo()  # one round trip
            props = info["properties"]
            t_lon = np.asarray(props["longitude"], dtype=np.float32)
            t_lat = np.asarray(props["latitude"], dtype=np.float32)
            t_val = np.asarray(props["fdi"], dtype=np.float32)
            if t_lon.shape == t_lat.shape == t_val.shape:
                # .ravel() instead of .flatten(): view, not a copy, when the
                # array is already contiguous -- cheaper across 143 tiles.
                all_lon.append(t_lon.ravel())
                all_lat.append(t_lat.ravel())
                all_val.append(t_val.ravel())
            else:
                print(f"  [WARN] tile ({lo:.4f},{la:.4f}) shape mismatch -- skipping")
        except Exception as e:
            print(f"  [WARN] tile ({lo:.4f},{la:.4f}) failed: {e}")

        done = i + 1
        if done % 10 == 0 or done == n_tiles:
            tile_timer.update(done)

        if done % checkpoint_every == 0 and done < n_tiles:
            with open(cache_path, "wb") as f:
                pickle.dump({"date": date, "complete": False, "done_idx": done,
                             "lon_parts": all_lon, "lat_parts": all_lat,
                             "val_parts": all_val}, f)
            hb(f"[S2-FDI] checkpoint saved at tile {done}/{n_tiles}")

    lon = np.concatenate(all_lon) if all_lon else np.array([])
    lat = np.concatenate(all_lat) if all_lat else np.array([])
    val = np.concatenate(all_val) if all_val else np.array([])
    valid = val != -9999
    n_valid = int(valid.sum())
    hb(f"[S2-FDI] {n_valid} valid FDI pixels for {date}")
    assert n_valid > 0, f"[S2-FDI] Zero valid pixels for {date} -- check masks/date."

    lon_valid, lat_valid, fdi_valid = lon[valid], lat[valid], val[valid]
    assert np.nanmax(np.abs(fdi_valid)) < 1.0, (
        "[S2-FDI] values outside [-1,1] -- reflectance scaling wrong "
        "(applied twice, or skipped). Stop here, do not proceed to Panel A."
    )

    # --- Save the COMPLETE cache; replaces any partial checkpoint. ---
    with open(cache_path, "wb") as f:
        pickle.dump({"date": date, "complete": True,
                     "lon": lon_valid, "lat": lat_valid, "fdi": fdi_valid}, f)
    hb(f"[S2-FDI] complete cache saved to {cache_path}")

    return lon_valid, lat_valid, fdi_valid
PREV_N_CANDIDATES_PROXY = 163_816

# ===== NB07 cell index 28 =====
import pickle, os
CBERS_CACHE_DIR = "/content/drive/MyDrive/Marine_Plastic_Detection/cache"
def _cbers_window_cache_path(item_id, date):
    os.makedirs(CBERS_CACHE_DIR, exist_ok=True)
    return os.path.join(CBERS_CACHE_DIR, f"cbers_window_{item_id}_{date}.pkl")
def _get_cbers_window(item, date):
    cache_path = _cbers_window_cache_path(item.id, date)
    if os.path.exists(cache_path):
        hb(f"[CBERS-fast] window cache hit for {item.id} -- skipping the 90s+ bulk read")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    blue_href = item.assets["BAND1"].href
    nir_href = item.assets["BAND4"].href
    abscal_blue, sun_elev = parse_calib_xml(item.assets["BAND1_xml"].href, 1)
    abscal_nir, _ = parse_calib_xml(item.assets["BAND4_xml"].href, 4)

    with rasterio.Env(GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=2), \
         rasterio.open(blue_href) as blue_ds, rasterio.open(nir_href) as nir_ds:
        b = blue_ds.bounds
        # Window covers the AOI's full extent in this raster's CRS, same
        # bbox used everywhere else -- not point-dependent, computed once.
        xs, ys = warp_transform("EPSG:4326", blue_ds.crs,
                                 [AOI["min_lon"], AOI["max_lon"]],
                                 [AOI["min_lat"], AOI["max_lat"]])
        r0, c0 = blue_ds.index(xs[0], ys[1])
        r1, c1 = blue_ds.index(xs[1], ys[0])
        row0, row1 = sorted([max(r0, 0), min(r1, blue_ds.height)])
        col0, col1 = sorted([max(c0, 0), min(c1, blue_ds.width)])
        window = rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)

        blue_win = blue_ds.read(1, window=window)
        nir_win = nir_ds.read(1, window=window)
        transform = blue_ds.transform
        crs = blue_ds.crs
    hb(f"[CBERS-fast] bulk-read window {blue_win.shape} in {time.time()-t0:.1f}s for {item.id}")

    cached = {"blue_win": blue_win, "nir_win": nir_win, "row0": row0, "col0": col0,
              "transform": transform, "crs": crs,
              "abscal_blue": abscal_blue, "abscal_nir": abscal_nir, "sun_elev": sun_elev}
    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)
    hb(f"[CBERS-fast] window cache saved to {cache_path}")
    return cached
def sample_cbers_contrast_at_points_fast(items, points_lonlat, date, return_nir=False):
    """Same output as the original sample_cbers_contrast_at_points, plus an
    optional parallel NIR-reflectance array (return_nir=True) for the glint
    sanity check."""
    n = len(points_lonlat)
    results = [None] * n
    nir_vals = [None] * n if return_nir else None
    assigned = np.zeros(n, dtype=bool)
    lons = np.array([p[0] for p in points_lonlat])
    lats = np.array([p[1] for p in points_lonlat])
    doy = dt.date.fromisoformat(date).timetuple().tm_yday

    for item in items:
        if assigned.all():
            break
        w = _get_cbers_window(item, date)
        xs, ys = warp_transform("EPSG:4326", w["crs"], lons.tolist(), lats.tolist())
        xs, ys = np.array(xs), np.array(ys)
        inv = ~w["transform"]
        cols_f, rows_f = inv * (xs, ys)
        rows_i = np.clip(np.round(rows_f).astype(int), 0, w["blue_win"].shape[0] + w["row0"] - 1)
        cols_i = np.clip(np.round(cols_f).astype(int), 0, w["blue_win"].shape[1] + w["col0"] - 1)
        local_r = rows_i - w["row0"]
        local_c = cols_i - w["col0"]
        in_window = ((local_r >= 0) & (local_r < w["blue_win"].shape[0]) &
                     (local_c >= 0) & (local_c < w["blue_win"].shape[1]) & (~assigned))

        n_valid_here = 0
        for i in np.where(in_window)[0]:
            assigned[i] = True
            b_dn = int(w["blue_win"][local_r[i], local_c[i]])
            n_dn = int(w["nir_win"][local_r[i], local_c[i]])
            if b_dn == 0 or n_dn == 0:
                continue
            r_blue = dn_to_reflectance(b_dn, w["abscal_blue"], w["sun_elev"], ESUN["Blue"], doy)
            r_nir = dn_to_reflectance(n_dn, w["abscal_nir"], w["sun_elev"], ESUN["NIR"], doy)
            results[i] = r_nir - r_blue
            if return_nir:
                nir_vals[i] = r_nir
            n_valid_here += 1
        hb(f"[CBERS-fast] {item.id}: {int(in_window.sum())} in bounds, {n_valid_here} valid")

    n_valid = sum(r is not None for r in results)
    hb(f"[CBERS-fast] {n_valid}/{n} valid, {n - int(assigned.sum())} outside all footprints")
    return (results, nir_vals) if return_nir else results

# ===== NB07 cell index 31 =====
import numpy as np, math, json, os, datetime as dt
from math import sqrt
COL_BIN     = 100
MIN_PER_BIN = 5
FPR_PCT     = 90
BLOCK_PX    = 125
BLOCK_SENS  = (62, 250)
BOOT_B      = 300
BOOT_SEED   = 0
STD_BIN     = 1000
STD_MIN_BG  = 30
SGLI_BOOT_B = 2000
P0          = 0.5
CODE_VERSION = "phase2-v1"
RESULTS_DIR = "/content/drive/MyDrive/Marine_Plastic_Detection/results"
def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
def cbers_k_factors(abscal_blue, abscal_nir, sun_elev, doy, esun_blue, esun_nir):
    """DN -> TOA reflectance is linear per band: R = k * DN. Same formula as dn_to_reflectance."""
    theta = math.radians(90 - sun_elev)
    d2 = 1 / (1 + 0.033 * math.cos(2 * math.pi * doy / 365))
    k = lambda ab, es: math.pi * ab * d2 / (es * math.cos(theta))
    return k(abscal_blue, esun_blue), k(abscal_nir, esun_nir)
def _baseline_lookup(known, vals, cols):
    cb = (cols // COL_BIN) * COL_BIN
    i = np.searchsorted(known, cb)
    i0 = np.clip(i - 1, 0, len(known) - 1); i1 = np.clip(i, 0, len(known) - 1)
    pick = np.where(np.abs(cb - known[i0]) <= np.abs(cb - known[i1]), i0, i1)  # ties -> lower bin
    return vals[pick]
def deglint_metrics(cand, ccol, wc, bg, bcol, wb):
    """1D cross-track de-glint (baseline = median of BACKGROUND ONLY per column bin), fixed-FPR
    threshold, rates, LR+, AUC. wc/wb are bootstrap multiplicities (all ones = point estimate).
    Background is expanded by its integer multiplicities so np.median / np.percentile behave exactly
    as in the original 09-09 run (the point estimate reproduces it bit-for-bit on DN contrast)."""
    wb_i = np.rint(wb).astype(int)
    bg_x, bcol_x = np.repeat(bg, wb_i), np.repeat(bcol, wb_i)
    bb = (bcol_x // COL_BIN) * COL_BIN
    base = {u: np.median(bg_x[bb == u]) for u in np.unique(bb) if (bb == u).sum() >= MIN_PER_BIN}
    known = np.array(sorted(base)); vals = np.array([base[k] for k in known])
    cd = cand - _baseline_lookup(known, vals, ccol)
    bd = bg_x - _baseline_lookup(known, vals, bcol_x)
    thr = np.percentile(bd, FPR_PCT)
    pn = (wc * (cd >= thr)).sum() / wc.sum()
    pb = (bd >= thr).mean()
    s_ = np.sort(bd)
    lo, hi = np.searchsorted(s_, cd, side="left"), np.searchsorted(s_, cd, side="right")
    auc = (wc * ((lo + 0.5 * (hi - lo)) / len(s_))).sum() / wc.sum()   # threshold-free; ties get half credit
    return dict(p_cand=pn, p_bg=pb, gap=pn - pb, lr=pn / pb, auc=auc, n_bins=len(known))
def block_bootstrap(cand, ccol, cblk, bg, bcol, bblk, B=BOOT_B, seed=BOOT_SEED):
    """Two-sample spatial-block bootstrap. The baseline is re-fitted in every replicate, so the
    interval includes de-glint uncertainty. Candidate and background blocks resampled independently."""
    rng = np.random.default_rng(seed)
    cu, cinv = np.unique(cblk, return_inverse=True)
    bu, binv = np.unique(bblk, return_inverse=True)
    out = []
    for _ in range(B):
        wc = rng.multinomial(len(cu), np.ones(len(cu)) / len(cu))[cinv].astype(float)
        wb = rng.multinomial(len(bu), np.ones(len(bu)) / len(bu))[binv].astype(float)
        try:
            r = deglint_metrics(cand, ccol, wc, bg, bcol, wb)
            out.append((r["gap"], r["lr"], r["auc"]))
        except Exception:
            pass
    a = np.array(out)
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]
    return dict(n_reps=len(a), gap_ci=ci(a[:, 0]), lr_ci=ci(a[:, 1]), auc_ci=ci(a[:, 2]))
def fuse(lrs, p0=P0):
    """Sum rule in log-odds (Kittler et al. 1998) == naive-Bayes pooling. Missing sensor = omit."""
    z = math.log(p0 / (1 - p0)) + sum(math.log(x) for x in lrs)
    return 1 / (1 + math.exp(-z))
from rasterio.warp import transform as warp_transform
def cbers_dn_at_points(w, points_lonlat):
    """Same indexing/clipping logic as the export cell whose counts reproduced the original run
    (candidates 144,047 valid; background 1,542 valid on 2024-09-09)."""
    pts = np.asarray(points_lonlat)
    xs, ys = warp_transform("EPSG:4326", w["crs"], pts[:, 0].tolist(), pts[:, 1].tolist())
    cols_f, rows_f = (~w["transform"]) * (np.array(xs), np.array(ys))
    H, W = w["blue_win"].shape
    rows_g = np.clip(np.round(rows_f).astype(int), 0, H + w["row0"] - 1)
    cols_g = np.clip(np.round(cols_f).astype(int), 0, W + w["col0"] - 1)
    r, c = rows_g - w["row0"], cols_g - w["col0"]
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    blue = np.zeros(len(pts)); nir = np.zeros(len(pts))
    blue[inside] = w["blue_win"][r[inside], c[inside]]
    nir[inside] = w["nir_win"][r[inside], c[inside]]
    valid = inside & (blue != 0) & (nir != 0)
    return dict(r=r, c=c, blue=blue, nir=nir, valid=valid)
def analyze_cbers(near, bg, kb, kn, window_width):
    """near/bg: dicts from cbers_dn_at_points. kb/kn: DN->reflectance factors."""
    vn, vb = near["valid"], bg["valid"]
    refl_n = kn * near["nir"][vn] - kb * near["blue"][vn]
    refl_b = kn * bg["nir"][vb] - kb * bg["blue"][vb]
    cn, cb = near["c"][vn], bg["c"][vb]
    rn, rb = near["r"][vn], bg["r"][vb]
    res = dict(n_cand_valid=int(vn.sum()), n_cand_total=int(len(vn)),
               n_bg_valid=int(vb.sum()), n_bg_total=int(len(vb)))

    # (a) raw TOA reflectance contrast at the physical zero (no de-glint) — the original convention
    pn0, pb0 = float((refl_n > 0).mean()), float((refl_b > 0).mean())
    res["raw_zero_threshold"] = dict(p_cand=pn0, p_bg=pb0, gap=pn0 - pb0, lr=pn0 / pb0 if pb0 > 0 else None)
    res["nir_refl_mean"] = dict(cand=float((kn * near["nir"][vn]).mean()), bg=float((kn * bg["nir"][vb]).mean()))

    # (b) PRIMARY: reflectance-space 1D de-glint, point estimate
    one_c, one_b = np.ones(len(refl_n)), np.ones(len(refl_b))
    res["deglint_reflectance"] = deglint_metrics(refl_n, cn, one_c, refl_b, cb, one_b)
    for k_, v_ in res["deglint_reflectance"].items():
        res["deglint_reflectance"][k_] = float(v_)

    # (c) SENSITIVITY: same procedure on DN contrast (what the 2026-09-21 run used; mixes band gains)
    dn_n = near["nir"][vn] - near["blue"][vn]; dn_b = bg["nir"][vb] - bg["blue"][vb]
    r_dn = deglint_metrics(dn_n, cn, one_c, dn_b, cb, one_b)
    res["deglint_dn_sensitivity"] = {k_: float(v_) for k_, v_ in r_dn.items()}

    # (d) spatial block-bootstrap intervals for the primary result (baseline re-fitted per replicate)
    boots = {}
    for blk in (BLOCK_PX,) + tuple(BLOCK_SENS):
        cblk = (rn // blk) * 100000 + (cn // blk); bblk = (rb // blk) * 100000 + (cb // blk)
        boots[f"block_{blk}px"] = block_bootstrap(refl_n, cn, cblk, refl_b, cb, bblk)
        boots[f"block_{blk}px"]["n_cand_blocks"] = int(len(np.unique(cblk)))
    res["bootstrap"] = boots

    # (e) footprint diagnostic: are candidates and background spread alike across the swath?
    W = window_width; edges = np.linspace(0, W, 8)
    share = []
    for i in range(7):
        mc = (near["c"] >= edges[i]) & (near["c"] < edges[i + 1]); mb = (bg["c"] >= edges[i]) & (bg["c"] < edges[i + 1])
        share.append(dict(col_from=int(edges[i]), cand_share=float(mc.mean()), bg_share=float(mb.mean()),
                          cand_valid=float(vn[mc].mean()) if mc.any() else None,
                          bg_valid=float(vb[mb].mean()) if mb.any() else None))
    res["footprint_shares"] = share
    # standardise raw zero-threshold rates to the candidates' cross-track distribution
    ce = np.arange(0, W + STD_BIN, STD_BIN); rows = []
    for i in range(len(ce) - 1):
        mn = (cn >= ce[i]) & (cn < ce[i + 1]); mb_ = (cb >= ce[i]) & (cb < ce[i + 1])
        if mn.sum() > 0 and mb_.sum() >= STD_MIN_BG:
            rows.append((mn.sum(), (refl_n[mn] > 0).mean(), (refl_b[mb_] > 0).mean()))
    if rows:
        a = np.array(rows); wgt = a[:, 0] / a[:, 0].sum()
        pn_s, pb_s = float((wgt * a[:, 1]).sum()), float((wgt * a[:, 2]).sum())
        res["footprint_standardised_raw"] = dict(p_cand=pn_s, p_bg=pb_s, gap=pn_s - pb_s,
                                                 lr=pn_s / pb_s if pb_s > 0 else None,
                                                 cand_coverage=float(a[:, 0].sum() / len(refl_n)), n_bins=len(rows))
    return res
from scipy.spatial import cKDTree
def analyze_sgli(sgli_lat, sgli_lon, sgli_bsi, thr, near_lonlat, bg_lonlat, tol_m, seed=BOOT_SEED):
    sgli_lat = np.asarray(sgli_lat); sgli_lon = np.asarray(sgli_lon)   # keep dtype: mirrors match_sgli_to_points
    m_lat = 111320; m_lon = 111320 * math.cos(math.radians(sgli_lat.mean()))
    tree = cKDTree(np.column_stack([sgli_lon * m_lon, sgli_lat * m_lat]))

    def match(pts):
        pts = np.asarray(pts)
        d, i = tree.query(np.column_stack([pts[:, 0] * m_lon, pts[:, 1] * m_lat]), distance_upper_bound=tol_m)
        ok = np.isfinite(d)
        return ok, i[ok]

    okn, idn = match(near_lonlat); okb, idb = match(bg_lonlat)
    pos = np.asarray(sgli_bsi) > thr
    res = dict(n_valid_sgli_px=int(len(sgli_bsi)), n_cand_matched=int(okn.sum()), n_bg_matched=int(okb.sum()),
               distinct_px_cand=int(len(np.unique(idn))), distinct_px_bg=int(len(np.unique(idb))))
    if okn.sum() == 0 or okb.sum() == 0:
        res["note"] = "no matches — SGLI leg not evaluable for this date"; return res
    pn, pb = float(pos[idn].mean()), float(pos[idb].mean())
    res.update(p_cand=pn, p_bg=pb, gap=pn - pb, lr=pn / pb if pb > 0 else None)
    cc = np.bincount(idn, minlength=len(pos)); bc = np.bincount(idb, minlength=len(pos))
    pu, bu = np.where(cc > 0)[0], np.where(bc > 0)[0]
    rng = np.random.default_rng(seed); out = []
    for _ in range(SGLI_BOOT_B):
        a = rng.choice(pu, len(pu)); b = rng.choice(bu, len(bu))
        rn_ = (cc[a] * pos[a]).sum() / cc[a].sum(); rb_ = (bc[b] * pos[b]).sum() / bc[b].sum()
        out.append((rn_ - rb_, rn_ / rb_ if rb_ > 0 else np.nan))
    out = np.array(out)
    res["cluster_bootstrap"] = dict(gap_ci=[float(v) for v in np.nanpercentile(out[:, 0], [2.5, 97.5])],
                                    lr_ci=[float(v) for v in np.nanpercentile(out[:, 1], [2.5, 97.5])])
    lo = res["cluster_bootstrap"]["lr_ci"][0]
    res["evaluable"] = bool(lo > 1.0)   # LR+ CI excludes 1 -> informative; otherwise report as inconclusive
    return res
def record_current_date(use_sgli=False, allow_sgli_date_mismatch=False):
    """Uses session globals: S2_DATE, CBERS_DATE, cbers_items, s2_fdi, near_pts_fdi, bg_pts_fdi,
    ESUN, SGLI_MATCH_TOLERANCE_M. SGLI is OFF by default: pass use_sgli=True only after Cell 16 has
    just been run for THIS date (guards against silently reusing a previous date's SGLI arrays)."""
    # ---- stale-state guards: every input must belong to the date being recorded ----
    assert CBERS_DATE.replace("-", "") in cbers_items[0].id, \
        f"cbers_items[0]={cbers_items[0].id} does not match CBERS_DATE={CBERS_DATE} — rerun Cell 12"
    frac = len(near_pts_fdi) / len(s2_fdi)
    assert abs(frac - (100 - S2_TOP_PCT) / 100) < 0.002, \
        f"candidate set is {frac:.4f} of s2_fdi, expected ~{(100 - S2_TOP_PCT) / 100} — rerun Cells 13 and 15"
    w = _get_cbers_window(cbers_items[0], CBERS_DATE)          # cache hit after Cell 14
    doy = dt.date.fromisoformat(CBERS_DATE).timetuple().tm_yday
    kb, kn = cbers_k_factors(w["abscal_blue"], w["abscal_nir"], w["sun_elev"], doy, ESUN["Blue"], ESUN["NIR"])
    near = cbers_dn_at_points(w, near_pts_fdi); bg = cbers_dn_at_points(w, bg_pts_fdi)
    cb = analyze_cbers(near, bg, kb, kn, w["blue_win"].shape[1])

    sg = None
    if use_sgli:
        import re
        m = re.search(r"GC1SG1_(\d{4})(\d{2})(\d{2})", os.path.basename(SGLI_FILE))
        sgli_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
        if sgli_date != S2_DATE and not allow_sgli_date_mismatch:
            raise ValueError(f"SGLI_FILE date {sgli_date} != S2_DATE {S2_DATE}; set SGLI_FILE in Cell 4, rerun "
                             f"Cell 16, or pass allow_sgli_date_mismatch=True (then state the offset in the paper)")
        sg = analyze_sgli(sgli_lat_bsi, sgli_lon_bsi, sgli_bsi, float(sgli_bsi_threshold),
                          near_pts_fdi[:, :2], bg_pts_fdi[:, :2], SGLI_MATCH_TOLERANCE_M)

    lr_c = cb["deglint_reflectance"]["lr"]
    lr_c_lo = cb["bootstrap"][f"block_{BLOCK_PX}px"]["lr_ci"][0]      # conservative: block-bootstrap lower bound
    fused = dict(s2_cbers_point=fuse([lr_c]), s2_cbers_conservative=fuse([lr_c_lo]))
    if sg and sg.get("lr"):
        fused["triple_secondary_point_only"] = fuse([lr_c, sg["lr"]])
        fused["triple_note"] = ("SECONDARY. SGLI evaluable=%s (LR+ 95%% cluster CI %s). Do not headline unless evaluable."
                                % (sg.get("evaluable"), [round(x, 2) for x in sg["cluster_bootstrap"]["lr_ci"]]))

    s2d, cbd = dt.date.fromisoformat(S2_DATE), dt.date.fromisoformat(CBERS_DATE)
    rec = dict(code_version=CODE_VERSION, s2_date=S2_DATE, cbers_date=CBERS_DATE,
               gap_days=abs((s2d - cbd).days), cbers_item=cbers_items[0].id,
               sgli_file=(os.path.basename(SGLI_FILE) if use_sgli else None),
               n_cbers_items=len(cbers_items), n_water_px=int(len(s2_fdi)),
               fdi_p99=float(np.percentile(s2_fdi, S2_TOP_PCT)), n_candidates=int(len(near_pts_fdi)),
               n_background=int(len(bg_pts_fdi)), cbers=cb, sgli=sg, fused=fused,
               params=dict(COL_BIN=COL_BIN, MIN_PER_BIN=MIN_PER_BIN, FPR_PCT=FPR_PCT, BLOCK_PX=BLOCK_PX,
                           BOOT_B=BOOT_B, BOOT_SEED=BOOT_SEED, P0=P0, S2_TOP_PCT=S2_TOP_PCT,
                           N_BACKGROUND_SAMPLES=N_BACKGROUND_SAMPLES),
               wording="floating marine-debris/macroplastic aggregation candidates; no ground truth")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"mvp2_{S2_DATE}_cbers{CBERS_DATE}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    d = cb["deglint_reflectance"]; bt = cb["bootstrap"][f"block_{BLOCK_PX}px"]
    print(f"\n=== {S2_DATE} (S2) x {CBERS_DATE} (CBERS), gap {rec['gap_days']} d — floating marine-debris/"
          f"macroplastic aggregation candidates ===")
    print(f"CBERS valid: cand {cb['n_cand_valid']}/{cb['n_cand_total']}, bg {cb['n_bg_valid']}/{cb['n_bg_total']}")
    print(f"raw (physical zero): cand {cb['raw_zero_threshold']['p_cand']*100:.1f}%  bg {cb['raw_zero_threshold']['p_bg']*100:.1f}%")
    print(f"de-glint REFLECTANCE: cand {d['p_cand']*100:.1f}%  bg {d['p_bg']*100:.1f}%  gap {d['gap']*100:.1f}  "
          f"LR+ {d['lr']:.2f}  AUC {d['auc']:.3f}")
    print(f"  block-bootstrap 95% (1 km): gap {[round(x*100,1) for x in bt['gap_ci']]}  "
          f"LR+ {[round(x,2) for x in bt['lr_ci']]}  AUC {[round(x,3) for x in bt['auc_ci']]}")
    print(f"de-glint DN (sensitivity): gap {cb['deglint_dn_sensitivity']['gap']*100:.1f}  LR+ {cb['deglint_dn_sensitivity']['lr']:.2f}")
    if "footprint_standardised_raw" in cb:
        s = cb["footprint_standardised_raw"]
        print(f"footprint-standardised raw: gap {s['gap']*100:.1f}  LR+ {s['lr']:.1f}  (candidate coverage {s['cand_coverage']:.0%})")
    print(f"fused S2+CBERS (primary): {fused['s2_cbers_point']*100:.1f}%  (conservative {fused['s2_cbers_conservative']*100:.1f}%)")
    if sg:
        if "gap" in sg:
            print(f"SGLI (SECONDARY): {sg['n_cand_matched']} cand matches from only {sg['distinct_px_cand']} distinct px; "
                  f"gap {sg['gap']*100:.1f}; LR+ cluster-CI {[round(x,2) for x in sg['cluster_bootstrap']['lr_ci']]}; evaluable={sg['evaluable']}")
        else:
            print("SGLI (SECONDARY):", sg.get("note"))
    print("saved:", path)
    return rec
def load_all_records():
    """Aggregate every date's JSON into one table (H5 profile + time-gap comparison)."""
    import glob
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "mvp2_*.json")))]
    print(f"{'S2 date':<11}{'CBERS':<11}{'gap_d':>5} {'p_cand':>7} {'p_bg':>6} {'LR+':>6} {'LR+ 95% CI':>13} {'AUC':>6} {'AUC 95% CI':>14}")
    for r in sorted(recs, key=lambda r: (r["gap_days"], r["s2_date"])):
        d = r["cbers"]["deglint_reflectance"]; b = r["cbers"]["bootstrap"][f"block_{BLOCK_PX}px"]
        print(f"{r['s2_date']:<11}{r['cbers_date']:<11}{r['gap_days']:>5} {d['p_cand']*100:>6.1f}% {d['p_bg']*100:>5.1f}% "
              f"{d['lr']:>6.2f} [{b['lr_ci'][0]:.2f},{b['lr_ci'][1]:.2f}] {d['auc']:>6.3f} [{b['auc_ci'][0]:.3f},{b['auc_ci'][1]:.3f}]")
    return recs

# ===== NB07 cell index 33 =====
import numpy as np, math, json, os, glob, datetime as dt
from scipy.stats import rankdata
FDI_EDGES   = [0.0140, 0.0225, 0.0309, np.inf]
BAND_NAMES  = ["weak", "mid", "strong"]
STRONG_BAND = 2
N_PER_BAND  = 30000
N_BG_OVER   = 100000
BAND_SEED   = 7
CELL_M      = 1000.0
MIN_C, MIN_B = 5, 5
CELL_BOOT_B = 2000
DIST_CUTS_M = (200, 500, 1000)
EXPORT_DIR  = "/content/drive/MyDrive/Marine_Plastic_Detection/exports"
def cell_ids(lon, lat):
    """1-km cell key on a local metric grid anchored at the AOI corner. Identical to the key used in the
    09-12 / 09-14 exports (injective for this AOI), so saved files stay comparable."""
    lat0, lon0 = AOI["min_lat"], AOI["min_lon"]
    m_lat = 111320.0
    m_lon = 111320.0 * math.cos(math.radians((AOI["min_lat"] + AOI["max_lat"]) / 2))
    ix = np.floor((np.asarray(lon, np.float64) - lon0) * m_lon / CELL_M).astype(np.int64)
    iy = np.floor((np.asarray(lat, np.float64) - lat0) * m_lat / CELL_M).astype(np.int64)
    return ix * 100000 + iy
def distance_to_nonwater_m(water_lon, water_lat, query_lonlat, res=10.0, chunk=4_000_000):
    """Distance (m) from each query point to the nearest pixel that is NOT valid water in this date's S2
    array (land, cloud, no-data lumped together). Also returns distance to the grid boundary so points
    whose 'nearest non-water' is only the AOI edge can be flagged. Grid = EPSG:32723, 10 m."""
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32723", always_xy=True)
    n = len(water_lon)
    xs = np.empty(n); ys = np.empty(n)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        xs[a:b], ys[a:b] = tr.transform(np.asarray(water_lon[a:b], np.float64), np.asarray(water_lat[a:b], np.float64))
    x0, y0 = xs.min(), ys.min()
    ix = np.rint((xs - x0) / res).astype(np.int64); iy = np.rint((ys - y0) / res).astype(np.int64)
    H, W = int(iy.max()) + 1, int(ix.max()) + 1
    water = np.zeros((H, W), dtype=bool)
    water[iy, ix] = True
    n_unique = int(water.sum())
    del xs, ys, ix, iy
    dist = distance_transform_edt(water).astype(np.float32) * res
    q = np.asarray(query_lonlat)
    qx, qy = tr.transform(q[:, 0].astype(np.float64), q[:, 1].astype(np.float64))
    qi = np.clip(np.rint((np.asarray(qx) - x0) / res).astype(np.int64), 0, W - 1)
    qj = np.clip(np.rint((np.asarray(qy) - y0) / res).astype(np.int64), 0, H - 1)
    d_edge = np.minimum.reduce([qi, W - 1 - qi, qj, H - 1 - qj]) * res
    info = dict(grid_shape=(H, W), n_water_px_in=int(n), n_unique_grid_cells=n_unique)
    return dist[qj, qi], d_edge.astype(np.float32), info
def build_band_samples(with_distance=True):
    """Same sampling logic/order as the 09-12 and 09-14 exports (regression-checked in Colab)."""
    assert CBERS_DATE.replace("-", "") in cbers_items[0].id, "cbers_items does not match CBERS_DATE — rerun Cell 12"
    rng = np.random.default_rng(BAND_SEED)
    w = _get_cbers_window(cbers_items[0], CBERS_DATE)                    # cache hit
    lon, lat, fdi = np.asarray(s2_lon), np.asarray(s2_lat), np.asarray(s2_fdi)
    S, tot = {}, 0
    for i in range(len(FDI_EDGES) - 1):
        idx = np.where((fdi >= FDI_EDGES[i]) & (fdi < FDI_EDGES[i + 1]))[0]
        pick = rng.choice(idx, min(N_PER_BAND, len(idx)), replace=False) if len(idx) else idx
        pts = np.column_stack([lon[pick], lat[pick]])
        d = cbers_dn_at_points(w, pts)
        S[f"band{i}_lonlat"], S[f"band{i}_fdi"], S[f"band{i}_n_in_band"] = pts, fdi[pick], len(idx)
        S.update({f"band{i}_{k}": v for k, v in d.items()})
        tot += len(idx)
        hb(f"[bands] {BAND_NAMES[i]} [{FDI_EDGES[i]}, {FDI_EDGES[i+1]}): {len(idx)} px, sampled {len(pick)}, CBERS-valid {int(d['valid'].sum())}")
    pool = np.where(fdi <= np.percentile(fdi, 50))[0]
    bpick = rng.choice(pool, min(N_BG_OVER, len(pool)), replace=False)
    bpts = np.column_stack([lon[bpick], lat[bpick]])
    bd = cbers_dn_at_points(w, bpts)
    S["bgL_lonlat"], S["bgL_fdi"] = bpts, fdi[bpick]
    S.update({f"bgL_{k}": v for k, v in bd.items()})
    hb(f"[bands] background oversample: CBERS-valid {int(bd['valid'].sum())}/{len(bpick)} | band pixels total {tot}")
    S["s2_cells"] = np.unique(cell_ids(lon, lat))
    for k in ("abscal_blue", "abscal_nir", "sun_elev"):
        S[k] = w[k]
    S.update(doy=dt.date.fromisoformat(CBERS_DATE).timetuple().tm_yday, esun_blue=ESUN["Blue"], esun_nir=ESUN["NIR"],
             window_shape=np.array(w["blue_win"].shape), fdi_p99=np.percentile(fdi, S2_TOP_PCT),
             n_water_px=len(fdi), edges=np.array(FDI_EDGES[:-1]))
    if with_distance:
        hb("[dist] building 10 m valid-water grid + distance transform (~20-60 s)")
        allpts = np.vstack([S[f"band{i}_lonlat"] for i in range(len(FDI_EDGES) - 1)] + [S["bgL_lonlat"]])
        dist, dedge, info = distance_to_nonwater_m(lon, lat, allpts)
        a = 0
        for pre in [f"band{i}" for i in range(len(FDI_EDGES) - 1)] + ["bgL"]:
            n = len(S[f"{pre}_lonlat"]); S[f"{pre}_dist"] = dist[a:a + n]; S[f"{pre}_edge"] = dedge[a:a + n]; a += n
        hb(f"[dist] grid {info['grid_shape']}, unique water cells {info['n_unique_grid_cells']} vs input px {info['n_water_px_in']} "
           f"(equal or slightly lower is fine; a large gap means pixel collisions)")
    return S
def _auc(a, b):
    r = rankdata(np.concatenate([a, b])); n1, n0 = len(a), len(b)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def _prep(S, pre, domain_cells=None, min_dist=None):
    kb, kn = cbers_k_factors(float(S["abscal_blue"]), float(S["abscal_nir"]), float(S["sun_elev"]),
                             int(S["doy"]), float(S["esun_blue"]), float(S["esun_nir"]))
    ll = S[f"{pre}_lonlat"]; cell = cell_ids(ll[:, 0], ll[:, 1]); m = np.array(S[f"{pre}_valid"], bool)
    if domain_cells is not None:
        m &= np.isin(cell, domain_cells)
    if min_dist is not None and f"{pre}_dist" in S:
        m &= np.asarray(S[f"{pre}_dist"]) >= min_dist
    con = kn * np.asarray(S[f"{pre}_nir"]) - kb * np.asarray(S[f"{pre}_blue"])
    return dict(con=con[m], cell=cell[m])
def _cell_table(cand, bg):
    """Per shared cell: within-cell AUC (candidates vs background), weight n1*n0, n1."""
    cells = np.intersect1d(np.unique(cand["cell"]), np.unique(bg["cell"]))
    rows = []
    for c in cells:
        m, b = cand["cell"] == c, bg["cell"] == c
        n1, n0 = int(m.sum()), int(b.sum())
        if n1 >= MIN_C and n0 >= MIN_B:
            rows.append((c, _auc(cand["con"][m], bg["con"][b]), float(n1 * n0), n1))
    if not rows:
        return dict(cells=np.array([], np.int64), auc=np.array([]), w=np.array([]), n1=np.array([], int))
    a = np.array(rows)
    return dict(cells=a[:, 0].astype(np.int64), auc=a[:, 1], w=a[:, 2], n1=a[:, 3].astype(int))
def _agg(t, mult=None):
    m = np.ones(len(t["auc"])) if mult is None else mult
    den = (m * t["w"]).sum()
    return float((m * t["auc"] * t["w"]).sum() / den) if den > 0 else float("nan")
def _boot_one(t, B=CELL_BOOT_B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed); k = len(t["auc"])
    v = [_agg(t, rng.multinomial(k, np.ones(k) / k).astype(float)) for _ in range(B)]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
def analyze_band_samples(S, domain_cells=None):
    """Within-date result per FDI band (own CBERS footprint unless domain_cells is given)."""
    res = dict(bands={})
    bg = _prep(S, "bgL", domain_cells)
    for i in range(len(FDI_EDGES) - 1):
        cd = _prep(S, f"band{i}", domain_cells)
        t = _cell_table(cd, bg)
        r = dict(name=BAND_NAMES[i], n_in_band=int(S[f"band{i}_n_in_band"]), n_cand_valid=int(len(cd["con"])),
                 n_cells=int(len(t["auc"])), n_cand_used=int(t["n1"].sum()) if len(t["auc"]) else 0)
        if len(t["auc"]) >= 5:
            r["auc"] = _agg(t); r["auc_ci"] = _boot_one(t)
        else:
            r["auc"] = None; r["note"] = "fewer than 5 eligible cells — not evaluable"
        res["bands"][BAND_NAMES[i]] = r
    # strong band away from non-water (needs the distance arrays)
    if f"band{STRONG_BAND}_dist" in S:
        dres = {}
        for cut in DIST_CUTS_M:
            cd = _prep(S, f"band{STRONG_BAND}", domain_cells, cut); bgc = _prep(S, "bgL", domain_cells, cut)
            t = _cell_table(cd, bgc)
            share_c = float((np.asarray(S[f"band{STRONG_BAND}_dist"])[S[f"band{STRONG_BAND}_valid"]] < cut).mean())
            share_b = float((np.asarray(S["bgL_dist"])[S["bgL_valid"]] < cut).mean())
            e = dict(share_strong_cand_within=share_c, share_bg_within=share_b, n_cand=int(len(cd["con"])), n_cells=int(len(t["auc"])))
            if len(t["auc"]) >= 5:
                e["auc"] = _agg(t); e["auc_ci"] = _boot_one(t)
            dres[f">={cut}m"] = e
        res["strong_band_by_min_distance_to_nonwater"] = dres
    return res
def paired_compare(Sa, Sb, label_a="A", label_b="B", B=CELL_BOOT_B, seed=BOOT_SEED):
    """Same-strength, same-cells comparison of two S2 dates against the SAME CBERS image.
    Domain = 1-km cells with valid S2 water on both dates; then only cells eligible in both."""
    dom = np.intersect1d(Sa["s2_cells"], Sb["s2_cells"])
    bga, bgb = _prep(Sa, "bgL", dom), _prep(Sb, "bgL", dom)
    out = dict(domain_cells=int(len(dom)), a=label_a, b=label_b, bands={})
    rng = np.random.default_rng(seed)
    for i in range(len(FDI_EDGES) - 1):
        ta = _cell_table(_prep(Sa, f"band{i}", dom), bga); tb = _cell_table(_prep(Sb, f"band{i}", dom), bgb)
        shared = np.intersect1d(ta["cells"], tb["cells"])
        if len(shared) < 5:
            out["bands"][BAND_NAMES[i]] = dict(shared_cells=int(len(shared)), note="fewer than 5 shared cells — not evaluable"); continue
        ia, ib = np.isin(ta["cells"], shared), np.isin(tb["cells"], shared)
        sa = {k: v[ia] for k, v in ta.items()}; sb = {k: v[ib] for k, v in tb.items()}
        # align cell order between the two tables
        oa, ob = np.argsort(sa["cells"]), np.argsort(sb["cells"])
        sa = {k: v[oa] for k, v in sa.items()}; sb = {k: v[ob] for k, v in sb.items()}
        d = []
        for _ in range(B):
            m = rng.multinomial(len(shared), np.ones(len(shared)) / len(shared)).astype(float)
            d.append(_agg(sa, m) - _agg(sb, m))
        d = np.array(d)
        out["bands"][BAND_NAMES[i]] = dict(shared_cells=int(len(shared)), auc_a=_agg(sa), auc_b=_agg(sb),
                                           diff_a_minus_b=_agg(sa) - _agg(sb),
                                           diff_ci=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                                           p_a_greater=float((d > 0).mean()))
    return out
def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
def record_strength_bands(with_distance=True):
    S = build_band_samples(with_distance)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    tag = f"{S2_DATE}_cbers{CBERS_DATE}"
    npz_path = os.path.join(EXPORT_DIR, f"mvp2_bands_{tag}.npz")
    np.savez_compressed(npz_path, **S)
    res = analyze_band_samples(S)
    s2d, cbd = dt.date.fromisoformat(S2_DATE), dt.date.fromisoformat(CBERS_DATE)
    sun_elev = float(S["sun_elev"])
    rec = dict(code_version="phase2-cell22-v1", s2_date=S2_DATE, cbers_date=CBERS_DATE, gap_days=abs((s2d - cbd).days),
               cbers_item=cbers_items[0].id, sun_elev_deg=sun_elev, sun_zenith_deg=90 - sun_elev,
               note_glint="nadir-view offset from specular geometry ~ sun zenith (smaller = higher glint risk)",
               fdi_p99=float(S["fdi_p99"]), n_water_px=int(S["n_water_px"]), fdi_edges=[float(x) for x in FDI_EDGES[:-1]],
               window_shape=[int(x) for x in S["window_shape"]], n_s2_cells_1km=int(len(S["s2_cells"])),
               bg_oversample_valid=int(np.sum(S["bgL_valid"])), results=res,
               params=dict(N_PER_BAND=N_PER_BAND, N_BG_OVER=N_BG_OVER, BAND_SEED=BAND_SEED, CELL_M=CELL_M,
                           MIN_C=MIN_C, MIN_B=MIN_B, CELL_BOOT_B=CELL_BOOT_B, DIST_CUTS_M=list(DIST_CUTS_M)),
               wording="floating marine-debris/macroplastic aggregation candidates; no ground truth")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jpath = os.path.join(RESULTS_DIR, f"strength_{tag}.json")      # 'strength_' prefix: not picked up by load_all_records()
    with open(jpath, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"\n=== {S2_DATE} (S2) x {CBERS_DATE} (CBERS), gap {rec['gap_days']} d | sun elevation {sun_elev:.1f} deg | "
          f"FDI p99 {rec['fdi_p99']:.4f} — floating marine-debris/macroplastic aggregation candidates ===")
    for nm, r in res["bands"].items():
        if r["auc"] is not None:
            print(f"  {nm:>6}: AUC {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]  "
                  f"cells {r['n_cells']}  candidates used {r['n_cand_used']}  (px in band {r['n_in_band']})")
        else:
            print(f"  {nm:>6}: not evaluable ({r.get('note')})  px in band {r['n_in_band']}, CBERS-valid {r['n_cand_valid']}")
    for k, e in res.get("strong_band_by_min_distance_to_nonwater", {}).items():
        a = f"AUC {e['auc']:.3f} [{e['auc_ci'][0]:.3f}, {e['auc_ci'][1]:.3f}]" if "auc" in e else "not evaluable"
        print(f"  strong band, points {k} from non-water: {a} | cells {e['n_cells']} | share of strong candidates "
              f"within cut {e['share_strong_cand_within']:.2f} vs background {e['share_bg_within']:.2f}")
    print("saved:", jpath, "|", npz_path)
    return rec
def compare_saved_bands(path_a, path_b, label_a="A", label_b="B"):
    out = paired_compare(load_npz(path_a), load_npz(path_b), label_a, label_b)
    print(f"paired, same CBERS image, {out['domain_cells']} shared S2 1-km cells | AUC {label_a} minus {label_b}")
    for nm, r in out["bands"].items():
        if "auc_a" in r:
            print(f"  {nm:>6}: shared cells {r['shared_cells']:3d} | AUC {label_a} {r['auc_a']:.3f}  {label_b} {r['auc_b']:.3f} | "
                  f"diff {r['diff_a_minus_b']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]  P({label_a}>{label_b})={r['p_a_greater']:.2f}")
        else:
            print(f"  {nm:>6}: {r['note']}")
    return out
def load_strength_records():
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "strength_*.json")))]
    print(f"{'S2 date':<11}{'CBERS':<11}{'gap':>3} {'sun':>5} {'FDI p99':>8}  " + "  ".join(f"{n:>22}" for n in BAND_NAMES))
    for r in sorted(recs, key=lambda r: (r["s2_date"], r["cbers_date"])):
        cells = []
        for n in BAND_NAMES:
            b = r["results"]["bands"][n]
            cells.append(f"{b['auc']:.3f} [{b['auc_ci'][0]:.2f},{b['auc_ci'][1]:.2f}]" if b.get("auc") is not None else "n/e")
        print(f"{r['s2_date']:<11}{r['cbers_date']:<11}{r['gap_days']:>3} {r['sun_elev_deg']:>5.1f} {r['fdi_p99']:>8.4f}  "
              + "  ".join(f"{c:>22}" for c in cells))
    return recs

# ===== NB07 cell index 35 =====
import numpy as np, math, os, glob, shutil, time, datetime as dt
from scipy.ndimage import distance_transform_edt
R_TILE_PX   = 320
R_CACHE_DIR = "/content/drive/MyDrive/Marine_Plastic_Detection/cache"
SCL_LAND    = (4, 5)
SCL_CLOUD   = (3, 8, 9, 10)
FDI_W_OURS  = (833.0 - 740.0) / (1610.0 - 740.0)
FDI_W_PAPER = (833.0 - 665.0) / (1610.0 - 665.0)
GRID_RES    = 10.0
def _raster_paths(date):
    return (os.path.join(R_CACHE_DIR, f"s2_raster_{date}.npz"),
            os.path.join(R_CACHE_DIR, f"s2_raster_tiles_{date}"))
def fetch_s2_raster_tiles(date, max_retries=3):
    """Per-tile files on Drive (resumable). Unmasked: water AND non-water pixels are kept, with SCL."""
    cache_path, tiles_dir = _raster_paths(date)
    os.makedirs(tiles_dir, exist_ok=True)
    aoi = aoi_ee_geometry()
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi)
            .filterDate(date, (dt.date.fromisoformat(date) + dt.timedelta(days=1)).isoformat()))
    n_images = coll.size().getInfo()
    assert n_images > 0, f"[S2-RASTER] no S2 images for {date}"
    hb(f"[S2-RASTER] {n_images} granule(s) on {date} — mosaicking B6, B8, B11, SCL (unmasked)")
    stack = (coll.mosaic().select(["B6", "B8", "B11", "SCL"]).unmask(0)
             .addBands(ee.Image.pixelLonLat()).reproject(crs="EPSG:32723", scale=10))
    lat0 = (AOI["min_lat"] + AOI["max_lat"]) / 2
    tile_deg_lat = R_TILE_PX * 10 / 111320
    tile_deg_lon = R_TILE_PX * 10 / (111320 * math.cos(math.radians(lat0)))
    lon_edges = np.arange(AOI["min_lon"], AOI["max_lon"], tile_deg_lon)
    lat_edges = np.arange(AOI["min_lat"], AOI["max_lat"], tile_deg_lat)
    tiles = [(lo, la) for lo in lon_edges for la in lat_edges]
    hb(f"[S2-RASTER] {len(lon_edges)}x{len(lat_edges)} = {len(tiles)} tiles of {R_TILE_PX} px")
    timer = ProgressTimer(len(tiles), "S2-RASTER tiling")
    failed = []
    for i, (lo, la) in enumerate(tiles):
        tp = os.path.join(tiles_dir, f"tile_{i:04d}.npz")
        if not os.path.exists(tp):
            region = ee.Geometry.Rectangle([lo, la, min(lo + tile_deg_lon, AOI["max_lon"]),
                                            min(la + tile_deg_lat, AOI["max_lat"])])
            ok = False
            for attempt in range(max_retries):
                try:
                    p = stack.sampleRectangle(region=region).getInfo()["properties"]
                    arrs = {k: np.asarray(p[k]) for k in ("B6", "B8", "B11", "SCL", "longitude", "latitude")}
                    assert all(a.shape == arrs["SCL"].shape for a in arrs.values()), "shape mismatch"
                    np.savez_compressed(tp, B6=arrs["B6"].astype(np.int16), B8=arrs["B8"].astype(np.int16),
                                        B11=arrs["B11"].astype(np.int16), SCL=arrs["SCL"].astype(np.uint8),
                                        lon=arrs["longitude"].astype(np.float32), lat=arrs["latitude"].astype(np.float32))
                    ok = True; break
                except Exception as e:
                    hb(f"  [WARN] tile {i} attempt {attempt+1}/{max_retries} failed: {str(e)[:120]}"); time.sleep(2 * (attempt + 1))
            if not ok:
                failed.append(i)
        if (i + 1) % 20 == 0 or i + 1 == len(tiles):
            timer.update(i + 1)
    if failed:
        raise RuntimeError(f"[S2-RASTER] {len(failed)} tile(s) failed {failed[:10]} — rerun this cell; finished tiles are kept")
    return tiles_dir
def _phase(v, res=GRID_RES):
    ang = 2 * np.pi * (np.asarray(v, np.float64) % res) / res
    return float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * res / (2 * np.pi)) % res)
def assemble_s2_raster(tiles_dir):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32723", always_xy=True)
    files = sorted(glob.glob(os.path.join(tiles_dir, "tile_*.npz")))
    assert files, "no tile files"
    cx, cy = tr.transform([AOI["min_lon"], AOI["max_lon"], AOI["min_lon"], AOI["max_lon"]],
                          [AOI["min_lat"], AOI["min_lat"], AOI["max_lat"], AOI["max_lat"]])
    X0 = math.floor(min(cx) / GRID_RES) * GRID_RES - 1000.0
    Y0 = math.floor(min(cy) / GRID_RES) * GRID_RES - 1000.0
    W = int(math.ceil((max(cx) - X0) / GRID_RES)) + 100
    H = int(math.ceil((max(cy) - Y0) / GRID_RES)) + 100
    z0 = np.load(files[0]); x, y = tr.transform(z0["lon"].ravel().astype(np.float64), z0["lat"].ravel().astype(np.float64))
    phx, phy = _phase(x), _phase(y)
    R = dict(B6=np.zeros((H, W), np.int16), B8=np.zeros((H, W), np.int16), B11=np.zeros((H, W), np.int16),
             SCL=np.zeros((H, W), np.uint8), X0=X0, Y0=Y0, phx=phx, phy=phy, res=GRID_RES)
    filled = np.zeros((H, W), bool); n_px = 0
    for f in files:
        z = np.load(f)
        x, y = tr.transform(z["lon"].ravel().astype(np.float64), z["lat"].ravel().astype(np.float64))
        ix = np.rint((np.asarray(x) - phx - X0) / GRID_RES).astype(np.int64)
        iy = np.rint((np.asarray(y) - phy - Y0) / GRID_RES).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        for k in ("B6", "B8", "B11", "SCL"):
            R[k][iy[ok], ix[ok]] = z[k].ravel()[ok]
        filled[iy[ok], ix[ok]] = True; n_px += int(ok.sum())
    R["n_tile_px"] = n_px; R["n_filled"] = int(filled.sum())
    return R
def get_s2_raster(date):
    cache_path, tiles_dir = _raster_paths(date)
    if os.path.exists(cache_path):
        z = np.load(cache_path); R = {k: z[k] for k in z.files}
        for k in ("X0", "Y0", "phx", "phy", "res"): R[k] = float(R[k])
        hb(f"[S2-RASTER] cache hit {cache_path} ({R['SCL'].shape})"); return R
    fetch_s2_raster_tiles(date)
    R = assemble_s2_raster(tiles_dir)
    np.savez_compressed(cache_path, **R)
    hb(f"[S2-RASTER] saved {cache_path} ({os.path.getsize(cache_path)/1e6:.0f} MB); tile files kept in {tiles_dir} "
       f"(delete that folder after the checks below pass)")
    return R
def water_mask(R):
    return (R["SCL"] == 6) & (R["B8"] > 0) & (R["B6"] > 0) & (R["B11"] > 0)
def fdi_from_dn(b6, b8, b11, w):
    r = lambda a: np.asarray(a, np.float64) / 10000.0
    return r(b8) - (r(b6) + (r(b11) - r(b6)) * w * 10.0)
def to_grid_index(R, lon, lat, chunk=4_000_000):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32723", always_xy=True)
    n = len(lon); ix = np.empty(n, np.int64); iy = np.empty(n, np.int64)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        x, y = tr.transform(np.asarray(lon[a:b], np.float64), np.asarray(lat[a:b], np.float64))
        ix[a:b] = np.rint((np.asarray(x) - R["phx"] - R["X0"]) / R["res"]).astype(np.int64)
        iy[a:b] = np.rint((np.asarray(y) - R["phy"] - R["Y0"]) / R["res"]).astype(np.int64)
    H, W = R["SCL"].shape
    return np.clip(ix, 0, W - 1), np.clip(iy, 0, H - 1), (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
def _dist(mask, res=GRID_RES):
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return (distance_transform_edt(~mask) * res).astype(np.float32)
def class_distance_maps(R):
    """Distances (m) from every cell to the nearest LAND / CLOUD / OTHER / NODATA non-water pixel."""
    w = water_mask(R); nonwater = ~w; scl = R["SCL"]
    land = nonwater & np.isin(scl, SCL_LAND)
    cloud = nonwater & np.isin(scl, SCL_CLOUD)
    nodata = nonwater & (scl == 0)
    other = nonwater & ~land & ~cloud & ~nodata
    return dict(land=_dist(land), cloud=_dist(cloud), other=_dist(other), nodata=_dist(nodata),
                counts=dict(water=int(w.sum()), land=int(land.sum()), cloud=int(cloud.sum()),
                            other=int(other.sum()), nodata=int(nodata.sum())))
def augment_points(R, S, dm=None):
    """Adds to the sample dict S (band{i}_*, bgL_*): distances to land/cloud/other/nodata, nearest non-water class,
    and FDI recomputed with our weight and with the paper's printed weight."""
    dm = dm or class_distance_maps(R)
    out = dict(S)
    for pre in [k[:-7] for k in S if k.endswith("_lonlat")]:
        ll = S[f"{pre}_lonlat"]; ix, iy, inside = to_grid_index(R, ll[:, 0], ll[:, 1])
        dl, dc, do, dn = dm["land"][iy, ix], dm["cloud"][iy, ix], dm["other"][iy, ix], dm["nodata"][iy, ix]
        stack = np.vstack([dl, dc, do])
        nn = np.where(np.isfinite(stack.min(axis=0)), stack.argmin(axis=0) + 1, 0).astype(np.int8)   # 1 land, 2 cloud, 3 other, 0 none
        b6, b8, b11 = R["B6"][iy, ix], R["B8"][iy, ix], R["B11"][iy, ix]
        out.update({f"{pre}_dland": dl, f"{pre}_dcloud": dc, f"{pre}_dother": do, f"{pre}_dnodata": dn,
                    f"{pre}_nnclass": nn, f"{pre}_scl": R["SCL"][iy, ix],
                    f"{pre}_fdi_ours": fdi_from_dn(b6, b8, b11, FDI_W_OURS).astype(np.float32),
                    f"{pre}_fdi_paper": fdi_from_dn(b6, b8, b11, FDI_W_PAPER).astype(np.float32),
                    f"{pre}_inside": inside})
    out["scl_counts"] = np.array([dm["counts"][k] for k in ("water", "land", "cloud", "other", "nodata")])
    return out
def compare_fdi_weights(R, dm, top_pct=99, near_m=100.0):
    """Whole-scene effect of the interpolation weight (ours 740 nm vs printed lambda_RED) on the top-1% candidate set."""
    w = water_mask(R)
    fo = fdi_from_dn(R["B6"][w], R["B8"][w], R["B11"][w], FDI_W_OURS)
    fp = fdi_from_dn(R["B6"][w], R["B8"][w], R["B11"][w], FDI_W_PAPER)
    to, tp = np.percentile(fo, top_pct), np.percentile(fp, top_pct)
    a, b = fo >= to, fp >= tp
    dmin = np.minimum.reduce([dm["land"][w], dm["cloud"][w], dm["other"][w]])
    hb(f"[weights] top-{100-top_pct}% thresholds: ours {to:.4f}, paper-weight {tp:.4f} | water px {int(w.sum())}")
    hb(f"[weights] candidate sets: ours {int(a.sum())}, paper {int(b.sum())}, overlap {float((a & b).sum() / a.sum()):.3f} of ours "
       f"| corr(FDI ours, FDI paper) {float(np.corrcoef(fo[::50], fp[::50])[0, 1]):.3f}")
    hb(f"[weights] share within {near_m:.0f} m of land/cloud/other: all water {float((dmin < near_m).mean()):.3f} | "
       f"top set ours {float((dmin[a] < near_m).mean()):.3f} | top set paper {float((dmin[b] < near_m).mean()):.3f}")
    return dict(thr_ours=float(to), thr_paper=float(tp), overlap=float((a & b).sum() / a.sum()))
def verify_raster_vs_cached_fdi(R, chunk=4_000_000):
    """Recompute FDI (our weight) from the raster at every cached FDI location and compare."""
    w = water_mask(R); n_unique = int(w.sum())
    hb(f"[verify] unique water cells in raster: {n_unique} | cached FDI pixels: {len(s2_fdi)} "
       f"(cached list contains tile-seam duplicates, so raster <= cached by ~1%)")
    ix, iy, inside = to_grid_index(R, np.asarray(s2_lon), np.asarray(s2_lat))
    maxd, n_mismatch = 0.0, 0
    for a in range(0, len(ix), chunk):
        b = min(a + chunk, len(ix)); sl = slice(a, b)
        re = fdi_from_dn(R["B6"][iy[sl], ix[sl]], R["B8"][iy[sl], ix[sl]], R["B11"][iy[sl], ix[sl]], FDI_W_OURS)
        d = np.abs(re - np.asarray(s2_fdi[sl], np.float64)); maxd = max(maxd, float(d.max()))
        n_mismatch += int((d > 1e-6).sum())
    hb(f"[verify] FDI recomputed vs cached: max |diff| {maxd:.2e}, points with diff > 1e-6: {n_mismatch} of {len(ix)} "
       f"| cached points on raster water cells: {float(w[iy, ix].mean()):.4f} | outside grid: {int((~inside).sum())}")
    return dict(n_unique_water=n_unique, max_abs_diff=maxd, n_mismatch=n_mismatch)
def augment_saved_samples(date_s2=None, date_cbers=None):
    """Loads mvp2_bands_<tag>.npz (Cell 22), adds the new per-point columns, saves ..._aug.npz."""
    d1, d2 = date_s2 or S2_DATE, date_cbers or CBERS_DATE
    tag = f"{d1}_cbers{d2}"
    S = load_npz(os.path.join(EXPORT_DIR, f"mvp2_bands_{tag}.npz"))
    assert int(S["n_water_px"]) == len(s2_fdi), (
        f"s2_fdi in memory has {len(s2_fdi)} px but the {tag} samples were built from {int(S['n_water_px'])} — "
        f"reload this date: s2_lon, s2_lat, s2_fdi = get_s2_fdi_array(S2_DATE)")
    R = get_s2_raster(d1)
    hb(f"[aug] raster {R['SCL'].shape}, tile px {R.get('n_tile_px')}, filled cells {R.get('n_filled')}")
    verify_raster_vs_cached_fdi(R)
    hb("[aug] distance maps (4 EDTs, ~1-2 min)")
    dm = class_distance_maps(R)
    hb(f"[aug] non-water class counts (cells): {dm['counts']}")
    compare_fdi_weights(R, dm)
    out = augment_points(R, S, dm)
    # consistency with Cell 22's distance (old = distance to any non-valid-water pixel inside the valid bbox)
    for pre in ("band2", "bgL"):
        if f"{pre}_dist" in S:
            new = np.minimum.reduce([out[f"{pre}_dland"], out[f"{pre}_dcloud"], out[f"{pre}_dother"], out[f"{pre}_dnodata"]])
            old = np.asarray(S[f"{pre}_dist"]); ok = np.isfinite(new)
            hb(f"[aug] {pre}: new min-distance == Cell-22 distance (within 1 px) for {float((np.abs(new[ok]-old[ok]) <= 14.2).mean()):.3f} of points")
        b = out[f"{pre}_nnclass"]; v = np.asarray(S[f"{pre}_valid"])
        hb(f"[aug] {pre} nearest non-water class (CBERS-valid pts): land {float((b[v]==1).mean()):.2f}, "
           f"cloud {float((b[v]==2).mean()):.2f}, other {float((b[v]==3).mean()):.2f}, none {float((b[v]==0).mean()):.2f}")
        hb(f"[aug] {pre} FDI recomputed (ours) vs stored: max |diff| {float(np.abs(out[f'{pre}_fdi_ours']-np.asarray(S[f'{pre}_fdi'])).max()):.2e}")
    path = os.path.join(EXPORT_DIR, f"mvp2_bands_{tag}_aug.npz")
    np.savez_compressed(path, **out)
    hb(f"[aug] saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path

# ===== NB07 cell index 37 =====
import numpy as np, json, os
CLASS_NAMES = {1: "land", 2: "cloud", 3: "other"}
MIN_CLASS_N = 15
OPEN_WATER_MIN_DIST_M = 500.0
def _prep_class(S, pre, cls_code, domain_cells=None):
    kb, kn = cbers_k_factors(float(S["abscal_blue"]), float(S["abscal_nir"]), float(S["sun_elev"]),
                             int(S["doy"]), float(S["esun_blue"]), float(S["esun_nir"]))
    ll = S[f"{pre}_lonlat"]; cell = cell_ids(ll[:, 0], ll[:, 1])
    m = np.asarray(S[f"{pre}_valid"], bool) & (np.asarray(S[f"{pre}_nnclass"]) == cls_code)
    if domain_cells is not None:
        m &= np.isin(cell, domain_cells)
    con = kn * np.asarray(S[f"{pre}_nir"]) - kb * np.asarray(S[f"{pre}_blue"])
    return dict(con=con[m], cell=cell[m])
def class_matched_auc(S, band_key="band2", domain_cells=None, min_n=MIN_CLASS_N):
    """Cell-matched AUC within each nearest-non-water class, for one FDI band vs the background pool."""
    out = {}
    for code, name in CLASS_NAMES.items():
        cd = _prep_class(S, band_key, code, domain_cells)
        bg = _prep_class(S, "bgL", code, domain_cells)
        cd["cell"], bg["cell"] = cd["cell"], bg["cell"]      # (kept for clarity; _cell_table needs 'cell'+'con')
        t = _cell_table({"con": cd["con"], "cell": cd["cell"]}, {"con": bg["con"], "cell": bg["cell"]})
        r = dict(n_cand=int(len(cd["con"])), n_bg=int(len(bg["con"])), n_cells=int(len(t["auc"])))
        r.update(auc=_agg(t), auc_ci=_boot_one(t)) if len(t["auc"]) >= 5 else r.update(auc=None, note="fewer than 5 eligible cells")
        out[name] = r
    return out
def open_water_other_only(S, band_key="band2", domain_cells=None):
    """Candidates AND background both restricted to 'other'-class edges (excludes land, cloud outright)."""
    return class_matched_auc(S, band_key, domain_cells).get("other")
def open_water_primary(S, band_key="band2", domain_cells=None, min_dist=OPEN_WATER_MIN_DIST_M):
    """PRIMARY open-water metric: candidates and background both restricted to dland >= min_dist,
    cell-matched. This is the headline result for the strong band away from land adjacency."""
    kb, kn = cbers_k_factors(float(S["abscal_blue"]), float(S["abscal_nir"]), float(S["sun_elev"]),
                             int(S["doy"]), float(S["esun_blue"]), float(S["esun_nir"]))
    def prep(pre):
        ll = S[f"{pre}_lonlat"]; cell = cell_ids(ll[:, 0], ll[:, 1])
        m = np.asarray(S[f"{pre}_valid"], bool) & (np.asarray(S[f"{pre}_dland"]) >= min_dist)
        if domain_cells is not None:
            m &= np.isin(cell, domain_cells)
        con = kn * np.asarray(S[f"{pre}_nir"]) - kb * np.asarray(S[f"{pre}_blue"])
        return dict(con=con[m], cell=cell[m])
    cd, bg = prep(band_key), prep("bgL")
    t = _cell_table(cd, bg)
    r = dict(min_dist_m=min_dist, n_cand=int(len(cd["con"])), n_bg=int(len(bg["con"])), n_cells=int(len(t["auc"])))
    if len(t["auc"]) >= 5:
        r.update(auc=_agg(t), auc_ci=_boot_one(t))
    else:
        r.update(auc=None, note=f"fewer than 5 eligible cells at dland>={min_dist:.0f}m — not evaluable for this date")
    return r
def record_class_analysis(aug_path, s2_date, cbers_date, gap_days):
    S = load_npz(aug_path)
    res = {b: class_matched_auc(S, f"band{i}") for i, b in enumerate(BAND_NAMES)}
    ow = open_water_primary(S)                          # PRIMARY headline metric, fixed rule
    rec = dict(s2_date=s2_date, cbers_date=cbers_date, gap_days=gap_days, by_band_by_class=res,
               open_water_primary=ow)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"classauc_{s2_date}_cbers{cbers_date}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"=== {s2_date} x {cbers_date} (gap {gap_days}d): class-matched AUC, strong band ===")
    for name, r in res["strong"].items():
        a = f"{r['auc']:.3f} [{r['auc_ci'][0]:.3f},{r['auc_ci'][1]:.3f}]" if r.get("auc") is not None else r.get("note")
        print(f"  {name:>6}: {a}  (cand {r['n_cand']}, bg {r['n_bg']}, cells {r['n_cells']})")
    a = f"{ow['auc']:.3f} [{ow['auc_ci'][0]:.3f},{ow['auc_ci'][1]:.3f}]" if ow.get("auc") is not None else ow.get("note")
    print(f"  PRIMARY open-water (dland>={OPEN_WATER_MIN_DIST_M:.0f}m): {a}  (cand {ow['n_cand']}, cells {ow['n_cells']})")
    print("saved:", path)
    return rec
def _selftest():
    # Expected values are CELL-MATCHED (same class AND same 1-km cell) — stricter than, and not expected to
    # equal, an earlier non-cell-matched same-class comparison reported in chat (0.933 / 0.785).
    a = class_matched_auc(load_npz(EXPORT_DIR + "/mvp2_bands_2024-09-12_cbers2024-09-14_aug.npz"))["land"]["auc"]
    b = class_matched_auc(load_npz(EXPORT_DIR + "/mvp2_bands_2024-09-14_cbers2024-09-14_aug.npz"))["cloud"]["auc"]
    ok = abs(a - 0.908) < 0.01 and abs(b - 0.602) < 0.01
    print(f"CHECK: 09-12 land AUC {a:.3f} (expect ~0.908), 09-14 cloud AUC {b:.3f} (expect ~0.602) -> {'PASS' if ok else 'FAIL, stop and report'}")

# ===== NB07 cell index 42 =====
import datetime as dt
def probe_cbers_dates(center_date, tolerance_days=4):
    """Tests REAL cached S2 candidate/background points against every nearby CBERS scene.
    Zero EE cost -- S2 must already be tiled for this date (true for 01-11)."""
    d0 = dt.date.fromisoformat(center_date)
    lo, hi = (d0 - dt.timedelta(days=tolerance_days)).isoformat(), (d0 + dt.timedelta(days=tolerance_days)).isoformat()
    catalog = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")
    wpm_id = next(c.id for c in catalog.get_collections() if "WPM" in c.id.upper())
    aoi_geojson = {"type": "Polygon", "coordinates": [[
        [AOI["min_lon"], AOI["min_lat"]], [AOI["max_lon"], AOI["min_lat"]],
        [AOI["max_lon"], AOI["max_lat"]], [AOI["min_lon"], AOI["max_lat"]], [AOI["min_lon"], AOI["min_lat"]]]]}
    items = list(catalog.search(collections=[wpm_id], intersects=aoi_geojson,
                                datetime=f"{lo}T00:00:00Z/{hi}T23:59:59Z").items())
    hb(f"[probe] {len(items)} scene(s) found in +/-{tolerance_days}d -- testing each against REAL cached candidates")
    results = []
    for it in items:
        idate = it.datetime.date().isoformat()
        w = _get_cbers_window(it, idate)     # pays the ~90s read once per new scene, cached after
        near = cbers_dn_at_points(w, near_pts_fdi)
        bg = cbers_dn_at_points(w, bg_pts_fdi)
        gap = abs((dt.date.fromisoformat(idate) - d0).days)
        results.append((near["valid"].sum(), bg["valid"].sum(), idate, gap, it))
        hb(f"  {idate}  gap {gap}d  {it.id}: cand valid {near['valid'].sum()}/{len(near_pts_fdi)}, "
           f"bg valid {bg['valid'].sum()}/{len(bg_pts_fdi)}")
    results.sort(key=lambda r: -r[0])
    return results

# ===== NB07 cell index 45 =====
def cbers_grid_coverage(item, date, n_side=25):
    """Fixed AOI-wide grid (no S2 dependency at all -- can run before any S2 tiling). Reports
    overall coverage and west-half coverage, since real candidates at this AOI have consistently
    clustered west of lon -44.30 on every date checked. This is what established the working rule:
    CBERS-4A/WPM path 200/row 142 reliably covers this AOI's west half (west-half 0.88-0.96 across
    every scene tested); path 199 and any row-143 scene consistently fail (0.00-0.47)."""
    w = _get_cbers_window(item, date)
    lons = np.linspace(AOI["min_lon"], AOI["max_lon"], n_side)
    lats = np.linspace(AOI["min_lat"], AOI["max_lat"], n_side)
    LO, LA = np.meshgrid(lons, lats)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    r = cbers_dn_at_points(w, pts)
    west = pts[:, 0] < -44.30
    return dict(date=date, id=item.id, overall=r["valid"].mean(),
               west_half=r["valid"][west].mean() if west.any() else float("nan"), n_pts=len(pts))
def scan_candidates(date_tolerance_pairs, n_side=25):
    """Surveys several candidate S2 dates at once, before tiling any of them. Cost: one ~90s CBERS
    window read per DISTINCT scene found (cached after), zero EE/S2 cost."""
    catalog = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")
    wpm_id = next(c.id for c in catalog.get_collections() if "WPM" in c.id.upper())
    aoi_geojson = {"type": "Polygon", "coordinates": [[
        [AOI["min_lon"], AOI["min_lat"]], [AOI["max_lon"], AOI["min_lat"]],
        [AOI["max_lon"], AOI["max_lat"]], [AOI["min_lon"], AOI["max_lat"]], [AOI["min_lon"], AOI["min_lat"]]]]}
    rows = []
    for center, tol in date_tolerance_pairs:
        d0 = dt.date.fromisoformat(center)
        lo, hi = (d0 - dt.timedelta(days=tol)).isoformat(), (d0 + dt.timedelta(days=tol)).isoformat()
        items = list(catalog.search(collections=[wpm_id], intersects=aoi_geojson,
                                    datetime=f"{lo}T00:00:00Z/{hi}T23:59:59Z").items())
        for it in items:
            idate = it.datetime.date().isoformat()
            cov = cbers_grid_coverage(it, idate, n_side)
            rows.append((center, idate, cov["overall"], cov["west_half"], it.id))
            hb(f"  {center} -> {idate}  overall {cov['overall']:.2f}  west-half {cov['west_half']:.2f}  {it.id}")
    return rows

# ===== NB07 cell index 47 =====
def own_threshold_open_water_auc(s2_date, cbers_date, cbers_item, pct=99, min_dist=500.0):
    """PRIMARY method (supersedes the fixed 0.0140/0.0225/0.0309 bands from Cell 13 for cross-date
    comparison). The fixed bands were set from the September dates' own p99 and turned out NOT to
    represent "top 1%" on dates with very different FDI magnitude -- confirmed concretely on
    2024-05-07, where the fixed-band result (0.712) reversed to 0.497 (chance) once corrected to
    that date's own top-1% threshold (own p99 = 0.0040, vs the fixed 0.0309 cut, an 8x difference).
    Every date's own-threshold result should be treated as the number of record; the fixed-band
    numbers in Cells 13/15's records remain useful only for the land/cloud/other class breakdown,
    not for the primary cross-date AUC comparison.

    IMPORTANT: cbers_item must be passed in explicitly and pre-filtered by the caller (do not call
    find_cbers_items(cbers_date)[0] inside this function) -- on 2024-02-05, two scenes intersect the
    AOI (_200_142_ good, _200_143_ dead) and STAC's default ordering returned the dead one first,
    silently producing a zero-candidate, zero-cell, uninterpretable result. Same failure class as
    Part 5's documented scene-selection bugs."""
    R = get_s2_raster(s2_date)
    w_mask = water_mask(R)
    fdi_full = fdi_from_dn(R["B6"], R["B8"], R["B11"], FDI_W_OURS)
    fdi_water = fdi_full[w_mask]
    own_thr = np.percentile(fdi_water, pct)
    hb(f"[own-pctile] {s2_date} own top-{100-pct}% threshold = {own_thr:.4f}")

    rows, cols = np.where(w_mask)
    strong = fdi_full[w_mask] >= own_thr
    rr, cc = rows[strong], cols[strong]
    hb(f"[own-pctile] {int(strong.sum())} pixels at own threshold")

    x = R["X0"] + R["phx"] + cc * R["res"]
    y = R["Y0"] + R["phy"] + rr * R["res"]
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    pts = np.column_stack([lon, lat])
    rng = np.random.default_rng(11)
    if len(pts) > 30000:
        pts = pts[rng.choice(len(pts), 30000, replace=False)]

    wnd = _get_cbers_window(cbers_item, cbers_date)      # uses the PASSED-IN item, not a fresh search
    cand = cbers_dn_at_points(wnd, pts)

    S = load_npz(EXPORT_DIR + f"/mvp2_bands_{s2_date}_cbers{cbers_date}_aug.npz")
    bg_valid = np.asarray(S["bgL_valid"], bool)
    bg_dland = np.asarray(S["bgL_dland"])
    bg_lonlat = S["bgL_lonlat"]
    kb, kn = cbers_k_factors(float(S["abscal_blue"]), float(S["abscal_nir"]), float(S["sun_elev"]),
                             int(S["doy"]), float(S["esun_blue"]), float(S["esun_nir"]))

    dm = class_distance_maps(R)
    ix, iy, inside = to_grid_index(R, pts[:, 0], pts[:, 1])
    dland_cand = np.minimum.reduce([dm["land"][iy, ix]])
    m_cand = cand["valid"] & (dland_cand >= min_dist)
    m_bg = bg_valid & (bg_dland >= min_dist)
    hb(f"[own-pctile] candidates total {len(pts)}, CBERS-valid {int(cand['valid'].sum())}, "
       f">={min_dist:.0f}m from land {int(m_cand.sum())} | background CBERS-valid {int(bg_valid.sum())}, "
       f">={min_dist:.0f}m from land {int(m_bg.sum())}")

    con_cand = kn * cand["nir"][m_cand] - kb * cand["blue"][m_cand]
    con_bg = kn * np.asarray(S["bgL_nir"])[m_bg] - kb * np.asarray(S["bgL_blue"])[m_bg]
    cell_c = cell_ids(pts[m_cand, 0], pts[m_cand, 1])
    cell_b = cell_ids(bg_lonlat[m_bg, 0], bg_lonlat[m_bg, 1])
    t = _cell_table({"con": con_cand, "cell": cell_c}, {"con": con_bg, "cell": cell_b})
    hb(f"[own-pctile] eligible cells = {len(t['auc'])}")

    if len(t["auc"]) < 5:
        return dict(auc=None, note=f"only {len(t['auc'])} eligible cells at dland>={min_dist:.0f}m -- not evaluable",
                    n_cand=int(m_cand.sum()), n_bg=int(m_bg.sum()), own_thr=float(own_thr))
    result = dict(auc=_agg(t), auc_ci=_boot_one(t), n_cand=int(m_cand.sum()), n_cells=int(len(t["auc"])),
                  own_thr=float(own_thr))
    hb(f"[own-pctile] open-water AUC at OWN threshold: {result}")
    return result

# ===== NB07 cell index 87 =====
import json, os
import numpy as np
from itertools import permutations
from scipy import stats
DATE_PAIRS = [
    ("2024-09-09", "2024-09-09", "CBERS_4A_WPM_20240909_200_142_L4", 0),
    ("2024-09-14", "2024-09-14", "CBERS_4A_WPM_20240914_199_142_L4", 0),
    ("2024-08-08", "2024-08-09", "CBERS_4A_WPM_20240809_200_142_L4", 1),
    ("2024-05-07", "2024-05-08", "CBERS_4A_WPM_20240508_200_142_L4", 1),
    ("2024-02-05", "2024-02-05", "CBERS_4A_WPM_20240205_200_142_L4", 0),  # NOT the _143_ scene
]
def recompute_all():
    """Re-derives each date's own-threshold open-water AUC from cached rasters/exports.
    No new EE cost -- get_s2_raster and _get_cbers_window are both cache hits for these dates."""
    results = []
    for s2_date, cbers_date, item_id, gap in DATE_PAIRS:
        hb(f"[trend-test] recomputing {s2_date} x {cbers_date}")
        item = next(it for it in find_cbers_items(cbers_date) if it.id == item_id)
        r = own_threshold_open_water_auc(s2_date, cbers_date, item)
        assert r.get("auc") is not None, f"{s2_date} came back not-evaluable -- stop and check before trusting the trend test"
        results.append(dict(s2_date=s2_date, cbers_date=cbers_date, gap_days=gap,
                            own_thr=r["own_thr"], auc=r["auc"], ci=list(r["auc_ci"]),
                            n_cand=r["n_cand"], n_cells=r["n_cells"]))
    return results
def save_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "own_threshold_summary_table13_1.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    hb(f"[trend-test] saved {path}")
    return path
def run_trend_test(results):
    """Spearman (exact permutation, appropriate at n=5 -- asymptotic p is unreliable here),
    weighted least squares (weights = inverse-variance from each date's own CI width), and an
    unweighted Pearson check. All three are reported; none should be cherry-picked post hoc."""
    fdi = np.array([r["own_thr"] for r in results])
    auc = np.array([r["auc"] for r in results])
    se = np.array([(r["ci"][1] - r["ci"][0]) / (2 * 1.96) for r in results])
    w = 1 / se ** 2
    n = len(results)

    rho, _ = stats.spearmanr(fdi, auc)
    ranks_fdi, ranks_auc = stats.rankdata(fdi), stats.rankdata(auc)
    obs = np.corrcoef(ranks_fdi, ranks_auc)[0, 1]
    count = total = 0
    for perm in permutations(range(n)):
        total += 1
        if np.corrcoef(ranks_fdi, ranks_auc[list(perm)])[0, 1] >= obs - 1e-9:
            count += 1
    p_spearman_exact = count / total

    X = np.column_stack([np.ones(n), fdi]); W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ auc)
    resid = auc - X @ beta; dof = n - 2
    sigma2 = (resid @ W @ resid) / dof
    se_beta = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ W @ X)))
    t_slope = beta[1] / se_beta[1]
    p_wls = 2 * (1 - stats.t.cdf(abs(t_slope), dof))

    r_pearson = np.corrcoef(fdi, auc)[0, 1]
    count2 = 0
    for perm in permutations(range(n)):
        if np.corrcoef(fdi, auc[list(perm)])[0, 1] >= r_pearson - 1e-9:
            count2 += 1
    p_pearson_exact = count2 / total

    out = dict(spearman_rho=float(rho), spearman_exact_p_onesided=float(p_spearman_exact),
               wls_slope=float(beta[1]), wls_slope_se=float(se_beta[1]), wls_p_twosided=float(p_wls),
               pearson_r=float(r_pearson), pearson_exact_p_onesided=float(p_pearson_exact))
    print(f"Spearman rho = {rho:.3f}, exact permutation p (one-sided) = {p_spearman_exact:.4f}")
    print(f"Weighted least squares: slope = {beta[1]:.2f} +/- {se_beta[1]:.2f}, two-sided p = {p_wls:.4f}")
    print(f"Unweighted Pearson r = {r_pearson:.3f}, exact permutation p (one-sided) = {p_pearson_exact:.4f}")
    return out

# ===== NB07 cell index 102 =====
def load_sgli_bsi_grid(filepath):
    hb(f"[SGLI-BSI] opening {filepath}")
    with h5py.File(filepath, "r") as f:
        lat_tie = f["Geometry_data/Latitude"][:].astype(np.float32)
        lon_tie = f["Geometry_data/Longitude"][:].astype(np.float32)

        blue_ds = f["Image_data/NWLR_443"]
        green_ds = f["Image_data/NWLR_565"]
        red_ds = f["Image_data/NWLR_670"]
        qa_ds = f["Image_data/QA_flag"]
        target_shape = red_ds.shape

        zoom_row = target_shape[0] / lat_tie.shape[0]
        zoom_col = target_shape[1] / lat_tie.shape[1]

        pad_deg = 0.05
        tie_in_aoi = (
            (lon_tie >= AOI["min_lon"] - pad_deg) & (lon_tie <= AOI["max_lon"] + pad_deg) &
            (lat_tie >= AOI["min_lat"] - pad_deg) & (lat_tie <= AOI["max_lat"] + pad_deg)
        )
        rows, cols = np.where(tie_in_aoi)
        assert len(rows) > 0, "[SGLI-BSI] No tie-points fall near the AOI even with padding."
        tie_row_min, tie_row_max = rows.min(), rows.max()
        tie_col_min, tie_col_max = cols.min(), cols.max()

        tie_pad = 1
        r0 = max(tie_row_min - tie_pad, 0)
        r1 = min(tie_row_max + tie_pad + 1, lat_tie.shape[0])
        c0 = max(tie_col_min - tie_pad, 0)
        c1 = min(tie_col_max + tie_pad + 1, lat_tie.shape[1])
        full_r0, full_r1 = int(r0 * zoom_row), min(int(r1 * zoom_row), target_shape[0])
        full_c0, full_c1 = int(c0 * zoom_col), min(int(c1 * zoom_col), target_shape[1])
        hb(f"[SGLI-BSI] windowed full-res read: rows {full_r0}:{full_r1}, cols {full_c0}:{full_c1}")

        blue = blue_ds[full_r0:full_r1, full_c0:full_c1].astype(np.float32)
        green = green_ds[full_r0:full_r1, full_c0:full_c1].astype(np.float32)
        red = red_ds[full_r0:full_r1, full_c0:full_c1].astype(np.float32)
        qa = qa_ds[full_r0:full_r1, full_c0:full_c1]

        blue = blue * blue_ds.attrs.get("Slope", 1.0) + blue_ds.attrs.get("Offset", 0.0)
        green = green * green_ds.attrs.get("Slope", 1.0) + green_ds.attrs.get("Offset", 0.0)
        red = red * red_ds.attrs.get("Slope", 1.0) + red_ds.attrs.get("Offset", 0.0)

        lat_tie_crop = lat_tie[r0:r1, c0:c1]
        lon_tie_crop = lon_tie[r0:r1, c0:c1]
        win_shape = red.shape
        zr = win_shape[0] / lat_tie_crop.shape[0]
        zc = win_shape[1] / lat_tie_crop.shape[1]
        lat_full = zoom(lat_tie_crop, (zr, zc), order=1)[:win_shape[0], :win_shape[1]].astype(np.float32)
        lon_full = zoom(lon_tie_crop, (zr, zc), order=1)[:win_shape[0], :win_shape[1]].astype(np.float32)

    del lat_tie, lon_tie
    gc.collect()

    baseline = blue + (red - blue) * ((565.0 - 443.0) / (670.0 - 443.0))
    bsi = (green - baseline).astype(np.float32)
    good = (qa == 0)

    n_good = int(good.sum())
    hb(f"[SGLI-BSI] windowed QA-valid pixels: {n_good} of {good.size} ({100 * n_good / good.size:.1f}%)")
    assert n_good > 0, "[SGLI-BSI] Zero QA-valid pixels in the windowed AOI region."

    result = lat_full[good], lon_full[good], bsi[good]
    del blue, green, red, qa, lat_full, lon_full, bsi, good
    gc.collect()
    return result
