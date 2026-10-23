# Frozen Analysis Protocol — Confirmatory Dates and Blind Candidate Characterization

**Project:** POLI/USP + IBGE, ODS 14.1.1(b) — Baía da Ilha Grande: Sentinel-2 × CBERS-4A/WPM (primary),
GCOM-C/SGLI (secondary), same dates
**Status:** FINAL — binding once timestamped (§7). Nothing in §2–§6 may change after that; implemented by NB08.
**Governing record:** `MVP2_composite_index_math_reasoning.md`, Sections 12–14.

**Terminology rule:** every output describes *floating marine-debris/macroplastic aggregation
candidates*. No ground truth of material identity exists; sensor agreement indicates an optical
surface anomaly, not confirmed plastic.

---

## 1. Why this protocol exists

The five dates in Table 13.1 were analysed while several rules were still being developed (the
500 m cut, the own-threshold correction, the de-glint method). They are therefore **exploratory**.
The magnitude trend they suggest (Spearman ρ = 0.800, exact one-sided p = 0.067, n = 5) can only
be tested fairly on dates that nobody has analysed, processed under rules fixed in advance. This
protocol fixes those rules, the date-selection criteria, the tests and the interpretation of every
possible outcome, **before** any new date is scouted.

## 2. Frozen analysis rules (identical to the `_final` notebook; no retuning)

| Element | Frozen value | Source |
|---|---|---|
| S2 product | `COPERNICUS/S2_SR_HARMONIZED`, SCL class 6 = water | Cell 9 / 14 |
| FDI | R_B8 − [R_B6 + (R_B11 − R_B6)·w·10], w = (833−740)/(1610−740) = 0.1069 | Cell 14 (`FDI_W_OURS`) |
| Candidates | each date's own top 1% of SCL-water FDI (p99) | Cell 17 |
| Candidate subsample | 30,000, `default_rng(11)` | Cell 17 |
| Background | `bgL` sample from `record_strength_bands` (N_BG_OVER = 100,000, BAND_SEED = 7) | Cell 13 |
| CBERS contrast | k_NIR·DN_NIR − k_Blue·DN_Blue (TOA reflectance), explicit scene ID | Cells 12, 17 |
| Open-water rule | distance to SCL land ≥ 500 m, candidates and background | Cell 15 |
| Matching | 1 km cells; MIN_C = MIN_B = 5; ≥ 5 eligible cells or "not evaluable" | Cell 13 |
| AUC | cell-weighted (n1·n0) within-cell AUC | Cell 13 (`_agg`) |
| CI | cell bootstrap, B = 2,000, seed 0, percentile 95% | Cell 13 (`_boot_one`) |

Code version: `nb07_core.py` — the `_final` notebook's definition cells extracted verbatim (load-time runs
stripped only) — and `nb08_confirm.py` (date selection, orchestration, tests). `nb07_core.py` also includes the
`load_sgli_bsi_grid` definition from the `_final` notebook's Archive (verbatim, definition only). SHA-256 of both in §7;
NB08 refuses to run if either differs. Any code change requires a dated, logged deviation (§6).

## 3. Date selection (objective, decided before scouting)

**Search window:** 2023-01-01 to 2025-12-31, excluding the seven dates already scouted
(2024-09-09, 09-12, 09-14, 08-08, 05-07, 02-05, 01-11).

**Eligibility — applied in this order; every scouted date and its outcome is logged:**

- **E1 — CBERS pairing.** A CBERS-4A/WPM L4 scene exists within 0 to 1 day of the S2 acquisition.
- **E2 — S2 pre-filter.** Mean `CLOUDY_PIXEL_PERCENTAGE` over all S2_SR_HARMONIZED granules intersecting
  the AOI on that UTC date ≤ 30% (metadata only; no pixels inspected).
- **E3 — Clear water.** After S2 tiling, ≥ 11,000,000 SCL-water pixels in the AOI raster (≈ 60% of
  the largest water count on record, 18,792,094 on 2024-05-07).
- **E4 — CBERS usability.** ≥ 50% of the date's own-threshold candidates are CBERS-valid on the
  chosen scene, tested with real candidate points (`probe_cbers_dates`). If more than one scene
  qualifies, the one with the highest candidate validity is used. Candidates for this test are generated
  exactly as in NB07 Cell 17 (own top 1%, 30,000 subsample, seed 11).
- **E5 — Evaluability.** ≥ 5 eligible cells at the 500 m rule (the function's own criterion).
  A date failing E5 is excluded from the trend test and reported, as 2024-09-12 was.

- **E6 — Independence.** One S2 date per CBERS scene. If two confirmed S2 dates use the same CBERS
  scene (possible under the ±1-day rule), the one with the smaller S2–CBERS gap counts; a tie goes to
  the earlier S2 date. The other is processed and reported as a sensitivity analysis only.

**Order and stopping rule.** Eligible dates are processed in chronological order from the start of
the window. Processing stops when **8 independent confirmatory dates** pass E1–E6, or when the window is
exhausted. No date may be skipped, replaced or re-ordered for any reason other than E1–E5. FDI
magnitude and AUC play no part in selection.

**Minimum for a confirmatory claim:** 6 dates. With fewer, results are reported as descriptive and
underpowered; no confirmatory conclusion is drawn.

## 4. Hypotheses and tests

**H-C1 (primary, confirmatory).** Across the new dates only, open-water AUC increases with the
date's own-threshold FDI p99.
- Test: Spearman ρ, one-sided, permutation p-value, α = 0.05.
- Permutations: exact enumeration for n ≤ 9 (p = count/n!); for n ≥ 10, Monte Carlo with 200,000
  permutations, seed 0, p = (1 + count)/(1 + 200,000). Verified to reproduce NB07 Cell 18 exactly on
  Table 13.1 (ρ = 0.800, p = 0.0667; Pearson p = 0.100; WLS p = 0.2081).
- Exact one-sided critical values at α = 0.05: n = 6 → ρ ≥ 0.829; n = 7 → 0.714; n = 8 → 0.643.

**H-C2 (secondary, descriptive).** The same test on all dates pooled (5 exploratory + new),
explicitly labelled as including exploratory data.

**Reported regardless of outcome (not used for the confirmatory conclusion):**
Pearson r (permutation) and WLS slope; the same tests at the 1,000 m and 200 m cuts and under the
printed FDI weight (λ_RED = 665 nm); per-date AUC with CI; the full exclusion log.

## 4b. SGLI leg — secondary hypothesis H-S1 (same dates, pre-declared)

**Role.** Tests whether the third sensor adds corroboration on the same confirmatory dates. Secondary:
it never affects which dates enter the S2 × CBERS test, and no primary conclusion rests on it.

**Data rule.** For each confirmed S2 date: GCOM-C/SGLI L2 NWLR, 1 km product (`NWLRK`, the standard
G-Portal product used for 2024-09-09), same UTC date as S2. All granules for that date are placed in the
input folder; the one with the most QA-valid pixels in the AOI window is used. No file, or no QA-valid
pixel, is logged as "SGLI not available" for that date.

**Analysis (frozen, identical to the exploratory 2024-09-09 SGLI leg).** BSI = R₅₆₅ − [R₄₄₃ + (R₆₇₀ −
R₄₄₃)·(565−443)/(670−443)], QA_flag = 0; positive = BSI above the scene's own 90th percentile; candidates =
all top-1% FDI points of the date; background = 20,000 random points with FDI ≤ p50 (seed 42); KDTree
match within 300 m; LR+ = candidate positive rate / background positive rate; cluster bootstrap over
distinct SGLI pixels, B = 2,000, seed 0.

**H-S1 (pooled).** Clusters from all dates with usable SGLI data are pooled (each date's pixels are
distinct clusters) and the same cluster bootstrap is applied. Pooling is pre-declared because a single
date yields only ~50 distinct matched pixels at 1 km (§13.7). Verified: for one date the pooled
estimator reproduces `analyze_sgli` exactly.

| Pooled LR+ 95% CI | Manuscript statement |
|---|---|
| lower bound > 1 | SGLI corroborates the S2 × CBERS candidates at 1 km, secondary evidence |
| upper bound < 1 | SGLI contradicts at 1 km |
| includes 1 | Inconclusive at the 1 km resolution available (as for 2024-09-09) |
| no usable date | Not assessable |

Per-date results are reported for every date. The fused three-sensor probability is **not** a
pre-declared outcome and is not reported as a finding.

## 5. Blind visual characterization of candidates

**Question:** what surface features do the open-water candidates correspond to, compared with
background water? (Addresses "what was detected" without claiming material identity.)

**Sample.** From each of the five exploratory dates: 20 own-threshold open-water candidates and
20 open-water background points (≥ 500 m from land), drawn at random with seed 2026. Total 200
points. Points are pooled, shuffled, and given anonymous IDs; the key linking IDs to
candidate/background status is withheld from the labelers until labeling is complete.

**Imagery.** For each point, a 500 m × 500 m Sentinel-2 SR chip from the same acquisition, 256 px,
EPSG:32723: true color (B4/B3/B2, 0–2000, γ 1.3) and false color (B8/B4/B3, 0–3000, γ 1.3), with a
magenta ring around the point that leaves the centre visible. Identical stretch for every chip.
Candidates are CBERS-valid and ≥ 500 m from land, exactly as in the primary metric.
Very-high-resolution imagery (e.g. Google Earth historical) may be consulted only for **persistent
fixed structures** (aquaculture rafts, moorings, buoys, rocks), since VHR dates will not match.

**Classes (fixed):**
1. Linear/patchy floating feature (slick, line, windrow, patch)
2. Foam / whitecap
3. Vessel, wake, or fixed structure (incl. aquaculture)
4. Glint or haze artifact
5. Cloud or cloud-shadow edge
6. Nothing visible / uniform water
7. Uncertain

**Procedure.** Two labelers work independently. Agreement is reported as Cohen's κ. Disagreements
are resolved by discussion after κ is computed, and the resolution is logged.

**Outcome.** Class proportions for candidates vs background, with Wilson 95% CIs, and a
χ² or Fisher test of the difference. All proportions are reported, including unfavourable ones
(e.g. a high share of vessels, structures or glint among candidates).

## 6. Interpretation fixed in advance

| Outcome of H-C1 | What the manuscript says |
|---|---|
| p < 0.05 | The magnitude-dependent trend replicated on independent dates under frozen rules. |
| p ≥ 0.05, ρ > 0 | Direction consistent, not confirmed; the trend remains a hypothesis. |
| ρ ≤ 0 | The exploratory trend did not replicate; reported as such. |
| < 6 eligible dates | Underpowered; descriptive results only. |

All four outcomes are publishable, and the paper's framing is written to accommodate any of them:
the primary contributions are the reproducible zero-cost pipeline and the methods findings; the
trend is the secondary, tested result.

**Deviations.** Any departure from §2–§5 (a bug fix, a service outage forcing a change) is logged
with date, reason and effect, and reported in the manuscript.

## 7. Timestamp (complete before scouting any new date)

- Archive this file together with the `_final` notebook on **OSF Registries** or **Zenodo** (both
  free; both issue a public timestamp and DOI), or at minimum a tagged GitHub release.
- Frozen code (verified by NB08 at every run):
  - `nb07_core.py` — SHA-256 `0c566d8886d499a13e34a3cb76cb86cc6671c233c27ba03dc2429208cf1027ce`
  - `nb08_confirm.py` — SHA-256 `aa29ef6915693a68bb5448ef65790587ab29dad09271fcfd9ae0a1624f89a64a`
- Upload this protocol, both `.py` files, `NB08_confirmatory.ipynb` and
  `NB07_MVP2_Phase2_workflow_final.ipynb` in one OSF/Zenodo deposit **before** running NB08 section 2.
- **Scouting record (metadata only).** NB08 §2 ran on 2026-09-23 at 18:17 (Colab time): 449 S2 dates,
  40 CBERS-4A/WPM dates, 20 queued. No pixel values, FDI values or AUCs were read. E6 was added after
  scouting revealed S2 dates sharing a CBERS pass and **before any date was processed**; it depends only on
  scene identity and day gap, never on data values. 2024-02-10 is in the queue: it was scouted but never
  analysed in the exploratory phase (its CBERS scene failed coverage), so it is processed under E1–E6 like
  any other date.
- Record here: registration DOI/URL ______ ; date/time (UTC) ______ .
