# Floating-debris FDI anomalies — Ilha Grande Bay, Brazil (Sentinel-2, CBERS-4A, GCOM-C/SGLI)
Code, protocol, logs and results for the exploratory (NB07) and pre-registered confirmatory (NB08) analyses.
Outputs describe *floating marine-debris/macroplastic aggregation candidates*; no material identification and no ground truth.

- `code/` frozen analysis modules (SHA-256 in `data/manifest.json`; NB08 refuses to run if they differ)
- `notebooks/` execution record with outputs · `protocol/` frozen protocol (deviations D1–D6 in `results/confirmatory/confirm_log.json`)
- `results/` every number reported in the manuscript · `data/manifest.json` exact public inputs (no satellite data in this repo)

**Run:** Google Colab; set `BASE` to your Drive folder; authenticate Earth Engine; download the SGLI files listed in the manifest from G-Portal.
**Licence:** code MIT; results and documentation CC BY 4.0.
