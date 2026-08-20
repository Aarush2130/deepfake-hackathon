# VeriChain QA Stress Test Log

| # | Test Case | Status (PASS/FAIL) | Notes / Error Details |
|---|---|---|---|
| 1 | Standard JPG/PNG Photo | PASS | Single & multi-subject face detection + cropping verified |
| 2 | Known Deepfake Photo | PASS | Isolated synthetic face artifacts flagged with high confidence |
| 3 | Multi-Face / Group Photo | PASS | Analyzes each face separately; highlights per-subject verdicts |
| 4 | Non-Face Image (Objects/Scenes) | PASS | Handled gracefully with "Inconclusive (No Face Detected)" |
| 5 | Special characters in metadata (&, #, ") | PASS | Handled in Streamlit session and PDF docket |
| 6 | Corrupted / renamed fake image (.txt -> .jpg) | PASS | Error caught gracefully via try/except in intake |
| 7 | Two consecutive uploads (no page refresh) | PASS | Stale results reset on file change |
| 8 | PDF Dossier Export & layout check | PASS | Formats chain of custody, multi-subject table, and heatmap |
