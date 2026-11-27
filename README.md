# Understood but Unflagged? A Study of LLM Refusal in Low-Resource Languages
*Author: Jesse-Paul Osemeke*
*[SAIR Fellowship 2026](https://susanajoke.org)*

## Pilot: comprehension vs action (Qwen3 4B / 8B / 14B)

Question: when a small open-weight model under-refuses a harmful request in a low-resource language, did it fail to read the request (comprehension) or read it and not act (action)? Cells: `eng`, `yor`, `yor-stripped` (diacritics removed), `swh`, `hau`, `hau-stripped` (hooked letters mapped to plain letters). Readings: **C** comprehension (logit-scored, no generation), **H** harm recognition (hidden states), **R** refusal (generation + judge). Pre-registered rules: [PREREG.md](PREREG.md).

## Layout
- `src/lrr/text.py` stripping and word shuffle; `data.py` pinned dataset loaders; `probe.py` U1 / U1-twin / Belebele; `hidden.py` last-token states; `analysis_h.py` harm direction, folds, layer choice; `pipeline.py` C+H for one model; `gen.py` vLLM generation; `judge.py` judge; `flags.py` output flags and judge parsing; `analysis.py` tables, residual gap, figures; `h_inlang.py` exploratory in-language direction; `calib.py` hand-label sheet; `report.py` cross-size tables; `sens_antirep.py` decoding sensitivity (PREREG Addendum B); `summary_figs.py` the cross-size figures in `figures/` (aggregate charts only). Failure-locus checks (PREREG Addenda C-E): `lm_probe.py` + `diag_lm.py` (C3-C5), `translate.py` (C5), `diag_align.py` (C4), `diag_local.py` (C1-C2), `steer.py` (C6, C7, D2, E: ablation, steering, layer sweeps), `diag_interv.py` (C6/C7/D2/E analysis), `diag_layers.py` (D1).
- `modal_app.py` Modal entrypoints (H100 80GB). `tests/` offline tests plus data-alignment tests (download on first run).

## Run
```bash
uv sync
uv run pytest -q                      # 58 tests; tests/test_data.py downloads the pinned datasets
PYTHONPATH=src modal run modal_app.py --model Qwen/Qwen3-4B --stage ch      # C and H
PYTHONPATH=src modal run modal_app.py --model Qwen/Qwen3-4B --stage gen     # R: responses (private Volume)
PYTHONPATH=src modal run modal_app.py --model Qwen/Qwen3-4B --stage judge   # R: labels
modal volume get lrr-outputs Qwen3-4B outputs/Qwen3-4B                      # fetch (states are large; fetch files selectively)
# Swahili/Hausa: add  --tag Qwen3-4B-swh-hau --cells eng,swh,hau,hau-stripped  to the ch and gen stages
# decoding sensitivity: --stage gen --tag Qwen3-4B-antirep --cells eng,yor,swh,hau --sampling antirep
PYTHONPATH=src uv run python -m lrr.analysis outputs/Qwen3-4B
# failure-locus checks (Addenda C-E): stages lm, translate, steer, steer_benign, steer_sweep, steer_dense, then --stage judge on each tag
PYTHONPATH=src modal run modal_app.py --model Qwen/Qwen3-8B --stage steer_dense --tag Qwen3-8B-dense
PYTHONPATH=src uv run python -m lrr.diag_interv outputs outputs/diagnostics   # C6, C7, D2, E tables
PYTHONPATH=src uv run python -m lrr.diag_layers outputs outputs/diagnostics   # D1
```
Any `Qwen/Qwen3-*` id works unchanged (8B, 14B tested). Seed 20261113 everywhere; dataset revisions pinned in `data.py`; every run writes `summary.json` with versions, GPU and a code hash.

## Results
[results.md](results.md) has the full write-up (pilot, Swahili/Hausa, decoding check, failure-locus checks C1-C7, D, E). Aggregate numbers behind the later checks are in `results/diagnostics/` (JSON, numbers only); figures in `figures/`.

## Privacy and licences
`outputs/`, `data/` and `*.jsonl` are gitignored (the aggregate-only files in `results/` are the exception, checked to contain no prompt or response text and no per-item rows): raw generations and harmful prompts stay in private storage (local disk and the private Modal Volume `lrr-outputs`). TukaBench is CC-BY-NC 4.0 and derived data inherits that; JBB is MIT; Belebele is CC-BY-SA 4.0. No harmful prompt or response is sent to a hosted API: the judge is an open model run on Modal.

The `private/` folder (gitignored) holds notes that are not for the public repo.
