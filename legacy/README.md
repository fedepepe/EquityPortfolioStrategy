# Legacy code

The flat-module version of the project, from before the `equity_strategy` package existed, plus the
one-off thesis and experiment scripts built on it. Everything here is kept for reference and is **not maintained**:

- `main.py`, `portfolio*.py`, `stock_picker.py`, `compute_sev.py`, `yahoo_data_*.py`, `definitions.py`,
  `proposed_estim.py`, `symbols_string.py` were replaced by the modules in `equity_strategy/`.
- The `main_*.py` and other scripts import those old modules. Several were already broken before the
  restructuring (e.g. `main_vs_nstock.py` imports a missing `portfolio_sim` module, `test_dataclass.py`
  imports a missing `DefaultPerfMetrics`).

The old modules resolve data paths relative to the repository root, so they no longer find the data from this folder.
