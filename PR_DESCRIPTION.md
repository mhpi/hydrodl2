## Issue Addressed

Introduces a shared `BasePhysicsModel` interface for HBV models, deduplicating repeated code and adding new opt-in Hbv_2 features, plus fixes for regressions the refactor introduced.

## Description

- Added `hydrodl2.models.base.BasePhysicsModel` and shared helpers in `core/calc/utils.py` (e.g. `trim_warmup`); `hbv.py`, `hbv_1_1p.py`, `hbv_2.py`, `hbv_2_hourly.py` now subclass it, removing large amounts of duplicated boilerplate across the four models.
- `hbv_2.py`: added opt-in flags `elev_parTT` (elevation-based parTT override), `gage_agg` (in-loop gage-level aggregation), `all_output`, and an optional `routing_state` carryover input.
- `hbv_2_hourly.py`: same `BasePhysicsModel`/`trim_warmup` refactor; removed dead `self.initialize` branch.
- `api/methods.py`: `load_model()` now also tries a PascalCase name conversion (e.g. `hbv_adj` -> `HbvAdj`) before falling back to the first class found in the module.
- Fixed two regressions surfaced by the refactor, verified against `hydrodl2_master` and against `dmg`'s test suite:
  - `load_model()`'s fallback could resolve to the wrong class (e.g. `BasePhysicsModel` itself, or an unrelated imported Hbv variant) once modules started importing `BasePhysicsModel`/other classes. Fixed by adding a case-insensitive name-match step and restricting the last-resort fallback to classes actually defined in the target module.
  - `elev_parTT` default was `False`, silently dropping the original model's always-on elevation-based parTT override with no config anywhere setting the key. Reverted default to `True`.
- `hbv_2.py`/`hbv_2_hourly.py`: `SLZ` lateral-flow clamp floor raised from `min=0.0` to `min=self.nearzero` — `min=0.0` lets `SLZ` collapse to exactly 0 for some parameter draws, permanently zeroing capillary rise and, in one observed case (dmg's test config), producing an all-zero/dead streamflow output.
- Added regression tests to `tests/test_model_structure.py`: `load_model()` identity checks for all loadable HBV variants, and an explicit default-flag check for Hbv_2 (`elev_parTT`/`gage_agg`/`all_output`) so a future default flip fails immediately instead of surfacing only as a numeric drift.

## Type of Change

- [x] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] Code cleanup/refactor
- [ ] Documentation update

Other (please specify):

## Checklist

- [x] Branch is up to date with master
- [x] Updated tests or added new tests
- [x] Tests pass locally (117 passed, 1 skipped in this repo; 166 passed, 10 skipped in dmg's suite against this branch)
- [ ] Updated documentation (if applicable)
- [x] Code follows established style and conventions
