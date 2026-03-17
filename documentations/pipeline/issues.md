# Pipeline Issues Register

This document records known implementation and documentation risks found in a deep review of the current 7-stage pipeline.

## Severity key

- High: likely to cause wrong outputs or expensive debugging cycles
- Medium: operationally risky or likely to create partial/ambiguous runs
- Low: important quality/documentation debt that should be scheduled

## High severity

### PL-ISSUE-001: Band normalization constants duplicated across stages

- Severity: High
- Evidence:
  - `src/pipeline/02_preprocess.py` (`BAND_MEANS`, `BAND_STDS`)
  - `src/pipeline/04_polymer.py` (`BAND_MEANS`, `BAND_STDS`)
  - `src/config/config.yaml` (`preprocessing.band_means`, `preprocessing.band_stds`)
- Impact: Stage 4 denormalization/classification can drift if constants are edited in only one location.
- Workaround: Update all three locations together before running.
- Recommended fix: Move shared band stats to one source module and assert consistency at startup.

### PL-ISSUE-002: Weak checkpoint schema validation in detection stage

- Severity: High
- Evidence:
  - `src/pipeline/03_detect.py` (`load_model`)
- Impact: Incorrect or legacy checkpoint files may fail late during inference with shape/key errors.
- Workaround: Manually verify checkpoint keys (`model_state`, encoder, classes, bands) before run.
- Recommended fix: Add strict schema and compatibility checks in `load_model` and fail fast.

## Medium severity

### PL-ISSUE-003: Fallback external data behavior is under-documented

- Severity: Medium
- Evidence:
  - `src/pipeline/05_backtrack.py` (synthetic currents/wind fallback path)
  - `src/pipeline/06_attribute.py` (heuristic fallback scoring)
  - `documentations/pipeline/stage-05-backtrack.md`
  - `documentations/pipeline/stage-06-attribute.md`
- Impact: Users may treat results as fully data-backed when synthetic or heuristic fallbacks were used.
- Workaround: Inspect logs and attribution output context for fallback clues.
- Recommended fix: Add explicit flags in outputs and stronger warning text in docs.

### PL-ISSUE-004: Exception context can be flattened by orchestration logging

- Severity: Medium
- Evidence:
  - `src/pipeline/run_pipeline.py` (stage failure handling)
- Impact: Root causes can be hard to trace from aggregated run errors.
- Workaround: Re-run failed stage directly to capture full traceback.
- Recommended fix: Preserve chained exceptions and include structured error code per failure.

### PL-ISSUE-005: GeoJSON schema assumptions are implicit across stages

- Severity: Medium
- Evidence:
  - `src/pipeline/03_detect.py` (detection geojson emit)
  - `src/pipeline/04_polymer.py` (schema-dependent reads/writes)
  - `src/pipeline/07_report.py` (expects downstream fields)
- Impact: Malformed or sparse GeoJSON causes downstream errors or silent omissions.
- Workaround: Validate GeoJSON columns and geometry before Stage 4 and Stage 7.
- Recommended fix: Define and enforce explicit schema contracts per stage.

### PL-ISSUE-006: CRS assumptions are fragile without strict validation

- Severity: Medium
- Evidence:
  - `src/pipeline/02_preprocess.py` (scene metadata and transforms)
  - `src/pipeline/03_detect.py` (polygon conversion and CRS assignment)
  - `src/pipeline/04_polymer.py` and `src/pipeline/07_report.py` (downstream CRS expectations)
- Impact: Geometries can be mislocated if CRS metadata is inconsistent or legacy artifacts are reused.
- Workaround: Regenerate invalid cached outputs and verify geographic bounds.
- Recommended fix: Add CRS assertion checks at stage boundaries and reject invalid coordinate ranges.

### PL-ISSUE-007: Stage input validation is inconsistent

- Severity: Medium
- Evidence:
  - `src/pipeline/02_preprocess.py` (input band presence assumptions)
  - `src/pipeline/03_detect.py` (patch/index assumptions)
  - `src/pipeline/04_polymer.py` (detections availability assumptions)
- Impact: Misconfigured runs fail late instead of failing fast with clear diagnostics.
- Workaround: Manually pre-check expected files before starting a run.
- Recommended fix: Add explicit entry validation checklist to every stage `run()`.

### PL-ISSUE-008: Multi-scene partial-failure semantics are easy to misread

- Severity: Medium
- Evidence:
  - `src/pipeline/run_pipeline.py` (per-scene continuation behavior)
- Impact: One scene may fail earlier stages while other scenes complete, yielding mixed output completeness.
- Workaround: Always inspect run summary before consuming outputs.
- Recommended fix: Document scene-level completion matrix and expose it in API/UI responses.

### PL-ISSUE-009: Cache checks are presence-based, not validity-based

- Severity: Medium
- Evidence:
  - `src/pipeline/utils/cache_utils.py` (`stage_output_exists`)
- Impact: Corrupted or stale artifacts can be reused silently.
- Workaround: Delete per-stage output directories to force recompute.
- Recommended fix: Add freshness/integrity checks (mtime relationships or checksums + schema parse).

## Low severity

### PL-ISSUE-010: Config precedence and required keys are not fully explicit

- Severity: Low
- Evidence:
  - `src/config/config.yaml`
  - `src/pipeline/run_pipeline.py` argument/config interactions
- Impact: Operators may edit the wrong layer and get unexpected runtime behavior.
- Workaround: Standardize operating procedure: request args override config values.
- Recommended fix: Publish a formal config reference table with required/optional fields.

### PL-ISSUE-011: Edge-case test coverage gaps remain

- Severity: Low
- Evidence:
  - `tests/test_stage_1.py` to `tests/test_stage_7.py` (limited edge-case coverage)
- Impact: Regressions in empty/no-data/failure-path scenarios can slip through.
- Workaround: Add manual smoke tests for no-detections and fallback-only runs.
- Recommended fix: Add focused edge-case tests and one lightweight end-to-end integration test.

### PL-ISSUE-012: Deprecation warnings globally suppressed in backtracking stage

- Severity: Low
- Evidence:
  - `src/pipeline/05_backtrack.py` (broad deprecation warning suppression)
- Impact: Future compatibility issues may be hidden until hard failures happen.
- Workaround: Temporarily enable warnings during validation runs.
- Recommended fix: Suppress only known noisy warnings, not entire categories.

### PL-ISSUE-013: Caching and cleanup behavior lacks operational guidance

- Severity: Low
- Evidence:
  - `src/pipeline/run_pipeline.py` (`cleanup_patches` behavior)
  - `src/pipeline/utils/cache_utils.py`
- Impact: Teams may retain stale intermediates or delete useful artifacts inadvertently.
- Workaround: Use stage-specific cleanup and rerun procedures in runbooks.
- Recommended fix: Add explicit cache lifecycle and cleanup guidance in pipeline docs.

## Quick operational checklist

- [ ] Verify model checkpoint schema before Stage 3
- [ ] Verify band stats consistency before Stage 2/4 changes
- [ ] Verify CRS sanity of detection outputs before report generation
- [ ] Check run summary for scene-level failures before consuming reports
- [ ] Force stage rerun when inputs or thresholds changed
- [ ] Track fallback usage as data-quality metadata in outputs
