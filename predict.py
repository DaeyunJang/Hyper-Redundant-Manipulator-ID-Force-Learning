"""Predict from CSV files with saved models and export comparisons, summaries and model sheets.

Test workflow: select_and_evaluate. Prediction computation: predict_trials.
Excel columns and sheets: _branch_export_frame / ModelWorkbook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from data_utils import (
    ROOT, DEFAULT_REGIONS, ensure_driver_runtime, input_columns, csv_paths,
    load_trial, load_prediction_trials, trial_endpoints,
    force_metrics, location_confusion, location_metrics,
    location_predictions, optimize_regions,
)

if __name__ == "__main__":
    ensure_driver_runtime()

import numpy as np
import pandas as pd
import torch
import xlsxwriter


def _engine():
    # Import lazily: train.py calls the saved-validation exporter after training.
    import train
    return train


def restore_bundle(*args, **kwargs):
    return _engine().restore_bundle(*args, **kwargs)


def write_json(*args, **kwargs):
    return _engine().write_json(*args, **kwargs)


def fresh_directory(*args, **kwargs):
    return _engine().fresh_directory(*args, **kwargs)


DEFAULT_FORCE_TOLERANCE_MN = 1e-3


# AUTHOR: Daeyun
# Restore saved data/model contracts and align predictions with source rows.
def _path(value: str | Path, bundle: dict) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(bundle["config"].get("project_root", ROOT)) / path).resolve()


def _known_trials(bundle: dict) -> list[dict]:
    return bundle.get("data_audit", {}).get("trials", [])


def _tip_verified(trial: Any, bundle: dict) -> bool:
    """An ID18 filename alone cannot certify a tip-force evaluation trial."""
    if bundle.get("force_scope", bundle["config"].get("force_scope", "tip")) != "tip":
        return True
    return any(info.get("sha256") == trial.sha256 and info.get("is_tip") for info in _known_trials(bundle))


def _warn_prediction_overlap(trials: list, bundle: dict) -> None:
    """Optionally flag overlap in saved metadata without changing inputs or predictions."""
    known = _known_trials(bundle)
    indices = set(bundle.get('split', {}).get('train', [])) | set(bundle.get('split', {}).get('validation', []))
    references = [('train/validation', known[i]) for i in sorted(indices)
                  if isinstance(i, int) and 0 <= i < len(known)]
    protected = set(bundle['config'].get('protected_development_hashes', []))
    for trial in trials:
        matches = []
        origin = trial.info.get('timestamp_origin_ns')
        for role, info in references:
            previous = info.get('timestamp_origin_ns')
            if trial.sha256 == info.get('sha256'):
                reason = 'identical file contents'
            elif (isinstance(origin, int) and origin > 0 and isinstance(previous, int) and previous > 0
                  and info.get('rows') == len(trial.x) and abs(previous - origin) <= 20000):
                reason = 'possible overlap: matching start timestamp and valid-row count'
            else:
                continue
            matches.append(f"{role}: {info.get('path', '(no saved path)')} ({reason})")
        if trial.sha256 in protected and not matches:
            matches.append('matches a previous training/validation file hash')
        if matches:
            # Write to stderr so PYTHONWARNINGS=error does not stop prediction.
            print(f"[WARNING] Data overlap check: {trial.path}\n  " + '\n  '.join(matches)
                  + '\n  Continuing prediction without excluding the file.', file=sys.stderr, flush=True)
        references.append(('prediction input', trial.info))


def _initial_frame(trial: Any, bundle: dict, checkpoint: Path) -> pd.DataFrame:
    size = len(trial.x)
    has_id = trial.info.get("has_location_target", True)
    scope_verified = _tip_verified(trial, bundle)
    states = np.asarray(trial.load_state)
    state_names = {-2: "invalid", -1: "uncertain", 0: "unloaded", 1: "loaded"}
    frame = pd.DataFrame({
        "source_file": trial.path,
        "source_sha256": trial.sha256,
        "source_row": trial.source_rows if getattr(trial, 'source_rows', None) is not None else np.arange(size),
        "source_time_ns": np.asarray(trial.timestamps, dtype=np.int64) + int(trial.info.get("timestamp_origin_ns", 0)),
        "elapsed_s": np.asarray(trial.timestamps) / 1e9,
        "mode": bundle["mode"],
        "force_model": bundle["force_name"],
        "location_model": bundle["location_name"],
        "bundle": str(checkpoint),
        "actual_id_raw": trial.segment_id_raw if has_id else np.nan,
        "actual_id_effective": trial.effective_id if has_id else np.nan,
        "load_state_gt": [state_names[int(value)] for value in states],
        "input_valid": trial.input_valid,
        "force_target_valid": trial.force_valid,
        "force_scope": bundle.get("force_scope", bundle["config"].get("force_scope", "tip")),
        "force_scope_verified": scope_verified,
        "force_ready": False,
        "location_ready": False,
        "force_status": np.where(trial.input_valid, "warming_up", "invalid_input"),
        "location_status": np.where(trial.input_valid, "warming_up", "invalid_input"),
        "location_decision_rule": "class_argmax" if bundle["mode"] == "class0" else "unavailable",
        "deployment_gate_validated": False,
        "evaluation_scope": bundle['config'].get('test_evaluation_scope', 'historical_held_out_recordings'),
    })
    if not trial.info.get('timestamps_available', True):
        frame[['source_time_ns', 'elapsed_s']] = np.nan
    columns = {}
    for axis, name in enumerate(("fx", "fy", "fz")):
        columns[f"true_{name}_N"] = trial.force[:, axis]
        columns[f"true_{name}_mN"] = trial.force[:, axis] * 1000.0
        columns[f"pred_{name}_N"] = np.full(size, np.nan)
        columns[f"pred_{name}_mN"] = np.full(size, np.nan)
    for segment in range(19):
        columns[f"p{segment}"] = np.full(size, np.nan)
    columns["conditional_id"] = np.full(size, np.nan)
    columns["final_id"] = np.full(size, np.nan)
    return pd.concat([frame, pd.DataFrame(columns)], axis=1)


def _add_regions(frame: pd.DataFrame, ready: np.ndarray, probabilities: np.ndarray,
                 bundle: dict, regions: dict, prefix: str) -> None:
    active = bundle["config"].get("active_ids", list(range(1, 19)))
    decoded = location_predictions(probabilities, bundle["mode"], active, regions)
    additions: dict[str, Any] = {}
    for index, name in enumerate(decoded["region_names"]):
        values = np.full(len(frame), np.nan)
        values[ready] = decoded["region_probabilities"][:, index]
        additions[f"{prefix}_region_{name}_probability"] = values
    for key in ("region_candidate", "region_final"):
        additions[f"{prefix}_{key}"] = np.full(len(frame), None, dtype=object)
    additions[f"{prefix}_region_id_candidate"] = np.full(len(frame), np.nan)
    names = np.asarray(decoded["region_names"], dtype=object)
    chosen = decoded["conditional_region_index"]
    additions[f"{prefix}_region_candidate"][ready] = names[chosen]
    if bundle["mode"] == "class0":
        additions[f"{prefix}_region_final"][ready] = np.where(decoded["final_id"] == 0, "unloaded", names[chosen])
    segment = probabilities[:, int(bundle["mode"] == "class0"):]
    active_array = np.asarray(active)
    within_id = np.full(len(ready), np.nan)
    for index, name in enumerate(names):
        rows = chosen == index
        supported = np.isin(active_array, regions[name])
        if supported.any():
            within_id[rows] = active_array[supported][segment[rows][:, supported].argmax(axis=1)]
    additions[f"{prefix}_region_id_candidate"][ready] = within_id
    for name, values in additions.items():
        frame[name] = values


def predict_trials(model: Any, bundle: dict, trials: list, checkpoint: str | Path,
                   optimized_regions: dict | None = None, device: str = "cpu") -> pd.DataFrame:
    """Predict loaded trial histories with original row mapping and raw force output."""
    cfg = bundle["config"]
    mode = bundle["mode"]
    checkpoint = Path(checkpoint).resolve()
    expected = bundle.get("feature_columns")
    if expected is not None and list(expected) != input_columns(cfg):
        raise ValueError("Current feature mapping differs from the saved bundle input order")
    frames = [_initial_frame(trial, bundle, checkpoint) for trial in trials]
    indices = [index for index, trial in enumerate(trials) if len(trial.predict_endpoints)]
    for branch in ("force", "location"):
        if not indices:
            continue
        scaler = {name: np.asarray(values) for name, values in bundle["scalers"][branch].items()}
        store = _engine().WindowStore(trials, indices, scaler, cfg, device, branch, mode, prediction=True)
        network = model.force_net if branch == "force" else model.location_net
        predicted = _engine().predict_branch(network, store, int(cfg["batch_size"]))
        refs = np.asarray(store.refs, dtype=int)
        for index in indices:
            rows = refs[:, 0] == index
            source_rows = refs[rows, 1]
            values = predicted[rows]
            frame = frames[index]
            frame.loc[source_rows, f"{branch}_ready"] = True
            if branch == "force":
                for axis, name in enumerate(("fx", "fy", "fz")):
                    frame.loc[source_rows, f"pred_{name}_N"] = values[:, axis]
                    frame.loc[source_rows, f"pred_{name}_mN"] = values[:, axis] * 1000.0
                status = "ready" if _tip_verified(trials[index], bundle) else "scope_unverified"
                frame.loc[source_rows, "force_status"] = status
            else:
                active = cfg.get("active_ids", list(range(1, 19)))
                decoded = location_predictions(values, mode, active, cfg.get("regions"))
                frame.loc[source_rows, "p0"] = decoded["no_load_probability"]
                frame.loc[source_rows, [f"p{segment}" for segment in range(1, 19)]] = decoded["segment_probabilities"]
                frame.loc[source_rows, "conditional_id"] = decoded["conditional_id"]
                frame.loc[source_rows, "final_id"] = decoded["final_id"]
                status = np.where(decoded["final_id"] == 0, "unloaded", "valid") if mode == "class0" else "unverified"
                frame.loc[source_rows, "location_status"] = status
    for frame in frames:
        ready = np.flatnonzero(frame["location_ready"].to_numpy())
        classes = ([0] if mode == "class0" else []) + list(cfg.get("active_ids", range(1, 19)))
        probabilities = frame.loc[ready, [f"p{segment}" for segment in classes]].to_numpy(dtype=float)
        _add_regions(frame, ready, probabilities, bundle, cfg.get("regions", DEFAULT_REGIONS), "baseline")
        if optimized_regions is not None:
            _add_regions(frame, ready, probabilities, bundle, optimized_regions, "optimized")
    if not frames:
        raise ValueError("No CSV trials were supplied for prediction")
    return pd.concat(frames, ignore_index=True)


def evaluate_frame(frame: pd.DataFrame, bundle: dict, regions: dict | None = None) -> dict:
    """Evaluate scope-verified force and conditional/detection/system location separately."""
    force_valid = (frame["force_ready"] & frame["force_target_valid"] & frame["force_scope_verified"]).to_numpy()
    prediction = frame[[f"pred_{axis}_N" for axis in ("fx", "fy", "fz")]].to_numpy(dtype=float)
    target = frame[[f"true_{axis}_N" for axis in ("fx", "fy", "fz")]].to_numpy(dtype=float)
    metrics = force_metrics(prediction, target, force_valid)
    states = frame["load_state_gt"].to_numpy()
    for state in ("loaded", "unloaded"):
        metrics.update({f"{state}_{key}": value for key, value in force_metrics(prediction, target, force_valid & (states == state)).items()})
    active = bundle["config"].get("active_ids", list(range(1, 19)))
    classes = ([0] if bundle["mode"] == "class0" else []) + list(active)
    ready = frame["location_ready"].to_numpy()
    probability = frame.loc[ready, [f"p{segment}" for segment in classes]].to_numpy(dtype=float)
    ids = frame.loc[ready, "actual_id_effective"].to_numpy(dtype=float)
    selected_states = states[ready]
    metrics.update(location_metrics(probability, ids, selected_states, bundle["mode"], active, regions))
    metrics.update({
        "source_rows": len(frame),
        "force_prediction_rows": int(frame["force_ready"].sum()),
        "location_prediction_rows": int(ready.sum()),
        "force_scope_unverified_prediction_rows": int((frame["force_ready"] & ~frame["force_scope_verified"]).sum()),
    })
    return metrics


def prediction_report(frame: pd.DataFrame, bundle: dict, optimized_regions: dict | None) -> dict:
    baseline = bundle["config"].get("regions", DEFAULT_REGIONS)
    active = bundle["config"].get("active_ids", list(range(1, 19)))
    classes = ([0] if bundle["mode"] == "class0" else []) + list(active)
    ready = frame["location_ready"].to_numpy()
    probability = frame.loc[ready, [f"p{segment}" for segment in classes]].to_numpy(dtype=float)
    actual = frame.loc[ready, "actual_id_effective"].to_numpy(dtype=float)
    states = frame.loc[ready, "load_state_gt"].to_numpy()
    report = {
        "overall": evaluate_frame(frame, bundle, baseline),
        "files": {str(path): evaluate_frame(group, bundle, baseline) for path, group in frame.groupby("source_file", sort=False)},
        "baseline_regions": baseline,
        "optimized_regions": optimized_regions,
        "conditional_confusion": location_confusion(probability, actual, states, bundle["mode"], active),
        "system_confusion": location_confusion(probability, actual, states, bundle["mode"], active, system=True),
        "limitations": [
            "Raw force estimates are emitted outside the verified force scope; force metrics exclude those rows.",
            "Class0 uses the declared class-argmax decision rule; deployment gate calibration is not established by replay.",
            "Masked outputs are conditional location diagnostics without a deployment load detector.",
            "A test set containing only ID18 does not independently validate body-wide location accuracy or the three-region boundaries.",
            "Region boundaries are frozen from validation; validation search scores are not independent test results.",
        ],
    }
    if optimized_regions is not None:
        report["optimized_overall"] = evaluate_frame(frame, bundle, optimized_regions)
        report["optimized_files"] = {str(path): evaluate_frame(group, bundle, optimized_regions) for path, group in frame.groupby("source_file", sort=False)}
    return report


# Select the best combination using validation results only.
def select_comparison_rows(comparison: pd.DataFrame, force_tolerance_mN: float = DEFAULT_FORCE_TOLERANCE_MN,
                           location_metric: str = 'location_macro_recall') -> list[tuple[int, pd.Series]]:
    """Select one existing pair per mode using validation columns only.

    The default 1e-3 mN (1 micro-newton) window absorbs CUDA rounding drift
    between independent same-seed force runs. It is a numerical tie policy,
    not a clinically meaningful performance margin or statistical equivalence.
    """
    if location_metric not in ('location_macro_recall', 'load_balanced_accuracy'):
        raise ValueError('Location selection must use a supported validation metric')
    if location_metric == 'load_balanced_accuracy' and set(comparison['mode']) != {'class0'}:
        raise ValueError('Load-balanced selection requires class0-only comparison rows')
    required = {"mode", "force_rmse_xyz_mN", location_metric, "force_model", "location_model", "bundle"}
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"Missing validation comparison fields: {sorted(missing)}")
    if force_tolerance_mN < 0 or not np.isfinite(force_tolerance_mN):
        raise ValueError("force_tolerance_mN must be finite and nonnegative")
    valid = comparison.copy()
    if "status" in valid:
        valid = valid.loc[valid["status"] == "complete"]
    valid = valid.loc[np.isfinite(valid["force_rmse_xyz_mN"]) & np.isfinite(valid[location_metric])]
    if valid.empty:
        raise ValueError("No complete comparison row has finite validation force and location metrics")
    selected = []
    for mode, rows in valid.groupby("mode", sort=True):
        if mode not in ("masked", "class0"):
            raise ValueError(f"Unsupported location mode in comparison: {mode}")
        minimum = rows["force_rmse_xyz_mN"].min()
        candidates = rows.loc[rows["force_rmse_xyz_mN"] <= minimum + force_tolerance_mN].copy()
        order, ascending = [location_metric], [False]
        if "location_validation_ce" in candidates:
            order.append("location_validation_ce")
            ascending.append(True)
        order.extend(["force_model", "location_model", "bundle"])
        ascending.extend([True, True, True])
        chosen = candidates.sort_values(order, ascending=ascending, kind="mergesort", na_position="last").iloc[0]
        selected.append((chosen.name, chosen))
    return selected


REPRESENTATIVE_POLICY = (
    "For each mode, force sheets use the first configured location partner and "
    "ID sheets use the first configured force partner. All completed combinations "
    "are compared; sheets show these fixed representative branches. No test-based selection."
)
AXES = ("fx", "fy", "fz")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(train_dir: Path, row: pd.Series) -> Path:
    value = Path(row["bundle"])
    candidates = [train_dir / "models" / (
        f"{row['mode']}__force_{row['force_model']}__location_{row['location_model']}"
    ) / "hrm_bundle.pt"]
    candidates.extend([value] if value.is_absolute() else [ROOT / value, train_dir / value])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing completed bundle: {value}")


def _completed_comparison(train_dir: Path) -> pd.DataFrame:
    comparison = pd.read_csv(train_dir / "comparison.csv")
    required = {"mode", "force_model", "location_model", "bundle"}
    if missing := required - set(comparison):
        raise ValueError(f"Comparison is missing columns: {sorted(missing)}")
    if "status" in comparison:
        comparison = comparison.loc[comparison["status"] == "complete"].copy()
    if comparison.empty:
        raise ValueError("No completed model combinations")
    if comparison.duplicated(["mode", "force_model", "location_model"]).any():
        raise ValueError("Duplicate completed model combinations")
    if not set(comparison["mode"]).issubset({"masked", "class0"}):
        raise ValueError("Unsupported location mode in comparison")
    return comparison


def _assert_bundle_row(bundle: dict, row: pd.Series) -> None:
    if bundle.get("kind") != "independent_hrm":
        raise ValueError("Expected an independent HRM bundle")
    if bundle["config"].get("preprocessing", "audited") != "numeric_compact":
        raise ValueError("This bundle uses historical preprocessing; use the archived implementation to preserve its saved policy")
    bundle["config"].setdefault("tip_scope_by_id", False)
    if (bundle["mode"], bundle["force_name"], bundle["location_name"]) != (
        row["mode"], row["force_model"], row["location_model"]
    ):
        raise ValueError("Comparison row does not match its saved bundle")
    if list(bundle.get("feature_columns") or input_columns(bundle["config"])) != input_columns(bundle["config"]):
        raise ValueError("Saved input columns differ from the bundle configuration")


def _contract(bundle: dict) -> str:
    """Require the same data, partitions and semantics before sharing loaded rows."""
    return json.dumps({key: bundle.get(key) for key in (
        "config", "feature_columns", "calibration", "split", "data_audit",
        "force_scope", "force_unit", "training_sign_multiplier",
    )}, sort_keys=True, ensure_ascii=False)


def representative_comparison_rows(comparison: pd.DataFrame, config: dict) -> list[dict]:
    """Choose fixed partners from configuration order, never a measured score."""
    forces = list(config.get("force_candidates") or comparison["force_model"].unique())
    locations = list(config.get("location_candidates") or comparison["location_model"].unique())
    modes = list(config.get("no_load_modes") or comparison["mode"].unique())
    result = []
    for mode in modes:
        rows = comparison.loc[comparison["mode"] == mode]
        if rows.empty:
            continue
        for branch, names, partner in (("force", forces, locations[0]), ("location", locations, forces[0])):
            for name in names:
                own_column = "force_model" if branch == "force" else "location_model"
                other_column = "location_model" if branch == "force" else "force_model"
                if name not in set(rows[own_column]):
                    continue
                matching = rows.loc[(rows[own_column] == name) & (rows[other_column] == partner)]
                if len(matching) != 1:
                    raise ValueError(f"Missing fixed-partner bundle for {mode}/{branch}/{name}, partner={partner}")
                row = matching.iloc[0]
                result.append({"comparison_row": row.name, "branch": branch, "mode": mode,
                               "model": name, "fixed_partner": partner, "row": row})
    return result


# Stream Excel rows and collect model, provenance and comparison sheets in one workbook.
class ModelWorkbook:
    """Small row-major XLSX writer; cells with missing values remain blank."""

    def __init__(self, path: str | Path, comparison: pd.DataFrame):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("xb")
        self.book = xlsxwriter.Workbook(self._handle, {
            "constant_memory": True, "strings_to_formulas": False,
            "strings_to_urls": False, "strings_to_numbers": False,
        })
        self.book.use_zip64()
        self.header = self.book.add_format({"bold": True, "bg_color": "#E5EDF5"})
        self.names: set[str] = set()
        self.models: list[dict] = []
        self.closed = False
        self._comparison = comparison.copy()

    @staticmethod
    def _cell(value):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, float):
            if math.isnan(value):
                return None
            if not math.isfinite(value):
                raise ValueError("Infinity cannot be exported to Excel")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2 ** 53:
            value = str(value)
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str) and len(value) > 32767:
            raise ValueError("Excel cell text exceeds 32767 characters")
        return value

    def _write(self, requested_name: str, frame: pd.DataFrame, *, model: bool = False) -> str:
        if self.closed:
            raise ValueError("Workbook is already closed")
        if len(frame) + 1 > 1_048_576 or not 0 < len(frame.columns) <= 16384:
            raise ValueError("Table exceeds Excel worksheet dimensions")
        if not frame.columns.is_unique:
            raise ValueError("Duplicate Excel table columns")
        base = re.sub(r"[\[\]:*?/\\]", "_", requested_name).strip("'")[:31] or "model"
        name, suffix = base, 1
        while name.casefold() in self.names:
            suffix += 1
            ending = f"_{suffix}"
            name = base[:31 - len(ending)] + ending
        self.names.add(name.casefold())
        sheet = self.book.add_worksheet(name)
        sheet.freeze_panes(1, 0)
        sheet.set_column(0, len(frame.columns) - 1, 19)
        sheet.write_row(0, 0, [self._cell(str(column)) for column in frame.columns], self.header)
        for row_number, values in enumerate(frame.itertuples(index=False, name=None), 1):
            sheet.write_row(row_number, 0, [self._cell(value) for value in values])
        sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
        if model and not self.models:
            sheet.activate()
            sheet.set_first_sheet()
        return name

    def add_model(self, frame: pd.DataFrame, metadata: dict) -> dict:
        prefix = "force" if metadata["branch"] == "force" else "id"
        sheet = self._write(f"{prefix}_{metadata['mode']}_{metadata['model']}", frame, model=True)
        entry = {**metadata, "sheet": sheet, "rows": len(frame), "columns": list(frame.columns)}
        self.models.append(entry)
        return entry

    def close(self, comparison: pd.DataFrame | None = None) -> None:
        if self.closed:
            return
        try:
            self._write("_comparison", self._comparison if comparison is None else comparison)
            self._write("_models", pd.DataFrame(self.models) if self.models else pd.DataFrame(columns=["model", "sheet"]))
        finally:
            try:
                self.book.close()
            finally:
                self._handle.close()
                self.closed = True

    def abort(self) -> None:
        if not self.closed:
            try:
                self.book.close()
            finally:
                self._handle.close()
                self.closed = True
        self.path.unlink(missing_ok=True)


def _sheet_metadata(job: dict, checkpoint: Path, bundle: dict, role: str) -> dict:
    return {key: job[key] for key in ("branch", "mode", "model", "fixed_partner")} | {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
        "dataset_role": role, "representative_policy": REPRESENTATIVE_POLICY,
        "source_row_basis": "zero-based original CSV data row, excluding header; not a compact-array index",
        "force_unit": "mN", "force_scope": bundle.get("force_scope", bundle["config"].get("force_scope", "tip")),
        "force_transform": "inverse target scaling, N to mN; recorded sign, no load-based zeroing",
        "id_semantics": "conditional argmax; final_id unavailable; never GT-gated" if bundle["mode"] == "masked"
                        else "argmax including learned class0; 0 denotes predicted no-load",
        "input_columns": input_columns(bundle["config"]),
        "active_ids": list(bundle["config"].get("active_ids", range(1, 19))),
        "probability_columns": "p0..p18; inactive classes and masked p0 are blank, not zero",
        "window_samples": bundle["config"]["window_samples"],
        "preprocessing": bundle["config"].get("preprocessing", "audited"),
        "validation_strategy": bundle["config"].get("validation_strategy", "whole_csv"),
        "continuous_block": "new source file, removed original rows, or nonincreasing/large source-time gap",
    }


def _base_export_frame(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    states = frame["load_state_gt"].to_numpy()
    result = pd.DataFrame({
        "source_file": frame["source_file"].to_numpy(),
        "source_row": frame["source_row"].to_numpy(),
        "elapsed_s": frame["elapsed_s"].to_numpy(),
        "load_state": states,
        "loaded_gt": states == "loaded",
        "contact_id_original": frame["actual_id_raw"].to_numpy(),
        "contact_id_effective": frame["actual_id_effective"].to_numpy(),
        "ground_truth_id": np.where(states == "loaded", frame["actual_id_effective"],
                                     np.where(states == "unloaded", 0, np.nan)),
    })
    begins = np.ones(len(result), dtype=bool)
    if len(result) > 1:
        times = result["elapsed_s"].to_numpy(dtype=float)
        delta = np.diff(times)
        same_file = result["source_file"].to_numpy()[1:] == result["source_file"].to_numpy()[:-1]
        positive = delta[same_file & np.isfinite(delta) & (delta > 0)]
        limit = np.median(positive) * float(bundle["config"].get("gap_factor", 1.5)) if len(positive) else np.inf
        begins[1:] = (~same_file | (np.diff(result["source_row"].to_numpy()) != 1)
                      | (np.isfinite(delta) & ((delta <= 0) | (delta > limit))))
    result.insert(3, "continuous_block", np.cumsum(begins, dtype=np.int64))
    return result


def _branch_export_frame(frame: pd.DataFrame, bundle: dict, branch: str) -> pd.DataFrame:
    if branch == "force":
        keep = frame["force_ready"] & frame["force_target_valid"] & frame["force_scope_verified"]
    else:
        keep = frame["location_ready"] & frame["force_target_valid"] & frame["load_state_gt"].isin(["loaded", "unloaded"])
    frame = frame.loc[keep].reset_index(drop=True)
    table = _base_export_frame(frame, bundle)
    if branch == "force":
        for axis in AXES:
            table[f"gt_{axis}_mN"] = frame[f"true_{axis}_N"].to_numpy(dtype=float) * 1000.0
            table[f"pred_{axis}_mN"] = frame[f"pred_{axis}_N"].to_numpy(dtype=float) * 1000.0
        if not np.isfinite(table[[f"{kind}_{axis}_mN" for kind in ("gt", "pred") for axis in AXES]].to_numpy()).all():
            raise ValueError("Nonfinite force on an eligible Excel row")
        leading = [f"{kind}_{axis}_mN" for kind in ("gt", "pred") for axis in AXES]
    else:
        table["pred_id"] = frame["final_id" if bundle["mode"] == "class0" else "conditional_id"].to_numpy()
        table["conditional_id"] = frame["conditional_id"].to_numpy()
        table["final_id"] = frame["final_id"].to_numpy()
        columns = [f"p{segment}" for segment in ([0] if bundle["mode"] == "class0" else [])
                   + list(bundle["config"].get("active_ids", range(1, 19)))]
        probability = frame[columns].to_numpy(dtype=float)
        location_predictions(probability, bundle["mode"], bundle["config"].get("active_ids", list(range(1, 19))), bundle["config"].get("regions"))
        for column in [f"p{segment}" for segment in range(19)]:
            table[column] = frame[column].to_numpy()
        leading = ["ground_truth_id", "pred_id"] + [f"p{segment}" for segment in range(19)]
    return table[leading + [column for column in table.columns if column not in leading]]


def _supervised_frame(frame: pd.DataFrame, trials: list, role: str) -> pd.DataFrame:
    """Excel uses valid supervised histories; the CSV retains input-only inference."""
    keep = np.zeros(len(frame), dtype=bool)
    for trial in trials:
        endpoints = trial_endpoints(trial, role)
        mapping = trial.source_rows if trial.source_rows is not None else np.arange(len(trial.x))
        keep |= ((frame["source_file"].to_numpy() == trial.path)
                 & np.isin(frame["source_row"].to_numpy(), mapping[endpoints]))
    return frame.loc[keep]


def _validation_trials(bundle: dict) -> list:
    known = _known_trials(bundle)
    trials = [None] * len(known)
    for index in bundle["split"]["validation"]:
        info = known[index]
        trial = load_trial(_path(info["path"], bundle), bundle["config"], bundle["calibration"], require_targets=True)
        if trial.sha256 != info["sha256"] or len(trial.x) != info["rows"]:
            raise ValueError(f"Frozen validation CSV changed: {trial.path}")
        if "role_rows" in info:
            trial.info["role_rows"] = info["role_rows"]
        if "window_endpoints" in info and len(trial.endpoints) != info["window_endpoints"]:
            raise ValueError(f"Frozen validation endpoints changed: {trial.path}")
        trials[index] = trial
    return trials


def _validation_frame(trials: list, bundle: dict, checkpoint: Path, cached: Any, branch: str) -> pd.DataFrame:
    pieces = []
    for index in bundle["split"]["validation"]:
        trial = trials[index]
        if branch == "force" and bundle.get("force_scope", bundle["config"].get("force_scope", "tip")) == "tip" and not trial.is_tip:
            continue
        endpoints = np.asarray(trial_endpoints(trial, "validation"), dtype=np.int64)
        pieces.append(np.column_stack((np.full(len(endpoints), index, dtype=np.int64), endpoints)))
    refs = np.concatenate(pieces) if pieces else np.empty((0, 2), dtype=np.int64)
    cached_refs = np.asarray(cached[f"{branch}_refs"])
    if cached_refs.dtype.kind not in "iu" or not np.array_equal(cached_refs, refs):
        raise ValueError(f"Saved validation {branch} references differ from the frozen role/window contract")
    values = np.asarray(cached["force_N" if branch == "force" else "location_probabilities"])
    columns = 3 if branch == "force" else len(bundle["config"].get("active_ids", range(1, 19))) + (bundle["mode"] == "class0")
    if values.shape != (len(refs), columns) or not np.isfinite(values).all():
        raise ValueError(f"Invalid saved validation {branch} prediction shape/values")
    frames = []
    for index in np.unique(refs[:, 0]):
        mask = refs[:, 0] == index
        rows = refs[mask, 1]
        trial = trials[int(index)]
        frame = _initial_frame(trial, bundle, checkpoint).iloc[rows].copy().reset_index(drop=True)
        frame[f"{branch}_ready"] = True
        if branch == "force":
            for axis, name in enumerate(AXES):
                frame[f"pred_{name}_N"] = values[mask, axis]
        else:
            if not np.array_equal(np.asarray(cached["actual_ids"])[mask], np.full(len(rows), trial.effective_id)):
                raise ValueError("Saved validation ID targets do not match source references")
            if not np.array_equal(np.asarray(cached["load_state"])[mask], trial.load_state[rows]):
                raise ValueError("Saved validation load targets do not match source references")
            active = bundle["config"].get("active_ids", list(range(1, 19)))
            decoded = location_predictions(values[mask], bundle["mode"], active, bundle["config"].get("regions"))
            frame["conditional_id"], frame["final_id"] = decoded["conditional_id"], decoded["final_id"]
            classes = ([0] if bundle["mode"] == "class0" else []) + list(active)
            frame[[f"p{segment}" for segment in classes]] = values[mask]
        frames.append(frame)
    if not frames:
        raise ValueError(f"No eligible validation {branch} rows")
    return _branch_export_frame(pd.concat(frames, ignore_index=True), bundle, branch)


def export_validation_workbook(train_dir: str | Path) -> Path:
    """Export frozen validation NPZ predictions without model execution or fitting."""
    train_dir = Path(train_dir).resolve()
    comparison = _completed_comparison(train_dir)
    reference_path = _checkpoint_path(train_dir, comparison.iloc[0])
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    _assert_bundle_row(reference, comparison.iloc[0])
    signature = _contract(reference)
    jobs = representative_comparison_rows(comparison, reference["config"])
    trials = _validation_trials(reference)
    path = train_dir / "validation.xlsx"
    writer = ModelWorkbook(path, comparison)
    try:
        for job in jobs:
            checkpoint = _checkpoint_path(train_dir, job["row"])
            bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
            _assert_bundle_row(bundle, job["row"])
            if _contract(bundle) != signature:
                raise ValueError("Representative bundles have different frozen data contracts")
            with np.load(checkpoint.parent / "validation_predictions.npz", allow_pickle=False) as cached:
                table = _validation_frame(trials, bundle, checkpoint, cached, job["branch"])
            metadata = _sheet_metadata(job, checkpoint, bundle, "validation")
            metadata["prediction_origin"] = "saved_validation_predictions.npz; no inference or retraining"
            writer.add_model(table, metadata)
            print(f"Validation Excel: {job['mode']}/{job['branch']}/{job['model']}", flush=True)
        _verify_unchanged([trial for trial in trials if trial is not None])
        writer.close()
    except BaseException:
        writer.abort()
        raise
    write_json(train_dir / "validation_workbook_manifest.json", {
        "path": str(path), "sha256": _sha256(path), "dataset_role": "validation",
        "all_completed_combinations": len(comparison), "model_sheets": writer.models,
        "representative_policy": REPRESENTATIVE_POLICY, "new_inference": False, "new_training": False,
        "sources": [trial.info for trial in trials if trial is not None],
    })
    return path


def _test_trials(bundle: dict, check_overlap: bool = False,
                 input_dir: str | Path | None = None) -> list:
    """Read all CSV files in the current test directory using saved preprocessing and calibration."""
    directory = (Path(input_dir).resolve() if input_dir is not None else
                 _path(bundle["config"].get("test_directory", "datasets/test"), bundle))
    paths = csv_paths(directory)
    if not paths:
        raise ValueError(f"No CSV files found in test directory: {directory}")
    trials = load_prediction_trials(paths, bundle["config"], bundle["calibration"])
    if check_overlap:
        _warn_prediction_overlap(trials, bundle)
    return trials


def _validation_regions(checkpoint: Path, bundle: dict) -> dict:
    with np.load(checkpoint.parent / "validation_predictions.npz", allow_pickle=False) as validation:
        return optimize_regions(
            validation["location_probabilities"], validation["actual_ids"], validation["load_state"],
            bundle["mode"], bundle["config"].get("active_ids", list(range(1, 19))),
            min_segments=int(bundle["config"].get("region_search_min_segments", 3)), split="validation",
        )


def _metric_columns(report: dict) -> dict:
    values = {"test_" + key: value for key, value in report["overall"].items()}
    values.update({"test_optimized_" + key: value for key, value in report.get("optimized_overall", {}).items()
                   if key.startswith("region_") or key in ("system_region_accuracy", "system_loaded_region_accuracy")})
    return values


def _verify_unchanged(trials: list) -> None:
    for trial in trials:
        if _sha256(trial.path) != trial.sha256:
            raise ValueError(f"Input CSV changed during prediction/export: {trial.path}")


# Full workflow: evaluate completed combinations, export representative sheets, summarize selected models.
def select_and_evaluate(train_dir: str | Path, output: str | Path | None = None,
                        device: str | None = None, location_metric: str = "location_macro_recall",
                        check_overlap: bool = False,
                        input_dir: str | Path | None = None) -> Path:
    """Evaluate every completed pair; the best pair per mode is frozen by validation."""
    train_dir = Path(train_dir).resolve()
    comparison = _completed_comparison(train_dir)
    selected = dict(select_comparison_rows(comparison, location_metric=location_metric))
    comparison["selected_by_validation"] = comparison.index.isin(selected)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    reference_path = _checkpoint_path(train_dir, comparison.iloc[0])
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    _assert_bundle_row(reference, comparison.iloc[0])
    signature = _contract(reference)
    jobs = representative_comparison_rows(comparison, reference["config"])
    trials = _test_trials(reference, check_overlap=check_overlap, input_dir=input_dir)
    test_directory = (Path(input_dir).resolve() if input_dir is not None else
                      _path(reference["config"].get("test_directory", "datasets/test"), reference))
    print(f"Test CSV: {len(trials)} files in {test_directory}", flush=True)
    destination = fresh_directory("predict", output)
    writer = ModelWorkbook(destination / "predictions.xlsx", comparison)
    reports, selected_reports, selections, updates = {}, {}, [], {}
    first_csv = True
    try:
        for number, (index, row) in enumerate(comparison.iterrows(), 1):
            checkpoint = _checkpoint_path(train_dir, row)
            model, bundle = restore_bundle(checkpoint, device)
            _assert_bundle_row(bundle, row)
            if _contract(bundle) != signature:
                raise ValueError("Completed bundles have different frozen data contracts")
            search = _validation_regions(checkpoint, bundle) if index in selected else None
            optimized = search["selected_regions"] if search else None
            frame = predict_trials(model, bundle, trials, checkpoint, optimized, device)
            report = prediction_report(frame, bundle, optimized)
            identifier = f"{bundle['mode']}__force_{bundle['force_name']}__location_{bundle['location_name']}"
            report.update({"checkpoint": str(checkpoint), "selected_by_validation": index in selected})
            if search:
                report["validation_region_search"] = search
            reports[identifier] = report
            updates[index] = _metric_columns(report)
            if index in selected:
                frame.to_csv(destination / "predictions.csv", mode="w" if first_csv else "a", header=first_csv, index=False)
                first_csv = False
                selected_reports[bundle["mode"]] = report
                selections.append({
                    "comparison_row": int(index), "mode": bundle["mode"], "checkpoint": str(checkpoint),
                    "force_model": bundle["force_name"], "location_model": bundle["location_name"],
                    "validation_force_rmse_xyz_mN": row["force_rmse_xyz_mN"],
                    "validation_location_macro_recall": row.get("location_macro_recall"),
                    "location_selection_metric": location_metric,
                    "validation_location_selection_score": row[location_metric],
                    "validation_location_ce": row.get("location_validation_ce"),
                    "test_metrics": report["overall"],
                })
            matching_jobs = [job for job in jobs if job["comparison_row"] == index]
            if matching_jobs:
                supervised = _supervised_frame(frame, trials, "test")
                for job in matching_jobs:
                    writer.add_model(_branch_export_frame(supervised, bundle, job["branch"]),
                                     _sheet_metadata(job, checkpoint, bundle, "test"))
            del model, frame
            print(f"Test [{number}/{len(comparison)}]: {identifier}", flush=True)
        _verify_unchanged(trials)
        added = pd.DataFrame.from_dict(updates, orient="index").reindex(comparison.index)
        comparison = pd.concat([comparison.drop(columns=added.columns.intersection(comparison.columns)), added], axis=1)
        comparison.to_csv(destination / "comparison.csv", index=False)
        writer.close(comparison)
    except BaseException:
        writer.abort()
        raise
    selection_rule = (
        "Within each mode, lowest validation force XYZ RMSE; within absolute 1e-3 mN numerical tie "
        f"tolerance, highest validation {location_metric}, lowest location CE, then deterministic model names. "
        "Test scores never select a model or region boundary."
    )
    write_json(destination / "metrics.json", {
        "training_directory": str(train_dir), "selection_rule": selection_rule,
        "force_numerical_tie_tolerance_mN": DEFAULT_FORCE_TOLERANCE_MN,
        "selection": selections, "modes": selected_reports, "all_combinations": reports,
        "completed_combinations_evaluated": len(comparison),
        "predictions_csv_scope": "one intact validation-selected bundle per trained mode",
        "comparison_csv_scope": "all completed combinations; original columns are validation metrics, test_ columns are test metrics",
    })
    write_json(destination / "summary.json", {
        "selection_rule": selection_rule, "best_by_mode": {item["mode"]: item for item in selections},
        "completed_combinations_evaluated": len(comparison),
        "comparison_csv": "comparison.csv", "predictions_csv": "predictions.csv", "workbook": "predictions.xlsx",
        "representative_policy": REPRESENTATIVE_POLICY,
        "test_file_policy": "all_csv_in_current_test_directory",
        "test_directory": str(test_directory), "test_csv_count": len(trials),
        "test_evaluation_scope": reference["config"].get("test_evaluation_scope", "historical_held_out_recordings"),
    })
    write_json(destination / "input_manifest.json", [trial.info for trial in trials])
    write_json(destination / "workbook_manifest.json", {
        "path": str(writer.path), "sha256": _sha256(writer.path), "dataset_role": "test",
        "all_completed_combinations": len(comparison), "model_sheets": writer.models,
        "representative_policy": REPRESENTATIVE_POLICY,
        "sources": [trial.info for trial in trials],
    })
    return destination


def predict_checkpoint(checkpoint: str | Path, input_dir: str | Path,
                       output: str | Path | None = None, device: str | None = None,
                       check_overlap: bool = False) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint).resolve()
    model, bundle = restore_bundle(checkpoint, device)
    paths = csv_paths(Path(input_dir).resolve())
    if not paths:
        raise ValueError(f"No CSV files found in {input_dir}")
    trials = load_prediction_trials(paths, bundle["config"], bundle["calibration"])
    if check_overlap:
        _warn_prediction_overlap(trials, bundle)
    frame = predict_trials(model, bundle, trials, checkpoint, device=device)
    report = prediction_report(frame, bundle, None)
    destination = fresh_directory("predict", output)
    row = {"mode": bundle["mode"], "force_model": bundle["force_name"], "location_model": bundle["location_name"],
           "bundle": str(checkpoint), "status": "complete", "selected_by_validation": False,
           **bundle.get("validation_metrics", {}), **_metric_columns(report)}
    comparison = pd.DataFrame([row])
    writer = ModelWorkbook(destination / "predictions.xlsx", comparison)
    try:
        supervised = _supervised_frame(frame, trials, "test")
        for branch, own, partner in (("force", "force_name", "location_name"), ("location", "location_name", "force_name")):
            job = {"branch": branch, "mode": bundle["mode"], "model": bundle[own], "fixed_partner": bundle[partner]}
            metadata = _sheet_metadata(job, checkpoint, bundle, "external_input")
            metadata["representative_policy"] = "Explicitly requested intact checkpoint; no automatic model selection"
            writer.add_model(_branch_export_frame(supervised, bundle, branch), metadata)
        _verify_unchanged(trials)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    frame.to_csv(destination / "predictions.csv", index=False)
    comparison.to_csv(destination / "comparison.csv", index=False)
    write_json(destination / "metrics.json", {"checkpoint": str(checkpoint), **report})
    write_json(destination / "summary.json", {"checkpoint": str(checkpoint), "selection": "explicit_checkpoint",
               "test_metrics": report["overall"], "workbook": "predictions.xlsx"})
    write_json(destination / "input_manifest.json", [trial.info for trial in trials])
    write_json(destination / "workbook_manifest.json", {"path": str(writer.path), "sha256": _sha256(writer.path),
               "model_sheets": writer.models, "sources": [trial.info for trial in trials]})
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint")
    choice.add_argument("--train-dir", help="Evaluate every completed bundle; best models are selected by validation")
    parser.add_argument("--input-dir", help="CSV directory override; default: saved test directory for --train-dir, datasets/test for --checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--check-overlap", action="store_true",
                        help="warn about duplicate files or matching start timestamps/row counts and continue prediction (default: skip check)")
    parser.add_argument("--location-selection-metric", default="location_macro_recall",
                        choices=["location_macro_recall", "load_balanced_accuracy"])
    args = parser.parse_args(argv)
    destination = (select_and_evaluate(args.train_dir, args.output, args.device, args.location_selection_metric,
                                       check_overlap=args.check_overlap, input_dir=args.input_dir) if args.train_dir else
                   predict_checkpoint(args.checkpoint, args.input_dir or "datasets/test", args.output, args.device,
                                      check_overlap=args.check_overlap))
    print(f"Predictions, comparison and Excel: {destination}")


if __name__ == "__main__":
    main()
