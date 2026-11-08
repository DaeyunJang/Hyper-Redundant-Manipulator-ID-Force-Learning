"""Analyze saved prediction results: force/ID metrics, combined GT/prediction workbooks and plots; no training, inference or source CSV reload.

bash scripts/hrm_python.sh analysis.py results/predict/run_directory
bash scripts/hrm_python.sh analysis.py 'results/predict/date_time_*'
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import glob
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter

from organize_results import WorkbookReader, KEYS, align_rows, force_tables


AXES = ("fx", "fy", "fz", "xyz")
GROUPS = {"all": set(range(1, 19)), "body": set(range(1, 18)), "tip": {18}}
STATES = ("all", "loaded", "unloaded")
MODE_STYLE = {"masked": "-", "class0": "--"}
INPUT_NAMES = {
    "cable_loadcell": "C+L", "cable_angle": "C+A", "loadcell_angle": "L+A",
    "cable_loadcell_angle": "C+L+A", "cable_loadcell_estimated_pose": "C+L+Estimated",
    "cable_loadcell_fk_tip_tf": "C+L+FK TF",
}


def read_json(path: Path, sources: list) -> object:
    raw = path.read_bytes()
    sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return json.loads(raw)


def percent(numerator: float, denominator: float) -> float:
    return 100 * numerator / denominator if denominator else np.nan


def pool_force(files: list[dict], state: str = "all") -> dict:
    """Pool file-level RMSE using sample counts and squared errors; N=0 is not evaluable.

    AUTHOR: Daeyun
    """
    prefix = "" if state == "all" else state + "_"
    supported = [(int(m[prefix + "force_support"]), m) for m in files]
    supported = [(n, m) for n, m in supported if n > 0]
    total = sum(n for n, _ in supported)
    result = {"n": total}
    for metric in ("rmse", "mae"):
        for axis in AXES:
            key = f"{metric}_{axis}_mN"
            if not total:
                result[key] = np.nan
                continue
            values = np.array([m.get(prefix + "force_" + key, np.nan) for _, m in supported], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Positive force_support requires {prefix + key}.")
            weights = np.array([n for n, _ in supported])
            result[key] = float(np.sqrt(np.average(values ** 2, weights=weights)) if metric == "rmse"
                                else np.average(values, weights=weights))
    return result


def final_confusion(report: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Use the Masked location argmax or final Class0 output including ID0."""
    saved = report["system_confusion" if mode == "class0" else "conditional_confusion"]
    labels = np.asarray(saved["labels"], dtype=int)
    matrix = np.asarray(saved["matrix"], dtype=np.int64)
    if not saved.get("available", True) or matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"The saved {mode} confusion matrix is missing or has the wrong shape.")
    return labels, matrix


def id_scores(labels: np.ndarray, matrix: np.ndarray, ids: set, tolerances: list[int]) -> dict:
    rows = np.isin(labels, list(ids)) & (labels > 0)
    counts = matrix[rows]
    support = counts.sum(axis=1)
    total = int(support.sum())
    distance = np.abs(labels[rows, None] - labels[None, :])
    result = {"n_loaded": total, "supported_ids": int((support > 0).sum())}
    for tolerance in tolerances:
        # ID0 is not a physical location, so it is always incorrect, even when close to GT1.
        hit = (counts * ((labels[None, :] > 0) & (distance <= tolerance))).sum(axis=1)
        result[f"hits_tol{tolerance}"] = int(hit.sum())
        result[f"accuracy_tol{tolerance}_pct"] = percent(hit.sum(), total)
        result[f"macro_tol{tolerance}_pct"] = float(100 * np.mean(hit[support > 0] / support[support > 0])) if total else np.nan
    for name, columns in (("id0", labels == 0), ("tip", labels == 18),
                          ("body", (labels >= 1) & (labels <= 17))):
        result[f"predicted_{name}_pct"] = percent(counts[:, columns].sum(), total)
    return result


def detection_scores(files: list[dict], mode: str) -> dict:
    """Pool body/tip detection per file because GT0 rows no longer identify the original contact ID."""
    if mode == "masked":
        return {"available": False}
    values = {name: sum(int(m["load_" + name]) for m in files) for name in ("tp", "fn", "tn", "fp")}
    tp, fn, tn, fp = (values[name] for name in ("tp", "fn", "tn", "fp"))
    return {"available": True, **values, "n_loaded": tp + fn, "n_unloaded": tn + fp,
            "loaded_recall_pct": percent(tp, tp + fn), "unloaded_recall_pct": percent(tn, tn + fp),
            "unloaded_false_positive_pct": percent(fp, tn + fp)}


def identifier(row: dict) -> str:
    return f"{row['mode']}__force_{row['force_model']}__location_{row['location_model']}"


def input_name(folder: Path) -> str:
    # Match input configurations even when run directories have been renamed to retain only the date.
    return re.sub(r"^\d{8}_(?:\d{6}_)?", "", folder.name) or folder.name


def _force_series_sources(folder: Path) -> list[dict]:
    """Read organized workbooks linked to current predictions, or fall back to saved force sheets."""
    source = folder / "predictions.xlsx"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads((folder / "workbook_manifest.json").read_text())
    if manifest.get("sha256") != source_hash:
        raise ValueError(f"SHA256 mismatch between predictions.xlsx and workbook_manifest: {folder}")
    models = pd.DataFrame(manifest["model_sheets"])
    models = models.loc[models.branch.eq("force") & models.rows.gt(0)].copy()
    if models.empty or models.duplicated(["mode", "model"]).any():
        raise ValueError(f"Representative force models are missing or duplicated: {folder}")
    if not models.force_unit.eq("mN").all():
        raise ValueError(f"Force workbook units must be mN: {folder}")
    modes = models["mode"].unique().tolist()
    organized = None
    candidates = sorted(folder.glob("organized_*"), key=lambda p: re.sub(
        r"^organized_(?:common_)?", "", p.name), reverse=True)
    for directory in candidates:
        books = {}
        for mode in modes:
            label = "loss-mask" if mode == "masked" else mode
            paths = list(directory.glob(f"force-estimation_{label}_id-tol-*_results.xlsx"))
            if len(paths) != 1:
                break
            reader = WorkbookReader(paths[0])
            try:
                if not {"_guide", "_models", "_rows", *AXES[:3]}.issubset(reader.sheets):
                    break
                guide = dict(reader.read("_guide").itertuples(index=False, name=None))
                # Accept English labels and the legacy Korean key in saved workbooks.
                matches = guide.get("Input SHA256", guide.get("\uc785\ub825 SHA256")) == source_hash
            finally:
                reader.close()
            if not matches:
                break
            books[mode] = paths[0]
        if len(books) == len(modes):
            organized = books
            break
    result = []
    for mode in modes:
        selected = models.loc[models["mode"].eq(mode)].copy()
        path = organized[mode] if organized else source
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        reader = WorkbookReader(path)
        try:
            if organized:
                rows = reader.read("_rows")
                axes = {axis: reader.read(axis) for axis in AXES[:3]}
                saved_models = reader.read("_models")
                saved_models = saved_models.loc[saved_models.branch.eq("force") & saved_models["mode"].eq(mode)]
                for model in selected.itertuples():
                    saved = saved_models.loc[saved_models.model.eq(model.model)]
                    if (len(saved) != 1 or saved.iloc[0].checkpoint_sha256 != model.checkpoint_sha256
                            or saved.iloc[0].fixed_partner != model.fixed_partner):
                        raise ValueError(f"Representative force-model metadata differs in the organized workbook: {path} / {model.model}")
            else:
                reference, frames = None, {}
                for model in selected.itertuples():
                    frame = reader.read(model.sheet)
                    reference = frame if reference is None else reference
                    frames[model.model] = align_rows(frame, reference)
                tables = force_tables(frames, reference)
                rows, axes = tables["_rows"], {axis: tables[axis] for axis in AXES[:3]}
        finally:
            reader.close()
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise ValueError(f"Force workbook changed while being read: {path}")
        expected = selected.model.tolist()
        for axis, table in axes.items():
            if len(table) != len(rows) or list(table) != [f"gt_{axis}_mN", *expected]:
                raise ValueError(f"Force sheets do not match the source-row/model lists: {path} / {axis}")
            if not np.isfinite(table.to_numpy(dtype=float)).all():
                raise ValueError(f"Force GT/predictions contain nonnumeric values: {path} / {axis}")
        if (rows[KEYS].isna().any().any() or rows.duplicated(KEYS).any()
                or not np.isfinite(rows.source_row.to_numpy(float)).all()
                or not rows.source_row.eq(np.floor(rows.source_row)).all() or rows.source_row.lt(0).any()):
            raise ValueError(f"Source file/row keys are missing, duplicated or noninteger: {path}")
        result.append({"mode": mode, "models": selected, "rows": rows, "axes": axes,
                       "source_workbook": str(path), "source_sha256": before,
                       "predictions_sha256": source_hash,
                       "source_kind": "organized" if organized else "predictions"})
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError(f"predictions.xlsx changed while being read: {folder}")
    return result


def combined_force_tables(folders: list[Path], force_files=()) -> dict:
    """Collect input/mode/representative-model columns and share GT only on intersecting source rows."""
    entries, reference = [], None
    names = [input_name(folder) for folder in folders]
    for folder, name in zip(folders, names):
        prefix = folder.name if names.count(name) > 1 else name
        for item in _force_series_sources(folder):
            rows, axes = item["rows"], item["axes"]
            item["original_rows"] = len(rows)
            if force_files:
                keep = rows.source_file.map(lambda p: Path(p).name).isin(force_files)
                rows = rows.loc[keep].reset_index(drop=True)
                axes = {axis: frame.loc[keep].reset_index(drop=True) for axis, frame in axes.items()}
            # Align using the original CSV file/row keys rather than Excel display row numbers.
            truth = rows.drop(columns=["excel_row"]).copy()
            for axis, frame in axes.items():
                truth[f"gt_{axis}_mN"] = frame[f"gt_{axis}_mN"].to_numpy()
            truth = truth.set_index(KEYS)
            if reference is None:
                reference = truth
            else:
                common = reference.index[reference.index.isin(truth.index)]
                reference = reference.loc[common]
                try:
                    pd.testing.assert_frame_equal(truth.loc[common, reference.columns], reference,
                                                  check_dtype=False, check_exact=True)
                except AssertionError as exc:
                    raise ValueError(f"GT/ID/load information differs for the same source rows: {folder} / {item['mode']}") from exc
            if reference.empty:
                raise ValueError("No force prediction rows are shared by every input/mode.")
            item.update(run=str(folder), input=name, prefix=prefix, rows=rows, axes=axes)
            entries.append(item)
        print(f"Read force time series: {folder.name}", flush=True)
    if not entries:
        raise ValueError("No prediction directories were provided for combining force time series.")
    common = reference.index
    values = {axis: {f"gt_{axis}_mN": reference[f"gt_{axis}_mN"].to_numpy(float)} for axis in AXES[:3]}
    columns, alignment, scores = [], [], []
    for item in entries:
        positions = pd.MultiIndex.from_frame(item["rows"][KEYS]).get_indexer(common)
        alignment.append({"input": item["input"], "mode": item["mode"], "run": item["run"],
                          **{key: item[key] for key in ("source_workbook", "source_sha256", "predictions_sha256", "source_kind", "original_rows")},
                          "filtered_rows": len(item["rows"]), "common_rows": len(common),
                          "excluded_rows": item["original_rows"] - len(common)})
        for model in item["models"].to_dict("records"):
            column = f"{item['prefix']}_{item['mode']}_{model['model']}_mN"
            if column in values["fx"]:
                raise ValueError(f"Duplicate force prediction column name: {column}")
            errors = []
            score = {"column": column, "input": item["input"], "mode": item["mode"],
                     "model": model["model"], "n": len(common)}
            for axis in AXES[:3]:
                prediction = item["axes"][axis][model["model"]].to_numpy(float)[positions]
                values[axis][column] = prediction
                error = prediction - values[axis][f"gt_{axis}_mN"]
                errors.append(error)
                score[f"rmse_{axis}_mN"] = float(np.sqrt(np.mean(error ** 2)))
                score[f"mae_{axis}_mN"] = float(np.mean(np.abs(error)))
            error = np.column_stack(errors)
            score.update(rmse_xyz_mN=float(np.sqrt(np.mean(error ** 2))), mae_xyz_mN=float(np.mean(np.abs(error))))
            scores.append(score)
            columns.append({**model, "column": column, "input": item["input"], "run": item["run"],
                            "source_workbook": item["source_workbook"]})
    rows = reference.drop(columns=[f"gt_{axis}_mN" for axis in AXES[:3]]).reset_index()
    rows.insert(0, "excel_row", np.arange(len(rows)) + 2)
    result = {axis: pd.DataFrame(data) for axis, data in values.items()}
    result.update(force_rows=rows, force_columns=pd.DataFrame(columns),
                  force_alignment=pd.DataFrame(alignment), force_common_metrics=pd.DataFrame(scores))
    print(f"Combined force time series: {len(rows):,} rows, 1 GT column + {len(columns)} prediction columns per axis", flush=True)
    return result


def choose_models(comparison: pd.DataFrame, summary: dict, reports: dict) -> tuple[set, set]:
    # Force evaluation retains the validation selection saved by predict.
    force = {identifier(row) for row in summary["best_by_mode"].values()}
    available = comparison[comparison.apply(lambda row: identifier(row) in reports, axis=1)].copy()
    for column in ("location_macro_recall", "location_validation_ce", "force_rmse_xyz_mN"):
        available[column] = pd.to_numeric(available[column], errors="coerce")
    available = available[np.isfinite(available.location_macro_recall)]
    ordered = available.sort_values(
        ["location_macro_recall", "location_validation_ce", "force_rmse_xyz_mN", "force_model", "location_model"],
        ascending=[False, True, True, True, True], na_position="last")
    location = {identifier(row) for row in ordered.groupby("mode", sort=False).head(1).to_dict("records")}
    if not force or not force.issubset(reports) or not location:
        raise ValueError("The saved validation model selection could not be verified.")
    return force, location


def analyze_run(folder: Path, force_files: list[str], tolerances: list[int], tables: dict, matrices: list) -> None:
    sources = []
    metrics = read_json(folder / "metrics.json", sources)
    if "all_combinations" not in metrics:
        raise ValueError("Provide an all-combination prediction directory created with --train-dir; single-checkpoint results are not supported.")
    manifest = read_json(folder / "workbook_manifest.json", sources)
    inputs = read_json(folder / "input_manifest.json", sources)
    summary = read_json(folder / "summary.json", sources)
    comparison_path = folder / "comparison.csv"
    sources.append({"path": str(comparison_path), "sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest()})
    comparison = pd.read_csv(comparison_path)
    reports = metrics["all_combinations"]
    force_best, id_best = choose_models(comparison, summary, reports)
    sheets = manifest["model_sheets"]
    scopes = {item["force_scope"] for item in sheets}
    if len(scopes) != 1:
        raise ValueError("force_scope is inconsistent within one run.")
    force_scope = scopes.pop()
    name = input_name(folder)
    run = str(folder)
    base = {"run": run, "input": name, "force_scope": force_scope}
    trial_map = {item["path"]: item for item in inputs}
    if force_files:
        missing = set(force_files) - {Path(path).name for path in trial_map}
        if missing:
            raise ValueError(f"{folder.name}: Requested force files were not found: {sorted(missing)}")
    representative = {"force": set(), "location": set()}
    for item in sheets:
        force_model = item["model"] if item["branch"] == "force" else item["fixed_partner"]
        location_model = item["model"] if item["branch"] == "location" else item["fixed_partner"]
        representative[item["branch"]].add(identifier({"mode": item["mode"], "force_model": force_model,
                                                        "location_model": location_model}))
    tables["runs"].append({**base, "combinations": len(reports), "test_files": len(inputs),
        "force_file_filter": ", ".join(force_files) or "all saved test files",
        "input_columns": ", ".join(sheets[0]["input_columns"]), "window_samples": sheets[0]["window_samples"],
        "test_evaluation_scope": summary.get("test_evaluation_scope", "not recorded; independence not verified")})
    for item in inputs:
        tables["test_files"].append({**base, **{key: item.get(key) for key in
            ("path", "sha256", "effective_id", "rows", "source_rows_total", "dropped_rows", "prediction_endpoints")}})
    indexed = {identifier(row): row for row in comparison.to_dict("records")}
    for key, report in reports.items():
        row = indexed[key]
        model = {**base, "mode": row["mode"], "force_model": row["force_model"], "location_model": row["location_model"],
                 "selected_force": key in force_best, "selected_id": key in id_best,
                 "representative_force": key in representative["force"], "representative_id": key in representative["location"],
                 "checkpoint": report.get("checkpoint", row.get("bundle"))}
        tables["models"].append({**model, **{"validation_" + field: row.get(field) for field in
            ("force_rmse_xyz_mN", "location_macro_recall", "location_validation_ce", "force_best_epoch", "location_best_epoch")}})
        file_rows = [(path, trial_map[path]["effective_id"], values) for path, values in report["files"].items()]
        selected_files = [(p, segment, values) for p, segment, values in file_rows if not force_files or Path(p).name in force_files]
        labels, matrix = final_confusion(report, row["mode"])
        for group, ids in GROUPS.items():
            for state in STATES:
                score = {**model, "group": group, "load_state": state,
                         **pool_force([m for _, segment, m in selected_files if segment in ids], state)}
                tables["force"].append(score)
                if key in force_best:
                    tables["force_selected"].append(score)
            score = {**model, "group": group, **id_scores(labels, matrix, ids, tolerances)}
            tables["location"].append(score)
            detection = {**model, "group": group, **detection_scores([m for _, segment, m in file_rows if segment in ids], row["mode"])}
            tables["detection"].append(detection)
            if key in id_best:
                tables["id_selected"].append({**score, **{"detection_" + k: v for k, v in detection.items() if k not in model and k != "group"}})
        # Per-ID/file detail tables use validation-selected models and the same representative branches as organized workbooks.
        if key in force_best | representative["force"]:
            for path, segment, values in selected_files:
                for state in STATES:
                    tables["force_by_file"].append({**model, "file": path, "contact_id": segment, "load_state": state,
                                                    **pool_force([values], state)})
            for segment in sorted({segment for _, segment, _ in selected_files}):
                for state in STATES:
                    tables["force_by_id"].append({**model, "contact_id": segment, "load_state": state,
                        **pool_force([m for _, sid, m in selected_files if sid == segment], state)})
        if key in id_best | representative["location"]:
            for segment in labels[labels > 0]:
                tables["id_by_id"].append({**model, "gt_id": int(segment), **id_scores(labels, matrix, {int(segment)}, tolerances)})
        if key in id_best:
            matrices.append((model, labels, matrix))
            for i, gt in enumerate(labels):
                for j, pred in enumerate(labels):
                    tables["confusion"].append({**model, "gt_id": int(gt), "predicted_id": int(pred),
                        "count": int(matrix[i, j]), "row_support": int(matrix[i].sum()),
                        "row_pct": percent(matrix[i, j], matrix[i].sum())})
    tables["sources"].extend({"run": run, **source} for source in sources)
    print(f"Analyzed: {folder.name} / force_scope={force_scope} / {len(reports)} combinations", flush=True)


def guide(force_files: list[str], tolerances: list[int]) -> pd.DataFrame:
    return pd.DataFrame([
        ("Start here", "force_body_summary / id_body_summary summarize body results. force_file_detail covers the selected separate force file. With --previous, force_file_comparison compares previous/current runs."),
        ("MATLAB", "Select this workbook in plot_analysis.m. Multiple prediction directories are already combined here, so no separate workbook merge is needed."),
        ("Units", "Force is in mN, accuracy in %, and n counts evaluated rows. Blank cells are not evaluable, rather than zero error or 0% accuracy."),
        ("Force scope", "Uses the saved force_scope. Body support is N=0 for tip-trained models. all_single_contact evaluates measured force at each contact location."),
        ("Force evaluation files", ", ".join(force_files) or "All saved test files. Use --force-file to select a particular independent experiment."),
        ("Load states", "Uses saved labels without reloading the current YAML or reapplying thresholds. Force metrics are separated into all/loaded/unloaded."),
        ("Force pooling", "RMSE=sqrt(sum(N_file*RMSE_file^2)/sum N_file); MAE=sum(N_file*MAE_file)/sum N_file."),
        ("XYZ error", "RMSE/MAE pooled across all three axis components, not force-vector norm error or the arithmetic mean of per-axis RMSE."),
        ("ID evaluation", "The denominator includes GT-loaded rows only. Masked uses ID1–18 argmax; Class0 uses final ID0–18 argmax. A prediction of 0 fails at every tolerance."),
        ("ID tolerance", f"{tolerances}: predicted ID>0 AND abs(predicted ID-GT)<=n. This is accuracy, not summed probability. Success within ±n does not imply correct tip/body discrimination."),
        ("Macro", "Equally weighted mean accuracy across GT IDs with support. Micro accuracy pools all rows."),
        ("Detection", "Class0 TP/FN/TN/FP, loaded detection rate and unloaded accuracy. Masked has no independent detection output to evaluate."),
        ("Force model selection", "Retains the validation force-priority pair from predict summary.json; does not reselect by minimum test error."),
        ("ID model selection", "Maximizes validation location_macro_recall, breaking ties by CE, force RMSE and model name. Class0 validation scores use candidates excluding ID0; test reporting uses final outputs."),
        ("Selected models", "force_selected and id_selected may refer to different bundles. Their scores must not be combined as results from a single bundle."),
        ("All combinations", "models, force, location and detection sheets include every saved combination. Force covers overall/body/tip × all/loaded/unloaded."),
        ("Detail and representative models", "force_by_file/force_by_id/id_by_id include validation selections and the same fixed-partner representative branches as organized workbooks. Use representative_* columns to select them."),
        ("Confusion matrix", "Uses the validation-selected ID model. The GT0 row contains all Class0 unloaded rows; detection sheets separate body/tip unloaded statistics."),
        ("ID file scope", "ID evaluation uses all saved test files. --force-file applies only to force analysis; saved confusion matrices cannot recover arbitrary per-file ±n scores."),
        ("Comparison interpretation", "Input configurations may differ in sample counts and training data in existing aggregates/plots. fx/fy/fz align intersecting source endpoints, without matching complete input-window histories or training data."),
        ("Row-wise force comparison", "fx/fy/fz contain one shared GT column plus input_mode_force-model_mN prediction columns for both Masked and Class0. force_rows identifies the source CSV/row/ID; force_columns records the checkpoint and location partner."),
        ("Time-series provenance", "Reads the latest organized force workbook linked to the current predictions.xlsx hash. If unavailable or stale, uses representative force sheets in predictions.xlsx. No training, inference, interpolation or relabeling is performed."),
        ("Common-row force metrics", "force_common_metrics recalculates representative-model RMSE/MAE on identical fx/fy/fz rows. force_alignment records source workbooks, hashes and row counts per input/mode. Denominators in existing all-combination aggregates and plots remain unchanged."),
        ("Representative force scope", "Row-wise comparisons use representative branches saved in predictions.xlsx, such as 2 modes × 11 force models per input. They do not contain row-wise predictions for every force/location pair and may differ from validation-selected models."),
        ("Previous comparisons", "--previous matches input names, modes and files. Error differences are shown only when evaluated source-row/GT/load-state sets in predictions.csv and file hashes match. This is not a causal comparison controlling window histories, training data or selected models."),
        ("Additional diagnostics", "ID accuracy by force magnitude and time-series diagnostics are separate analyses. Common-row fx/fy/fz predictions can be plotted directly from this workbook."),
    ], columns=["Item", "Description"])


def force_file_results(folder: Path, filename: str, experiment: str, sources: list) -> list[dict]:
    """Evaluate validation-selected models on a separate force trial and retain row/GT fingerprints for comparisons."""
    reports = read_json(folder / "metrics.json", sources)["all_combinations"]
    summary = read_json(folder / "summary.json", sources)
    inputs = read_json(folder / "input_manifest.json", sources)
    files = [item for item in inputs if Path(item["path"]).name == filename]
    if not files:
        print(f"Skipping separate force file: {folder.name} / {filename} not found", flush=True)
        return []
    if len(files) != 1:
        raise ValueError(f"Multiple test files have the same name: {folder} / {filename}")
    trial = files[0]
    csv = folder / "predictions.csv"
    rows = None
    if csv.exists():
        sources.append({"path": str(csv), "sha256": hashlib.sha256(csv.read_bytes()).hexdigest()})
        columns = ["source_file", "source_row", "mode", "force_model", "location_model", "load_state_gt"]
        columns += [f"{kind}_{axis}_N" for kind in ("true", "pred") for axis in AXES[:3]]
        rows = pd.read_csv(csv, usecols=columns)
        rows = rows[rows.source_file.eq(trial["path"])]
    result = []
    for mode, chosen in summary["best_by_mode"].items():
        metrics = reports[identifier(chosen)]["files"][trial["path"]]
        for state in STATES:
            score = pool_force([metrics], state)
            fingerprint = ""
            if rows is not None:
                part = rows[rows["mode"].eq(mode)]
                if not (part.force_model.eq(chosen["force_model"]) & part.location_model.eq(chosen["location_model"])).all():
                    raise ValueError(f"Selected models differ between predictions.csv and summary: {folder}")
                numeric = [f"{kind}_{axis}_N" for kind in ("true", "pred") for axis in AXES[:3]]
                part = part[np.isfinite(part[numeric].to_numpy(dtype=float)).all(axis=1)]
                if state != "all":
                    part = part[part.load_state_gt.eq(state)]
                part = part.sort_values("source_row")
                if len(part) != score["n"] or part.source_row.duplicated().any():
                    raise ValueError(f"Force evaluation row counts or duplicates do not match the saved metrics: {folder} / {mode} / {state}")
                data = part[["source_row", "true_fx_N", "true_fy_N", "true_fz_N"]].to_numpy(dtype="<f8")
                fingerprint = hashlib.sha256(data.tobytes()).hexdigest() if len(part) else ""
            result.append({"input": input_name(folder), "run": str(folder), "experiment": experiment,
                           "mode": mode, "force_model": chosen["force_model"], "location_model": chosen["location_model"],
                           "file": filename, "load_state": state, **score, "source_sha256": trial["sha256"],
                           "evaluation_rows_gt_sha256": fingerprint})
    return result


def add_summaries(frames: dict, folders: list[Path], previous: list[Path], filename: str) -> dict:
    """Prepend readable summaries and previous/current comparisons while retaining existing detail tables."""
    extra = {"force_body_summary": frames["force_selected"].query("group == 'body'").copy(),
             "id_body_summary": frames["id_selected"].query("group == 'body'").copy()}
    sources, rows = [], []
    for experiment, paths in (("current", folders), ("previous", previous)):
        for folder in paths:
            rows.extend(force_file_results(folder, filename, experiment, sources))
    if rows:
        detail = pd.DataFrame(rows)
        extra["force_file_detail"] = detail
        if previous:
            keys = ["input", "mode", "file", "load_state"]
            current = detail[detail.experiment.eq("current")]
            old = detail[detail.experiment.eq("previous")]
            if current.duplicated(keys).any() or old.duplicated(keys).any():
                raise ValueError("Specify one previous run per input configuration; multiple runs share the same input name.")
            joined = current.merge(old, on=keys, suffixes=("_new", "_old"), how="inner", validate="one_to_one")
            if joined.empty:
                raise ValueError("No matching input name, mode and separate force file exist across previous/current runs.")
            joined["same_evaluation_rows_gt"] = (joined.source_sha256_new.eq(joined.source_sha256_old) &
                joined.evaluation_rows_gt_sha256_new.ne("") &
                joined.evaluation_rows_gt_sha256_new.eq(joined.evaluation_rows_gt_sha256_old) &
                joined.n_new.eq(joined.n_old))
            for metric in ("rmse", "mae"):
                for axis in AXES:
                    col = f"{metric}_{axis}_mN"
                    joined[col + "_delta_new_minus_old"] = (joined[col + "_new"] - joined[col + "_old"]).where(joined.same_evaluation_rows_gt)
            extra["force_file_comparison"] = joined
    frames["sources"] = pd.concat([frames["sources"], pd.DataFrame(sources)], ignore_index=True).drop_duplicates("path")
    return {"guide": frames["guide"], **extra, **{key: value for key, value in frames.items() if key != "guide"}}


def write_excel(tables: dict, destination: Path) -> None:
    # Stream rows instead of accumulating millions of force-prediction cells in memory.
    options = {"constant_memory": True, "strings_to_formulas": False, "strings_to_urls": False}
    with xlsxwriter.Workbook(destination, options) as book:
        book.use_zip64()
        header = book.add_format({"bold": True, "text_wrap": True, "bg_color": "#173D59", "font_color": "white"})
        number = book.add_format({"num_format": "0.000000"})
        for name, frame in tables.items():
            if len(frame) >= 1048576 or len(frame.columns) > 16384:
                raise ValueError(f"{name}: Excel worksheet size limit exceeded.")
            # Place scores on the left and provenance, long source paths and selection flags on the right.
            preferred = ["input", "mode", "force_model", "location_model", "group", "load_state", "contact_id", "gt_id", "n", "n_loaded"]
            preferred += [f"{metric}_{axis}_mN" for metric in ("rmse", "mae") for axis in AXES]
            preferred += [column for column in frame.columns if column.startswith(("accuracy_tol", "macro_tol"))]
            front = [column for column in preferred if column in frame.columns]
            frame = frame[front + [column for column in frame.columns if column not in front]]
            sheet = book.add_worksheet(name)
            sheet.write_row(0, 0, list(frame), header)
            sheet.set_row(0, 60 if name in AXES[:3] else 32)
            sheet.freeze_panes(1, 1 if name in AXES[:3] else 0)
            for index, row in enumerate(frame.itertuples(index=False, name=None), 1):
                values = []
                for value in row:
                    if isinstance(value, (list, tuple, dict)):
                        value = json.dumps(value, ensure_ascii=False)
                    elif isinstance(value, np.generic):
                        value = value.item()
                    if pd.isna(value):
                        value = None
                    values.append(value)
                sheet.write_row(index, 0, values)
            if len(frame):
                sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            sheet.set_column(0, len(frame.columns) - 1, 28 if name in AXES[:3] else 19,
                             number if name in AXES[:3] else None)
            if name == "guide":
                sheet.set_column(0, 0, 22)
                sheet.set_column(1, 1, 115, book.add_format({"text_wrap": True}))
            else:
                for index, column in enumerate(frame.columns):
                    if column in ("run", "file", "path", "checkpoint", "input_columns"):
                        sheet.set_column(index, index, 32)


def plot_summary(tables: dict, output: Path, tolerances: list[int]) -> None:
    force = tables["force_selected"].query("group == 'all' and load_state == 'all'").reset_index(drop=True)
    location = tables["id_selected"].query("group == 'body'").reset_index(drop=True)
    runs = tables["runs"].run.tolist()
    colors = {run: plt.get_cmap("tab10")(i % 10) for i, run in enumerate(runs)}
    names = {}
    runs_table = tables["runs"]
    for index, row in enumerate(runs_table.itertuples(), 1):
        label = INPUT_NAMES.get(row.input, row.input)
        if runs_table.force_scope.nunique() > 1 or runs_table.input.duplicated(keep=False).any():
            label += f" [{index}: {row.force_scope}]"
        names[row.run] = label
    many_curves = len(location) > 6
    fig, axes = plt.subplots(2, 2, figsize=(max(13, len(force) * .95), 13 if many_curves else 11))
    x = np.arange(len(force))
    labels = [f"{names[r.run]}\n{r.mode}\nN={int(r.n):,}" for r in force.itertuples()]
    a, b, c, d = axes.flat
    a.bar(x, force.rmse_xyz_mN, color=[colors[r] for r in force.run])
    a.set(title="A. Force: XYZ RMSE (validation-selected force model)", ylabel="RMSE [mN]")
    a.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    for i, value in enumerate(force.rmse_xyz_mN):
        if np.isfinite(value):
            a.annotate(f"{value:.2f}", (i, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for offset, axis in enumerate(AXES[:3]):
        b.bar(x + (offset - 1) * .25, force[f"rmse_{axis}_mN"], width=.25, label=axis.upper())
    b.set(title="B. Force: axis RMSE (same models / rows as A)", ylabel="RMSE [mN]")
    b.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    b.legend()
    for row in location.to_dict("records"):
        c.plot(tolerances, [row[f"accuracy_tol{n}_pct"] for n in tolerances], MODE_STYLE[row["mode"]],
               marker="o", color=colors[row["run"]], label=f"{names[row['run']]} {row['mode']} (N={int(row['n_loaded']):,})")
    c.set(title="C. Body ID: final-output accuracy on GT-loaded rows", xlabel="Allowed segment error (±n)", ylabel="Accuracy [%]", ylim=(0, 105))
    c.set_xticks(tolerances)
    if many_curves:
        handles, legend_labels = c.get_legend_handles_labels()
        fig.legend(handles, legend_labels, fontsize=8, loc="lower center", bbox_to_anchor=(.5, .035), ncol=3)
    else:
        c.legend(fontsize=8, loc="best")
    detection = location[location["mode"].eq("class0")]
    xd = np.arange(len(detection))
    if len(detection):
        for shift, column, label in ((-.18, "loaded_recall_pct", "Loaded recall"), (.18, "unloaded_recall_pct", "Unloaded recall")):
            d.bar(xd + shift, detection["detection_" + column], width=.36, label=label)
        d.set_xticks(xd, [f"{names[r.run]}\nloaded N={int(r.n_loaded):,}\nunloaded N={int(r.detection_n_unloaded):,}" for r in detection.itertuples()],
                      rotation=30, ha="right", fontsize=8)
        d.legend(fontsize=9)
    else:
        d.text(.5, .5, "Masked has no load detector", transform=d.transAxes, ha="center")
    d.set(title="D. Class0: body-file load detection (ID-selected model)", ylabel="Recall [%]", ylim=(0, 115))
    d.set_yticks(range(0, 101, 20))
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    filters = tables["runs"].force_file_filter.unique()
    fig.suptitle("Saved test results | force training scope: " + ", ".join(tables["runs"].force_scope.unique()) +
                 "\nForce files: " + "; ".join(filters), fontsize=12)
    fig.text(.5, .008, "Force and ID models are selected separately on validation. N may differ across inputs. Class0 ID0 fails every ID tolerance.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .15 if many_curves else .025, 1, .94))
    fig.savefig(output / "force_and_id_summary.png", dpi=180)
    plt.close(fig)


def plot_confusions(matrices: list, output: Path) -> None:
    by_run = defaultdict(list)
    for model, labels, matrix in matrices:
        by_run[model["run"]].append((model, labels, matrix))
    for index, items in enumerate(by_run.values(), 1):
        fig, axes = plt.subplots(1, len(items), figsize=(9 * len(items), 8), squeeze=False, constrained_layout=True)
        for ax, (model, labels, matrix) in zip(axes.flat, items):
            rows = labels > 0
            support = matrix[rows].sum(axis=1)
            values = np.divide(100. * matrix[rows], support[:, None],
                               out=np.full(matrix[rows].shape, np.nan), where=support[:, None] > 0)
            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("lightgray")
            img = ax.imshow(values, vmin=0, vmax=100, aspect="auto", cmap=cmap)
            ax.set_xticks(np.arange(len(labels)), labels)
            ax.set_yticks(np.arange(rows.sum()), [f"{sid} (N={n})" for sid, n in zip(labels[rows], support)])
            name = INPUT_NAMES.get(model["input"], model["input"])
            ax.set(title=f"{name} / {model['mode']} / force scope: {model['force_scope']}\nForce {model['force_model']} + ID {model['location_model']}",
                   xlabel="Predicted ID (0 = unloaded)", ylabel="GT contact ID on loaded rows")
            fig.colorbar(img, ax=ax, label="Within-GT-ID percentage [%]", shrink=.8)
        fig.savefig(output / f"id_confusion_{index:02d}_{items[0][0]['input']}.png", dpi=160)
        plt.close(fig)


def plot_force_details(tables: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    for ax, state in zip(axes.flat, ("all", "loaded")):
        data = tables["force_body_summary"].query("load_state == @state")
        labels = [f"{INPUT_NAMES.get(r.input, r.input)}\n{r.mode} / N={r.n:,}" for r in data.itertuples()]
        ax.bar(np.arange(len(data)), data.rmse_xyz_mN)
        ax.set_xticks(np.arange(len(data)), labels, rotation=50, ha="right", fontsize=8)
        ax.set_title(f"Body force / {state} GT rows")
    ax = axes[1, 0]
    detail = tables.get("force_file_detail", pd.DataFrame())
    if not detail.empty:
        data = detail.query("experiment == 'current' and load_state == 'all'")
        ax.bar(np.arange(len(data)), data.rmse_xyz_mN)
        labels = [f"{INPUT_NAMES.get(r.input, r.input)}\n{r.mode} / N={r.n:,}" for r in data.itertuples()]
        ax.set_xticks(np.arange(len(data)), labels, rotation=50, ha="right", fontsize=8)
        ax.set_title("Separate force file: " + str(detail.file.iloc[0]))
    else:
        ax.set_title("Separate force file: unavailable")
    ax = axes[1, 1]
    comparison = tables.get("force_file_comparison", pd.DataFrame())
    if not comparison.empty:
        data = comparison.query("load_state == 'all' and same_evaluation_rows_gt")
        x = np.arange(len(data))
        ax.bar(x - .18, data.rmse_xyz_mN_old, .36, label="Previous")
        ax.bar(x + .18, data.rmse_xyz_mN_new, .36, label="Current")
        labels = [f"{INPUT_NAMES.get(r.input, r.input)}\n{r.mode} / N={r.n_new:,}" for r in data.itertuples()]
        ax.set_xticks(x, labels, rotation=50, ha="right", fontsize=8)
        ax.legend()
        ax.set_title("Previous vs current / same force rows and GT")
    else:
        ax.text(.5, .5, "Previous comparison: use --previous", transform=ax.transAxes, ha="center")
    for ax in axes.flat:
        ax.set_ylabel("XYZ RMSE [mN]")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Validation-selected force models | Body and separate force file\nN differs across inputs; missing support is not zero error")
    fig.savefig(output / "body_force_and_tip_comparison.png", dpi=180)
    plt.close(fig)


def expand_folders(patterns: list[str]) -> list[Path]:
    folders = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            raise ValueError(f"Result directory not found: {pattern}")
        for path in matches:
            folder = Path(path).resolve()
            if not (folder / "metrics.json").is_file():
                raise ValueError(f"Provide a prediction run directory containing metrics.json: {folder}")
            if folder not in folders:
                folders.append(folder)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predict_dirs", nargs="+", help="one or more prediction result directories or quoted glob patterns")
    parser.add_argument("--output", type=Path, help="new analysis directory (default: results/analysis/date_time_comparison)")
    parser.add_argument("--force-file", nargs="+", default=[], help="CSV filenames for force evaluation; defaults to all test files; ID evaluation always uses all files")
    parser.add_argument("--id-tolerances", nargs="+", type=int, default=[2, 3], help="default: 2 3; exact ID (0) and ±1 are always computed")
    parser.add_argument("--previous", nargs="+", default=[], help="optional previous prediction directories/globs for comparing separate force trials with matching inputs")
    parser.add_argument("--comparison-force-file", default="summary_18_test_2.csv",
                        help="filename for separate force-trial sheets/plots (default: summary_18_test_2.csv); independent of the --force-file aggregation filter")
    args = parser.parse_args()
    if any(n < 0 or n > 17 for n in args.id_tolerances):
        parser.error("ID tolerance must be between 0 and 17.")
    tolerances = sorted({0, 1, *args.id_tolerances})
    try:
        folders = expand_folders(args.predict_dirs)
        previous = expand_folders(args.previous)
    except ValueError as exc:
        parser.error(str(exc))
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or Path("results/analysis") / (stamp + "_comparison")).resolve()
    if output.exists():
        parser.error(f"Existing results are preserved; specify a new --output path: {output}")
    tables, matrices = defaultdict(list), []
    for folder in folders:
        analyze_run(folder, args.force_file, tolerances, tables, matrices)
    frames = {"guide": guide(args.force_file, tolerances), **{name: pd.DataFrame(rows) for name, rows in tables.items()}}
    frames = add_summaries(frames, folders, previous, args.comparison_force_file)
    series = combined_force_tables(folders, args.force_file)
    frames = {"guide": frames["guide"], **series, **{name: frame for name, frame in frames.items() if name != "guide"}}
    source_rows = series["force_alignment"][["source_workbook", "source_sha256"]].rename(
        columns={"source_workbook": "path", "source_sha256": "sha256"})
    frames["sources"] = pd.concat([frames["sources"], source_rows], ignore_index=True).drop_duplicates("path")
    output.mkdir(parents=True)
    write_excel(frames, output / "force_and_id_all_combinations.xlsx")
    plot_summary(frames, output, tolerances)
    plot_confusions(matrices, output)
    plot_force_details(frames, output)
    print(f"Analysis complete: {output}")
    print("fx/fy/fz: shared GT + force predictions by input/mode/model. force_common_metrics: RMSE/MAE on identical rows")
    print("One workbook: force_body_summary / id_body_summary for body results; force_file_comparison for previous/current comparisons")
    print("MATLAB: run plot_analysis.m and select this workbook; no separate workbook merge is needed.")


if __name__ == "__main__":
    main()
