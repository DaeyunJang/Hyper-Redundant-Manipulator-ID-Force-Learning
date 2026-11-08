"""저장된 predict 결과 → 힘/ID 지표·조합별 GT/예측 통합 Excel + 그림. 학습·추론·원본 CSV 재로딩 없음.

bash scripts/hrm_python.sh analysis.py results/predict/실행폴더
bash scripts/hrm_python.sh analysis.py 'results/predict/날짜시간_*'
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
    """파일별 RMSE를 표본 수와 제곱오차로 합친다. N=0은 평가 불가다."""
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
                raise ValueError(f"양수 force_support에 {prefix + key}가 없습니다.")
            weights = np.array([n for n, _ in supported])
            result[key] = float(np.sqrt(np.average(values ** 2, weights=weights)) if metric == "rmse"
                                else np.average(values, weights=weights))
    return result


def final_confusion(report: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Masked의 위치 argmax / Class0의 ID0 포함 최종 출력."""
    saved = report["system_confusion" if mode == "class0" else "conditional_confusion"]
    labels = np.asarray(saved["labels"], dtype=int)
    matrix = np.asarray(saved["matrix"], dtype=np.int64)
    if not saved.get("available", True) or matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"{mode}의 저장 혼동행렬이 없거나 크기가 맞지 않습니다.")
    return labels, matrix


def id_scores(labels: np.ndarray, matrix: np.ndarray, ids: set, tolerances: list[int]) -> dict:
    rows = np.isin(labels, list(ids)) & (labels > 0)
    counts = matrix[rows]
    support = counts.sum(axis=1)
    total = int(support.sum())
    distance = np.abs(labels[rows, None] - labels[None, :])
    result = {"n_loaded": total, "supported_ids": int((support > 0).sum())}
    for tolerance in tolerances:
        # ID0은 물리 위치가 아니므로 GT1과 가깝더라도 항상 실패다.
        hit = (counts * ((labels[None, :] > 0) & (distance <= tolerance))).sum(axis=1)
        result[f"hits_tol{tolerance}"] = int(hit.sum())
        result[f"accuracy_tol{tolerance}_pct"] = percent(hit.sum(), total)
        result[f"macro_tol{tolerance}_pct"] = float(100 * np.mean(hit[support > 0] / support[support > 0])) if total else np.nan
    for name, columns in (("id0", labels == 0), ("tip", labels == 18),
                          ("body", (labels >= 1) & (labels <= 17))):
        result[f"predicted_{name}_pct"] = percent(counts[:, columns].sum(), total)
    return result


def detection_scores(files: list[dict], mode: str) -> dict:
    """GT0 행은 원 접촉 ID를 잃으므로 body/tip 검출 통계는 파일별로 합친다."""
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
    # 사용자가 폴더를 날짜만 남겨 정리한 경우도 같은 입력으로 연결한다.
    return re.sub(r"^\d{8}_(?:\d{6}_)?", "", folder.name) or folder.name


def _force_series_sources(folder: Path) -> list[dict]:
    """현재 predictions와 연결된 organized를 읽고, 없으면 저장된 힘 시트를 사용합니다."""
    source = folder / "predictions.xlsx"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads((folder / "workbook_manifest.json").read_text())
    if manifest.get("sha256") != source_hash:
        raise ValueError(f"predictions.xlsx와 workbook_manifest의 SHA256이 다릅니다: {folder}")
    models = pd.DataFrame(manifest["model_sheets"])
    models = models.loc[models.branch.eq("force") & models.rows.gt(0)].copy()
    if models.empty or models.duplicated(["mode", "model"]).any():
        raise ValueError(f"대표 힘 모델이 없거나 중복됐습니다: {folder}")
    if not models.force_unit.eq("mN").all():
        raise ValueError(f"힘 Excel 단위는 mN이어야 합니다: {folder}")
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
                matches = guide.get("입력 SHA256") == source_hash
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
                        raise ValueError(f"organized 대표 힘 모델 정보가 다릅니다: {path} / {model.model}")
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
            raise ValueError(f"읽는 중 힘 Excel이 변경됐습니다: {path}")
        expected = selected.model.tolist()
        for axis, table in axes.items():
            if len(table) != len(rows) or list(table) != [f"gt_{axis}_mN", *expected]:
                raise ValueError(f"힘 시트와 원본 행/모델 목록이 다릅니다: {path} / {axis}")
            if not np.isfinite(table.to_numpy(dtype=float)).all():
                raise ValueError(f"힘 GT/예측에 비수치 값이 있습니다: {path} / {axis}")
        if (rows[KEYS].isna().any().any() or rows.duplicated(KEYS).any()
                or not np.isfinite(rows.source_row.to_numpy(float)).all()
                or not rows.source_row.eq(np.floor(rows.source_row)).all() or rows.source_row.lt(0).any()):
            raise ValueError(f"원본 파일/행 키가 없거나 중복/비정수입니다: {path}")
        result.append({"mode": mode, "models": selected, "rows": rows, "axes": axes,
                       "source_workbook": str(path), "source_sha256": before,
                       "predictions_sha256": source_hash,
                       "source_kind": "organized" if organized else "predictions"})
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError(f"읽는 중 predictions.xlsx가 변경됐습니다: {folder}")
    return result


def combined_force_tables(folders: list[Path], force_files=()) -> dict:
    """입력·방식·대표 모델을 열로 모읍니다. 실제 원본 행 교집합에서만 GT를 공유합니다."""
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
            # Excel의 표시 행 번호가 아니라 원 CSV의 파일/행을 기준으로 맞춥니다.
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
                    raise ValueError(f"같은 원본 행의 GT/ID/부하 정보가 다릅니다: {folder} / {item['mode']}") from exc
            if reference.empty:
                raise ValueError("모든 입력/방식에 공통인 힘 예측 행이 없습니다.")
            item.update(run=str(folder), input=name, prefix=prefix, rows=rows, axes=axes)
            entries.append(item)
        print(f"힘 시계열 읽음: {folder.name}", flush=True)
    if not entries:
        raise ValueError("힘 시계열을 합칠 predict 폴더가 없습니다.")
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
                raise ValueError(f"힘 예측 열 이름이 중복됩니다: {column}")
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
    print(f"통합 힘 시계열: {len(rows):,}행, 축당 GT 1열 + 예측 {len(columns)}열", flush=True)
    return result


def choose_models(comparison: pd.DataFrame, summary: dict, reports: dict) -> tuple[set, set]:
    # Force는 predict가 저장한 validation 선택을 그대로 사용한다.
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
        raise ValueError("저장된 validation 모델 선택을 확인할 수 없습니다.")
    return force, location


def analyze_run(folder: Path, force_files: list[str], tolerances: list[int], tables: dict, matrices: list) -> None:
    sources = []
    metrics = read_json(folder / "metrics.json", sources)
    if "all_combinations" not in metrics:
        raise ValueError("--train-dir로 predict한 전조합 결과 폴더를 지정하세요. 단일 checkpoint 결과는 지원하지 않습니다.")
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
        raise ValueError("한 실행의 force_scope가 일관되지 않습니다.")
    force_scope = scopes.pop()
    name = input_name(folder)
    run = str(folder)
    base = {"run": run, "input": name, "force_scope": force_scope}
    trial_map = {item["path"]: item for item in inputs}
    if force_files:
        missing = set(force_files) - {Path(path).name for path in trial_map}
        if missing:
            raise ValueError(f"{folder.name}: 요청한 force 파일이 없습니다: {sorted(missing)}")
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
        # ID/파일별 상세 표는 validation 선택 모델과 organized와 같은 대표분기다.
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
    print(f"분석: {folder.name} / force_scope={force_scope} / {len(reports)}조합", flush=True)


def guide(force_files: list[str], tolerances: list[int]) -> pd.DataFrame:
    return pd.DataFrame([
        ("먼저 볼 시트", "force_body_summary / id_body_summary: body 요약. force_file_detail: 선택한 별도 힘 파일. --previous 사용 시 force_file_comparison: 이전/현재 비교."),
        ("MATLAB", "plot_analysis.m에서 이 Excel 하나만 선택. 여러 predict 폴더는 이미 이 파일 안에 합쳐져 있으므로 별도 Excel 병합이 필요 없음."),
        ("단위", "힘 mN, 정확도 %, n은 평가행 수. 빈 셀은 평가 불가이며 0 오차/0%가 아님."),
        ("힘 범위", "저장된 force_scope 사용. tip 학습의 body는 N=0. all_single_contact는 각 접촉점의 실측 힘을 평가."),
        ("힘 평가 파일", ", ".join(force_files) or "저장된 전체 test 파일. 특정 독립 실험만 보려면 --force-file 사용."),
        ("부하 구분", "저장 라벨 사용. 현재 YAML을 읽거나 문턱을 다시 적용하지 않음. 힘은 all/loaded/unloaded로 분리."),
        ("힘 pooling", "RMSE=sqrt(sum(N_file*RMSE_file^2)/sum N_file); MAE=sum(N_file*MAE_file)/sum N_file."),
        ("XYZ 오차", "3축 성분 전체의 RMSE/MAE. 힘 벡터 norm 오차나 각 축 RMSE의 산술평균이 아님."),
        ("ID 평가", "GT 유부하만 분모. Masked=ID1–18 argmax, Class0=최종 ID0–18 argmax. 예측0은 모든 tolerance에서 실패."),
        ("ID tolerance", f"{tolerances}: 예측ID>0 AND abs(예측ID-GT)<=n. 확률 합이 아닌 정답률. ±n 성공은 tip/body 구분 성공과 다름."),
        ("Macro", "지원행이 있는 GT ID별 성공률의 동일 가중 평균. micro accuracy는 전체 행 기준."),
        ("검출", "Class0의 TP/FN/TN/FP 및 유부하 검출률, 무부하 정답률. Masked 자체 검출은 평가 불가."),
        ("힘 모델 선택", "predict summary.json의 validation 힘 우선 선택 쌍 그대로. Test 최소 오차로 재선택하지 않음."),
        ("ID 모델 선택", "validation location_macro_recall 최대, CE/힘 RMSE/모델명 순 동률 처리. Class0 validation 점수는 ID0 제외 후보이며 test 보고는 최종 출력."),
        ("선택 모델", "force_selected와 id_selected는 서로 다른 bundle일 수 있음. 두 성능을 한 bundle의 결과로 합치지 말 것."),
        ("전체 조합", "models, force, location, detection 시트는 모든 저장 조합. force는 전체/body/tip × all/loaded/unloaded."),
        ("상세 및 대표모델", "force_by_file/force_by_id/id_by_id는 validation 선택 + organized와 동일한 고정 파트너 대표분기. representative_* 열로 선택."),
        ("혼동행렬", "ID validation 선택 모델. GT0 행은 Class0 무부하 전체; body/tip 무부하별 통계는 detection 시트."),
        ("ID 파일 범위", "ID는 모든 저장 test를 평가. --force-file은 힘 분석에만 적용; 저장 혼동행렬만으로 임의 파일별 ±n는 계산 불가."),
        ("비교 해석", "기존 집계/그림의 입력별 N과 학습 자료가 다를 수 있음. fx/fy/fz는 원본 끝행 교집합으로 비교하지만 입력창 전체 이력이나 학습 자료까지 맞추지는 않음."),
        ("행별 힘 비교", "fx/fy/fz: 공통 GT 1열 + 입력조합_방식_힘모델_mN 예측 열. Masked/Class0 모두 포함. force_rows로 원CSV/행/ID를 찾고 force_columns에서 실제 checkpoint와 위치 파트너 확인."),
        ("시계열의 출처", "현재 predictions.xlsx hash에 연결된 최신 organized 힘 Excel을 읽음. organized가 없거나 오래됐으면 저장 predictions.xlsx의 대표 힘 시트 사용. 학습/추론/보간/재라벨 없음."),
        ("공통행 힘 지표", "force_common_metrics는 fx/fy/fz의 동일 공통행에서 재계산한 대표 힘 모델 RMSE/MAE. force_alignment에 입력/방식별 사용 원Excel/hash/행 수 기록. 기존 전체 조합 집계 및 그림의 분모는 바꾸지 않음."),
        ("대표 힘 모델 범위", "행별 비교는 입력별 2방식×11힘모델 등 predictions.xlsx에 저장된 대표 분기. 모든 힘×위치 조합의 행별 예측을 의미하지 않으며 validation 최선모델과 다를 수 있음."),
        ("이전 비교", "--previous는 동일 입력명/방식/파일을 연결. predictions.csv의 평가 원행·GT·부하별 집합과 파일 hash가 같을 때만 오차 차이를 표시. 시간창 이력/학습자료/선택모델까지 통제한 인과 비교는 아님."),
        ("추가 진단", "힘 크기별 ID 정확도와 시계열 그래프는 별도 분석. fx/fy/fz의 공통행 힘 예측은 이 Excel에서 바로 그래프로 그릴 수 있음."),
    ], columns=["항목", "설명"])


def force_file_results(folder: Path, filename: str, experiment: str, sources: list) -> list[dict]:
    """별도 힘 시험의 validation 선택 모델. 이전/현재 비교용 행·GT 지문도 보존한다."""
    reports = read_json(folder / "metrics.json", sources)["all_combinations"]
    summary = read_json(folder / "summary.json", sources)
    inputs = read_json(folder / "input_manifest.json", sources)
    files = [item for item in inputs if Path(item["path"]).name == filename]
    if not files:
        print(f"별도 힘 파일 생략: {folder.name} / {filename} 없음", flush=True)
        return []
    if len(files) != 1:
        raise ValueError(f"동일 이름의 시험 파일이 여러 개입니다: {folder} / {filename}")
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
                    raise ValueError(f"predictions.csv와 summary의 선택 모델이 다릅니다: {folder}")
                numeric = [f"{kind}_{axis}_N" for kind in ("true", "pred") for axis in AXES[:3]]
                part = part[np.isfinite(part[numeric].to_numpy(dtype=float)).all(axis=1)]
                if state != "all":
                    part = part[part.load_state_gt.eq(state)]
                part = part.sort_values("source_row")
                if len(part) != score["n"] or part.source_row.duplicated().any():
                    raise ValueError(f"힘 평가행 수/중복이 저장 지표와 맞지 않습니다: {folder} / {mode} / {state}")
                data = part[["source_row", "true_fx_N", "true_fy_N", "true_fz_N"]].to_numpy(dtype="<f8")
                fingerprint = hashlib.sha256(data.tobytes()).hexdigest() if len(part) else ""
            result.append({"input": input_name(folder), "run": str(folder), "experiment": experiment,
                           "mode": mode, "force_model": chosen["force_model"], "location_model": chosen["location_model"],
                           "file": filename, "load_state": state, **score, "source_sha256": trial["sha256"],
                           "evaluation_rows_gt_sha256": fingerprint})
    return result


def add_summaries(frames: dict, folders: list[Path], previous: list[Path], filename: str) -> dict:
    """기존 상세표를 유지하고 읽기 쉬운 요약/동일 파일 전후 비교를 앞에 둔다."""
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
                raise ValueError("이전 비교는 입력당 한 실행씩 지정하세요. 동일 입력 이름의 실행이 여러 개입니다.")
            joined = current.merge(old, on=keys, suffixes=("_new", "_old"), how="inner", validate="one_to_one")
            if joined.empty:
                raise ValueError("이전/현재에 동일 입력명·방식·별도 힘 파일이 없습니다.")
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
    # 행별 힘 예측 수백만 셀을 메모리에 쌓지 않고 한 행씩 저장합니다.
    options = {"constant_memory": True, "strings_to_formulas": False, "strings_to_urls": False}
    with xlsxwriter.Workbook(destination, options) as book:
        book.use_zip64()
        header = book.add_format({"bold": True, "text_wrap": True, "bg_color": "#173D59", "font_color": "white"})
        number = book.add_format({"num_format": "0.000000"})
        for name, frame in tables.items():
            if len(frame) >= 1048576 or len(frame.columns) > 16384:
                raise ValueError(f"{name}: Excel 시트 크기를 초과했습니다.")
            # 점수는 왼쪽에, 긴 원본 경로와 선택 여부 등 출처 정보는 오른쪽에 둔다.
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
            raise ValueError(f"결과 폴더를 찾을 수 없습니다: {pattern}")
        for path in matches:
            folder = Path(path).resolve()
            if not (folder / "metrics.json").is_file():
                raise ValueError(f"metrics.json이 있는 predict 실행 폴더를 지정하세요: {folder}")
            if folder not in folders:
                folders.append(folder)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predict_dirs", nargs="+", help="predict 결과 폴더 하나/여러 개 또는 따옴표로 감싼 glob")
    parser.add_argument("--output", type=Path, help="새 분석 폴더; 기본 results/analysis/날짜시간_comparison")
    parser.add_argument("--force-file", nargs="+", default=[], help="힘 평가에 포함할 CSV 파일명. 생략하면 전체 test; ID 평가는 전체 유지")
    parser.add_argument("--id-tolerances", nargs="+", type=int, default=[2, 3], help="기본 2 3; 정확 ID(0)와 ±1도 항상 계산")
    parser.add_argument("--previous", nargs="+", default=[], help="선택: 이전 predict 폴더/glob. 동일 입력의 별도 힘 파일 전후 비교")
    parser.add_argument("--comparison-force-file", default="summary_18_test_2.csv",
                        help="별도 힘 시험 시트/그림에 사용할 파일명. 기본 summary_18_test_2.csv; --force-file 집계 필터와 별개")
    args = parser.parse_args()
    if any(n < 0 or n > 17 for n in args.id_tolerances):
        parser.error("ID tolerance는 0–17이어야 합니다.")
    tolerances = sorted({0, 1, *args.id_tolerances})
    try:
        folders = expand_folders(args.predict_dirs)
        previous = expand_folders(args.previous)
    except ValueError as exc:
        parser.error(str(exc))
    stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or Path("results/analysis") / (stamp + "_comparison")).resolve()
    if output.exists():
        parser.error(f"기존 결과를 보존합니다. 새 --output 경로를 지정하세요: {output}")
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
    print(f"분석 완료: {output}")
    print("fx/fy/fz: 공통 GT + 입력조합_방식_모델별 힘 예측. force_common_metrics: 같은 행의 RMSE/MAE")
    print("Excel 하나: force_body_summary / id_body_summary → body, force_file_comparison → 이전 비교")
    print("MATLAB: plot_analysis.m 실행 후 이 Excel 선택. 별도 Excel 병합 불필요.")


if __name__ == "__main__":
    main()
