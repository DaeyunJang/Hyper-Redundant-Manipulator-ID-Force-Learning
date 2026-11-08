"""predictions.xlsx를 GT + 모델별 열의 비교 Excel로 정리합니다. 학습/추론은 하지 않습니다.

실행: python organize_results.py results/predict/날짜_시간/predictions.xlsx --id-tolerance 2
여러 Excel/와일드카드를 주면 입력 조합 모두에 있는 공통 원본 행만 같은 순서로 정리합니다.
수정: force_tables는 힘 시트, id_tables는 ID 시트, write_workbook은 표시 형식입니다.
입력 Excel만 필요하며 모델·CSV·현재 train.yaml을 다시 읽지 않습니다.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import glob
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import xlsxwriter

DEFAULT_REGIONS = "1-4,5-10,11-18"  # 기존 비교의 고정 구간. test로 다시 선택하지 않음.
AXES = ("fx", "fy", "fz")
KEYS = ["source_file", "source_row"]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


class WorkbookReader:
    """프로젝트 Excel의 inline/shared 문자열과 숫자를 읽습니다. 추가 Excel 의존성 없음."""

    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        try:
            z = self.archive
            rels = {e.attrib["Id"]: e.attrib["Target"] for e in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
            self.sheets = {}
            for sheet in ET.fromstring(z.read("xl/workbook.xml")).findall("m:sheets/m:sheet", NS):
                target = rels[sheet.attrib[RID]]
                self.sheets[sheet.attrib["name"]] = target.lstrip("/") if target.startswith("/") else "xl/" + target
            self.strings = []
            if "xl/sharedStrings.xml" in z.namelist():
                self.strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        except BaseException:
            self.close()
            raise

    def close(self):
        self.archive.close()

    def read(self, name):
        headers, rows, indices = None, [], {}
        with self.archive.open(self.sheets[name]) as stream:
            for _, row in ET.iterparse(stream, events=["end"]):
                if row.tag != "{" + NS["m"] + "}row":
                    continue
                values = {}
                for cell in row:
                    letters = cell.attrib["r"].rstrip("0123456789")
                    if letters not in indices:
                        index = 0
                        for ch in letters:
                            index = index * 26 + ord(ch) - 64
                        indices[letters] = index - 1
                    node = cell.find("m:v", NS)
                    kind = cell.attrib.get("t")
                    if kind == "inlineStr":
                        value = "".join(cell.find("m:is", NS).itertext())
                    elif kind == "e":
                        raise ValueError(f"{name}/{cell.attrib['r']}: Excel 오류 셀")
                    elif node is None or node.text is None:
                        value = None
                    elif kind == "s":
                        value = self.strings[int(node.text)]
                    elif kind == "str":
                        value = node.text
                    else:
                        value = float(node.text)
                    values[indices[letters]] = value
                if headers is None:
                    headers = [values.get(i) for i in range(max(values) + 1)]
                    if not all(isinstance(v, str) for v in headers) or len(set(headers)) != len(headers):
                        raise ValueError(f"{name}: 열 이름이 없거나 중복됐습니다.")
                else:
                    if int(row.attrib["r"]) != len(rows) + 2:
                        raise ValueError(f"{name}: 데이터 중간에 빈 행이 있습니다.")
                    rows.append([values.get(i) for i in range(len(headers))])
                row.clear()
        return pd.DataFrame(rows, columns=headers)


def parse_regions(text):
    groups = []
    for token in text.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", token)
        if match is None:
            raise ValueError("구간 예: --regions 1-4,5-10,11-18")
        start, stop = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= stop <= 18:
            raise ValueError("구간은 ID1–18 범위 안이어야 합니다.")
        groups.append(list(range(start, stop + 1)))
    if [i for group in groups for i in group] != list(range(1, 19)):
        raise ValueError("구간은 ID1–18을 순서대로 중복·누락 없이 나눠야 합니다.")
    return groups


def align_rows(frame, reference):
    """행 순서가 달라도 원본 파일/행으로 대응하고, GT 차이는 숨기지 않습니다."""
    for table in (frame, reference):
        if table[KEYS].isna().any().any() or table.duplicated(KEYS).any():
            raise ValueError("원본 파일/행 키가 없거나 중복됐습니다.")
    target = pd.MultiIndex.from_frame(reference[KEYS])
    source = frame.set_index(KEYS)
    if len(source) != len(target) or len(target.difference(source.index)):
        raise ValueError("모델 간 원본 행 집합이 다릅니다. 서로 다른 결과를 합칠 수 없습니다.")
    frame = source.reindex(target).reset_index()
    truth = [c for c in reference if c.startswith("gt_") or c in
             ["ground_truth_id", "contact_id_original", "contact_id_effective", "load_state", "loaded_gt"]]
    pd.testing.assert_frame_equal(frame[truth], reference[truth], check_dtype=False, check_exact=True)
    return frame


def row_table(frame):
    states = frame.load_state.to_numpy()
    return pd.DataFrame({
        "excel_row": np.arange(len(frame)) + 2,
        "source_file": frame.source_file.to_numpy(),
        "source_row": frame.source_row.to_numpy(),
        "source_csv_line": frame.source_row.to_numpy() + 2,
        "recorded_contact_id": frame.contact_id_original.to_numpy(),
        "gt_id": frame.ground_truth_id.to_numpy(),
        "load_state_gt": states,
        "loaded_gt": np.where(states == "loaded", 1, np.where(states == "unloaded", 0, np.nan)),
    })


def force_tables(frames, reference):
    tables = {"force_all": reference[[f"gt_{a}_mN" for a in AXES]].copy()}
    tables.update({a: pd.DataFrame({f"gt_{a}_mN": reference[f"gt_{a}_mN"]}) for a in AXES})
    summary = []
    for model, frame in frames.items():
        pred = frame[[f"pred_{a}_mN" for a in AXES]].to_numpy(float)
        gt = frame[[f"gt_{a}_mN" for a in AXES]].to_numpy(float)
        if not np.isfinite(pred).all() or not np.isfinite(gt).all():
            raise ValueError(f"{model}: 힘 GT/예측에 비수치 값이 있습니다.")
        error = pred - gt
        row = {"model": model, "support": len(frame)}
        for i, axis in enumerate(AXES):
            tables["force_all"][f"pred_{axis}_{model}_mN"] = pred[:, i]
            tables[axis][model] = pred[:, i]
            row[f"rmse_{axis}_mN"] = np.sqrt(np.mean(error[:, i] ** 2))
            row[f"mae_{axis}_mN"] = np.mean(abs(error[:, i]))
        row.update(rmse_xyz_mN=np.sqrt(np.mean(error ** 2)), mae_xyz_mN=np.mean(abs(error)))
        summary.append(row)
    tables["_summary"] = pd.DataFrame(summary)
    tables["_rows"] = row_table(reference)
    return tables


def validate_id_tolerance(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 17:
        raise ValueError("id_tolerance은 0–17 정수여야 합니다. 0은 정확 ID 일치입니다.")
    return int(value)


def id_tables(modes, reference, regions, id_tolerance=2):
    id_tolerance = validate_id_tolerance(id_tolerance)
    tolerance_key = f"within{id_tolerance}"
    actual = reference.contact_id_effective.to_numpy(float)
    gt = reference.ground_truth_id.to_numpy(float)
    states = reference.load_state.to_numpy()
    if not np.isin(actual, np.arange(1, 19)).all() or not np.isin(states, ["loaded", "unloaded"]).all():
        raise ValueError("ID 결과는 ID1–18 및 확정된 loaded/unloaded 행이어야 합니다.")
    loaded = states == "loaded"
    if not np.array_equal(gt, np.where(loaded, actual, 0)):
        raise ValueError("GT ID와 부하 라벨이 일치하지 않습니다.")
    base = pd.DataFrame({"gt_id": gt, "loaded_gt": loaded.astype(int)})
    lookup = {i: g + 1 for g, members in enumerate(regions) for i in members}
    true_region = np.array([lookup[i] for i in actual])
    region_base = pd.DataFrame({"gt_region": np.where(loaded, true_region, 0), "loaded_gt": loaded.astype(int)})
    tables, summary = {}, []
    mean = lambda values: float(np.mean(values)) if len(values) else np.nan
    for mode, frames in modes.items():
        prefix = "loss-mask" if mode == "masked" else mode
        names = {"id": prefix + "_id", "error": prefix + "_ID_error", "within": prefix + "_" + tolerance_key,
                 "region": prefix + "_region"}
        if mode == "class0":
            names.update(conditional="class0_id_conditional", error="class0_ID_error_conditional",
                         within=f"class0_{tolerance_key}_conditional", region="class0_region_conditional",
                         loaded="class0_loaded", within_detected=f"class0_{tolerance_key}_detected",
                         region_detected="class0_region_detected")
        for key, name in names.items():
            tables[name] = (region_base if key.startswith("region") else base).copy()
        for model, frame in frames.items():
            active = [i for i in range(1, 19) if not frame[f"p{i}"].isna().all()]
            prob = frame[[f"p{i}" for i in active]].to_numpy(float)
            if not active or not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
                raise ValueError(f"{mode}/{model}: ID 확률이 유효하지 않습니다.")
            conditional = np.array(active)[prob.argmax(axis=1)]
            final, detected = conditional, None
            if mode == "class0":
                full = np.column_stack([frame.p0, prob])
                if not np.isfinite(full).all() or ((full < 0) | (full > 1)).any():
                    raise ValueError(f"{model}: class0 확률이 유효하지 않습니다.")
                final = np.array([0] + active)[full.argmax(axis=1)]
                detected = final != 0
            else:
                full = prob
                if frame.p0.notna().any() or frame.final_id.notna().any():
                    raise ValueError("Masked 결과에 무부하 검출 출력이 들어 있습니다.")
            if not np.allclose(full.sum(1), 1, rtol=0, atol=2e-6):
                raise ValueError(f"{mode}/{model}: 확률 합이1이 아닙니다.")
            if not np.array_equal(conditional, frame.conditional_id) or not np.array_equal(final, frame.pred_id):
                raise ValueError(f"{mode}/{model}: 예측 ID와 확률이 일치하지 않습니다.")
            if mode == "class0" and not np.array_equal(final, frame.final_id):
                raise ValueError(f"{model}: class0 최종 ID가 일치하지 않습니다.")
            distance = abs(conditional - actual)
            within = distance <= id_tolerance
            sums = np.column_stack([prob[:, np.isin(active, members)].sum(1) for members in regions])
            region = sums.argmax(1) + 1
            values = {"id": final, "error": np.where(loaded, distance, np.nan),
                      "within": np.where(loaded, within.astype(int), np.nan), "region": region}
            if detected is not None:
                values.update(conditional=conditional, loaded=detected.astype(int),
                              within_detected=np.where(loaded, (detected & within).astype(int), np.nan),
                              region_detected=np.where(detected, region, 0))
            for key, values_array in values.items():
                tables[names[key]][model] = values_array
            for scope, include in [("all_ID1_18", np.ones(len(frame), bool)), ("body_ID1_17", actual < 18), ("tip_ID18", actual == 18)]:
                l, u = include & loaded, include & ~loaded
                row = {"mode": mode, "model": model, "scope": scope, "id_tolerance": id_tolerance,
                       "loaded_support": int(l.sum()),
                       "unloaded_support": int(u.sum()), "unsupported_loaded": int((l & ~np.isin(actual, active)).sum()),
                       "exact_ID_accuracy": mean(distance[l] == 0), "within1_accuracy": mean(distance[l] <= 1),
                       "within2_accuracy": mean(distance[l] <= 2), "mean_abs_ID_error": mean(distance[l]),
                       "region_conditional_accuracy": mean(region[l] == true_region[l]),
                       "tip_body_conditional_accuracy": mean((conditional[l] == 18) == (actual[l] == 18))}
                row[tolerance_key + "_accuracy"] = mean(within[l])
                for name, success in {"within2": distance <= 2, tolerance_key: within,
                                      "region": region == true_region}.items():
                    row[name + "_ID_macro_accuracy"] = mean([mean(success[l & (actual == i)]) for i in np.unique(actual[l])])
                row.update(loaded_recall=mean(detected[l]) if detected is not None else np.nan,
                           unloaded_recall=mean(~detected[u]) if detected is not None else np.nan,
                           false_positive_rate=mean(detected[u]) if detected is not None else np.nan,
                           within2_detected_accuracy=mean((detected & (distance <= 2))[l]) if detected is not None else np.nan,
                           region_detected_accuracy=mean((detected & (region == true_region))[l]) if detected is not None else np.nan)
                row[tolerance_key + "_detected_accuracy"] = mean((detected & within)[l]) if detected is not None else np.nan
                summary.append(row)
    tables["_summary"] = pd.DataFrame(summary)
    tables["_rows"] = row_table(reference)
    return tables


def write_workbook(path, tables):
    """GT는 왼쪽, 모델은 열로 배치합니다. 숫자는 그대로, 해당없음은 빈칸으로 저장합니다."""
    with path.open("xb") as handle:
        book = xlsxwriter.Workbook(handle, {"constant_memory": True, "strings_to_formulas": False, "strings_to_urls": False})
        book.use_zip64()
        header = book.add_format({"bold": True, "bg_color": "#173D59", "font_color": "white", "text_wrap": True})
        number = book.add_format({"num_format": "0.000000"})
        integer = book.add_format({"num_format": "0"})
        percent = book.add_format({"num_format": "0.00%"})
        wrap = book.add_format({"text_wrap": True, "valign": "top"})
        good = book.add_format({"bg_color": "#D9EAD3"})
        bad = book.add_format({"bg_color": "#F4CCCC"})
        for name, frame in tables.items():
            if len(frame) >= 1048576 or len(frame.columns) > 16384:
                raise ValueError(f"{name}: Excel 시트 크기를 초과했습니다.")
            sheet = book.add_worksheet(name)
            sheet.freeze_panes(1, 2 if "loaded_gt" in frame else 1)
            sheet.set_row(0, 32)
            sheet.write_row(0, 0, list(frame), header)
            for j, column in enumerate(frame):
                fmt = percent if any(s in column for s in ("accuracy", "recall", "rate")) else number if "mN" in column or name in AXES or column == "mean_abs_ID_error" else integer
                sheet.set_column(j, j, 21, fmt)
            if name == "_guide":
                sheet.set_column(0, 0, 30, wrap)
                sheet.set_column(1, 1, 105, wrap)
            for i, row in enumerate(frame.itertuples(index=False, name=None), 1):
                values = []
                for value in row:
                    if isinstance(value, np.generic):
                        value = value.item()
                    if pd.isna(value):
                        value = None
                    elif isinstance(value, float) and not np.isfinite(value):
                        raise ValueError(f"{name}: 무한대 값은 저장할 수 없습니다.")
                    values.append(value)
                sheet.write_row(i, 0, values)
                if name == "_guide":
                    sheet.set_row(i, 60)
            sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            if re.search(r"_within\d+(?:_|$)", name) and len(frame):
                for value, fmt in [(1, good), (0, bad)]:
                    sheet.conditional_format(1, 2, len(frame), len(frame.columns) - 1,
                                             {"type": "formula", "criteria": f"AND(ISNUMBER(C2),C2={value})", "format": fmt})
        book.close()


def organize_workbook(source, output=None, regions=DEFAULT_REGIONS, id_tolerance=2,
                      *, common_references=None, alignment=None):
    id_tolerance = validate_id_tolerance(id_tolerance)
    source = Path(source).expanduser().resolve()
    groups = parse_regions(regions)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if alignment is not None:
        expected = alignment.loc[alignment.source_workbook == str(source), "source_sha256"]
        if expected.empty or not expected.eq(source_hash).all():
            raise ValueError(f"공통 행 확인 후 원본 Excel이 변경됐습니다: {source}")
    reader = WorkbookReader(source)
    branches = {"force": {}, "id": {}}
    references, skipped = {}, []
    try:
        models = reader.read("_models") if "_models" in reader.sheets else pd.DataFrame()
        for sheet in reader.sheets:
            match = re.fullmatch(r"(force|id)_(masked|class0)_(.+)", sheet)
            if match is None:
                continue
            branch, mode, model = match.groups()
            frame = reader.read(sheet)
            if frame.empty:
                skipped.append(sheet)
                continue
            if branch not in references:
                references[branch] = frame
            frame = align_rows(frame, references[branch])
            if common_references is not None:
                target = common_references[branch]
                keys = pd.MultiIndex.from_frame(target[KEYS])
                # 원본 시트끼리의 정합성을 먼저 검사한 뒤 공통 끝행만 선택합니다.
                frame = align_rows(frame.set_index(KEYS).loc[keys].reset_index(), target)
            branches[branch].setdefault(mode, {})[model] = frame
            print(f"읽음: {sheet} ({len(frame):,}행)", flush=True)
    finally:
        reader.close()
    if not references:
        raise ValueError("GT가 있는 force_*/id_* 결과가 없습니다. 프로젝트 predictions.xlsx를 지정하세요.")
    if common_references is not None:
        references = common_references
    results = {}
    for mode, frames in branches["force"].items():
        label = "loss-mask" if mode == "masked" else mode
        results[f"force-estimation_{label}_id-tol-{id_tolerance}_results.xlsx"] = force_tables(frames, references["force"])
    if branches["id"]:
        results[f"id-estimation_all-models_id-tol-{id_tolerance}_results.xlsx"] = id_tables(branches["id"], references["id"], groups, id_tolerance)
    guide = pd.DataFrame([
        ("입력", str(source)), ("입력 SHA256", source_hash),
        ("처리", "저장된 Excel만 재정리했습니다. 학습·추론·모델 선택·현재 YAML 재라벨은 하지 않습니다."),
        ("힘", "첫 force_all 시트에 GT Fx/Fy/Fz와 모든 모델의 XYZ 예측 열을 모읍니다. fx/fy/fz 시트도 유지하며 첫 열은 GT, 다음 열들은 모델 예측입니다. 단위mN, 회귀값/부호/부하별 힘을 그대로 보존합니다."),
        ("ID", "gt_id는 유부하ID/무부하0, loaded_gt는 실측 기준 유부하1/무부하0입니다. class0_id의0은 모델이 무부하로 판단한 것입니다."),
        ("조건부 ID", "class0_id_conditional은 p0를 제외한 위치 후보입니다. Masked도 접촉ID 후보만 출력하며 자체 무부하 검출은 없습니다."),
        ("ID 허용 거리 n", id_tolerance),
        (f"ID_error / within{id_tolerance}", f"실제 유부하에서 위치 후보와 정답의 절대 ID차이를 평가합니다. 오차≤{id_tolerance}이면1, 아니면0입니다. 실제 무부하 행은 빈칸입니다. n은 평가 기준이며 학습이나 예측을 바꾸지 않습니다."),
        (f"within{id_tolerance}_detected", f"실제 유부하에서 접촉 검출과 ±{id_tolerance} 위치 성공을 모두 만족해야1입니다. 접촉을 놓치면0이며, 무부하 GT 행은 빈칸입니다."),
        ("class0_loaded", "각 모델의 유부하1/무부하0 판정입니다. Masked에는 이 출력을 임의로 만들지 않습니다."),
        ("구간", f"고정 비교 구간: {regions}. 순서대로 구간1부터 번호를 붙이고 ID 확률 합으로 선택합니다. test로 구간을 선택하지 않습니다. --regions로 명시할 수 있습니다."),
        ("region_detected", "Class0가 접촉을 검출하면 확률합 구간, 무부하를 예측하면0입니다. 구간 후보는 최대확률ID가 속한 구간과 다를 수 있습니다."),
        ("Tip/body", "정답17에 예측18이면 ±1이지만 tip/body 구별은 오류입니다. _summary에서 별도 평가합니다. 구간에 tip이 포함되면 tip/body 구별 성능으로 해석하지 않습니다."),
        ("_summary", "조건부/검출포함 위치 지표는 전체GT유부하가 분모입니다. 무부하 오검출률은GT무부하가 분모입니다. Body/Tip은 ID범위이며 독립실험 여부는 원 실험기록을 따릅니다."),
        ("원본 대응", "각 파일 안의 데이터 시트는 동일 행=동일 원본입니다. _rows의 source_row는 헤더 제외0기준, source_csv_line은 헤더 포함1기준입니다. 힘/ID 파일의 행 수는 서로 다를 수 있습니다."),
        ("제외/불확실", "원본 Excel에서 이미 제외된 무효·warmup·불확실 ID 행을 복구하지 않습니다. 힘 표의 불확실 부하 행은 유효 회귀값이라 유지하며 loaded_gt는 빈칸입니다."),
        ("모델", "모델 개수·순서는 입력 시트를 따릅니다. 원본의 대표 파트너/checkpoint는 _models에 보존하며 새로 모델을 선택하지 않습니다."),
        ("빈 원본 시트", ", ".join(skipped) or "없음"),
    ], columns=["항목", "설명"])
    if alignment is not None:
        guide = pd.concat([guide, pd.DataFrame([
            ("공통 행 비교", "_alignment의 모든 입력 Excel에 존재하는 source_file/source_row의 교집합입니다. 힘/ID별로 첫 입력의 행 순서를 따르며 공통 GT도 대조합니다. 행 수만 잘라 맞추거나 보간하지 않습니다."),
            ("공통 행 지표", "_summary의 RMSE/MAE와 ID 지표는 이 파일에 남긴 공통 행으로 다시 계산했습니다. 원래 metrics.json 및 analysis.py의 집계 결과는 바꾸지 않습니다."),
            ("비교 범위", "저장 Excel의 대표 모델 분기만 비교합니다. 같은 예측 끝행이어도 결측 연결 후 과거 29행 이력은 입력별로 다를 수 있습니다. 공통 행 필터는 재학습이나 동일 이력으로 재예측하는 처리가 아닙니다."),
        ], columns=["항목", "설명"])], ignore_index=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("정리 중 원본 Excel이 변경됐습니다.")
    destination = Path(output).expanduser().resolve() if output else source.parent / (
        "organized_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_id-tol-{id_tolerance}")
    destination.mkdir(parents=True, exist_ok=False)
    for name, tables in results.items():
        tables["_guide"] = guide
        if alignment is not None:
            tables["_alignment"] = alignment
        if not models.empty:
            tables["_models"] = models
        write_workbook(destination / name, tables)
        print(f"저장: {destination / name}", flush=True)
    return destination


def organize_workbooks(sources, output=None, regions=DEFAULT_REGIONS, id_tolerance=2):
    """여러 입력의 공통 예측 끝행을 찾고, 기존 정리/지표 함수를 그대로 사용합니다."""
    id_tolerance = validate_id_tolerance(id_tolerance)
    parse_regions(regions)
    sources = list(dict.fromkeys(Path(p).expanduser().resolve() for p in sources))
    if not sources:
        raise ValueError("입력 Excel이 없습니다.")
    if len(sources) == 1:
        return [organize_workbook(sources[0], output, regions, id_tolerance)]
    references, records = {}, []
    # 전체 44개 시트를 여섯 입력 모두 동시에 메모리에 올리지 않습니다.
    for source in sources:
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        reader = WorkbookReader(source)
        found = {}
        try:
            for sheet in reader.sheets:
                match = re.fullmatch(r"(force|id)_(masked|class0)_(.+)", sheet)
                if match is None or match[1] in found:
                    continue
                frame = reader.read(sheet)
                if frame.empty:
                    continue
                branch = match[1]
                found[branch] = align_rows(frame, frame)
        finally:
            reader.close()
        if not found or (references and set(found) != set(references)):
            raise ValueError(f"비교 Excel의 force/id 분기 구성이 다릅니다: {source}")
        for branch, frame in found.items():
            if branch not in references:
                references[branch] = frame
            else:
                reference = references[branch]
                keep = pd.MultiIndex.from_frame(reference[KEYS]).isin(pd.MultiIndex.from_frame(frame[KEYS]))
                reference = reference.loc[keep].reset_index(drop=True)
                if reference.empty:
                    raise ValueError(f"{branch}: 모든 입력에 공통인 예측 행이 없습니다.")
                keys = pd.MultiIndex.from_frame(reference[KEYS])
                align_rows(frame.set_index(KEYS).loc[keys].reset_index(), reference)
                references[branch] = reference
            records.append({"source_workbook": str(source), "source_sha256": source_hash,
                            "branch": branch, "original_rows": len(frame)})
        print(f"공통 행 확인: {source.parent.name} " +
              ", ".join(f"{k} {len(v):,}행" for k, v in found.items()), flush=True)
    alignment = pd.DataFrame(records)
    alignment["common_rows"] = alignment.branch.map({k: len(v) for k, v in references.items()})
    alignment["excluded_rows"] = alignment.original_rows - alignment.common_rows
    print("최종 공통 행: " + ", ".join(f"{k} {len(v):,}행" for k, v in references.items()), flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output:
        root = Path(output).expanduser().resolve()
        if root.exists():
            raise FileExistsError(f"새 출력 폴더를 지정하세요: {root}")
        destinations = [root / source.parent.name for source in sources]
    else:
        destinations = [source.parent / f"organized_common_{stamp}_id-tol-{id_tolerance}" for source in sources]
    if len(set(destinations)) != len(destinations) or any(p.exists() for p in destinations):
        raise ValueError("출력 폴더가 중복됩니다. 입력 Excel을 서로 다른 폴더에 두세요.")
    return [organize_workbook(source, destination, regions, id_tolerance,
                              common_references=references, alignment=alignment)
            for source, destination in zip(sources, destinations)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", help="predictions.xlsx 경로 또는 와일드카드. 여러 파일이면 공통 행으로 비교")
    parser.add_argument("--output", help="새 출력 폴더. 여러 입력이면 그 아래 입력 폴더별 저장. 생략하면 각 입력 옆 organized_[common_]날짜_시간_id-tol-N")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="고정 구간. 기본: %(default)s")
    parser.add_argument("--id-tolerance", type=int, choices=range(18), default=2,
                        help="±n ID 허용 거리(0–17). 0은 정확 ID 일치. 기본: %(default)s")
    args = parser.parse_args(argv)
    try:
        sources = []
        for pattern in args.input:
            expanded = str(Path(pattern).expanduser())
            matches = sorted(glob.glob(expanded)) if glob.has_magic(expanded) else [expanded]
            if not matches:
                raise ValueError(f"일치하는 입력 Excel이 없습니다: {pattern}")
            sources.extend(matches)
        destinations = organize_workbooks(sources, args.output, args.regions, args.id_tolerance)
    except (ValueError, AssertionError, KeyError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"결과 정리 실패: {error}\n")
    for destination in destinations:
        print(f"결과 정리 완료: {destination}")


if __name__ == "__main__":
    main()
