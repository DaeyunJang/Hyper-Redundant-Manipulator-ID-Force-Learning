"""Reorganize predictions.xlsx into comparison workbooks with GT and per-model columns; no training or inference.

Usage: python organize_results.py results/predict/date_time/predictions.xlsx --id-tolerance 2
With multiple workbooks or wildcards, align source rows shared by every input configuration.
Edit force_tables for force sheets, id_tables for ID sheets, and write_workbook for formatting.
Only the input workbooks are read; models, CSV files and the current train.yaml are not reloaded.
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

DEFAULT_REGIONS = "1-4,5-10,11-18"  # Fixed regions from the existing comparison; never reselected using test data.
AXES = ("fx", "fy", "fz")
KEYS = ["source_file", "source_row"]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


class WorkbookReader:
    """Read inline/shared strings and numbers from project workbooks without extra Excel dependencies."""

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
                        raise ValueError(f"{name}/{cell.attrib['r']}: Excel error cell")
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
                        raise ValueError(f"{name}: Column names are missing or duplicated.")
                else:
                    if int(row.attrib["r"]) != len(rows) + 2:
                        raise ValueError(f"{name}: An empty row occurs within the data.")
                    rows.append([values.get(i) for i in range(len(headers))])
                row.clear()
        return pd.DataFrame(rows, columns=headers)


def parse_regions(text):
    groups = []
    for token in text.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", token)
        if match is None:
            raise ValueError("Example regions: --regions 1-4,5-10,11-18")
        start, stop = int(match[1]), int(match[2] or match[1])
        if not 1 <= start <= stop <= 18:
            raise ValueError("Regions must be within ID1–18.")
        groups.append(list(range(start, stop + 1)))
    if [i for group in groups for i in group] != list(range(1, 19)):
        raise ValueError("Regions must cover ID1–18 in order without overlaps or gaps.")
    return groups


# AUTHOR: Daeyun
# Align source rows and compute force/localization comparison tables.
def align_rows(frame, reference):
    """Align by source file/row regardless of order, and report GT mismatches."""
    for table in (frame, reference):
        if table[KEYS].isna().any().any() or table.duplicated(KEYS).any():
            raise ValueError("Source file/row keys are missing or duplicated.")
    target = pd.MultiIndex.from_frame(reference[KEYS])
    source = frame.set_index(KEYS)
    if len(source) != len(target) or len(target.difference(source.index)):
        raise ValueError("Models have different source-row sets; incompatible results cannot be merged.")
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
            raise ValueError(f"{model}: Force GT/predictions contain nonnumeric values.")
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
        raise ValueError("id_tolerance must be an integer from 0 to 17; 0 requires an exact ID match.")
    return int(value)


def id_tables(modes, reference, regions, id_tolerance=2):
    id_tolerance = validate_id_tolerance(id_tolerance)
    tolerance_key = f"within{id_tolerance}"
    actual = reference.contact_id_effective.to_numpy(float)
    gt = reference.ground_truth_id.to_numpy(float)
    states = reference.load_state.to_numpy()
    if not np.isin(actual, np.arange(1, 19)).all() or not np.isin(states, ["loaded", "unloaded"]).all():
        raise ValueError("ID results require ID1–18 and definite loaded/unloaded labels.")
    loaded = states == "loaded"
    if not np.array_equal(gt, np.where(loaded, actual, 0)):
        raise ValueError("GT IDs and load labels do not agree.")
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
                raise ValueError(f"{mode}/{model}: ID probabilities are invalid.")
            conditional = np.array(active)[prob.argmax(axis=1)]
            final, detected = conditional, None
            if mode == "class0":
                full = np.column_stack([frame.p0, prob])
                if not np.isfinite(full).all() or ((full < 0) | (full > 1)).any():
                    raise ValueError(f"{model}: class0 probabilities are invalid.")
                final = np.array([0] + active)[full.argmax(axis=1)]
                detected = final != 0
            else:
                full = prob
                if frame.p0.notna().any() or frame.final_id.notna().any():
                    raise ValueError("Masked results contain no-load detection outputs.")
            if not np.allclose(full.sum(1), 1, rtol=0, atol=2e-6):
                raise ValueError(f"{mode}/{model}: Probabilities do not sum to 1.")
            if not np.array_equal(conditional, frame.conditional_id) or not np.array_equal(final, frame.pred_id):
                raise ValueError(f"{mode}/{model}: Predicted IDs do not agree with the probabilities.")
            if mode == "class0" and not np.array_equal(final, frame.final_id):
                raise ValueError(f"{model}: Final class0 IDs do not agree.")
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
    """Place GT on the left and models in columns, preserving numbers and leaving inapplicable cells blank."""
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
                raise ValueError(f"{name}: Excel worksheet size limit exceeded.")
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
                        raise ValueError(f"{name}: Infinite values cannot be written.")
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
            raise ValueError(f"Source workbook changed after checking common rows: {source}")
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
                # Validate consistency across source sheets before selecting common window-ending rows.
                frame = align_rows(frame.set_index(KEYS).loc[keys].reset_index(), target)
            branches[branch].setdefault(mode, {})[model] = frame
            print(f"Read: {sheet} ({len(frame):,} rows)", flush=True)
    finally:
        reader.close()
    if not references:
        raise ValueError("No force_*/id_* results with GT were found; provide a project predictions.xlsx workbook.")
    if common_references is not None:
        references = common_references
    results = {}
    for mode, frames in branches["force"].items():
        label = "loss-mask" if mode == "masked" else mode
        results[f"force-estimation_{label}_id-tol-{id_tolerance}_results.xlsx"] = force_tables(frames, references["force"])
    if branches["id"]:
        results[f"id-estimation_all-models_id-tol-{id_tolerance}_results.xlsx"] = id_tables(branches["id"], references["id"], groups, id_tolerance)
    guide = pd.DataFrame([
        ("Input", str(source)), ("Input SHA256", source_hash),
        ("Processing", "Reorganizes saved Excel results only; does not train, infer, select models or relabel using the current YAML."),
        ("Force", "The first force_all sheet collects GT Fx/Fy/Fz and XYZ predictions from every model. The fx/fy/fz sheets also retain GT first, followed by model predictions. Values are in mN; regression values, signs and forces for each load state are preserved."),
        ("ID", "gt_id is the contact ID when loaded and 0 when unloaded; loaded_gt is the measured load label (1/0). A class0_id of 0 means the model predicts no load."),
        ("Conditional ID", "class0_id_conditional is the location candidate excluding p0. Masked also outputs contact-ID candidates only and has no independent no-load detector."),
        ("ID tolerance n", id_tolerance),
        (f"ID_error / within{id_tolerance}", f"For GT-loaded rows, evaluate the absolute difference between the location candidate and GT ID. A difference ≤{id_tolerance} yields 1, otherwise 0. GT-unloaded rows are blank. The tolerance is an evaluation criterion and does not change training or predictions."),
        (f"within{id_tolerance}_detected", f"For GT-loaded rows, both contact detection and localization within ±{id_tolerance} must succeed to yield 1. Missed contact yields 0; GT-unloaded rows are blank."),
        ("class0_loaded", "Each model's loaded/unloaded decision (1/0). This output is not synthesized for Masked."),
        ("Regions", f"Fixed comparison regions: {regions}. Regions are numbered from 1 in order and selected by summed ID probabilities. Regions are not selected using test data; specify them with --regions."),
        ("region_detected", "For Class0, use the probability-sum region when contact is detected and 0 when no load is predicted. The candidate region may differ from the region containing the most probable ID."),
        ("Tip/body", "Predicting 18 for GT17 is within ±1 but misclassifies tip versus body; _summary reports this separately. A region containing the tip does not measure tip/body discrimination."),
        ("_summary", "Conditional and detection-inclusive localization metrics use all GT-loaded rows as the denominator. The unloaded false-positive rate uses GT-unloaded rows. Body/Tip denote ID ranges; independence follows the original experiment records."),
        ("Source alignment", "Matching rows across data sheets in a workbook refer to the same source. In _rows, source_row is zero-based without the header; source_csv_line is one-based including the header. Force and ID workbooks may contain different row counts."),
        ("Excluded/uncertain", "Invalid, warmup and uncertain-ID rows already excluded from the source workbook are not restored. Valid regression rows with uncertain load state remain in force tables with blank loaded_gt."),
        ("Models", "Model counts and order follow the input sheets. _models preserves representative partners/checkpoints from the source; no new model selection is performed."),
        ("Empty source sheets", ", ".join(skipped) or "None"),
    ], columns=["Item", "Description"])
    if alignment is not None:
        guide = pd.concat([guide, pd.DataFrame([
            ("Common-row comparison", "The source_file/source_row intersection across all input workbooks in _alignment. Force and ID rows follow the first input's order, and shared GT is checked. Rows are not matched by truncating counts or interpolation."),
            ("Common-row metrics", "RMSE/MAE and ID metrics in _summary are recalculated on the common rows retained in this workbook. Original metrics.json and analysis.py aggregate results are unchanged."),
            ("Comparison scope", "Only representative model branches from saved workbooks are compared. Even for matching prediction endpoints, the preceding 29 retained rows can differ across inputs after missing-row removal. Common-row filtering does not retrain or rerun predictions on identical histories."),
        ], columns=["Item", "Description"])], ignore_index=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source workbook changed during reorganization.")
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
        print(f"Saved: {destination / name}", flush=True)
    return destination


def organize_workbooks(sources, output=None, regions=DEFAULT_REGIONS, id_tolerance=2):
    """Find shared prediction window-ending rows and reuse the existing organization and metric functions."""
    id_tolerance = validate_id_tolerance(id_tolerance)
    parse_regions(regions)
    sources = list(dict.fromkeys(Path(p).expanduser().resolve() for p in sources))
    if not sources:
        raise ValueError("No input Excel files were provided.")
    if len(sources) == 1:
        return [organize_workbook(sources[0], output, regions, id_tolerance)]
    references, records = {}, []
    # Avoid loading all 44 sheets from all six input configurations into memory at once.
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
            raise ValueError(f"Comparison workbooks have different force/ID branch configurations: {source}")
        for branch, frame in found.items():
            if branch not in references:
                references[branch] = frame
            else:
                reference = references[branch]
                keep = pd.MultiIndex.from_frame(reference[KEYS]).isin(pd.MultiIndex.from_frame(frame[KEYS]))
                reference = reference.loc[keep].reset_index(drop=True)
                if reference.empty:
                    raise ValueError(f"{branch}: No prediction rows are shared by all inputs.")
                keys = pd.MultiIndex.from_frame(reference[KEYS])
                align_rows(frame.set_index(KEYS).loc[keys].reset_index(), reference)
                references[branch] = reference
            records.append({"source_workbook": str(source), "source_sha256": source_hash,
                            "branch": branch, "original_rows": len(frame)})
        print(f"Checked common rows: {source.parent.name} " +
              ", ".join(f"{k} {len(v):,} rows" for k, v in found.items()), flush=True)
    alignment = pd.DataFrame(records)
    alignment["common_rows"] = alignment.branch.map({k: len(v) for k, v in references.items()})
    alignment["excluded_rows"] = alignment.original_rows - alignment.common_rows
    print("Final common rows: " + ", ".join(f"{k} {len(v):,} rows" for k, v in references.items()), flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output:
        root = Path(output).expanduser().resolve()
        if root.exists():
            raise FileExistsError(f"Specify a new output directory: {root}")
        destinations = [root / source.parent.name for source in sources]
    else:
        destinations = [source.parent / f"organized_common_{stamp}_id-tol-{id_tolerance}" for source in sources]
    if len(set(destinations)) != len(destinations) or any(p.exists() for p in destinations):
        raise ValueError("Output directories overlap; place input workbooks in separate directories.")
    return [organize_workbook(source, destination, regions, id_tolerance,
                              common_references=references, alignment=alignment)
            for source, destination in zip(sources, destinations)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", help="predictions.xlsx paths or wildcards; compare shared rows when multiple files are supplied")
    parser.add_argument("--output", help="new output directory; multiple inputs get separate subdirectories; defaults to organized_[common_]date_time_id-tol-N beside each input")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="fixed regions (default: %(default)s)")
    parser.add_argument("--id-tolerance", type=int, choices=range(18), default=2,
                        help="ID tolerance ±n (0–17); 0 requires an exact ID match (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        sources = []
        for pattern in args.input:
            expanded = str(Path(pattern).expanduser())
            matches = sorted(glob.glob(expanded)) if glob.has_magic(expanded) else [expanded]
            if not matches:
                raise ValueError(f"No input workbooks match: {pattern}")
            sources.extend(matches)
        destinations = organize_workbooks(sources, args.output, args.regions, args.id_tolerance)
    except (ValueError, AssertionError, KeyError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Result reorganization failed: {error}\n")
    for destination in destinations:
        print(f"Result reorganization complete: {destination}")


if __name__ == "__main__":
    main()
