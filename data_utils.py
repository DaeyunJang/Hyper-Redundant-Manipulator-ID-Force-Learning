"""Shared CSV column selection, numeric row compaction, splitting and evaluation.

Input handling: input_columns/load_trial. Splitting: prepare_data.
Metrics: force_metrics/location_metrics. Models and optimizers are defined in
model_zoo.py and train.py, respectively.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
import copy
import csv
import hashlib
import json
import math
import os
import sys
import re

import numpy as np
import pandas as pd
import yaml

from analyze_no_load import calibrate_no_load, load_states

# In a frozen build, __file__ points inside the bundle; use the working directory for data/results.
ROOT = Path.cwd() if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent

# Apply the optional local driver workaround to this process only.
def ensure_driver_runtime():
    root = ROOT
    version_file = Path('/proc/driver/nvidia/version')
    if not version_file.exists():
        return
    match = re.search(r'\b(\d{3}\.\d+\.\d+)\b', version_file.read_text())
    if not match:
        return
    version = match.group(1)
    local = root / '.runtime' / ('nvidia-' + version)
    system = Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
    if system.exists() and system.resolve().name == 'libcuda.so.' + version:
        return
    if not (local / 'libcuda.so.1').exists() or os.environ.get('HRM_DRIVER_RUNTIME') == version:
        return
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(local) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['HRM_DRIVER_RUNTIME'] = version
    arguments = sys.argv[1:] if getattr(sys, 'frozen', False) else sys.argv
    os.execve(sys.executable, [sys.executable, *arguments], env)


# Historical label and file-role contracts apply only to verified hashes; source CSVs stay unchanged.
DEFAULT_DATA_CONTRACTS = {
    "label_overrides_by_sha256": {
        "269578346b29158aafc7989051c872e71954f91d0a992665cedaf95d11b7c3af": {
            "original_id": 1,
            "effective_id": 18,
            "basis": "User-confirmed right_2 ID18; this is the original/replayed CSV before the ID correction. All sensor rows match the corrected source hash 96a85df0c67903a8ad6c21abff43533e5c56a1f3c0e19ecb2cd689f7473fe203."
        }
    },
    "protected_test_hashes": [
        "4a37bf418ae1de06fb30dbe3076f383f81c58dedfb831c80d007ed0865f0050c",
        "ee628bcc6cf97e603ff3efaba0d28195d6addc53227d3a786dd0e49003022648"
    ],
    "protected_development_hashes": [
        "269578346b29158aafc7989051c872e71954f91d0a992665cedaf95d11b7c3af",
        "96a85df0c67903a8ad6c21abff43533e5c56a1f3c0e19ecb2cd689f7473fe203"
    ],
    "excluded_files_by_sha256": {
        "c1c6ec1e7bbee8dfd7a731dd6e6d1c24cf69aca4ba964a37eb25f8c9fa06fb1d": {
            "reason": "Static ID10 has no valid Kalman aligned force/frame on all 48360 rows; preserve original, no raw fallback, exclude from supervised training and validation."
        }
    }
}

# Column groups for older configs; new configs specify columns directly in input_columns.
FEATURE_GROUPS = {
    'wire_length': [f'wire.Cable #{i} length' for i in range(1, 5)],
    'loadcell_tension': [f'loadcell.Loadcell #{i} tension' for i in range(1, 5)],
    'relative_angle': [f'relative_angle_{i}' for i in range(1, 19)],
    'motor_position': [f'motor.Motor #{i} position [count]' for i in range(1, 5)],
    'wire_velocity': [f'wire_velocity.Cable #{i} velocity' for i in range(1, 5)],
    'relative_angular_velocity': [f'relative_angular_velocity_{i}_rad_s' for i in range(1, 19)],
    **{f'{axis}_{kind}': [f'{axis}_{kind}_{i:02d}_rad' for i in range(1, 19)]
       for axis in ('pan', 'tilt') for kind in ('relative', 'absolute')},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n')


json_write = write_json


def _path(value, config):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(config.get('project_root', ROOT)) / path).resolve()


def csv_paths(directory):
    """Sort uniform trainN/testN names numerically; preserve lexical order otherwise."""
    paths = sorted(path for path in Path(directory).glob('*.csv') if path.is_file())
    matches = [re.fullmatch(r'(train|test)([0-9]+)', path.stem) for path in paths]
    if matches and all(matches) and len({match.group(1) for match in matches}) == 1:
        return [path for match, path in sorted(zip(matches, paths), key=lambda item: int(item[0].group(2)))]
    return paths


def input_columns(config):
    """Return the input column order shared by training, saved bundles and prediction."""
    columns = config.get('input_columns', config.get('feature_columns'))
    if columns is None:
        columns = [column for group in config.get('feature_groups', ['wire_length', 'loadcell_tension', 'relative_angle'])
                   for column in FEATURE_GROUPS[group]]
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) and c for c in columns):
        raise ValueError('input_columns must be a nonempty list of column names')
    if len(columns) != len(set(columns)):
        raise ValueError('Duplicate input columns')
    return list(columns)


def load_config(path):
    """Read user YAML and resolve reproducibility defaults and input column order."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping')
    for key, value in DEFAULT_DATA_CONTRACTS.items():
        config.setdefault(key, copy.deepcopy(value))
    defaults = dict(project_root=str(ROOT), preprocessing='numeric_compact',
                    validation_strategy='per_file_tail', validation_fraction=.2,
                    window_samples=30, id_column='contact_segment_id', timestamp_column='source_time_ns',
                    source_force_unit='mN', force_frame='unverified', force_scope='tip', tip_scope_by_id=True,
                    active_ids=list(range(1, 19)), seed=42, training_sign_multiplier=1)
    for key, value in defaults.items():
        config.setdefault(key, value)
    config['config_path'] = str(path)
    if config['preprocessing'] != 'numeric_compact':
        raise ValueError('The simplified runner supports numeric_compact only; audited loaders are preserved in archive')
    if config['validation_strategy'] not in ('whole_csv', 'per_file_tail'):
        raise ValueError('validation_strategy must be whole_csv or per_file_tail')
    if not 0 < float(config['validation_fraction']) < 1:
        raise ValueError('validation_fraction must be strictly between zero and one')
    if type(config['window_samples']) is not int or config['window_samples'] < 1:
        raise ValueError('window_samples must be a positive integer')
    if config['source_force_unit'] not in ('mN', 'N'):
        raise ValueError('source_force_unit must be mN or N')
    if config['training_sign_multiplier'] != 1:
        raise ValueError('Training preserves the recorded force sign; training_sign_multiplier must be 1')
    columns = config.setdefault('force_columns', [f'fts_kalman.aligned_f{axis}' for axis in 'xyz'])
    if not isinstance(columns, list) or len(columns) != 3 or len(set(columns)) != 3 or not all(isinstance(c, str) and c for c in columns):
        raise ValueError('force_columns must list three distinct column names in Fx, Fy, Fz order')
    if not isinstance(config['id_column'], str) or not config['id_column']:
        raise ValueError('id_column must be a column name')
    if config['id_column'] in columns:
        raise ValueError('ID and force targets must use distinct columns')
    config['input_columns'] = config['feature_columns'] = input_columns(config)
    comparison = config.get('comparison_columns', [])
    if (not isinstance(comparison, list) or
            not all(isinstance(c, str) and c for c in comparison) or len(comparison) != len(set(comparison))):
        raise ValueError('comparison_columns must list distinct column names (or be empty)')
    forbidden = set(columns) | {config['id_column'], config.get('timestamp_column'),
                               'contact_segment_id', 'elapsed_s', 'trial_id', 'experiment_id', 'session_id',
                               'load_state', 'is_loaded', 'source_row', 'source_file'}
    for column in config['input_columns']:
        lower = column.lower()
        if (column in forbidden or lower.startswith(('fts.', 'fts_kalman.')) or
                any(name in lower for name in ('timestamp', 'time_ns', 'time_difference_ms'))):
            raise ValueError(f'Target/ID/time metadata cannot be a model input: {column}')
    for name in ('force_candidates', 'location_candidates', 'no_load_modes'):
        values = config.get(name)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f'{name} must be a nonempty list without duplicates')
    if not set(config['no_load_modes']) <= {'masked', 'class0'}:
        raise ValueError('no_load_modes supports masked and class0')
    ids = config['active_ids']
    if not ids or any(type(i) is not int or i not in range(1, 19) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('active_ids must contain distinct integer segment IDs in 1..18')
    calibrate_no_load(config)  # Validate manual range units and bounds without reading a CSV.
    return config


# x/force/timestamps use compacted row indices; source_rows preserves original CSV row numbers.
@dataclass
class Trial:
    path: str
    sha256: str
    segment_id_raw: int
    effective_id: int
    x: np.ndarray
    force: np.ndarray
    timestamps: np.ndarray
    input_valid: np.ndarray
    force_valid: np.ndarray
    load_state: np.ndarray
    endpoints: np.ndarray
    predict_endpoints: np.ndarray
    is_tip: bool
    info: dict = field(default_factory=dict)
    source_rows: np.ndarray | None = None


def trial_endpoints(trial, role=None, prediction=False):
    ends = trial.predict_endpoints if prediction else trial.endpoints
    spans = getattr(trial, 'info', {}).get('role_rows', {})
    if role in spans:
        start, stop = spans[role]
        ends = ends[(ends >= start + trial.info['window_samples'] - 1) & (ends < stop)]
    elif spans and role is None and not prediction:
        raise ValueError('A partitioned trial requires an explicit train/validation role')
    return ends


def _frame(path, columns):
    with Path(path).open(newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError(f'Duplicate CSV headers: {path}')
    missing = set(columns) - set(header)
    if missing:
        raise ValueError(f'Missing selected columns in {path}: {sorted(missing)}')
    return pd.read_csv(path, usecols=list(dict.fromkeys(columns)), dtype=str, keep_default_na=False)


def _number(frame, columns):
    return frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)


def _timestamp(value):
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number < 0 or number != number.to_integral_value():
            raise ValueError('Invalid integer timestamp')
        return int(number)
    except (InvalidOperation, ValueError):
        return None


# AUTHOR: Daeyun
# Project data preparation implementation: numeric row selection and train/validation partitioning.
def load_trial(path, config, calibration, require_targets=True):
    """Compact finite selected X/Y rows; comparison_columns only filters comparison rows."""
    if config.get('preprocessing', 'numeric_compact') != 'numeric_compact':
        raise ValueError('Audited legacy preprocessing requires the archived runner; no automatic policy change')
    path = Path(path).resolve()
    before = sha256(path)
    features = input_columns(config)
    targets = config.get('force_columns', [f'fts_kalman.aligned_f{a}' for a in 'xyz'])
    id_column = config.get('id_column', 'contact_segment_id')
    time_column = config.get('timestamp_column', 'source_time_ns')
    with path.open(newline='') as stream:
        header = next(csv.reader(stream), [])
    has_targets = all(column in header for column in targets)
    required = features + (targets + [id_column] if require_targets else [c for c in targets + [id_column] if c in header])
    comparison = config.get('comparison_columns', [])
    required += [c for c in comparison if c in header]
    if time_column and time_column in header:
        required.append(time_column)
    frame = _frame(path, required)
    x = _number(frame, features)
    force = _number(frame, targets) if has_targets else np.full((len(frame), 3), np.nan)
    label = pd.to_numeric(frame.get(id_column, pd.Series(np.nan, index=frame.index)), errors='coerce').to_numpy(dtype=float)
    x_ok, y_ok = np.isfinite(x).all(1), np.isfinite(force).all(1)
    # Evaluation uses valid X/Y rows; inference without targets uses valid X rows only.
    evaluating = require_targets or (has_targets and y_ok.any())
    keep = x_ok & y_ok if evaluating else x_ok.copy()
    comparison_applied = bool(comparison and evaluating)
    comparison_ok = np.ones(len(frame), dtype=bool)
    if comparison_applied:
        missing = set(comparison) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing comparison columns in {path}: {sorted(missing)}')
        comparison_ok = np.isfinite(_number(frame, comparison)).all(1)
        keep &= comparison_ok
    if require_targets or (comparison_applied and id_column in frame):
        keep &= np.isfinite(label)
    excluded = config.get('excluded_rows_by_sha256', {}).get(before)
    if excluded:
        rows = excluded.get('rows', [])
        if not isinstance(rows, list) or not excluded.get('reason') or any(type(i) is not int or not 0 <= i < len(frame) for i in rows):
            raise ValueError(f'Invalid exact-hash excluded row rule: {path}')
        keep[rows] = False
    source_rows = np.flatnonzero(keep)
    if not len(source_rows):
        raise ValueError(f'No finite selected input/target rows: {path}')
    label = label[keep]
    has_id = bool(np.isfinite(label).all() and len(np.unique(label)) == 1 and label[0] == int(label[0]))
    if require_targets and not has_id:
        raise ValueError(f'Each retained CSV must have one integer contact ID: {path}')
    raw_id = int(label[0]) if has_id else 0
    effective = raw_id
    override = config.get('label_overrides_by_sha256', {}).get(before)
    if override:
        if override.get('original_id') != raw_id or not override.get('basis'):
            raise ValueError(f'Invalid exact-hash ID override: {path}')
        effective = int(override['effective_id'])
    if require_targets and effective not in range(1, 19):
        raise ValueError(f'Expected ID1..18: {path}')
    match = re.search(r'seg-id-(\d+)|summary_(?:test_)?(\d+)', path.stem)
    if has_id and match and int(next(v for v in match.groups() if v is not None)) != effective:
        raise ValueError(f'Filename and recorded/effective ID disagree: {path}')
    times = [_timestamp(value) for value in frame[time_column].to_numpy()[keep]] if time_column in frame else []
    available = bool(times and all(value is not None and value <= np.iinfo(np.int64).max for value in times))
    origin = times[0] if available else 0
    timestamps = np.asarray(times, dtype=np.int64) - origin if available else np.zeros(len(source_rows), dtype=np.int64)
    positive = np.diff(timestamps)
    positive = positive[positive > 0]
    unit = config.get('source_force_unit', 'mN')
    if unit not in ('mN', 'N'):
        raise ValueError('source_force_unit must be mN or N')
    x, force = x[keep], force[keep] * (.001 if unit == 'mN' else 1.)
    force_valid = y_ok[keep]
    window = int(config['window_samples'])
    pred_ends = np.arange(window - 1, len(x), dtype=np.int64)
    ends = pred_ends[force_valid[pred_ends]] if has_targets else np.empty(0, dtype=np.int64)
    explicit_tip = {_path(p, config) for p in config.get('tip_train_files', [])}
    is_tip = effective == 18 if config.get('tip_scope_by_id', True) else (
        path in explicit_tip or (effective == 18 and path.parent == _path(config['test_directory'], config)))
    states = load_states(force, force_valid, calibration)
    info = {'path': str(path), 'sha256': before, 'rows': len(x), 'source_rows_total': len(frame),
            'dropped_rows': int((~keep).sum()), 'input_nonfinite_rows': int((~x_ok).sum()),
            'target_nonfinite_rows': int((~y_ok).sum()), 'explicit_row_exclusion': excluded,
            'segment_id_raw': raw_id, 'effective_id': effective, 'label_override': override,
            'is_tip': bool(is_tip), 'has_location_target': has_id, 'has_force_targets': has_targets,
            'timestamp_origin_ns': origin, 'timestamps_available': available,
            'timestamp_column': time_column, 'median_dt_ms': float(np.median(positive)/1e6) if len(positive) else None,
            'preprocessing': 'numeric_compact', 'window_samples': window,
            'window_endpoints': len(ends), 'prediction_endpoints': len(pred_ends),
            'input_invalid_rows': 0, 'force_invalid_rows': int((~force_valid).sum()),
            'load_state_counts': {str(i): int((states[ends] == i).sum()) for i in (-2, -1, 0, 1)},
            'source_row_policy': 'source_rows maps compact positions to zero-based original CSV data rows',
            'timestamp_policy': 'metadata only; never a model input or a window boundary',
            'force_frame_basis': config.get('force_frame', 'user-selected XYZ columns; no frame-flag checks')}
    if comparison:
        info.update(comparison_columns=comparison, comparison_filter_applied=comparison_applied,
                    comparison_nonfinite_rows=int((~comparison_ok).sum()))
    if sha256(path) != before:
        raise ValueError(f'CSV changed while reading: {path}')
    return Trial(str(path), before, raw_id, effective, x, force, timestamps,
                 np.ones(len(x), dtype=bool), force_valid, states, ends, pred_ends, is_tip, info, source_rows)


def _reject_duplicates(trials):
    seen = {}
    for trial in trials:
        if trial.sha256 in seen:
            raise ValueError(f'Duplicate CSV bytes: {seen[trial.sha256]} and {trial.path}')
        seen[trial.sha256] = trial.path


def prepare_data(config, run_dir=None):
    """Keep the test directory fixed and derive validation only from training recordings."""
    calibration = calibrate_no_load(config)
    train_root, test_root = _path(config['train_directory'], config), _path(config['test_directory'], config)
    if train_root == test_root:
        raise ValueError('Train and test directories must differ')
    excluded = []
    def discover(directory):
        selected = []
        for path in csv_paths(directory):
            digest = sha256(path)
            rule = config.get('excluded_files_by_sha256', {}).get(digest)
            if rule:
                if not rule.get('reason'):
                    raise ValueError(f'Excluded file needs an explicit reason: {path}')
                excluded.append({'path': str(path), 'sha256': digest, **rule})
            else:
                selected.append(path)
        return selected
    development, tests = discover(train_root), discover(test_root)
    if not development:
        raise ValueError(f'No training CSVs: {train_root}')
    trials = [load_trial(path, config, calibration) for path in development + tests]
    _reject_duplicates(trials)
    for index, trial in enumerate(trials):
        if index < len(development) and trial.sha256 in config.get('protected_test_hashes', []):
            raise ValueError(f'Historical held-out test cannot enter development: {trial.path}')
        if index >= len(development) and trial.sha256 in config.get('protected_development_hashes', []):
            raise ValueError(f'Historical development cannot become independent test: {trial.path}')
        if not len(trial.endpoints):
            raise ValueError(f'Not enough finite rows for a complete input window: {trial.path}')
    split = {'train': [], 'validation': [], 'test': list(range(len(development), len(trials)))}
    groups = {}
    for index, trial in enumerate(trials[:len(development)]):
        groups.setdefault(trial.effective_id, []).append(index)
    strategy = config.get('validation_strategy', 'per_file_tail')
    for segment_id, indices in sorted(groups.items()):
        if strategy == 'per_file_tail':
            for index in indices:
                trial = trials[index]
                count = len(trial.x)
                cut = int(count * (1 - float(config['validation_fraction'])))
                if min(cut, count - cut) < int(config['window_samples']):
                    raise ValueError(f'Not enough retained rows for separate train/validation windows: {trial.path}')
                trial.info['role_rows'] = {'train': [0, cut], 'validation': [cut, count]}
                trial.info['role_window_counts'] = {role: len(trial_endpoints(trial, role)) for role in ('train', 'validation')}
                split['train'].append(index)
                split['validation'].append(index)
        elif strategy == 'whole_csv':
            if len(indices) < 2:
                raise ValueError(f'ID{segment_id} needs two independent CSVs for whole_csv validation')
            rng = np.random.default_rng(int(config['seed']) + segment_id)
            count = min(len(indices)-1, max(1, math.ceil(len(indices)*float(config['validation_fraction']))))
            validation = set(rng.permutation(indices)[:count].tolist())
            split['validation'].extend(sorted(validation))
            split['train'].extend(i for i in indices if i not in validation)
        else:
            raise ValueError('Unknown validation strategy')
    declared_tip = {_path(path, config) for path in config.get('tip_train_files', [])}
    if declared_tip - set(development):
        raise ValueError(f'Missing explicitly declared tip files: {declared_tip - set(development)}')
    # Use the user-defined train/test directories without comparing X/Y row values across them.
    audit = {'schema_version': 2, 'feature_columns': input_columns(config),
             'force_columns': config.get('force_columns', [f'fts_kalman.aligned_f{a}' for a in 'xyz']), 'force_unit': 'N', 'force_frame': config.get('force_frame', 'unverified'),
             'source_files': len(trials), 'development_files': len(development), 'test_files': len(tests),
             'trials': [trial.info for trial in trials], 'split': split, 'excluded_files': excluded,
             'split_policy': ('test directory fixed; per-file retained-row prefix train / tail validation; windows stay within roles'
                              if strategy == 'per_file_tail' else 'whole_CSV_per_ID_seeded'),
             'scaler_policy': 'fit eligible training rows only; performed separately for each branch by train.py',
             'test_usage': 'fixed file identity checked; no model/scaler/region selection',
             'preprocessing': 'numeric_compact; finite selected X/Y only; remaining rows joined within each file and role',
             'timestamp_policy': 'provenance only; no flag/frame/matching-time filtering or time-gap window breaks',
             'test_evaluation_scope': config.get('test_evaluation_scope', 'user_defined')}
    if config.get('comparison_columns'):
        audit['comparison_columns'] = config['comparison_columns']
        audit['preprocessing'] += '; also require finite comparison_columns; these do not expand model inputs'
    if run_dir is not None:
        output = Path(run_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'calibration.json', calibration)
        write_json(output/'data_audit.json', audit)
        write_json(output/'split_manifest.json', {part: [trials[i].info for i in indices] for part, indices in split.items()})
        write_json(output/'resolved_config.json', config)
    return {'trials': trials, 'split': split, 'calibration': calibration, 'audit': audit,
            'feature_columns': audit['feature_columns'], 'force_columns': audit['force_columns']}


def load_prediction_trials(paths, config, calibration):
    """Use saved training calibration and allow prediction inputs without targets."""
    trials = [load_trial(_path(path, config), config, calibration, require_targets=False) for path in paths]
    # predict.py --check-overlap optionally warns about overlap without excluding files.
    return trials


# Metrics accept forces in N and report mN; ground truth never changes predicted ID probabilities.
DEFAULT_REGIONS = {
    "lower": list(range(1, 11)),
    "middle": list(range(11, 16)),
    "upper": list(range(16, 19)),
}


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _mean(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def force_metrics(pred_N: Any, target_N: Any, valid: Any = None) -> dict[str, Any]:
    """Return axis/pooled XYZ RMSE and MAE; finite valid targets define support.

    Invalid target rows can be excluded, but nonfinite predictions on otherwise
    valid targets raise an error instead of making a failing model look better.
    ``valid`` is a row mask, independent of the location loss mask.
    """
    prediction = np.asarray(pred_N, dtype=float)
    target = np.asarray(target_N, dtype=float)
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Force prediction and target must both have shape (N, 3)")
    selected = np.isfinite(target).all(axis=1)
    if valid is not None:
        mask = np.asarray(valid)
        if mask.shape != (len(target),) or not np.isin(mask, [0, 1]).all():
            raise ValueError("valid must contain one boolean / 0 or 1 per row")
        selected &= mask.astype(bool)
    if not np.isfinite(prediction[selected]).all():
        raise ValueError("Nonfinite force prediction on a valid target")
    error = (prediction[selected] - target[selected]) * 1000.0
    result: dict[str, Any] = {
        "force_support": int(selected.sum()),
        "force_excluded": int((~selected).sum()),
    }
    for column, axis in enumerate(("fx", "fy", "fz")):
        result[f"force_rmse_{axis}_mN"] = float(np.sqrt(np.mean(error[:, column] ** 2))) if len(error) else float("nan")
        result[f"force_mae_{axis}_mN"] = _mean(np.abs(error[:, column]))
    result["force_rmse_xyz_mN"] = float(np.sqrt(np.mean(error ** 2))) if len(error) else float("nan")
    result["force_mae_xyz_mN"] = _mean(np.abs(error).ravel())
    return result


def _probabilities(probabilities: Any, mode: str, active_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    if mode not in ("masked", "class0"):
        raise ValueError("mode must be masked or class0")
    ids = np.asarray(active_ids)
    if ids.ndim != 1 or not len(ids) or not np.isin(ids, np.arange(1, 19)).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("active_ids must contain distinct integer segment IDs in 1..18")
    ids = ids.astype(int)
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != len(ids) + (mode == "class0"):
        raise ValueError("Probability columns must follow the saved active_ids and class0 mapping")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("Each probability row must sum to 1; do not pass logits")
    return probability, ids


def _regions(regions: Mapping[str, Sequence[int]] | None, ids: np.ndarray) -> dict[str, list[int]]:
    source = DEFAULT_REGIONS if regions is None else regions
    if not source:
        raise ValueError("At least one region is required")
    result = {}
    seen: set[int] = set()
    for name, members in source.items():
        if not isinstance(name, str) or not name or not all(char.isalnum() or char == "_" for char in name):
            raise ValueError("Region names must contain letters, digits, or underscores")
        values = np.asarray(members)
        if values.ndim != 1 or not len(values) or not np.isin(values, np.arange(1, 19)).all():
            raise ValueError("Region IDs must be integer segments in 1..18")
        group = values.astype(int).tolist()
        if len(set(group)) != len(group) or seen.intersection(group):
            raise ValueError("Region IDs must not overlap or repeat")
        result[name] = group
        seen.update(group)
    if not set(ids.tolist()).issubset(seen):
        raise ValueError("Regions must cover every active segment exactly once")
    return result


def region_probabilities(
    probabilities: Any, mode: str, active_ids: Sequence[int],
    regions: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, np.ndarray]:
    """Sum original segment probabilities; class0 sums remain 1 - p0."""
    probability, ids = _probabilities(probabilities, mode, active_ids)
    groups = _regions(regions, ids)
    segment = probability[:, int(mode == "class0"):]
    return {name: segment[:, np.isin(ids, members)].sum(axis=1) for name, members in groups.items()}


def location_predictions(
    probabilities: Any, mode: str, active_ids: Sequence[int],
    regions: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Decode without GT; masked has no operational ID or no-load detector.

    Class0's candidate rule is argmax over all classes, including zero. Region
    candidates use summed probabilities independently of the global ID argmax.
    This explicit rule is evaluable; it does not itself establish deployment
    calibration. Ties follow saved class / region order.
    """
    probability, ids = _probabilities(probabilities, mode, active_ids)
    groups = _regions(regions, ids)
    segment = probability[:, int(mode == "class0"):]
    sums = region_probabilities(probability, mode, ids, groups)
    region_matrix = np.column_stack(list(sums.values()))
    region_index = region_matrix.argmax(axis=1)
    conditional = ids[segment.argmax(axis=1)]
    full = np.full((len(probability), 18), np.nan)
    full[:, ids - 1] = segment
    final = np.full(len(probability), np.nan)
    system_region = np.full(len(probability), np.nan)
    p0 = np.full(len(probability), np.nan)
    if mode == "class0":
        final = np.r_[0, ids][probability.argmax(axis=1)]
        system_region = np.where(final == 0, -1, region_index)
        p0 = probability[:, 0].copy()
    return {
        "conditional_id": conditional,
        "final_id": final,
        "segment_probabilities": full,
        "no_load_probability": p0,
        "region_names": list(groups),
        "region_probabilities": region_matrix,
        "conditional_region_index": region_index,
        "system_region_index": system_region,
        "decision_rule": "class_argmax" if mode == "class0" else "unavailable",
    }


def _labels(actual_ids: Any, load_state: Any, length: int) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual_ids, dtype=float)
    states = np.asarray(load_state, dtype=object)
    if actual.shape != (length,) or states.shape != (length,):
        raise ValueError("actual_ids and load_state must each have one value per row")
    mapping = {1: "loaded", 0: "unloaded", -1: "uncertain", -2: "invalid"}
    states = np.asarray([mapping.get(value, value) for value in states], dtype=object)
    if not np.isin(states, ["loaded", "unloaded", "uncertain", "invalid"]).all():
        raise ValueError("load_state must be loaded/unloaded/uncertain/invalid (or 1/0/-1/-2)")
    return actual, states


def _detection_metrics(true_loaded: np.ndarray, predicted_loaded: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(true_loaded & predicted_loaded))
    tn = int(np.sum(~true_loaded & ~predicted_loaded))
    fp = int(np.sum(~true_loaded & predicted_loaded))
    fn = int(np.sum(true_loaded & ~predicted_loaded))
    recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    return {
        "load_support": len(true_loaded),
        "load_tp": tp, "load_tn": tn, "load_fp": fp, "load_fn": fn,
        "load_precision": _ratio(tp, tp + fp),
        "load_recall": recall,
        "load_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "load_false_positive_rate": _ratio(fp, fp + tn),
        "load_false_negative_rate": _ratio(fn, fn + tp),
        "load_accuracy": _ratio(tp + tn, len(true_loaded)),
        "load_balanced_accuracy": (recall + specificity) / 2,
        "unload_precision": _ratio(tn, tn + fn),
        "unload_recall": specificity,
        "unload_f1": _ratio(2 * tn, 2 * tn + fp + fn),
    }


def location_metrics(
    probabilities: Any, actual_ids: Any, load_state: Any, mode: str,
    active_ids: Sequence[int], regions: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Return flat metrics with diagnostic, detection, and system denominators.

    Location diagnostics use GT loaded rows whose ID is active, even if class0
    predicts no load. Detection/system metrics use those rows plus all known
    unloaded rows. Unsupported loaded contacts are counted separately, never
    relabelled as unloaded. ``region_id_macro_recall`` averages region success
    per supported true segment. ``region_balanced_id_macro_recall`` first
    averages those segment recalls within each region, then weights regions
    equally; it is undefined if any region has no supported GT samples.
    """
    probability, ids = _probabilities(probabilities, mode, active_ids)
    groups = _regions(regions, ids)
    actual, states = _labels(actual_ids, load_state, len(probability))
    decoded = location_predictions(probability, mode, ids, groups)
    selected = (states == "loaded") & np.isin(actual, ids)
    count = int(selected.sum())
    conditional = decoded["conditional_id"]
    lookup = {segment: index for index, members in enumerate(groups.values()) for segment in members}
    true_region = np.asarray([lookup.get(value, -1) for value in actual], dtype=int)
    region_prediction = decoded["conditional_region_index"]
    correct = conditional == actual
    region_correct = region_prediction == true_region
    result: dict[str, Any] = {
        "location_mode": mode,
        "location_interpretation": "loaded_conditional_diagnostic",
        "location_support": count,
        "location_loaded_outside_support": int(np.sum((states == "loaded") & ~np.isin(actual, ids))),
        "location_unloaded_support": int(np.sum(states == "unloaded")),
        "location_uncertain_support": int(np.sum(states == "uncertain")),
        "location_invalid_support": int(np.sum(states == "invalid")),
        "location_accuracy": _ratio(np.sum(correct[selected]), count),
        "location_within1_accuracy": _ratio(np.sum(np.abs(conditional[selected] - actual[selected]) <= 1), count),
        "region_accuracy": _ratio(np.sum(region_correct[selected]), count),
        "region_predicted_probability_mean": _mean(decoded["region_probabilities"][selected].max(axis=1)),
        "region_true_probability_mean": _mean(decoded["region_probabilities"][np.flatnonzero(selected), true_region[selected]]),
        "load_evaluation_scope": "active_loaded_and_all_unloaded",
        "system_decision_rule": decoded["decision_rule"],
        "system_available": mode == "class0",
    }
    id_recalls, id_region_recalls = [], []
    for segment in range(1, 19):
        subset = selected & (actual == segment)
        support = int(subset.sum())
        recall = _ratio(np.sum(correct[subset]), support)
        region_recall = _ratio(np.sum(region_correct[subset]), support)
        result[f"location_id_{segment:02d}_supported"] = bool(segment in ids)
        result[f"location_id_{segment:02d}_support"] = support
        result[f"location_id_{segment:02d}_recall"] = recall
        result[f"region_id_{segment:02d}_recall"] = region_recall
        id_recalls.append(recall)
        id_region_recalls.append(region_recall)
    result["location_macro_recall"] = _mean(id_recalls)
    result["location_macro_class_count"] = int(np.isfinite(id_recalls).sum())
    result["region_id_macro_recall"] = _mean(id_region_recalls)
    region_recalls, region_id_means = [], []
    for index, name in enumerate(groups):
        subset = selected & (true_region == index)
        support = int(subset.sum())
        recall = _ratio(np.sum(region_correct[subset]), support)
        result[f"region_{name}_support"] = support
        result[f"region_{name}_recall"] = recall
        result[f"region_{name}_true_probability_mean"] = _mean(decoded["region_probabilities"][subset, index])
        id_mean = _mean([result[f"region_id_{segment:02d}_recall"] for segment in groups[name]])
        result[f"region_{name}_id_macro_recall"] = id_mean
        region_id_means.append(id_mean)
        region_recalls.append(recall)
    result["region_macro_recall"] = _mean(region_recalls)
    result["region_macro_class_count"] = int(np.isfinite(region_recalls).sum())
    result["region_balanced_id_macro_recall"] = float(np.mean(region_id_means))
    evaluable = selected | (states == "unloaded")
    empty_detection = _detection_metrics(np.array([], dtype=bool), np.array([], dtype=bool))
    if mode == "class0":
        final = decoded["final_id"]
        result.update(_detection_metrics(states[evaluable] == "loaded", final[evaluable] != 0))
        target = np.where(states == "unloaded", 0, actual)
        final_region = decoded["system_region_index"]
        target_region = np.where(states == "unloaded", -1, true_region)
        total = int(evaluable.sum())
        result.update({
            "system_support": total,
            "system_id_accuracy": _ratio(np.sum(final[evaluable] == target[evaluable]), total),
            "system_region_accuracy": _ratio(np.sum(final_region[evaluable] == target_region[evaluable]), total),
            "system_loaded_id_accuracy": _ratio(np.sum(final[selected] == actual[selected]), count),
            "system_loaded_region_accuracy": _ratio(np.sum(final_region[selected] == true_region[selected]), count),
        })
    else:
        result.update({key: float("nan") for key in empty_detection})
        result.update({key: float("nan") for key in (
            "system_support", "system_id_accuracy", "system_region_accuracy",
            "system_loaded_id_accuracy", "system_loaded_region_accuracy",
        )})
    return result


def location_confusion(
    probabilities: Any, actual_ids: Any, load_state: Any, mode: str,
    active_ids: Sequence[int], *, system: bool = False,
) -> dict[str, Any]:
    """Return labels and count matrix separately from scalar comparison rows."""
    probability, ids = _probabilities(probabilities, mode, active_ids)
    actual, states = _labels(actual_ids, load_state, len(probability))
    decoded = location_predictions(probability, mode, ids)
    selected = (states == "loaded") & np.isin(actual, ids)
    labels = ids
    truth, predicted = actual, decoded["conditional_id"]
    if system:
        if mode != "class0":
            return {"available": False, "labels": [], "matrix": []}
        selected |= states == "unloaded"
        labels = np.r_[0, ids]
        truth = np.where(states == "unloaded", 0, actual)
        predicted = decoded["final_id"]
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    lookup = {value: index for index, value in enumerate(labels)}
    for target, estimate in zip(truth[selected], predicted[selected]):
        matrix[lookup[target], lookup[estimate]] += 1
    return {"available": True, "labels": labels.tolist(), "matrix": matrix.tolist()}


def optimize_regions(
    probabilities: Any, actual_ids: Any, load_state: Any, mode: str,
    active_ids: Sequence[int], *, min_segments: int = 3, split: str = "validation",
) -> dict[str, Any]:
    """Evaluate all 136 contiguous 3-region partitions on validation only.

    Every boundary pair is retained with its constraint/support status. Select
    by region-balanced true-ID macro recall, then ordinary region macro recall,
    then true-ID macro region recall. Equal scores favor boundaries closest to
    the baseline (10, 15), then lower IDs.
    Each selected region must have at least one GT loaded validation sample;
    missing IDs remain explicit and are not treated as evidence of accuracy.
    The result is a validation search result, not an independent test estimate.
    """
    if split != "validation":
        raise ValueError("Region boundaries may only be selected using validation data")
    if isinstance(min_segments, bool) or not isinstance(min_segments, (int, np.integer)) or not 1 <= min_segments <= 6:
        raise ValueError("min_segments must be an integer in 1..6")
    probability, ids = _probabilities(probabilities, mode, active_ids)
    actual, states = _labels(actual_ids, load_state, len(probability))
    baseline = location_metrics(probability, actual, states, mode, ids, DEFAULT_REGIONS)
    selected = (states == "loaded") & np.isin(actual, ids)
    truth = actual[selected].astype(int)
    segment = probability[selected, int(mode == "class0"):]
    # Validate and select rows once. Use the same summation order as inference;
    # cumulative subtraction can change exact ties through floating-point error.
    id_support = np.bincount(truth, minlength=19)
    supported = id_support > 0
    common = {key: value for key, value in baseline.items() if key.startswith("location_id_") or key in ("location_support", "location_macro_class_count")}
    candidates = []
    for first in range(1, 17):
        for second in range(first + 1, 18):
            sums = np.column_stack((
                segment[:, ids <= first].sum(axis=1),
                segment[:, (ids > first) & (ids <= second)].sum(axis=1),
                segment[:, ids > second].sum(axis=1),
            ))
            prediction = sums.argmax(axis=1)
            target_region = (truth > first).astype(int) + (truth > second).astype(int)
            correct = prediction == target_region
            true_support = np.bincount(target_region, minlength=3)
            true_correct = np.bincount(target_region, weights=correct, minlength=3)
            recalls = np.full(3, np.nan)
            np.divide(true_correct, true_support, out=recalls, where=true_support > 0)
            id_correct = np.bincount(truth, weights=correct, minlength=19)
            id_recalls = np.full(19, np.nan)
            np.divide(id_correct, id_support, out=id_recalls, where=supported)
            metrics = {
                "region_accuracy": _ratio(correct.sum(), len(truth)),
                "region_macro_recall": _mean(recalls),
                "region_macro_class_count": int(np.sum(true_support > 0)),
                "region_id_macro_recall": _mean(id_recalls),
                "region_predicted_probability_mean": _mean(sums.max(axis=1)),
                "region_true_probability_mean": _mean(sums[np.arange(len(truth)), target_region]),
                **common,
            }
            within_region_means = []
            for index, (name, start, end) in enumerate((
                ("lower", 1, first), ("middle", first + 1, second), ("upper", second + 1, 18),
            )):
                metrics[f"region_{name}_support"] = int(true_support[index])
                metrics[f"region_{name}_recall"] = float(recalls[index])
                metrics[f"region_{name}_true_probability_mean"] = _mean(sums[target_region == index, index])
                id_mean = _mean(id_recalls[start:end + 1])
                metrics[f"region_{name}_id_macro_recall"] = id_mean
                within_region_means.append(id_mean)
            metrics["region_balanced_id_macro_recall"] = float(np.mean(within_region_means))
            for value in range(1, 19):
                metrics[f"region_id_{value:02d}_recall"] = float(id_recalls[value])
            sizes = (first, second - first, 18 - second)
            size_valid = min(sizes) >= min_segments
            has_support = bool(np.all(true_support > 0))
            row = {
                "lower_end": first, "middle_end": second,
                "lower_segments": sizes[0], "middle_segments": sizes[1], "upper_segments": sizes[2],
                "min_segments": int(min_segments), "meets_min_segments": size_valid,
                "has_support_all_regions": has_support,
                "eligible": bool(size_valid and has_support and np.isfinite(metrics["region_balanced_id_macro_recall"])),
                "is_baseline": first == 10 and second == 15,
                "selected": False,
                **metrics,
            }
            candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    best = max(eligible, key=lambda row: (
        row["region_balanced_id_macro_recall"], row["region_macro_recall"], row["region_id_macro_recall"],
        -(abs(row["lower_end"] - 10) + abs(row["middle_end"] - 15)),
        -row["lower_end"], -row["middle_end"],
    )) if eligible else None
    chosen = None
    if best is not None:
        best["selected"] = True
        first, second = best["lower_end"], best["middle_end"]
        chosen = {"lower": list(range(1, first + 1)), "middle": list(range(first + 1, second + 1)), "upper": list(range(second + 1, 19))}
    return {
        "selection_split": "validation",
        "selection_metric": "region_balanced_id_macro_recall",
        "selection_metric_definition": "Mean per-true-ID region recall within each region, then equal mean across all three regions; undefined when any region lacks GT support. Original probability sums are unchanged.",
        "selection_tie_breakers": ["region_macro_recall", "region_id_macro_recall", "closest_to_baseline_10_15", "smaller_boundaries"],
        "selection_status": "selected" if best is not None else "no_eligible_candidate",
        "min_segments": int(min_segments),
        "selected_regions": chosen,
        "selected_metrics": location_metrics(probability, actual, states, mode, ids, chosen) if chosen is not None else None,
        "baseline_regions": {name: members.copy() for name, members in DEFAULT_REGIONS.items()},
        "baseline_metrics": baseline,
        "candidates": candidates,
    }
