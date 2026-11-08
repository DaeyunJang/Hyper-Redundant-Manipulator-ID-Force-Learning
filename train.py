#!/usr/bin/env python3
"""Train HRM models: prepare data, fit independent force/ID branches, save results.

User settings are in configs/train.yaml; model definitions are in model_zoo.py.
--check-data validates input files and splits without constructing or training models.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from data_utils import (
    ROOT, load_config, prepare_data, trial_endpoints, serializable, write_json,
    force_metrics, location_metrics, ensure_driver_runtime,
)
if __name__ == '__main__':
    ensure_driver_runtime()

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from model_zoo import build_estimator, MODEL_NAMES


# Output paths and reproducible initialization.
def fresh_directory(kind, explicit=None):
    path = Path(explicit) if explicit else ROOT / 'results' / kind / datetime.now().strftime('%Y%m%d_%H%M%S')
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def seed_all(seed, threads=4, deterministic=False, benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


# Normalization and input windows that stay within file and split boundaries.
def fit_scaler(trials, indices, scope):
    """Fit once on valid raw train rows, never on repeated overlapping windows."""
    selected = [trials[i] for i in indices if scope != 'tip' or trials[i].is_tip]
    if not selected:
        raise ValueError(f'No training trials for {scope}')
    def train_rows(t, valid):
        start, stop = getattr(t, 'info', {}).get('role_rows', {}).get('train', [0, len(t.x)])
        mask = valid.copy()
        mask[:start] = False
        mask[stop:] = False
        return mask
    xs = [t.x[train_rows(t, t.input_valid)] for t in selected]
    x = np.concatenate(xs).astype(np.float64)
    ys = [t.force[train_rows(t, t.force_valid)] for t in selected]
    y = np.concatenate(ys).astype(np.float64)
    if not len(x) or not len(y):
        raise ValueError(f'No finite training rows for {scope}')
    return {'x_mean': x.mean(0), 'x_std': np.maximum(x.std(0), 1e-8),
            'y_mean': y.mean(0), 'y_std': np.maximum(y.std(0), 1e-6)}


class WindowStore:
    """Keep raw rows once on device; gather only the requested past windows."""

    def __init__(self, trials, indices, scaler, config, device, branch='location',
                 mode='masked', prediction=False, role=None):
        self.device = torch.device(device)
        self.window = int(config['window_samples'])
        self.branch = branch
        self.mode = mode
        self.scaler = scaler
        active = list(config.get('active_ids', range(1, 19)))
        class_map = {seg: i + int(mode == 'class0') for i, seg in enumerate(active)}
        xx, ends, yy, ids, states, labels, refs, force_valid = [], [], [], [], [], [], [], []
        offset = 0
        for ti in indices:
            trial = trials[ti]
            if not prediction and branch == 'force' and config.get('force_scope', 'tip') == 'tip' and not trial.is_tip:
                continue
            ep = np.asarray(trial_endpoints(trial, role, prediction), dtype=np.int64)
            normalized = (trial.x - scaler['x_mean']) / scaler['x_std']
            # Invalid rows cannot enter a valid window; finite fillers only avoid GPU NaNs
            # in storage. Original arrays/validity are kept unchanged.
            xx.append(np.nan_to_num(normalized, nan=0, posinf=0, neginf=0).astype(np.float32))
            ends.append(ep + offset)
            yy.append(trial.force[ep])
            force_valid.append(trial.force_valid[ep])
            ids.append(np.full(len(ep), trial.effective_id, dtype=np.int64))
            states.append(trial.load_state[ep])
            label = np.full(len(ep), -100, dtype=np.int64)
            loaded = trial.load_state[ep] == 1
            if trial.effective_id in class_map:
                label[loaded] = class_map[trial.effective_id]
            if mode == 'class0':
                label[trial.load_state[ep] == 0] = 0
            labels.append(label)
            refs.extend((ti, int(row)) for row in ep)
            offset += len(trial.x)
        if not xx or not sum(map(len, ends)):
            raise ValueError(f'No usable {branch}/{mode} endpoints')
        self.x = torch.as_tensor(np.concatenate(xx), device=device)
        self.ends = torch.as_tensor(np.concatenate(ends), device=device)
        self.force_N = np.concatenate(yy).astype(np.float64)
        self.force_valid = np.concatenate(force_valid).astype(bool)
        self.actual_ids = np.concatenate(ids)
        self.load_state = np.concatenate(states)
        self.refs = refs
        self.targets = torch.as_tensor(np.concatenate(labels), device=device)
        self.y_mean = torch.as_tensor(scaler['y_mean'], dtype=torch.float32, device=device)
        self.y_std = torch.as_tensor(scaler['y_std'], dtype=torch.float32, device=device)
        target = (self.force_N - scaler['y_mean']) / scaler['y_std']
        self.y = torch.as_tensor(target.astype(np.float32), device=device)
        self.offsets = torch.arange(1 - self.window, 1, device=device)
        eligible = self.force_valid & np.isfinite(self.force_N).all(1) if branch == 'force' else np.concatenate(labels) >= 0
        self.eligible = torch.as_tensor(np.flatnonzero(eligible), device=device)

    def __len__(self):
        return len(self.ends)

    def batch(self, ix, current_only=False):
        ends = self.ends[ix]
        if current_only:
            return self.x[ends].unsqueeze(1)
        return self.x[ends[:, None] + self.offsets[None, :]]


@torch.inference_mode()
def validate_branch(net, store, batch_size):
    net.eval()
    total = torch.zeros((), dtype=torch.float64, device=store.device)
    count = 0
    for begin in range(0, len(store.eligible), batch_size):
        ix = store.eligible[begin:begin + batch_size]
        out = net(store.batch(ix, net.name == 'mlp'))
        if store.branch == 'force':
            total += (((out - store.y[ix]) * store.y_std).double() ** 2).sum()
            count += len(ix) * 3
        else:
            total += F.cross_entropy(out, store.targets[ix], reduction='sum').double()
            count += len(ix)
    if not count:
        raise ValueError(f'No valid validation labels: {store.branch}/{store.mode}')
    metric = float(total / count)
    return math.sqrt(metric) * 1000 if store.branch == 'force' else metric


# AUTHOR: Daeyun
# Project training implementation: optimizer, loss and early stopping.
def train_branch(net, train, val, cfg, seed, progress_name):
    """New optimizer on every call; no reuse, resume, or sibling forward."""
    seed_all(seed, cfg.get('threads', 4), cfg.get('cudnn_deterministic', False), cfg.get('cudnn_benchmark', True))
    if not len(train.eligible) or not len(val.eligible):
        raise ValueError('The branch requires train and validation supervision')
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.get(f'{train.branch}_learning_rate', cfg['learning_rate']),
                                 weight_decay=cfg.get('weight_decay', 0.0001))
    history, best_state = [], None
    best, stale, best_epoch = math.inf, 0, 0
    start = time.perf_counter()
    batch_size = int(cfg['batch_size'])
    for epoch in range(1, int(cfg['max_epochs']) + 1):
        net.train()
        perm = train.eligible[torch.randperm(len(train.eligible), device=train.device)]
        total = torch.zeros((), device=train.device)
        for begin in range(0, len(perm), batch_size):
            ix = perm[begin:begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            out = net(train.batch(ix, net.name == 'mlp'))
            if train.branch == 'force':
                loss_fn = {'mse': F.mse_loss, 'smooth_l1': F.smooth_l1_loss}[cfg.get('force_loss', 'mse')]
                loss = loss_fn(out, train.y[ix])
            else:
                loss = F.cross_entropy(out, train.targets[ix])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {progress_name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.get('gradient_clip_norm', 1.0))
            optimizer.step()
            total += loss.detach() * len(ix)
        metric = validate_branch(net, val, batch_size)
        if not math.isfinite(metric):
            raise FloatingPointError(f'Nonfinite validation: {progress_name}')
        history.append({'epoch': epoch, 'train_loss': float(total / len(perm)),
                        'validation_metric': metric, 'seconds': time.perf_counter() - start})
        if metric < best:
            best, best_epoch, stale = metric, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= cfg['patience']:
            print(f'{progress_name} epoch={epoch} best={best:.6f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        if stale >= cfg['patience']:
            break
    net.load_state_dict(best_state)
    net.eval()
    return {'best_epoch': best_epoch, 'best_metric': best,
            'metric': 'rmse_xyz_mN' if train.branch == 'force' else 'cross_entropy',
            'epochs': len(history), 'train_seconds': time.perf_counter() - start,
            'seed': seed, 'train_samples': len(train.eligible), 'validation_samples': len(val.eligible),
            'parameters': sum(p.numel() for p in net.parameters()), 'history': history}


@torch.inference_mode()
def predict_branch(net, store, batch_size):
    net.eval()
    outputs = []
    for begin in range(0, len(store), batch_size):
        ix = torch.arange(begin, min(begin + batch_size, len(store)), device=store.device)
        out = net(store.batch(ix, net.name == 'mlp'))
        out = out * store.y_std + store.y_mean if store.branch == 'force' else out.softmax(-1)
        outputs.append(out.cpu().numpy())
    return np.concatenate(outputs)


# Save normalization, input order and calibration alongside the model weights.
def bundle_for(model, cfg, prepared, scalers, stats, mode, force_name, location_name):
    return {'schema_version': 2, 'kind': 'independent_hrm', 'mode': mode,
            'force_name': force_name, 'location_name': location_name,
            'force_params': model.force_net.params, 'location_params': model.location_net.params,
            'input_dim': model.force_net.input_dim, 'num_classes': model.location_net.output_dim,
            'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'config': serializable(cfg), 'scalers': serializable(scalers),
            'calibration': serializable(prepared['calibration']),
            'feature_columns': prepared.get('feature_columns', cfg.get('feature_columns')),
            'split': prepared['split'], 'data_audit': serializable(prepared['audit']),
            'training': serializable(stats), 'training_sign_multiplier': 1,
            'future_hrm_control_multiplier': -1, 'force_unit': 'N',
            'force_scope': cfg.get('force_scope', 'tip'),
            'deployment_gate': 'class0_argmax' if mode == 'class0' else None}


def restore_bundle(path, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('kind') != 'independent_hrm':
        raise ValueError('Expected an independent HRM bundle')
    cfg = bundle['config']
    # Older bundles without this option used audited preprocessing; preserve that policy.
    cfg.setdefault('preprocessing', 'audited')
    cfg.setdefault('tip_scope_by_id', False)
    # Match training execution/precision policy without reseeding caller RNGs.
    torch.set_num_threads(cfg.get('threads', 4))
    torch.backends.cudnn.deterministic = cfg.get('cudnn_deterministic', False)
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_estimator(bundle['force_name'], bundle['location_name'], bundle['input_dim'],
                            cfg['window_samples'], bundle['num_classes'], bundle['force_params'],
                            bundle['location_params']).to(device)
    model.load_state_dict(bundle['state_dict'], strict=True)
    model.eval()
    return model, bundle


# Main training workflow: start here when modifying the experiment runner.
def train_experiment(config_path, output=None, epochs=None, models=None, modes=None, check_data=False):
    cfg = load_config(config_path)
    if epochs is not None:
        cfg['max_epochs'] = epochs
    if models:
        cfg['force_candidates'] = cfg['location_candidates'] = models
    if modes:
        cfg['no_load_modes'] = modes
    for key in ('max_epochs', 'patience', 'batch_size'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('force_candidates', 'location_candidates'):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or set(cfg[key]) - set(MODEL_NAMES):
            raise ValueError(f'Invalid/duplicate {key}')
    if not cfg['no_load_modes'] or len(set(cfg['no_load_modes'])) != len(cfg['no_load_modes']) or set(cfg['no_load_modes']) - {'masked', 'class0'}:
        raise ValueError('Invalid/duplicate no_load_modes')
    if cfg.get('force_loss', 'mse') not in ('mse', 'smooth_l1'):
        raise ValueError('force_loss must be mse or smooth_l1')
    if cfg.get('force_scope', 'tip') not in ('tip', 'all_single_contact'):
        raise ValueError('force_scope must be tip or all_single_contact')
    outdir = fresh_directory('train', output)
    print(f'Preparing dataset: {outdir}', flush=True)
    prepared = prepare_data(cfg, outdir)
    cfg = prepared.get('config', cfg)
    (outdir / 'config_resolved.yaml').write_text(yaml.safe_dump(serializable(cfg), allow_unicode=True, sort_keys=False))
    if check_data:
        print('Data check complete (no model training):',
              {role: {'files': len(indices),
                      'windows': sum(len(trial_endpoints(prepared['trials'][i], role)) for i in indices)}
               for role, indices in prepared['split'].items()}, flush=True)
        return outdir, prepared, []
    seed_all(cfg['seed'], cfg.get('threads', 4), cfg.get('cudnn_deterministic', False), cfg.get('cudnn_benchmark', True))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_json(outdir / 'environment.json', {'python': sys.version, 'executable': sys.executable,
               'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'device': str(device),
               'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
               'cudnn_deterministic': torch.backends.cudnn.deterministic,
               'cudnn_benchmark': torch.backends.cudnn.benchmark})
    (outdir / 'environment_packages.txt').write_text('\n'.join(sorted(
        f'{dist.metadata["Name"]}=={dist.version}' for dist in importlib.metadata.distributions())) + '\n')
    trials, split = prepared['trials'], prepared['split']
    scalers = {'force': fit_scaler(trials, split['train'], cfg.get('force_scope', 'tip')),
               'location': fit_scaler(trials, split['train'], 'all')}
    if not cfg.get('target_scaling', True):
        scalers['force']['y_mean'] = np.zeros(3)
        scalers['force']['y_std'] = np.ones(3)
    force_stores = {part: WindowStore(trials, split[part], scalers['force'], cfg, device, 'force', role=part)
                    for part in ('train', 'validation')}
    loc_stores = {mode: {part: WindowStore(trials, split[part], scalers['location'], cfg, device, 'location', mode, role=part)
                        for part in ('train', 'validation')} for mode in cfg['no_load_modes']}
    rows = []
    active = cfg.get('active_ids', list(range(1, 19)))
    total = len(cfg['force_candidates']) * len(cfg['location_candidates']) * len(cfg['no_load_modes'])
    for mode in cfg['no_load_modes']:
        for fn in cfg['force_candidates']:
            for ln in cfg['location_candidates']:
                name = f'{mode}__force_{fn}__location_{ln}'
                dest = outdir / 'models' / name
                dest.mkdir(parents=True, exist_ok=False)
                print(f'[{len(rows)+1}/{total}] {name}', flush=True)
                seed_all(cfg['seed'], cfg.get('threads', 4), cfg.get('cudnn_deterministic', False), cfg.get('cudnn_benchmark', True))
                model = build_estimator(fn, ln, force_stores['train'].x.shape[1], cfg['window_samples'],
                           len(active) + int(mode == 'class0'), cfg.get('force_params', {}).get(fn),
                           cfg.get('location_params', {}).get(ln), cfg['seed'], cfg['seed'] + 1).to(device)
                try:
                    force_stats = train_branch(model.force_net, force_stores['train'], force_stores['validation'],
                                               cfg, cfg['seed'], name + '/force')
                    loc_stats = train_branch(model.location_net, loc_stores[mode]['train'], loc_stores[mode]['validation'],
                                             cfg, cfg['seed'] + 1, name + '/location')
                except Exception as error:
                    write_json(dest / 'failure.json', {'status': 'failed', 'error': repr(error)})
                    write_json(outdir / 'progress.json', {'complete': len(rows), 'total': total,
                               'last': name, 'status': 'failed', 'error': repr(error)})
                    raise
                fp = predict_branch(model.force_net, force_stores['validation'], cfg['batch_size'])
                lp = predict_branch(model.location_net, loc_stores[mode]['validation'], cfg['batch_size'])
                metrics = force_metrics(fp, force_stores['validation'].force_N, force_stores['validation'].force_valid)
                for state, label in ((0, 'unloaded'), (1, 'loaded')):
                    sub = force_metrics(fp, force_stores['validation'].force_N,
                                        (force_stores['validation'].load_state == state) & force_stores['validation'].force_valid)
                    metrics.update({label + '_' + k: v for k, v in sub.items()})
                ls = loc_stores[mode]['validation']
                metrics.update(location_metrics(lp, ls.actual_ids, ls.load_state, mode, active, cfg.get('regions')))
                stats = {'force': force_stats, 'location': loc_stats}
                bundle = bundle_for(model, cfg, prepared, scalers, stats, mode, fn, ln)
                bundle['validation_metrics'] = serializable(metrics)
                torch.save(bundle, dest / 'hrm_bundle.pt')
                write_json(dest / 'training.json', stats)
                np.savez_compressed(dest / 'validation_predictions.npz', force_N=fp, location_probabilities=lp,
                                    force_refs=np.asarray(force_stores['validation'].refs),
                                    location_refs=np.asarray(ls.refs), actual_ids=ls.actual_ids, load_state=ls.load_state)
                row = {'mode': mode, 'force_model': fn, 'location_model': ln, 'status': 'complete',
                       'force_best_epoch': force_stats['best_epoch'], 'location_best_epoch': loc_stats['best_epoch'],
                       'force_validation_rmse_mN': force_stats['best_metric'],
                       'location_validation_ce': loc_stats['best_metric'],
                       'force_train_seconds': force_stats['train_seconds'],
                       'location_train_seconds': loc_stats['train_seconds'],
                       'force_parameters': force_stats['parameters'], 'location_parameters': loc_stats['parameters'],
                       **metrics, 'bundle': str((dest / 'hrm_bundle.pt').relative_to(ROOT))
                       if (dest / 'hrm_bundle.pt').is_relative_to(ROOT) else str(dest / 'hrm_bundle.pt')}
                rows.append(row)
                pd.DataFrame(rows).to_csv(outdir / 'comparison.csv', index=False)
                write_json(outdir / 'progress.json', {'complete': len(rows), 'total': total, 'last': name})
                del model
    # Select models using validation only. Test files are not evaluated here.
    from predict import select_comparison_rows, export_validation_workbook
    comparison = pd.DataFrame(rows)
    selected = select_comparison_rows(comparison)
    write_json(outdir / 'summary.json', {
        'dataset_role': 'validation', 'trained_combinations': len(rows),
        'selection_basis': 'Validation force RMSE, then location macro recall; never test',
        'best_by_mode': {row['mode']: row.to_dict() for _, row in selected},
        'force_scope': cfg.get('force_scope', 'tip'),
        'workbook_policy': 'One sheet per branch/model/mode, fixed first partner; all pairs in comparison',
    })
    if cfg.get('export_excel', True):
        export_validation_workbook(outdir)
    write_json(outdir / 'progress.json', {'complete': len(rows), 'total': total, 'status': 'complete'})
    print(f'Training complete: {outdir}', flush=True)
    return outdir, prepared, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--output')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--no-load-modes', nargs='+', choices=['masked', 'class0'])
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args(argv)
    train_experiment(args.config, args.output, args.epochs, args.models, args.no_load_modes, args.check_data)


if __name__ == '__main__':
    main()
