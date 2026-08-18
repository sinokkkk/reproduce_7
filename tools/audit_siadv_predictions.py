#!/usr/bin/env python3
"""Recompute SI-Adv clean/adv predictions and strict attack success rates."""

import argparse
import csv
import importlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIADV_ROOT = REPO_ROOT / 'SI-Adv'
if str(SIADV_ROOT) not in sys.path:
    sys.path.insert(0, str(SIADV_ROOT))
if str(SIADV_ROOT / 'model' / 'classifier') not in sys.path:
    sys.path.insert(0, str(SIADV_ROOT / 'model' / 'classifier'))

RUNS = (
    ('ifgm_pointnet', SIADV_ROOT / 'output' / 'modelnet40-ifgm-pointnet'),
    ('ifgm_dgcnn', SIADV_ROOT / 'output' / 'modelnet40-ifgm-dgcnn'),
    ('query_dgcnn_paconv', SIADV_ROOT / 'output' / 'modelnet40-query-dgcnn-paconv'),
)

CSV_FIELDS = [
    'index',
    'sample_id',
    'label',
    'clean_pred',
    'adv_pred_saved',
    'adv_pred_recomputed',
    'prediction_match',
    'clean_correct',
    'raw_success',
    'strict_success',
    'file',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit saved SI-Adv predictions and compute strict clean-correct ASR.'
    )
    parser.add_argument('--device', default=None, help='Torch device, e.g. cuda:0 or cpu.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_ROOT / 'results' / 'siadv-prediction-audit',
        help='New directory for per-sample audits and summaries.',
    )
    parser.add_argument(
        '--runs',
        default=None,
        help='Comma-separated run names; default audits all three committed runs.',
    )
    return parser.parse_args()


def load_json(path):
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_checkpoint(model, model_name, checkpoint_dir, torch):
    base = checkpoint_dir / model_name
    checkpoint_path = next(
        (candidate for candidate in (base.with_suffix(ext) for ext in ('.pth', '.t7', '.tar'))
         if candidate.is_file()),
        None,
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            'No checkpoint found for {} under {}'.format(model_name, checkpoint_dir)
        )
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
    except RuntimeError:
        if not isinstance(checkpoint, dict):
            raise
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint)
    return model, checkpoint_path


def parse_rows(run_dir):
    csv_path = run_dir / 'per_sample.csv'
    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('No rows in {}'.format(csv_path))
    for index, row in enumerate(rows):
        if int(row['index']) != index:
            raise ValueError('{} has non-contiguous index'.format(csv_path))
        path = (run_dir / row['file']).resolve()
        if not path.is_file():
            raise FileNotFoundError('Missing NPZ {}'.format(path))
        row['_path'] = path
    return rows


def make_model(model_name, torch):
    module = importlib.import_module(model_name)
    model = module.get_model(40, normal_channel=False)
    return model


def predict(model, points, batch_size, device, torch):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.from_numpy(np.asarray(points[start:start + batch_size], dtype=np.float32))
            logits = model(batch.transpose(1, 2).contiguous().to(device))
            predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return np.asarray(predictions, dtype=np.int64)


def audit_run(name, run_dir, batch_size, device, torch, output_dir):
    config = load_json(run_dir / 'config.json')
    rows = parse_rows(run_dir)
    target_model_name = config.get('target_model')
    if not target_model_name:
        raise ValueError('{} has no target_model'.format(run_dir))
    model = make_model(target_model_name, torch)
    model, checkpoint_path = load_checkpoint(
        model, target_model_name, SIADV_ROOT / 'checkpoint' / config['dataset'], torch
    )
    model = model.to(device).eval()

    labels = np.empty(len(rows), dtype=np.int64)
    clean_predictions = np.empty(len(rows), dtype=np.int64)
    saved_adv_predictions = np.empty(len(rows), dtype=np.int64)
    clean_points = []
    adv_points = []
    for row in rows:
        with np.load(str(row['_path']), allow_pickle=False) as payload:
            clean = payload['clean_xyz']
            adv = payload['adv_xyz']
            if clean.shape != (1024, 3) or adv.shape != (1024, 3):
                raise ValueError('{} has invalid point shape'.format(row['_path']))
            if clean.dtype != np.float32 or adv.dtype != np.float32:
                raise ValueError('{} has invalid point dtype'.format(row['_path']))
            clean_points.append(clean)
            adv_points.append(adv)
        labels[int(row['index'])] = int(row['label'])
        saved_adv_predictions[int(row['index'])] = int(row['adv_pred'])

    clean_predictions[:] = predict(model, np.asarray(clean_points), batch_size, device, torch)
    recomputed_adv = predict(model, np.asarray(adv_points), batch_size, device, torch)
    clean_correct = clean_predictions == labels
    raw_success = recomputed_adv != labels
    prediction_match = recomputed_adv == saved_adv_predictions
    strict_success = clean_correct & raw_success

    output_csv = output_dir / '{}.csv'.format(name)
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                'index': index,
                'sample_id': row['sample_id'],
                'label': int(labels[index]),
                'clean_pred': int(clean_predictions[index]),
                'adv_pred_saved': int(saved_adv_predictions[index]),
                'adv_pred_recomputed': int(recomputed_adv[index]),
                'prediction_match': int(prediction_match[index]),
                'clean_correct': int(clean_correct[index]),
                'raw_success': int(raw_success[index]),
                'strict_success': int(strict_success[index]),
                'file': row['file'],
            })

    clean_correct_count = int(clean_correct.sum())
    summary = {
        'run': name,
        'run_dir': str(run_dir.resolve()),
        'target_model': target_model_name,
        'checkpoint': str(checkpoint_path.resolve()),
        'sample_count': len(rows),
        'clean_accuracy': float(clean_correct.mean()),
        'clean_correct_count': clean_correct_count,
        'raw_asr': float(raw_success.mean()),
        'adversarial_accuracy': float((recomputed_adv == labels).mean()),
        'strict_asr': (
            float(strict_success.sum() / clean_correct_count)
            if clean_correct_count else None
        ),
        'saved_prediction_match_rate': float(prediction_match.mean()),
        'saved_adv_pred_disagreements': int((~prediction_match).sum()),
        'csv': output_csv.name,
    }
    with (output_dir / '{}.json'.format(name)).open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    return summary


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('SI-Adv audit requires PyTorch: {}'.format(exc))

    device = torch.device(
        args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    selected = [name.strip() for name in args.runs.split(',') if name.strip()] if args.runs else [name for name, _ in RUNS]
    known = dict(RUNS)
    unknown = sorted(set(selected) - set(known))
    if not selected or unknown:
        raise ValueError('Unknown or empty run selection: {}'.format(unknown))

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite {}'.format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix='.{}-'.format(output_dir.name), dir=str(output_dir.parent)))
    try:
        summaries = []
        for name in selected:
            summary = audit_run(name, known[name], args.batch_size, device, torch, temporary_dir)
            summaries.append(summary)
            print(
                '{} clean_acc={:.6f} raw_asr={:.6f} strict_asr={}'.format(
                    name,
                    summary['clean_accuracy'],
                    summary['raw_asr'],
                    '{:.6f}'.format(summary['strict_asr']) if summary['strict_asr'] is not None else 'undefined',
                )
            )
        metadata = {
            'schema_version': 1,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'device': str(device),
            'batch_size': args.batch_size,
            'summaries': summaries,
        }
        with (temporary_dir / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
            handle.write('\n')
        os.replace(str(temporary_dir), str(output_dir))
    except Exception:
        import shutil
        shutil.rmtree(str(temporary_dir), ignore_errors=True)
        raise
    print('Wrote prediction audit to {}'.format(output_dir))


if __name__ == '__main__':
    main()
