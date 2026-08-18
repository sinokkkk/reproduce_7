#!/usr/bin/env python3
"""Evaluate GPointNet score and full energy on aligned SI-Adv arrays."""

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
GPOINTNET_ROOT = SCRIPT_DIR.parent
REPO_ROOT = GPOINTNET_ROOT.parent
if str(GPOINTNET_ROOT) not in sys.path:
    sys.path.insert(0, str(GPOINTNET_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CSV_FIELDS = [
    'index',
    'sample_id',
    'label',
    'attack',
    'score_clean',
    'prior_clean',
    'energy_clean',
    'score_adv',
    'prior_adv',
    'energy_adv',
    'delta_score',
    'delta_energy',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate GPointNet learned scores and full energies on the aligned '
            '1024-point SI-Adv corpus.'
        )
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing manifest.json and aligned .npy arrays.',
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='A GPointNet Lightning .ckpt trained with num_point=1024.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='New directory for energy CSV, arrays, and metadata.',
    )
    stats = parser.add_mutually_exclusive_group(required=True)
    stats.add_argument(
        '--train-data',
        type=Path,
        nargs='+',
        help=(
            'One or more raw GPointNet training .npy files used together to '
            'derive the global ebp min/max.'
        ),
    )
    stats.add_argument(
        '--train-stats',
        type=Path,
        help='JSON containing scalar train_min and train_max values.',
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=None,
        help='Reference Gaussian sigma; defaults to checkpoint ref_sigma.',
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--device',
        default=None,
        help='Torch device, for example cuda:0 or cpu; defaults to CUDA when available.',
    )
    parser.add_argument(
        '--attacks',
        default=None,
        help='Comma-separated manifest attack names; default evaluates every attack.',
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError('Missing JSON file: {}'.format(path))
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def scalar(value, field, source):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError('{} must be a scalar in {}'.format(field, source))
    return array.item()


def read_stats(args):
    if args.train_data is not None:
        files = []
        minima = []
        maxima = []
        for supplied_path in args.train_data:
            path = supplied_path.resolve()
            if not path.is_file():
                raise FileNotFoundError('Missing training data: {}'.format(path))
            data = np.load(str(path), mmap_mode='r', allow_pickle=False)
            if data.ndim < 2 or data.shape[-1] != 3:
                raise ValueError(
                    'Training data must have shape [..., 3], got {} in {}'.format(
                        data.shape, path
                    )
                )
            if not np.isfinite(data).all():
                raise ValueError('Training data contains non-finite values: {}'.format(path))
            minima.append(float(data.min()))
            maxima.append(float(data.max()))
            files.append({
                'path': str(path),
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'sha256': sha256_file(path),
            })
        train_min = min(minima)
        train_max = max(maxima)
        source = {
            'kind': 'train_data',
            'files': files,
        }
    else:
        path = args.train_stats.resolve()
        stats = load_json(path)
        try:
            train_min = float(stats['train_min'])
            train_max = float(stats['train_max'])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                '{} must contain scalar train_min and train_max'.format(path)
            )
        source = {
            'kind': 'train_stats',
            'path': str(path),
            'sha256': sha256_file(path),
        }

    if not np.isfinite(train_min) or not np.isfinite(train_max):
        raise ValueError('Training min/max must be finite')
    if train_max <= train_min:
        raise ValueError(
            'Training max must be greater than min, got {} and {}'.format(
                train_min, train_max
            )
        )
    source.update({'train_min': train_min, 'train_max': train_max})
    return train_min, train_max, source


def load_model(checkpoint_path, device):
    try:
        import torch
        from src.model_point_torch import GPointNet
    except ImportError as exc:
        raise RuntimeError(
            'GPointNet evaluation requires the project PyTorch/Lightning environment: {}'.format(exc)
        )

    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError('Missing checkpoint: {}'.format(checkpoint_path))
    try:
        model = GPointNet.load_from_checkpoint(
            str(checkpoint_path), map_location=device
        )
    except Exception as exc:
        raise RuntimeError(
            'Could not load Lightning checkpoint {}. It must be a GPointNet '
            'checkpoint with saved hyperparameters; original error: {}'.format(
                checkpoint_path, exc
            )
        )
    model = model.to(device)
    model.eval()
    if not hasattr(model, 'C') or not hasattr(model.C, 'num_point'):
        raise ValueError('Checkpoint model does not expose C.num_point')
    if not hasattr(model.C, 'point_dim') or model.C.point_dim != 3:
        raise ValueError('Only point_dim=3 is supported')
    if not hasattr(model, 'energy_net'):
        raise ValueError('Checkpoint does not expose energy_net')
    if getattr(model.C, 'normalize', None) != 'ebp':
        raise ValueError(
            'Checkpoint normalize={!r}; this evaluator only implements GPointNet '
            'global ebp min/max normalization'.format(getattr(model.C, 'normalize', None))
        )
    if not hasattr(model.C, 'ref_sigma'):
        raise ValueError('Checkpoint does not expose ref_sigma')
    return model, checkpoint_path


def validate_input(manifest, input_dir, requested_attacks):
    if manifest.get('dataset') != 'ModelNet40' or manifest.get('split') != 'test':
        raise ValueError('Input manifest must describe the ModelNet40 test split')
    count = int(manifest.get('sample_count', 0))
    point_count = int(manifest.get('point_count', 0))
    if count <= 0 or point_count <= 0:
        raise ValueError('Manifest has invalid sample_count or point_count')
    if manifest.get('coordinate_dtype') != 'float32':
        raise ValueError('Input coordinates must be float32')

    attacks = manifest.get('attacks')
    arrays = manifest.get('arrays')
    if not isinstance(attacks, dict) or not attacks:
        raise ValueError('Manifest has no attacks')
    if not isinstance(arrays, dict) or 'clean_xyz' not in arrays:
        raise ValueError('Manifest has no clean_xyz array')
    if requested_attacks is None:
        attack_names = list(attacks)
    else:
        attack_names = [name.strip() for name in requested_attacks.split(',') if name.strip()]
        unknown = sorted(set(attack_names) - set(attacks))
        if not attack_names or unknown:
            raise ValueError('Unknown or empty attack selection: {}'.format(unknown))

    labels_path = input_dir / manifest['labels']['file']
    labels = np.load(str(labels_path), mmap_mode='r', allow_pickle=False)
    if labels.shape != (count,) or labels.dtype != np.int64:
        raise ValueError('Labels must have shape ({},) and dtype int64'.format(count))

    sample_entries = manifest.get('samples')
    if not isinstance(sample_entries, list) or len(sample_entries) != count:
        raise ValueError('Manifest sample table is incomplete')
    sample_ids = []
    for index, entry in enumerate(sample_entries):
        if int(entry.get('index', -1)) != index or not entry.get('sample_id'):
            raise ValueError('Invalid sample manifest entry at index {}'.format(index))
        if int(entry.get('label', -1)) != int(labels[index]):
            raise ValueError('Manifest label mismatch at index {}'.format(index))
        sample_ids.append(str(entry['sample_id']))
    if len(set(sample_ids)) != count:
        raise ValueError('Manifest sample IDs are not unique')

    array_paths = {}
    array_specs = [('clean_xyz', 'clean_xyz', arrays['clean_xyz']['file'])]
    for attack_name in attack_names:
        array_key = attack_name
        if array_key not in arrays:
            raise ValueError(
                'Manifest attack {} has no matching arrays entry'.format(attack_name)
            )
        expected_file = attacks[attack_name].get('array')
        actual_file = arrays[array_key].get('file')
        if expected_file != actual_file:
            raise ValueError(
                'Manifest attack {} references {!r}, but arrays entry uses {!r}'.format(
                    attack_name, expected_file, actual_file
                )
            )
        array_specs.append((attack_name, array_key, actual_file))

    for key, array_key, filename in array_specs:
        if key in array_paths:
            continue
        path = input_dir / filename
        array = np.load(str(path), mmap_mode='r', allow_pickle=False)
        expected_shape = (count, point_count, 3)
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(
                '{} must have shape {} and dtype float32, got {} {}'.format(
                    path, expected_shape, array.shape, array.dtype
                )
            )
        if not np.isfinite(array).all():
            raise ValueError('{} contains non-finite values'.format(path))
        array_paths[key] = path

    return count, point_count, labels, sample_ids, attack_names, array_paths


def normalize(points, train_min, train_max):
    return ((points - train_min) / (train_max - train_min)) * 2.0 - 1.0


def evaluate_array(model, array_path, count, batch_size, train_min, train_max, sigma, device):
    import torch

    array = np.load(str(array_path), mmap_mode='r', allow_pickle=False)
    scores = np.empty(count, dtype=np.float64)
    priors = np.empty(count, dtype=np.float64)
    energies = np.empty(count, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            points = normalize(array[start:stop], train_min, train_max)
            tensor = torch.from_numpy(np.asarray(points, dtype=np.float32))
            tensor = tensor.transpose(1, 2).contiguous().to(device)
            output = model.energy_net(tensor)
            if not torch.is_tensor(output) or output.ndim != 2 or output.shape[1] != 1:
                raise ValueError(
                    'energy_net must return [B, 1], got {}'.format(
                        getattr(output, 'shape', type(output))
                    )
                )
            score = output[:, 0]
            prior = tensor.square().sum(dim=(1, 2)) / (2.0 * sigma * sigma)
            energy = -score + prior
            scores[start:stop] = score.detach().cpu().numpy()
            priors[start:stop] = prior.detach().cpu().numpy()
            energies[start:stop] = energy.detach().cpu().numpy()
    if not np.isfinite(scores).all() or not np.isfinite(priors).all() or not np.isfinite(energies).all():
        raise ValueError('Model produced non-finite score or energy values')
    return scores, priors, energies


def write_npy(path, array):
    np.save(str(path), np.asarray(array), allow_pickle=False)
    return {
        'file': path.name,
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive')
    train_min, train_max, stats = read_stats(args)

    input_dir = args.input_dir.resolve()
    manifest_path = input_dir / 'manifest.json'
    manifest = load_json(manifest_path)
    count, point_count, labels, sample_ids, attack_names, array_paths = validate_input(
        manifest, input_dir, args.attacks
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'GPointNet evaluation requires PyTorch in the active environment: {}'.format(exc)
        )

    device = torch.device(
        args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model, checkpoint_path = load_model(args.checkpoint, device)
    model_point_count = int(model.C.num_point)
    if model_point_count != point_count:
        raise ValueError(
            'Checkpoint num_point={} does not match input point_count={}; '
            'train/load a 1024-point checkpoint.'.format(model_point_count, point_count)
        )
    sigma = float(args.sigma if args.sigma is not None else model.C.ref_sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be positive and finite')

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite output directory: {}'.format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix='.{}-'.format(output_dir.name), dir=str(output_dir.parent)))

    arrays = {}
    try:
        clean_scores, clean_priors, clean_energies = evaluate_array(
            model, array_paths['clean_xyz'], count, args.batch_size,
            train_min, train_max, sigma, device
        )
        arrays['clean_score'] = write_npy(temporary_dir / 'clean_score.npy', clean_scores)
        arrays['clean_prior'] = write_npy(temporary_dir / 'clean_prior.npy', clean_priors)
        arrays['clean_energy'] = write_npy(temporary_dir / 'clean_energy.npy', clean_energies)

        attack_results = {}
        csv_path = temporary_dir / 'per_sample_energy.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for attack_name in attack_names:
                score, prior, energy = evaluate_array(
                    model, array_paths[attack_name], count, args.batch_size,
                    train_min, train_max, sigma, device
                )
                prefix = attack_name
                arrays[prefix + '_score'] = write_npy(
                    temporary_dir / '{}_score.npy'.format(prefix), score
                )
                arrays[prefix + '_prior'] = write_npy(
                    temporary_dir / '{}_prior.npy'.format(prefix), prior
                )
                arrays[prefix + '_energy'] = write_npy(
                    temporary_dir / '{}_energy.npy'.format(prefix), energy
                )
                delta_score = score - clean_scores
                delta_energy = energy - clean_energies
                attack_results[attack_name] = {
                    'mean_score': float(score.mean()),
                    'mean_prior': float(prior.mean()),
                    'mean_energy': float(energy.mean()),
                    'mean_delta_score': float(delta_score.mean()),
                    'mean_delta_energy': float(delta_energy.mean()),
                    'std_delta_energy': float(delta_energy.std()),
                    'min_delta_energy': float(delta_energy.min()),
                    'max_delta_energy': float(delta_energy.max()),
                }
                for index in range(count):
                    writer.writerow({
                        'index': index,
                        'sample_id': sample_ids[index],
                        'label': int(labels[index]),
                        'attack': attack_name,
                        'score_clean': clean_scores[index],
                        'prior_clean': clean_priors[index],
                        'energy_clean': clean_energies[index],
                        'score_adv': score[index],
                        'prior_adv': prior[index],
                        'energy_adv': energy[index],
                        'delta_score': delta_score[index],
                        'delta_energy': delta_energy[index],
                    })

        metadata = {
            'schema_version': 1,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'input_manifest': {
                'path': str(manifest_path),
                'sha256': sha256_file(manifest_path),
            },
            'checkpoint': {
                'path': str(checkpoint_path),
                'sha256': sha256_file(checkpoint_path),
            },
            'model': {
                'num_point': model_point_count,
                'point_dim': int(model.C.point_dim),
                'net_type': getattr(model.C, 'net_type', None),
                'batch_norm': getattr(model.C, 'batch_norm', None),
                'normalize': getattr(model.C, 'normalize', None),
                'checkpoint_ref_sigma': getattr(model.C, 'ref_sigma', None),
                'device': str(device),
            },
            'normalization': {
                'method': 'GPointNet ebp global scalar min/max',
                'train_min': train_min,
                'train_max': train_max,
                'source': stats,
                'formula': '((x - train_min) / (train_max - train_min)) * 2 - 1',
                'same_transform_for_clean_and_adv': True,
                'random_sampling': False,
                'augmentation_noise': False,
            },
            'energy': {
                'sigma': sigma,
                'score_definition': 'energy_net(X) = f_theta(X)',
                'prior_definition': 'sum(X^2) / (2 * sigma^2)',
                'full_energy_definition': '-f_theta(X) + sum(X^2) / (2 * sigma^2)',
            },
            'sample_count': count,
            'point_count': point_count,
            'attacks': attack_results,
            'arrays': arrays,
            'csv': {
                'file': csv_path.name,
                'rows': count * len(attack_names),
                'columns': CSV_FIELDS,
            },
        }
        metadata_path = temporary_dir / 'manifest.json'
        with metadata_path.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=True)
            handle.write('\n')
        os.replace(str(temporary_dir), str(output_dir))
    except Exception:
        import shutil
        shutil.rmtree(str(temporary_dir), ignore_errors=True)
        raise

    print('Evaluated {} samples for {} attack variants'.format(count, len(attack_names)))
    print('Wrote energy results to {}'.format(output_dir))
    print('Manifest: {}'.format(output_dir / 'manifest.json'))


if __name__ == '__main__':
    main()
