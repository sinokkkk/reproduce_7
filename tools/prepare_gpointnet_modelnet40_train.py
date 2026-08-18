#!/usr/bin/env python3
"""Build GPointNet ModelNet40 training arrays with SI-Adv preprocessing."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Convert ModelNet40 normal-resampled training samples into per-category '
            'GPointNet arrays using the same first-N and unit-sphere normalization '
            'protocol as SI-Adv.'
        )
    )
    parser.add_argument('--modelnet-root', type=Path, required=True)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_ROOT / 'GPointNet' / 'data' / 'modelnet40-siadv-1024',
    )
    parser.add_argument('--num-point', type=int, default=1024)
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


def read_lines(path):
    if not path.is_file():
        raise FileNotFoundError('Missing ModelNet40 metadata: {}'.format(path))
    with path.open('r', encoding='utf-8') as handle:
        values = [line.strip() for line in handle if line.strip()]
    if not values:
        raise ValueError('Empty ModelNet40 metadata: {}'.format(path))
    return values


def category_from_sample_id(sample_id):
    parts = sample_id.rsplit('_', 1)
    if len(parts) != 2 or not parts[0]:
        raise ValueError('Invalid ModelNet40 sample ID: {!r}'.format(sample_id))
    return parts[0]


def normalize_like_siadv(points, sample_id):
    centroid = points.mean(axis=0)
    centered = points - centroid
    radius = np.sqrt(np.sum(centered ** 2, axis=1)).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Degenerate point cloud: {}'.format(sample_id))
    normalized = centered / radius
    if not np.isfinite(normalized).all():
        raise ValueError('Non-finite normalized point cloud: {}'.format(sample_id))
    return normalized.astype(np.float32, copy=False)


def main():
    args = parse_args()
    if args.num_point <= 0:
        raise ValueError('--num-point must be positive')
    root = args.modelnet_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError('Missing ModelNet40 root: {}'.format(root))
    categories = read_lines(root / 'modelnet40_shape_names.txt')
    sample_ids = read_lines(root / 'modelnet40_train.txt')
    category_set = set(categories)
    grouped = defaultdict(list)
    source_files = []

    global_min = float('inf')
    global_max = float('-inf')
    for index, sample_id in enumerate(sample_ids):
        category = category_from_sample_id(sample_id)
        if category not in category_set:
            raise ValueError(
                'Sample {} has unknown category {!r}'.format(sample_id, category)
            )
        source_path = root / category / '{}.txt'.format(sample_id)
        if not source_path.is_file():
            raise FileNotFoundError('Missing ModelNet40 sample: {}'.format(source_path))
        points = np.loadtxt(str(source_path), delimiter=',', dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError('{} has invalid shape {}'.format(source_path, points.shape))
        if points.shape[0] < args.num_point:
            raise ValueError(
                '{} has {} points; expected at least {}'.format(
                    source_path, points.shape[0], args.num_point
                )
            )
        xyz = np.asarray(points[:args.num_point, :3], dtype=np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError('{} contains non-finite XYZ'.format(source_path))
        normalized = normalize_like_siadv(xyz, sample_id)
        grouped[category].append(normalized)
        global_min = min(global_min, float(normalized.min()))
        global_max = max(global_max, float(normalized.max()))
        source_files.append({
            'index': index,
            'sample_id': sample_id,
            'category': category,
            'source': str(source_path),
        })

    missing = [category for category in categories if not grouped[category]]
    if missing:
        raise ValueError('No training samples for categories: {}'.format(missing))

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite {}'.format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix='.{}-'.format(output_dir.name), dir=str(output_dir.parent)
    ))
    arrays = {}
    try:
        for category in categories:
            array = np.stack(grouped[category], axis=0).astype(np.float32, copy=False)
            expected_shape = (len(grouped[category]), args.num_point, 3)
            if array.shape != expected_shape:
                raise ValueError('{} array has invalid shape {}'.format(category, array.shape))
            path = temporary_dir / '{}_train.npy'.format(category)
            np.save(str(path), array, allow_pickle=False)
            arrays[category] = {
                'file': path.name,
                'shape': list(array.shape),
                'dtype': str(array.dtype),
                'bytes': path.stat().st_size,
                'sha256': sha256_file(path),
            }

        stats_path = temporary_dir / 'ebp_stats.json'
        stats = {
            'train_min': global_min,
            'train_max': global_max,
            'normalization_applied_before_ebp': (
                'SI-Adv pc_normalize: subtract per-sample XYZ centroid and divide '
                'by per-sample maximum point radius'
            ),
            'ebp_formula': '((x - train_min) / (train_max - train_min)) * 2 - 1',
            'sample_count': len(sample_ids),
            'point_count': args.num_point,
        }
        with stats_path.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(stats, handle, indent=2)
            handle.write('\n')

        manifest = {
            'schema_version': 1,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': 'ModelNet40',
            'split': 'train',
            'source_root': str(root),
            'source_train_list': str(root / 'modelnet40_train.txt'),
            'source_train_list_sha256': sha256_file(root / 'modelnet40_train.txt'),
            'sample_count': len(sample_ids),
            'point_count': args.num_point,
            'coordinate_dtype': 'float32',
            'sampling': 'first num_point rows, matching SI-Adv uniform=False',
            'normalization': stats['normalization_applied_before_ebp'],
            'category_order': categories,
            'arrays': arrays,
            'ebp_stats': {
                'file': stats_path.name,
                'train_min': global_min,
                'train_max': global_max,
            },
            'samples': source_files,
        }
        manifest_path = temporary_dir / 'manifest.json'
        with manifest_path.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=True)
            handle.write('\n')
        os.replace(str(temporary_dir), str(output_dir))
    except Exception:
        shutil.rmtree(str(temporary_dir), ignore_errors=True)
        raise

    print('Converted {} ModelNet40 training samples'.format(len(sample_ids)))
    print('Wrote GPointNet training arrays to {}'.format(output_dir))
    print('Global ebp min/max: {} {}'.format(global_min, global_max))


if __name__ == '__main__':
    main()
