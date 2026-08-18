#!/usr/bin/env python3
"""Validate SI-Adv runs and build an ordered corpus for GPointNet evaluation."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NPZ_FIELDS = {
    'clean_xyz',
    'adv_xyz',
    'label',
    'adv_pred',
    'sample_id',
}
REQUIRED_CSV_FIELDS = {
    'index',
    'sample_id',
    'label',
    'adv_pred',
    'success',
    'query_count',
    'mse_l2',
    'linf',
    'chamfer',
    'hausdorff',
    'elapsed_seconds',
    'file',
}
RUN_DEFINITIONS = (
    {
        'name': 'ifgm_pointnet',
        'default_dir': REPO_ROOT / 'SI-Adv' / 'output' / 'modelnet40-ifgm-pointnet',
        'array_file': 'ifgm_pointnet_adv.npy',
        'config': {
            'dataset': 'ModelNet40',
            'input_point_nums': 1024,
            'batch_size': 1,
            'transfer_attack_method': 'ifgm_ours',
            'query_attack_method': None,
            'surrogate_model': 'pointnet_cls',
            'target_model': 'pointnet_cls',
        },
    },
    {
        'name': 'ifgm_dgcnn',
        'default_dir': REPO_ROOT / 'SI-Adv' / 'output' / 'modelnet40-ifgm-dgcnn',
        'array_file': 'ifgm_dgcnn_adv.npy',
        'config': {
            'dataset': 'ModelNet40',
            'input_point_nums': 1024,
            'batch_size': 1,
            'transfer_attack_method': 'ifgm_ours',
            'query_attack_method': None,
            'surrogate_model': 'dgcnn',
            'target_model': 'dgcnn',
        },
    },
    {
        'name': 'query_dgcnn_paconv',
        'default_dir': REPO_ROOT / 'SI-Adv' / 'output' / 'modelnet40-query-dgcnn-paconv',
        'array_file': 'query_dgcnn_paconv_adv.npy',
        'config': {
            'dataset': 'ModelNet40',
            'input_point_nums': 1024,
            'batch_size': 1,
            'transfer_attack_method': None,
            'query_attack_method': 'ours',
            'surrogate_model': 'dgcnn',
            'target_model': 'paconv',
        },
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Validate the three committed SI-Adv ModelNet40 runs and export '
            'aligned float32 arrays for GPointNet energy evaluation.'
        )
    )
    parser.add_argument(
        '--ifgm-pointnet-run',
        type=Path,
        default=RUN_DEFINITIONS[0]['default_dir'],
    )
    parser.add_argument(
        '--ifgm-dgcnn-run',
        type=Path,
        default=RUN_DEFINITIONS[1]['default_dir'],
    )
    parser.add_argument(
        '--query-dgcnn-paconv-run',
        type=Path,
        default=RUN_DEFINITIONS[2]['default_dir'],
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_ROOT / 'results' / 'gpointnet-inputs' / 'modelnet40-siadv-1024',
    )
    parser.add_argument(
        '--expected-samples',
        type=int,
        default=2468,
        help='Expected number of ModelNet40 test samples.',
    )
    return parser.parse_args()


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError('Missing JSON file: {}'.format(path))
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def validate_config(run_dir, definition):
    config_path = run_dir / 'config.json'
    config = load_json(config_path)
    mismatches = []
    for key, expected in definition['config'].items():
        actual = config.get(key)
        if actual != expected:
            mismatches.append('{}={!r} (expected {!r})'.format(key, actual, expected))
    if mismatches:
        raise ValueError(
            'Unexpected configuration in {}: {}'.format(
                config_path, ', '.join(mismatches)
            )
        )
    return config_path, config


def read_csv_rows(run_dir, expected_samples):
    csv_path = run_dir / 'per_sample.csv'
    if not csv_path.is_file():
        raise FileNotFoundError('Missing CSV file: {}'.format(csv_path))
    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_FIELDS - fields
        if missing:
            raise ValueError(
                '{} is missing columns: {}'.format(csv_path, sorted(missing))
            )
        rows = list(reader)

    if len(rows) != expected_samples:
        raise ValueError(
            '{} contains {} rows; expected {}'.format(
                csv_path, len(rows), expected_samples
            )
        )

    run_root = run_dir.resolve()
    sample_ids = set()
    referenced_files = set()
    for expected_index, row in enumerate(rows):
        index = parse_int(row, 'index', csv_path)
        if index != expected_index:
            raise ValueError(
                '{} row {} has index {}'.format(csv_path, expected_index + 2, index)
            )
        sample_id = row['sample_id']
        if not sample_id or sample_id in sample_ids:
            raise ValueError(
                '{} has an empty or duplicate sample_id {!r}'.format(
                    csv_path, sample_id
                )
            )
        sample_ids.add(sample_id)

        sample_path = (run_dir / row['file']).resolve()
        try:
            sample_path.relative_to(run_root)
        except ValueError:
            raise ValueError(
                '{} references a file outside its run: {}'.format(csv_path, sample_path)
            )
        if sample_path.suffix.lower() != '.npz' or not sample_path.is_file():
            raise FileNotFoundError('Missing sample NPZ: {}'.format(sample_path))
        if sample_path in referenced_files:
            raise ValueError('{} references a duplicate NPZ'.format(csv_path))
        referenced_files.add(sample_path)
        row['_sample_path'] = sample_path

    actual_files = {path.resolve() for path in (run_dir / 'samples').glob('*.npz')}
    if actual_files != referenced_files:
        missing = referenced_files - actual_files
        extra = actual_files - referenced_files
        raise ValueError(
            '{} NPZ linkage mismatch: {} missing, {} unreferenced'.format(
                run_dir, len(missing), len(extra)
            )
        )
    return csv_path, rows


def parse_int(row, field, source):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError('{} has invalid integer {}={!r}'.format(source, field, row.get(field)))


def parse_float(row, field, source):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError('{} has invalid float {}={!r}'.format(source, field, row.get(field)))
    if not np.isfinite(value):
        raise ValueError('{} has non-finite {}={!r}'.format(source, field, value))
    return value


def scalar_value(value, field, source):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError('{} field {} must be scalar'.format(source, field))
    return array.item()


def assert_close(actual, expected, field, source, atol=1e-6):
    if not np.isclose(actual, expected, rtol=1e-6, atol=atol):
        raise ValueError(
            '{} {} mismatch: stored {!r}, recomputed {!r}'.format(
                source, field, actual, expected
            )
        )


def portable_path(path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def array_record(path, shape, role, source_run):
    return {
        'file': path.name,
        'role': role,
        'dtype': 'float32',
        'shape': list(shape),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
        'source_run': source_run,
    }


def validate_cross_run_rows(contexts):
    reference = contexts[0]
    reference_data_path = reference['config'].get('data_path')
    reference_seed = reference['config'].get('seed')
    for context in contexts[1:]:
        if context['config'].get('data_path') != reference_data_path:
            raise ValueError('Runs use different source data paths')
        if context['config'].get('seed') != reference_seed:
            raise ValueError('Runs use different random seeds')
        for index, (expected, actual) in enumerate(
                zip(reference['rows'], context['rows'])):
            if actual['sample_id'] != expected['sample_id']:
                raise ValueError(
                    'Sample ordering differs at index {}: {!r} versus {!r}'.format(
                        index, expected['sample_id'], actual['sample_id']
                    )
                )
            if parse_int(actual, 'label', context['csv_path']) != parse_int(
                    expected, 'label', reference['csv_path']):
                raise ValueError('Labels differ at index {}'.format(index))


def process_run(context, clean_memmap, adv_memmap, labels, samples, is_reference):
    definition = context['definition']
    metric_totals = {
        'l2': 0.0,
        'linf': 0.0,
        'chamfer': 0.0,
        'hausdorff': 0.0,
        'elapsed_seconds': 0.0,
        'query_count': 0.0,
        'success': 0,
    }

    for index, row in enumerate(context['rows']):
        sample_path = row['_sample_path']
        with np.load(str(sample_path), allow_pickle=False) as payload:
            fields = set(payload.files)
            if fields != EXPECTED_NPZ_FIELDS:
                raise ValueError(
                    '{} fields are {}; expected {}'.format(
                        sample_path, sorted(fields), sorted(EXPECTED_NPZ_FIELDS)
                    )
                )
            clean_xyz = payload['clean_xyz']
            adv_xyz = payload['adv_xyz']
            npz_label = scalar_value(payload['label'], 'label', sample_path)
            npz_adv_pred = scalar_value(payload['adv_pred'], 'adv_pred', sample_path)
            npz_sample_id = str(
                scalar_value(payload['sample_id'], 'sample_id', sample_path)
            )

        expected_shape = (context['point_count'], 3)
        for field, points in (('clean_xyz', clean_xyz), ('adv_xyz', adv_xyz)):
            if points.shape != expected_shape:
                raise ValueError(
                    '{} {} has shape {}; expected {}'.format(
                        sample_path, field, points.shape, expected_shape
                    )
                )
            if points.dtype != np.float32:
                raise ValueError(
                    '{} {} has dtype {}; expected float32'.format(
                        sample_path, field, points.dtype
                    )
                )
            if not np.isfinite(points).all():
                raise ValueError('{} {} contains non-finite values'.format(sample_path, field))

        csv_label = parse_int(row, 'label', context['csv_path'])
        csv_adv_pred = parse_int(row, 'adv_pred', context['csv_path'])
        success = parse_int(row, 'success', context['csv_path'])
        query_count = parse_int(row, 'query_count', context['csv_path'])
        if npz_sample_id != row['sample_id']:
            raise ValueError('{} sample_id does not match CSV'.format(sample_path))
        if int(npz_label) != csv_label or int(npz_adv_pred) != csv_adv_pred:
            raise ValueError('{} predictions do not match CSV'.format(sample_path))
        if success not in (0, 1) or success != int(csv_adv_pred != csv_label):
            raise ValueError('{} has inconsistent success flag'.format(sample_path))
        if definition['config']['query_attack_method'] is None and query_count != 0:
            raise ValueError('{} has a query count for a white-box run'.format(sample_path))
        if definition['config']['query_attack_method'] is not None and query_count <= 0:
            raise ValueError('{} has a non-positive query count'.format(sample_path))

        delta = adv_xyz - clean_xyz
        l2 = float(np.sqrt(np.mean(delta ** 2) * delta.size))
        linf = float(np.abs(delta).max())
        stored_l2 = parse_float(row, 'mse_l2', context['csv_path'])
        stored_linf = parse_float(row, 'linf', context['csv_path'])
        assert_close(stored_l2, l2, 'mse_l2', sample_path)
        assert_close(stored_linf, linf, 'linf', sample_path)

        if is_reference:
            clean_memmap[index] = clean_xyz
            labels[index] = csv_label
            samples.append({
                'index': index,
                'sample_id': row['sample_id'],
                'label': csv_label,
                'sources': {},
                'attack_results': {},
            })
        else:
            if labels[index] != csv_label:
                raise ValueError('NPZ labels differ at index {}'.format(index))
            if not np.array_equal(clean_memmap[index], clean_xyz):
                max_difference = float(np.abs(clean_memmap[index] - clean_xyz).max())
                raise ValueError(
                    '{} clean_xyz differs from the reference at index {} '
                    '(max absolute difference {})'.format(
                        definition['name'], index, max_difference
                    )
                )

        adv_memmap[index] = adv_xyz
        chamfer = parse_float(row, 'chamfer', context['csv_path'])
        hausdorff = parse_float(row, 'hausdorff', context['csv_path'])
        elapsed = parse_float(row, 'elapsed_seconds', context['csv_path'])
        samples[index]['sources'][definition['name']] = portable_path(sample_path)
        samples[index]['attack_results'][definition['name']] = {
            'adv_pred': csv_adv_pred,
            'success': success,
            'query_count': query_count,
            'l2': stored_l2,
            'linf': stored_linf,
            'chamfer': chamfer,
            'hausdorff': hausdorff,
            'elapsed_seconds': elapsed,
        }

        metric_totals['l2'] += stored_l2
        metric_totals['linf'] += stored_linf
        metric_totals['chamfer'] += chamfer
        metric_totals['hausdorff'] += hausdorff
        metric_totals['elapsed_seconds'] += elapsed
        metric_totals['query_count'] += query_count
        metric_totals['success'] += success

    count = len(context['rows'])
    return {
        'raw_asr': metric_totals['success'] / count,
        'mean_l2': metric_totals['l2'] / count,
        'mean_linf': metric_totals['linf'] / count,
        'mean_chamfer': metric_totals['chamfer'] / count,
        'mean_hausdorff': metric_totals['hausdorff'] / count,
        'mean_elapsed_seconds': metric_totals['elapsed_seconds'] / count,
        'mean_query_count': metric_totals['query_count'] / count,
    }


def build_manifest(contexts, output_dir, array_records, labels_path, samples, summaries):
    attacks = {}
    for context in contexts:
        definition = context['definition']
        config = context['config']
        attacks[definition['name']] = {
            'attack_method': config.get('transfer_attack_method') or config.get('query_attack_method'),
            'attack_family': (
                'transfer_gradient' if config.get('transfer_attack_method') else 'query'
            ),
            'surrogate_model': config.get('surrogate_model'),
            'target_model': config.get('target_model'),
            'step_size': config.get('step_size'),
            'max_steps': config.get('max_steps'),
            'eps': config.get('eps'),
            'seed': config.get('seed'),
            'source_run': portable_path(context['run_dir']),
            'source_config': portable_path(context['config_path']),
            'source_git_commit': config.get('git_commit'),
            'array': definition['array_file'],
            'summary': summaries[definition['name']],
        }

    manifest = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'ModelNet40',
        'split': 'test',
        'sample_count': len(samples),
        'point_count': contexts[0]['point_count'],
        'coordinate_dtype': 'float32',
        'ordering': 'SI-Adv ModelNet40 test-list order; aligned by stable sample_id',
        'normalization': {
            'input_state': 'already_normalized_by_siadv',
            'source_implementation': 'SI-Adv/data_utils/ModelNetDataLoader.py:pc_normalize',
            'source_scope': 'per_sample',
            'source_formula': 'subtract XYZ centroid, then divide by maximum point radius',
            'additional_transform_applied_by_this_tool': False,
            'gpointnet_requirement': (
                'Apply the fixed statistics belonging to the 1024-point GPointNet '
                'training protocol identically to every aligned clean/adversarial pair.'
            ),
        },
        'integrity_checks': {
            'all_runs_have_complete_csv_npz_linkage': True,
            'all_npz_shapes_dtypes_and_finite_values_valid': True,
            'all_sample_ids_labels_and_order_match': True,
            'all_clean_arrays_bitwise_equal_across_runs': True,
            'no_resampling_or_coordinate_modification': True,
        },
        'metric_definitions': {
            'raw_asr': (
                'Fraction with adv_pred != label. Clean predictions were not stored, '
                'so this is not strict clean-correct ASR.'
            ),
            'l2': {
                'source_csv_column': 'mse_l2',
                'formula': 'sqrt(mean((adv-clean)^2) * number_of_coordinates)',
                'note': 'The legacy CSV column name is misleading; this value is global L2.',
            },
        },
        'arrays': array_records,
        'labels': {
            'file': labels_path.name,
            'dtype': 'int64',
            'shape': [len(samples)],
            'bytes': labels_path.stat().st_size,
            'sha256': sha256_file(labels_path),
        },
        'attacks': attacks,
        'samples': samples,
    }
    manifest_path = output_dir / 'manifest.json'
    with manifest_path.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write('\n')
    return manifest_path


def main():
    args = parse_args()
    if args.expected_samples <= 0:
        raise ValueError('--expected-samples must be positive')

    run_dirs = (
        args.ifgm_pointnet_run,
        args.ifgm_dgcnn_run,
        args.query_dgcnn_paconv_run,
    )
    contexts = []
    for definition, supplied_dir in zip(RUN_DEFINITIONS, run_dirs):
        run_dir = supplied_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError('Missing run directory: {}'.format(run_dir))
        config_path, config = validate_config(run_dir, definition)
        csv_path, rows = read_csv_rows(run_dir, args.expected_samples)
        contexts.append({
            'definition': definition,
            'run_dir': run_dir,
            'config_path': config_path,
            'config': config,
            'csv_path': csv_path,
            'rows': rows,
            'point_count': int(config['input_point_nums']),
        })

    validate_cross_run_rows(contexts)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite output directory: {}'.format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix='.{}-'.format(output_dir.name),
        dir=str(output_dir.parent),
    ))

    count = args.expected_samples
    point_count = contexts[0]['point_count']
    shape = (count, point_count, 3)
    labels = np.empty(count, dtype=np.int64)
    samples = []
    summaries = {}
    array_records = {}

    try:
        clean_path = temporary_dir / 'clean_xyz.npy'
        clean_memmap = np.lib.format.open_memmap(
            str(clean_path), mode='w+', dtype=np.float32, shape=shape
        )
        for run_index, context in enumerate(contexts):
            definition = context['definition']
            adv_path = temporary_dir / definition['array_file']
            adv_memmap = np.lib.format.open_memmap(
                str(adv_path), mode='w+', dtype=np.float32, shape=shape
            )
            summaries[definition['name']] = process_run(
                context=context,
                clean_memmap=clean_memmap,
                adv_memmap=adv_memmap,
                labels=labels,
                samples=samples,
                is_reference=run_index == 0,
            )
            adv_memmap.flush()
            del adv_memmap
            array_records[definition['name']] = array_record(
                adv_path,
                shape,
                'adversarial_xyz',
                definition['name'],
            )
            print('Validated {}: {} samples'.format(definition['name'], count))

        clean_memmap.flush()
        del clean_memmap
        array_records['clean_xyz'] = array_record(
            clean_path,
            shape,
            'clean_xyz',
            'all_runs_bitwise_verified',
        )
        labels_path = temporary_dir / 'labels.npy'
        np.save(str(labels_path), labels, allow_pickle=False)
        manifest_path = build_manifest(
            contexts=contexts,
            output_dir=temporary_dir,
            array_records=array_records,
            labels_path=labels_path,
            samples=samples,
            summaries=summaries,
        )
        os.replace(str(temporary_dir), str(output_dir))
    except Exception:
        shutil.rmtree(str(temporary_dir), ignore_errors=True)
        raise

    print('Wrote aligned GPointNet input corpus: {}'.format(output_dir))
    print('Manifest: {}'.format(output_dir / manifest_path.name))


if __name__ == '__main__':
    main()
