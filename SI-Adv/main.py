# -*- coding: utf-8 -*-

import os
import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader, TensorDataset


from utils.logging import Logging_str
from utils.utils import set_seed

from attacks import PointCloudAttack
from utils.set_distance import ChamferDistance, HausdorffDistance



def load_data(args):
    """Load the dataset from the given path.
    """
    print('Start Loading Dataset...')
    if args.dataset == 'ModelNet40':
        TEST_DATASET = ModelNetDataLoader(
            root=args.data_path,
            npoint=args.input_point_nums,
            split='test',
            normal_channel=True
        )
    elif args.dataset == 'ShapeNetPart':
        TEST_DATASET = PartNormalDataset(
            root=args.data_path,
            npoints=args.input_point_nums,
            split='test',
            normal_channel=True
        )
    else:
        raise NotImplementedError

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Finish Loading Dataset...')
    return testDataLoader



def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target


def save_tensor_as_txt(points, filename):
    """Save the torch tensor into a txt file.
    """
    points = points.squeeze(0).detach().cpu().numpy()
    with open(filename, "a") as file_object:
        for i in range(points.shape[0]):
            # msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2])
            msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) + \
                ' ' + str(points[i][3].item()) +' ' + str(points[i][3].item()) + ' '+ str(1-points[i][3].item())
            file_object.write(msg+'\n')
        file_object.close()
    print('Have saved the tensor into {}'.format(filename))


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _safe_sample_id(sample_id):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(sample_id)).strip('._')
    return value or 'sample'


def _prepare_output(args):
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'siadv')
    else:
        output_dir = os.path.abspath(args.output_dir)
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'config.json')
    csv_path = os.path.join(output_dir, 'per_sample.csv')
    if os.path.exists(config_path) or os.path.exists(csv_path):
        raise FileExistsError('Output directory already contains an experiment: {}'.format(output_dir))

    config = {
        'dataset': args.dataset,
        'data_path': args.data_path,
        'input_point_nums': args.input_point_nums,
        'batch_size': args.batch_size,
        'transfer_attack_method': args.transfer_attack_method,
        'query_attack_method': args.query_attack_method,
        'surrogate_model': args.surrogate_model,
        'target_model': args.target_model,
        'defense_method': args.defense_method,
        'step_size': args.step_size,
        'max_steps': args.max_steps,
        'eps': args.eps,
        'seed': args.seed,
        'max_samples': args.max_samples,
        'git_commit': _git_commit(),
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
    }
    with open(config_path, 'w') as handle:
        json.dump(config, handle, indent=2)

    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'index', 'sample_id', 'label', 'adv_pred', 'success', 'query_count',
        'mse_l2', 'linf', 'chamfer', 'hausdorff', 'elapsed_seconds', 'file'
    ])
    csv_writer.writeheader()
    return output_dir, samples_dir, csv_file, csv_writer


def _save_sample(samples_dir, csv_writer, sample_index, sample_id, clean_xyz,
                 adv_xyz, label, adv_pred, query_count, elapsed_seconds,
                 chamfer, hausdorff, eps, enforce_linf):
    clean_np = clean_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    adv_np = adv_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    if clean_np.shape != adv_np.shape or clean_np.ndim != 2 or clean_np.shape[1] != 3:
        raise ValueError('Expected clean and adversarial point clouds with shape [N, 3]')
    if not np.isfinite(clean_np).all() or not np.isfinite(adv_np).all():
        raise ValueError('Non-finite value found in point cloud')
    if not str(sample_id):
        raise ValueError('Empty sample ID')

    delta = adv_np - clean_np
    linf = float(np.abs(delta).max())
    if enforce_linf and linf > eps + 1e-6:
        raise ValueError('L_inf constraint exceeded for {}'.format(sample_id))

    filename = '{:06d}_{}.npz'.format(sample_index, _safe_sample_id(sample_id))
    relative_file = os.path.join('samples', filename)
    output_file = os.path.join(samples_dir, filename)
    if os.path.exists(output_file):
        raise FileExistsError('Refusing to overwrite {}'.format(output_file))
    np.savez_compressed(
        output_file,
        clean_xyz=clean_np,
        adv_xyz=adv_np,
        label=np.int64(label),
        adv_pred=np.int64(adv_pred),
        sample_id=np.asarray(str(sample_id)),
    )

    mse_l2 = float(np.sqrt(np.mean(delta ** 2) * delta.size))
    csv_writer.writerow({
        'index': sample_index,
        'sample_id': str(sample_id),
        'label': int(label),
        'adv_pred': int(adv_pred),
        'success': int(int(adv_pred) != int(label)),
        'query_count': int(query_count) if query_count is not None else 0,
        'mse_l2': mse_l2,
        'linf': linf,
        'chamfer': float(chamfer),
        'hausdorff': float(hausdorff),
        'elapsed_seconds': float(elapsed_seconds),
        'file': relative_file.replace(os.sep, '/'),
    })
    return linf


def main():
    if args.dataset != 'ModelNet40':
        raise NotImplementedError('Result export currently supports ModelNet40 only')
    if args.batch_size != 1:
        raise ValueError('SI-Adv result export requires --batch-size 1')

    # load data
    test_loader = load_data(args)
    sample_ids = test_loader.dataset.sample_ids
    if len(sample_ids) != len(test_loader.dataset):
        raise ValueError('ModelNet40 sample IDs do not match the dataset')

    num_class = 40
    args.num_class = num_class

    # load model
    attack = PointCloudAttack(args)
    output_dir, samples_dir, csv_file, csv_writer = _prepare_output(args)
    print('Saving adversarial point clouds to {}'.format(output_dir))

    # start attack
    atk_success = 0
    avg_query_costs = 0.
    avg_mse_dist = 0.
    avg_chamfer_dist = 0.
    avg_hausdorff_dist = 0.
    avg_time_cost = 0.
    processed_samples = 0
    chamfer_loss = ChamferDistance()
    hausdorff_loss = HausdorffDistance()
    try:
        for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            if args.max_samples is not None and processed_samples >= args.max_samples:
                break

            # prepare data for testing
            points, target = data_preprocess(data)
            target = target.long()

            # start attack
            t0 = time.perf_counter()
            adv_points, adv_target, query_costs = attack.run(points, target)
            t1 = time.perf_counter()
            elapsed_seconds = t1 - t0
            avg_time_cost += elapsed_seconds
            if not args.query_attack_method is None:
                print('>>>>>>>>>>>>>>>>>>>>>>>')
                print('Query cost: ', query_costs)
                print('>>>>>>>>>>>>>>>>>>>>>>>')
                avg_query_costs += query_costs

            label_value = int(target.item())
            adv_pred_value = int(adv_target.item()) if torch.is_tensor(adv_target) else int(adv_target)
            atk_success += 1 if adv_pred_value != label_value else 0

            clean_points = points[:, :, :3].detach()
            adv_points = adv_points.detach()
            pert_pos = torch.where(abs(adv_points-clean_points).sum(2))
            count_map = torch.zeros_like(clean_points.sum(2))
            count_map[pert_pos] = 1.

            mse_dist = float(np.sqrt(
                F.mse_loss(adv_points, clean_points).detach().cpu().numpy()
                * clean_points[0].numel()
            ))
            chamfer_dist = float(chamfer_loss(adv_points, clean_points).item())
            hausdorff_dist = float(hausdorff_loss(adv_points, clean_points).item())
            avg_mse_dist += mse_dist
            avg_chamfer_dist += chamfer_dist
            avg_hausdorff_dist += hausdorff_dist

            sample_id = sample_ids[batch_id]
            _save_sample(
                samples_dir=samples_dir,
                csv_writer=csv_writer,
                sample_index=processed_samples,
                sample_id=sample_id,
                clean_xyz=clean_points[0],
                adv_xyz=adv_points[0],
                label=label_value,
                adv_pred=adv_pred_value,
                query_count=query_costs if args.query_attack_method is not None else 0,
                elapsed_seconds=elapsed_seconds,
                chamfer=chamfer_dist,
                hausdorff=hausdorff_dist,
                eps=args.eps,
                enforce_linf=args.transfer_attack_method == 'ifgm_ours',
            )
            csv_file.flush()
            processed_samples += 1
    finally:
        csv_file.close()

    if processed_samples == 0:
        raise RuntimeError('No samples were processed')

    atk_success /= processed_samples
    print('Attack success rate: ', atk_success)
    avg_time_cost /= processed_samples
    print('Average time cost: ', avg_time_cost)
    if not args.query_attack_method is None:
        avg_query_costs /= processed_samples
        print('Average query cost: ', avg_query_costs)
    avg_mse_dist /= processed_samples
    print('Average MSE Dist:', avg_mse_dist)
    avg_chamfer_dist /= processed_samples
    print('Average Chamfer Dist:', avg_chamfer_dist)
    avg_hausdorff_dist /= processed_samples
    print('Average Hausdorff Dist:', avg_hausdorff_dist)
    print('Saved samples:', processed_samples)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shape-invariant 3D Adversarial Point Clouds')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', 
                        help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024,
                        help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='ModelNet40',
                        choices=['ModelNet40', 'ShapeNetPart'])
    parser.add_argument('--data_path', type=str, 
                        default='/root/autodl-tmp/dataset/modelnet40_normal_resampled')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Worker nums of data loading.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum test samples to process; default uses the full split')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for paired clean/adversarial point clouds')

    parser.add_argument('--transfer_attack_method', type=str, default=None,
                        choices=['ifgm_ours'])
    parser.add_argument('--query_attack_method', type=str, default=None,
                        choices=['simbapp', 'simba', 'ours'])
    parser.add_argument('--surrogate_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
    parser.add_argument('--target_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
    parser.add_argument('--defense_method', type=str, default=None,
                        choices=['sor', 'srs', 'dupnet'])
    parser.add_argument('--top5_attack', action='store_true', default=False,
                        help='Whether to attack the top-5 prediction [default: False]')

    parser.add_argument('--max_steps', default=50, type=int,
                        help='max iterations for black-box attack')
    parser.add_argument('--eps', default=0.16, type=float,
                        help='epsilon of perturbation')
    parser.add_argument('--step_size', default=0.07, type=float,
                        help='step-size of perturbation')
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error('--max_samples must be a positive integer')

    # basic configuration
    set_seed(args.seed)
    args.device = torch.device("cuda")

    # main loop
    main()
