# Workspace Guide

## Purpose

This repository combines two paper reproductions:

- `SI-Adv/`: generate shape-invariant adversarial ModelNet40 point clouds.
- `GPointNet/`: energy-based point-cloud model used to compare clean/adversarial energy.
- `论文/`: Markdown copies of both papers. Read these before changing algorithms.

The current research goal is to compare GPointNet energy changes for paired clean and SI-Adv point clouds, not merely to maximize classifier ASR.

## Environment Boundaries

Do not merge the original environments blindly.

- SI-Adv is run on AutoDL RTX 3080 with Python 3.8.18, PyTorch 1.10.1+cu113, NumPy 1.21.6, Pandas 1.3.5, tqdm 4.64.1, and Open3D 0.13.0.
- PAConv imports a JIT CUDA extension. It requires Ninja, nvcc/GCC compatibility, and should use `TORCH_EXTENSIONS_DIR` outside the repository.
- GPointNet is a separate PyTorch 1.8 / Lightning 1.2-era project. Its CUDA structural-loss extension is optional unless generation metrics are needed.
- Datasets and checkpoints live outside Git. The AutoDL ModelNet40 path is `/root/autodl-tmp/dataset/modelnet40_normal_resampled`.

## SI-Adv Workflow

Run from `SI-Adv/` because checkpoint lookup is relative to the working directory.

```bash
python main.py --dataset ModelNet40 \
  --transfer_attack_method ifgm_ours \
  --surrogate_model pointnet_cls --target_model pointnet_cls \
  --batch-size 1 --input_point_nums 1024 \
  --step_size 0.007 --max_steps 50 --eps 0.16 \
  --output_dir output/<unique-run-name>
```

Important constraints:

- Keep `batch_size=1`; attack and export logic explicitly require it.
- Use a new `output_dir` for every run; existing experiments are never overwritten.
- `--max_samples 1` is the supported smoke-test limiter.
- Each result NPZ stores paired `clean_xyz`, `adv_xyz` (`[1024,3]`, float32), `label`, `adv_pred`, and stable `sample_id`. `per_sample.csv` and `config.json` are part of the experiment contract.
- Committed white-box baselines are under `SI-Adv/output/modelnet40-ifgm-{pointnet,dgcnn}`.
- Changing only the IFGM target does not create new geometry: IFGM updates use the surrogate gradient and query the target only after generation. Do not rerun the same surrogate merely to obtain a different target evaluation.
- The fixed query path is `--query_attack_method ours --surrogate_model dgcnn --target_model paconv --step_size 0.32 --max_steps 1024`. Here `max_steps` is the number of ranked point bases searched; query attacks do not use the IFGM Linf protocol.
- Current printed `Attack success rate` is `adv_pred != label`; clean predictions are not stored, so it is not strict clean-correct ASR. Preserve this caveat in reports.

## GPointNet Integration

Do not feed SI-Adv output directly to a default checkpoint without checking its protocol.

- SI-Adv outputs 1024 points; GPointNet defaults to 2048 and may bind LayerNorm parameters to point count. Never silently pad/resample.
- Train/load a GPointNet checkpoint configured for 1024 points, or document and validate an explicit adaptation.
- GPointNet training uses fixed dataset-level normalization; clean and adversarial pairs must use identical statistics and ordering.
- `model.energy_net(x)` returns the learned score `f_theta(X)`, not the paper's full energy. Report both score and
  `E_theta(X) = -f_theta(X) + ||X||^2 / (2*sigma^2)` (default `sigma=0.3`).
- Compare energy only under the same checkpoint, point count, normalization, and sigma.

## Checks

No unified lint/test suite is configured. At minimum run:

```bash
python -m py_compile SI-Adv/main.py SI-Adv/attacks.py \
  SI-Adv/data_utils/ModelNetDataLoader.py
```

For SI-Adv changes, run one real AutoDL sample and verify NPZ shape/dtype/finite values, CSV linkage, final prediction consistency, and attack-specific constraints before a full 2468-sample run.

GPointNet entry points:

```bash
python src/model_point_torch.py -do_evaluation 0
python tools/test_torch.py -checkpoint_path <ckpt> -synthesis
```

## Git and Artifacts

Work directly on `main` unless isolation is genuinely needed. Pull before editing and stage explicit paths rather than `git add .`.

- `data/`, `dataset/`, `checkpoint/`, model weights, environments, caches, and compiled extensions remain ignored.
- Experiment NPZ/NPY, CSV/JSON, logs, figures, and Markdown are intentionally tracked so they can be pulled locally for analysis.
- Do not delete, regenerate, or rewrite committed experiment outputs unless explicitly requested. Treat each output directory plus its config/CSV/log as an immutable run.
- Do not commit PAConv/GPointNet CUDA build caches.
