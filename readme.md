# ULIP-2 One-Rest 3D Anomaly Detection

This codebase provides a fixed-prompt trainable baseline and an optional
fine-grained 3D geometric CAP/DAP stage. ULIP-2 and its text encoder remain
frozen. No fixed dent/bulge/crack prompt bank is used, and geometry is never
added or concatenated to visual patch features.

## 1. Fixed-prompt baseline

Train and test one source category:

```bash
python train.py \
  --config configs/one_rest.yaml \
  --protocol one_rest \
  --train_category car

python test.py \
  --config configs/one_rest.yaml \
  --protocol one_rest \
  --train_category car
```

This provides the fixed-prompt comparison checkpoint. It is not required by the default joint CAP/DAP configuration.

## 2. 3D geometric CAP + DAP

CAP learns one shared normal token sequence and K abnormal-specific token
sequences. DAP selects top-M visual patches, pools their evidence and a local
covariance descriptor, and maps them to a sample-wise prior. The prior is added
only to abnormal prompt tokens.

```bash
python train.py \
  --config configs/one_rest_geometric_cap_dap.yaml \
  --protocol one_rest \
  --train_category car

python test.py \
  --config configs/one_rest_geometric_cap_dap.yaml \
  --protocol one_rest \
  --train_category car
```

With `freeze_visual_adapter: false`, the four-layer adapter, CAP tokens, and DAP prior are trained jointly and no baseline checkpoint is required. Results are written to `outputs/one_rest_geometric_cap_dap/car/`.

Quick one-batch checks:

```bash
python train.py \
  --config configs/one_rest_geometric_cap_dap.yaml \
  --protocol one_rest \
  --train_category car \
  --debug --max_train_samples 4 --batch_size 4 --num_workers 0

python test.py \
  --config configs/one_rest_geometric_cap_dap.yaml \
  --protocol one_rest \
  --train_category car \
  --debug --max_test_samples_per_category 1 --batch_size 1 --num_workers 0
```

## 3. All Real3D-AD source categories

Run joint CAP/DAP for every source category:

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
CONFIG=configs/one_rest_geometric_cap_dap.yaml \
OUTPUT_ROOT=outputs/one_rest_geometric_cap_dap \
  bash scripts/run_one_rest_real3dad.sh
```

Each output root contains `summary.csv` and `summary.json` after all categories
finish.
