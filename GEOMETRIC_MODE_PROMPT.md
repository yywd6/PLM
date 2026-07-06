# SE(3)-Invariant Geometric Mode Prompting

这是与旧 geometric CAP/DAP 并存的独立版本。旧模块与
`configs/one_rest_geometric_cap_dap.yaml` 不需要修改即可继续运行。

## Prompt

默认类别语义模板：

```text
a point cloud patch of a {category}
```

以 `car` 为例：

```text
normal:  [V1] ... [VE] a point cloud patch of a car
mode k:  [V1] ... [VE] [A1_k + delta_A1_k] ... [AEa_k + delta_AEa_k]
         a point cloud patch of a car
```

测试时类别文本按当前目标样本替换，例如 `an airplane`。类别名称只作为
文本条件；one-rest 目标类别样本不会进入训练或微调。

## 核心路径

```text
points + PointBERT patch index
  -> SE3InvariantPatchGraphEncoder
  -> graph features [B,G,C]
  -> GeometricModeRouter
  -> mode weights [B,K] + mode features [B,K,C]
  -> ModeSpecificPromptModulator
  -> delta_A [B,K,Ea,token_dim]
  -> K dynamic abnormal text embeddings [B,K,D]
  -> weighted per-mode patch similarity
```

几何特征不加到、不拼接到 visual patch embedding。视觉侧仍使用 PointBERT
第 2/5/8/11 层 patch tokens 和 `MultiLayerPatchAdapter`。

## Loss

```text
L = focal + dice + lambda_obj*object
  + lambda_mode_diversity*L_mode_diversity
  + lambda_mode_entropy*L_mode_entropy
  + lambda_residual_suppression*L_residual_suppression
```

其中 mode entropy 在 patch graph 节点分配 `[B,G,K]` 上使用稳定的负载均衡目标：

```text
L_mode_entropy = 1 - H(mode) + beta * H(mode | sample)
```

熵均按 `log(K)` 归一化，`beta=0.5`。第一项直接惩罚全局单 mode 坍塌，
第二项避免每个样本始终采用完全均匀的路由；默认损失权重为 `0.05`。日志额外输出
conditional entropy、marginal entropy、mode information 和每个 mode 的样本间
usage 标准差。

`lambda_se3_sanity` 默认是0，只用于 debug 一致性检查。

## 单类别运行

```bash
python train.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest \
  --train_category car

python test.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest \
  --train_category car
```

## Debug

```bash
python train.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest \
  --train_category car \
  --debug --max_train_samples 8 --batch_size 4 --num_workers 0

python test.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest \
  --train_category car \
  --debug --max_test_samples_per_category 4 --num_workers 0
```

## 全量 one-rest

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
CONFIG=configs/one_rest_geometric_mode_prompt.yaml \
OUTPUT_BASE=outputs/one_rest_geometric_mode_prompt_all \
RESUME=1 BATCH_SIZE=4 \
bash scripts/run_all_one_rest_geometric_mode_prompt.sh
```

每个源类别额外生成 `mode_statistics.json`，保存每个目标类别的 prompt text、
mode usage distribution、mode entropy 和平均 residual norm。
