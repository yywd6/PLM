# SE(3)-Invariant Geometric Prompt Learning for Zero-Shot 3D Anomaly Detection

本项目基于 ULIP-2/PointBERT，实现 one-rest cross-category 3D 异常检测，包含：

- 固定 normal/abnormal prompt 的可训练 baseline；
- 2/5/8/11 层 PointBERT patch feature 对齐；
- 3D Geometric Compound Abnormality Prompt（3D-CAP）；
- SE(3)-Invariant Patch-Graph Data-dependent Abnormality Prior（3D-DAP）；
- Real3D-AD 与 AnomalyShapeNet 全类别 one-rest 训练、测试和矩阵汇总。

核心限制：

- 不使用固定的 dent、bulge、crack 等缺陷词表；
- 不把几何描述符加到或拼接到 visual patch feature；
- 不使用 BTP 的 GFCM/MGFEM；
- ULIP-2 与 text encoder 始终冻结；
- 测试目标类别不参与训练或微调。

## 1. 方法总览

```text
输入点云
 ├─ ULIP-2 / PointBERT（冻结）
 │   ├─ layer 2 patch tokens ─→ adapter ┐
 │   ├─ layer 5 patch tokens ─→ adapter ├─→ softmax layer fusion
 │   ├─ layer 8 patch tokens ─→ adapter ┤    → visual patch embedding
 │   └─ layer 11 patch tokens → adapter ┘
 │
 └─ 原始点云与 patch index
     → SE(3)-invariant patch graph
     → geometry-only soft routing
     → sample-wise geometric prior
     → 动态 abnormal prompt tokens
     → 冻结 text encoder

visual patch embedding × text embedding
→ patch anomaly score
→ patch-to-point mapping
→ point anomaly map
```

几何信息只通过 prompt 侧影响异常分数。视觉 patch embedding 在整个 DAP
过程中不接收任何几何加法或拼接。

## 2. 多层视觉对齐 baseline

从 PointBERT 第 `2/5/8/11` 层提取 patch tokens。每层分别进行：

```text
Linear(384, 1280) → LayerNorm → L2 Normalize
```

四层结果通过可学习权重融合：

```math
F_{patch}=\operatorname{Normalize}
\left(\sum_{l\in\{2,5,8,11\}}\operatorname{softmax}(w)_l P_l(F_l)\right)
```

该分支只使用多层语义 token，不使用几何特征、CLS token 或 ULIP 最终全局
`concat` embedding。最终全局 embedding 已失去 patch 空间位置，不用于 point map。

实现位置：`models/trainable_baseline.py::MultiLayerPatchAdapter`。

## 3. 3D Geometric CAP

`GeometricCompoundPromptLearner` 学习：

- 一组共享 normal geometric tokens；
- K 组 abnormal-specific tokens，默认 `K=10`；
- 使用类别语义模板 `a point cloud patch of {article} {category}`；normal 和 abnormal prompt 共享同一当前类别后缀。

Prompt 结构：

```text
normal:
[V1] ... [VE] a point cloud patch of a car

abnormal k:
[V1] ... [VE] [A1_k] ... [AEa_k] a point cloud patch of a car
```

默认 `E=4`、`Ea=4`。K 个 abnormal prompt 分别经过冻结 text encoder，
取平均得到 base abnormal prototype：

```math
t_a^{base}=\operatorname{Normalize}\left(\frac{1}{K}\sum_{k=1}^{K}t_{a,k}\right)
```

实现位置：`models/geometric_cap.py`。

## 4. SE(3)-Invariant Patch Graph

旧版的“3个特征值比例 + 1个局部半径”已替换为 patch kNN graph。

### 4.1 节点特征

每个 patch 使用7维 SE(3) 不变标量：

1. 三个归一化协方差特征值；
2. 曲率 `lambda_min / sum(lambda)`；
3. 相对局部密度；
4. 归一化 RMS 半径；
5. 到物体中心的归一化径向距离。

### 4.2 边特征

kNN graph 的每条边使用5维不变量：

1. patch 中心距离 / object scale；
2. `abs(normal_i dot normal_j)`；
3. 曲率差；
4. 密度对数比；
5. 局部尺度对数比。

不使用绝对 XYZ、法向量 XYZ 分量或中心方向向量。

### 4.3 图消息传递

```math
m_{ij}=\phi(h_i,h_j,e_{ij})
```

```math
h_i^{l+1}=\operatorname{LayerNorm}
\left(h_i^l+\psi\left(h_i^l,\frac{1}{|N(i)|}\sum_{j\in N(i)}m_{ij}\right)\right)
```

默认设置：

```yaml
geometry_graph_dim: 128
geometry_graph_k: 8
geometry_graph_layers: 2
```

输出：

```text
geometry_graph_feature: [B, G, 128]
```

实现位置：`models/geometric_dap.py::SE3InvariantPatchGraphEncoder`。

## 5. Geometry-only Soft Routing 与 3D-DAP

每个 graph node 预测一个 routing logit：

```math
r_i=\operatorname{softmax}_i
\left(\frac{W_r h_i}{\tau_r}\right)
```

所有 patch 进行软聚合：

```math
g=\sum_i r_i h_i
```

当前 prior MLP 的输入仅为：

```text
pooled invariant geometry
+ normal text embedding
+ base abnormal text prototype
```

输出为 prompt token 宽度的 sample-wise prior：

```text
prior: [B, 1664]   # ViT-bigG text token width
```

prior 只加到 abnormal tokens：

```math
A'_{k,j}=A_{k,j}+\Delta A(P)
```

当前 `top_m_abnormal_patches` 仅用于记录 routing 权重最高的 patch，
不参与 prior 的硬聚合。

实现位置：`models/geometric_dap.py::PointCloudAbnormalityPrior`。

## 6. 异常分数

```math
s_n=\cos(F_{patch},t_n)
```

```math
s_a^{base}=\cos(F_{patch},t_a^{base})
```

```math
s_a^{dynamic}=\cos(F_{patch},t_a^{dynamic})
```

启用 DAP 时：

```math
s_{patch}=0.5s_a^{base}+0.5s_a^{dynamic}-s_n
```

只启用 CAP 时：

```math
s_{patch}=s_a^{base}-s_n
```

`use_geometric_cap=false` 时完全使用原固定 normal/abnormal prompt baseline。

patch score 根据 PointBERT patch index 平均映射回原始2048个点。

## 7. 训练损失

检测损失：

```math
L_{det}=L_{focal}+L_{dice}+\lambda_{obj}L_{object}
```

Prompt 多样性：

```math
L_{orth}=\left\|TT^T-I\right\|_F^2
```

正常样本 prior 约束：

```math
L_{prior}=\mathbb{E}_{y=0}\left[\|\Delta A(P)\|_2^2\right]
```

SE(3) 一致性检查：

```math
L_{inv}=\|\Delta A(P)-\Delta A(RP+t)\|_2^2
```

总损失：

```math
L=L_{det}+\lambda_{orth}L_{orth}
+\lambda_{prior}L_{prior}+\lambda_{inv}L_{inv}
```

由于 graph feature 按结构严格使用 SE(3) 不变量，`L_inv` 通常接近机器精度并
显示为 `0.000000`。它主要作为一致性检查，可设置：

```yaml
lambda_geometry_invariance: 0.0
```

## 8. 训练参数

默认 CAP/DAP 配置采用联合训练：

```yaml
freeze_plm: true
freeze_text_encoder: true
freeze_visual_adapter: false
baseline_checkpoint: null
```

联合更新：

- 2/5/8/11 多层 visual adapter；
- CAP normal/abnormal tokens；
- SE(3) graph encoder；
- geometry soft router；
- DAP prior MLP。

冻结：

- ULIP-2/PointBERT；
- ViT-bigG text encoder。

若设置 `freeze_visual_adapter: true`，则必须提供相同多层结构的 baseline
checkpoint，此时只训练 CAP/DAP。

## 9. One-rest Zero-shot Protocol

以 `train_category: car` 为例：

- 训练数据只包含 car；
- car 的 normal、anomaly 和 point mask 可以参与源类监督；
- 测试集合自动计算为 `all_categories - car`；
- 目标类别样本不会进入训练、验证或微调；
- 训练 prompt 使用源类别名；测试 prompt 使用每个目标样本的已知类别名。仅使用类别文本，不使用目标样本训练或微调，因此保持数据层面的 zero-shot，但 prompt 不再是 class-agnostic。

训练和测试日志会打印实际类别，并对样本路径执行泄漏检查。

## 10. 配置

固定 prompt baseline：

```text
configs/one_rest.yaml
```

SE(3) graph CAP/DAP：

```text
configs/one_rest_geometric_cap_dap.yaml
```

主要参数：

```yaml
use_geometric_cap: true
use_geometric_dap: true
num_geometric_abnormal_prompts: 10
num_normal_tokens: 4
num_abnormal_tokens: 4
geometric_prompt_suffix: "a point cloud patch of {article} {category}"
feature_layers: [2, 5, 8, 11]
geometry_graph_dim: 128
geometry_graph_k: 8
geometry_graph_layers: 2
geometry_routing_temperature: 0.2
lambda_prompt_orthogonal: 0.1
lambda_prior: 0.1
lambda_geometry_invariance: 0.1
```

## 11. 单个源类别运行

### 11.1 固定 prompt baseline

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

### 11.2 SE(3) graph CAP/DAP

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

快速检查：

```bash
python train.py \
  --config configs/one_rest_geometric_cap_dap.yaml \
  --protocol one_rest \
  --train_category car \
  --debug --max_train_samples 8 --batch_size 4 --num_workers 0

python test.py \
  --config configs/one_rest_geometric_cap_dap.yaml \
  --protocol one_rest \
  --train_category car \
  --debug --max_test_samples_per_category 4 --num_workers 0
```

## 12. Real3D-AD + AnomalyShapeNet 全量 one-rest

前台运行：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
CONFIG=configs/one_rest_geometric_cap_dap.yaml \
OUTPUT_BASE=outputs/one_rest_geometric_cap_dap_all \
RESUME=1 \
bash scripts/run_all_one_rest_datasets.sh
```

后台运行：

```bash
mkdir -p outputs/one_rest_geometric_cap_dap_all && \
nohup env \
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
CONFIG=configs/one_rest_geometric_cap_dap.yaml \
OUTPUT_BASE=outputs/one_rest_geometric_cap_dap_all \
RESUME=1 \
bash scripts/run_all_one_rest_datasets.sh \
> outputs/one_rest_geometric_cap_dap_all/run_all.log 2>&1 &
```

查看进度：

```bash
tail -f outputs/one_rest_geometric_cap_dap_all/run_all.log
```

输出矩阵：

```text
outputs/one_rest_geometric_cap_dap_all/Real3D/summary.csv
outputs/one_rest_geometric_cap_dap_all/Real3D/summary.json
outputs/one_rest_geometric_cap_dap_all/AnomalyShapeNet/summary.csv
outputs/one_rest_geometric_cap_dap_all/AnomalyShapeNet/summary.json
```

矩阵定义：

- 行：训练源类别；
- 列：测试目标类别；
- 对角线：空值；
- 数值：point-level AUROC；
- 最后一列：其余目标类别均值。

## 13. 输出文件

每个源类别目录包含：

```text
best.pth
config.yaml
training_complete.yaml
train.log
test.log
per_category_metrics.json
mean_metrics.json
```

评估指标：

- object AUROC / AP；
- point AUROC / AP / PRO；
- 所有目标类别 mean。

注意：当前 PRO 基于采样后的2048点 mask，不是 full-resolution connected-region
PRO。

## 14. 自检

```bash
/home/objectdec/anaconda3/envs/B/bin/python -m pytest -q \
  tests/test_baseline.py \
  tests/test_one_rest_protocol.py \
  tests/test_geometric_prompt.py \
  tests/test_multi_layer_adapter.py
```

测试覆盖：

- one-rest 类别隔离与路径泄漏；
- CAP normal/K abnormal prompt shape；
- 多层 adapter 融合与梯度；
- graph feature/routing/prior 的 SE(3) 不变性；
- CAP 关闭时回退固定 prompt baseline；
- NaN/Inf 检查。

## 15. 当前尚未实现

- K 个 abnormal prompt 与不同潜在几何退化模式的独立对应；
- 多尺度 patch graph routing；
- 频域几何分支；
- full-resolution connected-region PRO。

旧的4D/top-M DAP checkpoint 与当前 `se3_graph_v1` 不兼容；固定通用 prompt checkpoint 与当前 `prompt_template_v1` 也不兼容，均必须重新训练。
