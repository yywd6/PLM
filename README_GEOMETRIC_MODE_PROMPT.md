# SE(3)-Invariant Geometric Mode Prompting

本文件说明项目中的独立改进版本：**SE(3)-Invariant Geometric Mode Prompting for Zero-shot 3D Anomaly Detection**。当前实现版本为 `se3_mode_prompt_v6_patch_routing`。旧版 fixed prompt、geometric CAP/DAP 及其配置保持可用。

## 1. 方法目标

方法在 one-rest cross-category 协议下训练：每次只使用一个源类别训练，在其余类别上直接测试。几何信息只用于文本 prompt 的动态调制，不与 PointBERT visual patch feature 相加或拼接。

核心思路是让多个可学习 abnormal prompts 表示潜在几何异常模式，并由 SE(3) 不变的 patch graph 根据输入点云进行路由。

## 2. Prompt 结构

默认类别语义模板：

```text
a point cloud patch of a {category}
```

以 `car` 为例：

```text
Normal:
[V1][V2][V3][V4] a point cloud patch of a car

Abnormal mode k:
[V1][V2][V3][V4]
[A1_k + delta_A1_k] ... [A4_k + delta_A4_k]
a point cloud patch of a car
```

- `V`：所有 normal/abnormal prompts 共享的 normal tokens。
- `A_k`：第 `k` 个 mode 独立的 abnormal tokens。
- `delta_A_k`：由当前点云的 SE(3) 不变几何图生成，只调制第 `k` 个 abnormal prompt。
- 测试时 `{category}` 自动替换成当前目标类别名称。

类别名称只是文本语义，不代表目标类别样本参与训练。

## 3. 整体流程

```text
Point cloud P
  ├─ PointBERT layers 2/5/8/11
  │    └─ MultiLayerPatchAdapter
  │         └─ visual patch embeddings F [B,G,D]
  │
  └─ patch indices + XYZ
       └─ SE3InvariantPatchGraphEncoder
            └─ graph features H [B,G,C]
                 └─ GeometricModeRouter
                      ├─ patch routing weights Q [B,G,K] -> point-level scoring
                      ├─ sample mode weights W [B,K] -> global scoring
                      └─ mode features [B,K,C]
                           └─ ModeSpecificPromptModulator
                                └─ delta_A [B,K,Ea,token_dim]
                                     └─ dynamic abnormal text embeddings [B,K,D]
```

视觉分支与几何分支只在最终的 mode-aware 相似度评分处发生联系。几何描述不会进入 visual adapter。

## 4. SE(3) 不变 patch graph

节点特征只使用旋转和平移不变量：

- 归一化协方差特征值；
- 曲率；
- 相对局部密度；
- 归一化 RMS 半径；
- 到物体中心的归一化径向距离。

边特征只使用：

- patch 中心距离除以 object scale；
- `abs(normal_i · normal_j)`；
- 曲率差；
- 密度对数比；
- 局部尺度对数比。

实现中不使用绝对 XYZ、法向量 XYZ 分量或中心方向向量。

## 5. 节点级 mode routing

Router 为每个 patch graph 节点输出 K 个 mode logits：

```text
Q[b,g,:] = softmax(router(H[b,g,:]) / temperature)
W[b,:]   = mean_g Q[b,g,:]
```

`Q` 表示 patch 节点属于各几何 mode 的概率，`W` 表示整个样本中各 mode 的使用比例。Router 最后一层采用接近零的初始化，避免低温度下由随机初始 logit 造成即时单-mode 塌缩。

“不同样本具有不同路由”意味着它们的 `W` 应随几何结构变化。如果所有样本始终产生相同的 `W`，即使没有集中到单一 mode，router 仍可能退化成固定混合权重。

## 6. Mode-specific prompt residual

每个 mode 聚合自己的 patch graph feature，并生成独立 residual：

```text
delta_A.shape = [B,K,Ea,token_dim]
A_dynamic[b,k,j] = A_base[k,j] + delta_A[b,k,j]
```

Mode identity 只通过乘性 gate 调制几何 hidden feature，不能绕过几何输入直接生成固定 residual。

与旧 DAP 不同，本方法没有把一个 shared prior 加到所有 abnormal tokens 上，也不会先把 K 个 abnormal prompts 平均成单一 prototype。

## 7. Mode-aware anomaly score

对每个 visual patch 和每个 dynamic abnormal mode 分别计算 cosine similarity：

```text
sim_a[b,g,k] = cosine(F[b,g], T_dynamic[b,k])
s_a[b,g]     = sum_k Q[b,g,k] * sim_a[b,g,k]
s_n[b,g]     = cosine(F[b,g], T_normal[b])
s_patch      = s_a - s_n
```

默认 `mode_score_type: weighted_sum`。

## 8. 损失函数

```text
L = L_focal + L_dice + lambda_obj * L_object
  + lambda_mode_diversity * L_mode_diversity
  + lambda_mode_entropy * L_mode_entropy
  + lambda_residual_suppression * L_residual_suppression
```

### Mode diversity

约束 K 个 abnormal text embeddings 不要重复，降低 mode 冗余。

### Node-level mode entropy

仅在源类别有标注的异常 patch 节点分配 `Q_anomaly` 上计算：

```text
L_mode_entropy = 1 - H(mean(Q_anomaly)) + beta * mean(H(Q_anomaly))
```

- 第一项抑制异常 patch 集中到少数 mode；
- 第二项让每个异常 patch 形成更明确的 mode 选择；
- 正常 patch 不参与异常 mode 的 entropy 统计，避免类别不平衡主导 Router；
- 当前 `beta=0.5`，`lambda_mode_entropy=0.05`。

### Residual suppression

正常样本上的 `delta_A` 被约束接近零，避免正常点云产生过强 abnormal residual。

`lambda_se3_sanity` 默认为 0，仅用于旋转、平移一致性的 debug 检查。

## 9. One-rest zero-shot 协议

以 `train_category=car` 为例：

- 训练数据只允许来自 `car`；
- 测试类别自动计算为 `all_categories - car`；
- 测试类别样本不能参与训练、验证或微调；
- 测试类别名可以作为 prompt 文本后缀；
- 训练和测试日志会打印类别列表及实际路径类别。

## 10. 默认配置

配置文件：[configs/one_rest_geometric_mode_prompt.yaml](configs/one_rest_geometric_mode_prompt.yaml)

关键参数：

```yaml
num_geometric_modes: 10
num_normal_tokens: 4
num_abnormal_tokens: 4
feature_layers: [2, 5, 8, 11]

geometry_graph_dim: 128
geometry_graph_k: 8
geometry_graph_layers: 2
mode_router_temperature: 0.2
mode_residual_scale: 0.1
use_patch_mode_routing: true
mode_anomaly_patch_threshold: 0.05

lambda_mode_diversity: 0.05
lambda_mode_entropy: 0.05
mode_conditional_entropy_weight: 0.5
lambda_residual_suppression: 0.01
```

## 11. 运行命令

### Car 单类别训练

```bash
/home/objectdec/anaconda3/envs/B/bin/python train.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest \
  --train_category car
```

### Car one-rest 测试

```bash
/home/objectdec/anaconda3/envs/B/bin/python test.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest \
  --train_category car
```

### 最小 debug

```bash
/home/objectdec/anaconda3/envs/B/bin/python train.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest --train_category car \
  --debug --max_train_samples 8 --batch_size 4 --num_workers 0

/home/objectdec/anaconda3/envs/B/bin/python test.py \
  --config configs/one_rest_geometric_mode_prompt.yaml \
  --protocol one_rest --train_category car \
  --debug --max_test_samples_per_category 4 --num_workers 0
```

### Real3D 和 AnomalyShapeNet 全量 one-rest

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
CONFIG=configs/one_rest_geometric_mode_prompt.yaml \
OUTPUT_BASE=outputs/one_rest_geometric_mode_prompt_all \
RESUME=1 BATCH_SIZE=4 \
bash scripts/run_all_one_rest_geometric_mode_prompt.sh
```

`RESUME=1` 只跳过带有当前 v5 完成标记的训练；旧版本 checkpoint 会重新训练。

## 12. 输出文件

每个源类别目录包含：

```text
best.pth
config.yaml
train.log
test.log
training_complete.yaml
per_category_metrics.json
mean_metrics.json
mode_statistics.json
```

`mode_statistics.json` 按目标类别记录：

- prompt text；
- mode usage distribution；
- mode usage standard deviation；
- conditional/marginal entropy；
- mode information；
- `delta_A` 平均范数。

## 13. Router 诊断

- `mode_weights_max` 接近 1：可能发生单-mode 塌缩。
- `mode_marginal_entropy` 很低：整体 mode 使用不均衡。
- `mode_information` 接近 0：不同样本的路由分布很相似。
- `mode_usage_std` 接近 0：样本级 mode 直方图几乎不变；patch-level scoring 仍需结合下面指标判断。
- `patch_mode_entropy` 很高：异常 patch 尚未形成明确 mode 选择。
- `anomaly_normal_route_gap` 接近 0：异常和正常 patch 的平均路由仍然相似。
- `delta_A_norm` 很大：prompt residual 可能过强。
- normal/abnormal residual norm 长期相同：residual suppression 或异常判别仍可能不足。

不能只看某一个统计量。例如 mode weights 没有集中到单个 mode，并不自动说明 router 已学到有意义的几何模式，最终仍需结合完整测试指标和跨类别 mode statistics 判断。

## 14. 与旧 geometric CAP/DAP 的区别

| 项目 | 旧 geometric CAP/DAP | 当前 geometric mode prompting |
|---|---|---|
| 动态信息 | 一个 sample-wise shared prior | K 个 mode-specific residual |
| abnormal prompts | 最终可平均为单 prototype | 保留 K 个动态 text embeddings |
| 路由 | top-M/聚合 prior | patch 节点级 K-mode routing |
| 几何作用位置 | prompt prior | prompt residual + mode score |
| visual feature | 不直接融合几何 | 不直接融合几何 |
| 防塌缩 | prompt orthogonal/prior loss | mode diversity + node entropy |

## 15. 测试

```bash
/home/objectdec/anaconda3/envs/B/bin/python -m pytest -q
```

当前测试覆盖 prompt shape、mode 权重归一化、residual shape、SE(3) 一致性、NaN/Inf 检查，以及几何特征不进入 visual patch feature。
