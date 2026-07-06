# SE(3)-Invariant Geometric Mode Prompting v7

本文件说明项目中的独立改进版本：**SE(3)-Invariant Geometric Mode Prompting for Zero-shot 3D Anomaly Detection**。当前实现版本为 `se3_mode_prompt_v7_gate_sinkhorn`。旧版 fixed prompt、geometric CAP/DAP 及其配置保持可用。

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
s_a[b,g]     = tau * logsumexp_k(sim_a[b,g,k] / tau + log Q[b,g,k])
s_n[b,g]     = cosine(F[b,g], T_normal[b])
s_patch      = s_a - s_n + gate_scale * gate_logit
```

v7 默认 `mode_score_type: logsumexp`，保留多个高响应 mode，避免 weighted-sum 在路由近似固定时退化为单一原型。

## 8. v7 的 gate、路由监督和损失

v7 在几何图上增加独立的 patch abnormal gate：

```text
g[b,g] = sigmoid(gate(H[b,g]))
s_patch = mode_logsumexp(sim_a, Q) - sim_n + 0.5 * gate_logit
```

Gate 只改变异常 logit，不进入 visual feature。训练目标使用源类别 point mask 下采样得到的 patch mask；损失由 class-balanced BCE、Dice 和正负 patch logit margin 组成。这样 router 不再只靠检测损失间接学习。

异常 patch 的 K-mode 分配采用 balanced Sinkhorn self-labeling：

```text
L = L_focal + L_dice + lambda_obj * L_object
  + lambda_mode_assignment * L_sinkhorn
  + lambda_geometry_gate * L_gate
  + lambda_mode_diversity * L_mode_diversity
  + lambda_residual_suppression * L_residual_suppression
```

- `L_sinkhorn`：只在有标注的异常 patch 上产生近似均衡的 mode 伪标签，减少单 mode 塌缩。
- `L_gate`：直接训练几何分支区分异常与正常 patch。
- `L_mode_diversity`：降低 K 个 abnormal text embeddings 的重复。
- `L_residual_suppression`：正常样本约束 `delta_A`，避免无异常时产生强 residual。
- v7 默认关闭旧 entropy loss，`lambda_se3_sanity=0` 只保留为 debug。

训练采用 10% linear warmup、cosine decay、最小学习率 `1e-6` 和梯度裁剪 `10`。

## 9. One-rest zero-shot 协议

以 `train_category=car` 为例：

- 训练数据只允许来自 `car`；
- 测试类别自动计算为 `all_categories - car`；
- 测试类别样本不能参与训练、验证或微调；
- 测试类别名可以作为 prompt 文本后缀；
- 训练和测试日志会打印类别列表及实际路径类别。

## 10. 默认配置

配置文件：[configs/one_rest_geometric_mode_prompt_v7.yaml](configs/one_rest_geometric_mode_prompt_v7.yaml)

```yaml
num_geometric_modes: 6
feature_layers: [2, 5, 8, 11]
mode_score_type: logsumexp
mode_score_temperature: 0.07
use_geometry_abnormal_gate: true
use_sinkhorn_mode_assignment: true
lambda_geometry_gate: 0.1
lambda_mode_assignment: 0.2
lambda_mode_diversity: 0.01
lambda_residual_suppression: 0.003
freeze_visual_adapter: true
global_alpha: 0.0
warmup_ratio: 0.1
gradient_clip_norm: 10.0
```

`global_alpha=0` 表示对象级分数由 top patch 聚合得到；5-epoch 检查中它比混合未充分训练的 global branch 更稳定。

## 11. 运行命令

v7 默认采用两阶段训练，避免 visual adapter 与 prompt/router 相互追逐。

### Stage 1：训练 car visual baseline

```bash
/home/objectdec/anaconda3/envs/B/bin/python train.py \
  --config configs/one_rest_visual_baseline_v7.yaml \
  --protocol one_rest --train_category car \
  --output_root outputs/one_rest_visual_baseline_v7/Real3D
```

### Stage 2：冻结 adapter，训练 car prompt/router

```bash
/home/objectdec/anaconda3/envs/B/bin/python train.py \
  --config configs/one_rest_geometric_mode_prompt_v7.yaml \
  --protocol one_rest --train_category car \
  --baseline_checkpoint outputs/one_rest_visual_baseline_v7/Real3D/car/best.pth \
  --output_root outputs/one_rest_geometric_mode_prompt_v7_all/Real3D
```

### 测试 car one-rest

```bash
/home/objectdec/anaconda3/envs/B/bin/python test.py \
  --config configs/one_rest_geometric_mode_prompt_v7.yaml \
  --protocol one_rest --train_category car \
  --output_root outputs/one_rest_geometric_mode_prompt_v7_all/Real3D
```

### 最小 debug

先用 baseline config 运行 debug，再把生成的 `best.pth` 通过 `--baseline_checkpoint` 传给 v7 prompt config。两次都可加：

```text
--debug --max_train_samples 8 --batch_size 4 --num_workers 0
```

测试可加：

```text
--debug --max_test_samples_per_category 4 --num_workers 0
```

### Real3D 与 AnomalyShapeNet 全量 one-rest

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
RESUME=1 BATCH_SIZE=4 \
bash scripts/run_all_one_rest_geometric_mode_prompt_v7.sh
```

脚本会为每个源类别依次完成 visual baseline、冻结 adapter 的 v7 训练、全目标测试和 summary。

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
- `delta_A` 平均范数；
- abnormal/normal gate probability 与 gap。

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
| 防塌缩 | prompt orthogonal/prior loss | mode diversity + Sinkhorn assignment + abnormal gate |

## 15. 测试

```bash
/home/objectdec/anaconda3/envs/B/bin/python -m pytest -q
```

当前测试覆盖 prompt shape、mode 权重归一化、residual shape、SE(3) 一致性、NaN/Inf 检查，以及几何特征不进入 visual patch feature。
