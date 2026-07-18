# 基线方法与实验总记录

更新日期：2026-07-18

本文档是仓库中唯一维护的 Markdown，统一记录现有基线、方法变体、已做实验、结果状态、复现入口和后续结论。以后新增方法或实验直接在本文末尾的“实验追加模板”基础上补充，不再新建独立 Markdown。

## 1. 当前结论速览

- 当前 two-rest 主线中，**Static Six Prompt 是可核验证据下的最强基线**。
- 当前维护的拟议主方法为 **NCRP-K1**；它只学习一个异常残差向量。旧 NCRP K=6、adaptive、cover/reject、dual-basis 和 QR 版本仅保留历史结果，配置、运行/汇总脚本与模型分支已从活动代码删除。
- GASP-6 和 HS6P 只作为历史实验记录保留在本文与旧 `outputs/` 中；其配置、模型、运行脚本和测试已从活动代码删除，不再是可执行方法。
- 早期 one-rest 主线中，**V7 uniform scoring 是历史保留基线**；V8–V22 的候选均未达到预注册保留条件，相关实现已归档或从当前活动路径移除。
- two-rest 与 one-rest 的训练类别数、测试集合和汇总方式不同，**两条主线的 AUROC 不能直接比较**。
- 当前部分输出文件为 0 字节或缺失。在补齐六个 two-rest 组合之前，不报告两个数据集的正式总均值。

### 1.1 结果可信度标记

| 标记 | 含义 |
| --- | --- |
| 可核验 | 当前工作区有非空 JSON、日志、checkpoint 或完成标记可交叉核对 |
| 历史正式记录 | 来自此前完整实验报告或正式汇总；当前实现/部分输出已归档或移除 |
| 待复核 | 只有先前会话读取记录，当前对应输出目录不存在或文件损坏 |
| 未完成 | 仅完成部分 epoch、部分 source 或没有测试结果 |

## 2. 实验协议

### 2.1 当前 two-rest 协议

- 数据集：Real3D、AnomalyShapeNet。
- 每次只使用同一数据集中的两个 source 类别训练，测试时排除这两个类别。
- Real3D：`car+chicken`、`candybar+fish`、`duck+starfish`。
- AnomalyShapeNet：`bowl5+cup0`、`bottle0+tap0`、`helmet0+jar0`。
- 同一数据集的三个组合使用六个互不重复的 source 类别。
- PointBERT 和 OpenCLIP 文本编码器冻结。
- Stage 1 训练轻量视觉 adapter；Stage 2 冻结视觉侧，只训练 Static Prompt，或 NCRP-K1 的正常 Prompt token 与单异常残差向量。
- 测试标签只用于最终 Object AUROC/AP、Point AUROC/AP 和 AUPRO 计算，不用于训练、Prompt 路由、原型构建或超参数选择。
- `mean_metrics.json` 是一个 source 组合在剩余测试类别上的宏平均，不是单类别结果。

这六个组合属于探索性配置。若作为论文正式协议，应预先固定组合，或另设 source-only validation，不能根据目标测试结果继续挑选组合或参数。

### 2.2 历史 one-rest 协议

- 数据集：Real3D-AD，共 12 个类别。
- 每次只使用一个 source 类别训练，其余 11 个 target 类别只用于最终评估。
- 正式结果对 12 个 source 的测试结果取均值。
- Point AUROC/AP 使用 raw patch logits 映射得到的 raw point scores。
- Object score 的定义预先固定，不使用 target calibration。
- V21/V22 正式实验均声明未使用 target test 做训练、checkpoint、超参数或分数方向选择。

## 3. 当前基线：Static Six Prompt

### 3.1 方法定位

Static Six Prompt 是当前活动基线。方法采用两阶段训练：先学习点云视觉特征到文本空间的 adapter，再冻结视觉分支并学习 1 个正常 Prompt 和 6 个异常 Prompt。当前版本不包含 SE(3) 图、几何条件化、路由、门控、Sinkhorn、Patch MoE 或多尺度模块。

主要入口：

```text
configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml
models/static_prompt.py
models/trainable_baseline.py
train.py
test.py
test_standard_aupro.py
scripts/run_selected_disjoint_two_rest_static_six_prompt_v1.sh
scripts/summarize_selected_two_rest.py
```

### 3.2 视觉编码与 adapter

输入点云为 $X\in\mathbb{R}^{N\times3}$。冻结的 ULIP-2 PointBERT 提取第 2、5、8、11 层 patch token：

$$
F^{(l)}\in\mathbb{R}^{G\times384},\qquad l\in\{2,5,8,11\}.
$$

每层由独立线性层和 LayerNorm 投影到 1280 维文本空间并归一化：

$$
Z^{(l)}=\operatorname{Norm}\!\left(\operatorname{LN}(W_lF^{(l)}+b_l)\right).
$$

四层使用可学习 softmax 权重做 feature-first 融合：

$$
Z=\operatorname{Norm}\!\left(\sum_l\alpha_l Z^{(l)}\right),
\qquad
\alpha_l=\frac{\exp(q_l)}{\sum_j\exp(q_j)}.
$$

Stage 1 只训练四个 adapter 投影、LayerNorm 和四个层融合 logit；PointBERT 和文本编码器保持冻结。

### 3.3 静态 Prompt 与打分

Prompt learner 包含 4 个共享正常上下文 token，以及 6 组相互独立、每组 4 个的异常 token。默认类别后缀为 `a point cloud patch of a {category}`。冻结的文本编码器输出一个正常嵌入 $t^N$ 和六个异常嵌入 $t^A_k$。

对归一化 patch 特征 $z_g$：

$$
s^N_g=z_g^\top t^N,
\qquad
s^A_{g,k}=z_g^\top t^A_k.
$$

六个异常 Prompt 使用统一 log-mean-exp 聚合，不做路由或类别选择：

$$
\bar{s}^A_g=\tau_p\left[
\log\sum_{k=1}^{6}\exp(s^A_{g,k}/\tau_p)-\log 6
\right],
$$

$$
\ell_g=\frac{\bar{s}^A_g-s^N_g}{\tau},
\qquad \tau_p=\tau=0.07.
$$

patch logit 通过邻域索引回填到原始点；一个点被多个 patch 覆盖时取平均。对象分数采用 top-$p$ patch 概率均值，当前 $p=0.2$、`global_alpha=0.0`。

### 3.4 训练目标

$$
\mathcal{L}=
\mathcal{L}_{focal}+
\mathcal{L}_{dice}+
0.5\mathcal{L}_{object}+
0.01\mathcal{L}_{div}.
$$

多样性损失约束六个异常文本嵌入的余弦 Gram 矩阵接近单位矩阵。Stage 2 加载并冻结 Stage 1 adapter，只训练正常/异常 Prompt token；梯度可穿过冻结文本编码器回传到 Prompt token。

### 3.5 运行命令

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 \
./scripts/run_selected_disjoint_two_rest_static_six_prompt_v1.sh
```

显存不足可设置 `BATCH_SIZE=2`；需要强制重训时设置 `RESUME=0`。输出目录沿用历史名称：

```text
outputs/selected_two_rest_ablation_static_six_prompt_v7_uniform_scoring
```

## 4. 当前方法变体

| 方法 | Stage 1 视觉侧 | Stage 2 打分 | 层级 token | 层融合 | 专门化 | 正常间隔 | 测试原型 | top ratio |
| --- | --- | --- | :---: | --- | :---: | :---: | :---: | ---: |
| Static Six Prompt | feature-first adapter | 先融合特征再打分 | 否 | adapter feature weights | 否 | 否 | 否 | 0.2 |
| HS6P V1 | 原 feature-first adapter | 四层分别打分 | 是 | fixed equal | 是 | 是 | 是 | 已完成结果实际为 0.01 |
| A1 | 原 feature-first adapter | 四层分别打分 | 否 | fixed equal | 否 | 否 | 否 | 0.2 |
| A2 | 原 feature-first adapter | 四层分别打分 | 是 | fixed equal | 否 | 否 | 否 | 0.2 |
| A3 | 原 feature-first adapter | 四层分别打分 | 是 | fixed equal | 是 | 否 | 否 | 0.2 |
| A4 | 原 feature-first adapter | 四层分别打分 | 是 | fixed equal | 是 | 是 | 否 | 0.2 |
| A5 | 原 feature-first adapter | 四层分别打分 | 是 | fixed equal | 是 | 是 | 是 | 0.2 |
| HS6P V2 | hierarchical deep-supervision adapter | 四层分别打分 | 是 | fixed equal | 是 | 是 | 是 | 0.2 |
| HS6P V3 | hierarchical deep-supervision adapter | 四层分别打分 | 是 | frozen visual weights | 否 | 否 | 否 | 0.2 |

对应配置：

```text
configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml
configs/two_rest_hierarchical_specialized_six_prompt_v1.yaml
configs/ablations/
configs/two_rest_hierarchical_visual_adapter_v1.yaml
configs/two_rest_hierarchical_specialized_six_prompt_v2_hierarchical_visual.yaml
configs/two_rest_hierarchical_six_prompt_v3_visual_weighted.yaml
```

### 4.1 HS6P V1

HS6P V1 在 Static 上增加：

1. 第 2/5/8/11 层分别做 Prompt 打分，再融合四层 logit；
2. shallow/deep/global level token；
3. 基于异常 Prompt 相似度的 assignment entropy 与 batch balance 专门化损失；
4. 正常 patch margin；
5. 测试时仅使用当前点云的 KMeans 正常多原型校准。

四层 patch logit 为 $\ell_{lg}$，最终分数为 $\ell_g=\sum_lw_l\ell_{lg}$。固定融合时 $w_l=1/4$。完整 HS6P 总损失在 Static 损失上增加：

$$
\lambda_aL_{assign}+\lambda_bL_{balance}+\lambda_nL_{normal}.
$$

原型校准只处理当前测试样本：先选低原始 logit 的候选 patch，KMeans 得到正常原型，再按最近原型余弦距离生成概率，并与文本概率按 `prototype_gamma=0.3` 融合。它不使用标签或跨样本 memory bank。

注意：已完成的 V1 `car+chicken` checkpoint 和日志实际记录 `top_ratio=0.01`。主配置后来改为 0.2，因此旧结果不能标成 top-0.2。

### 4.2 Hierarchical Visual Adapter 与 HS6P V2

原 Stage 1 只监督融合后的特征，但 HS6P score-first 推理会独立消费每一层。Hierarchical Visual Adapter 因此在 source 数据上同时监督融合分数和四个单层分数：

$$
L_{visual}=L_{fused}+\frac{1}{4}\sum_lL_l.
$$

V2 加载训练模式为 `hierarchical_deep_supervision` 的 adapter，冻结视觉侧，再训练 V1 的完整 HS6P 模块。加载不兼容的旧 adapter 时会显式报错。

### 4.3 HS6P V3

V3 复用已完成的 hierarchical visual adapter，只重训 Stage 2 Prompt：

- 保留四层 score-first 和 level token；
- 用冻结的 `softmax(adapter.layer_logits)` 替代等权 score 融合；
- 关闭 specialization、normal margin 和 prototype calibration；
- 保持 `top_ratio=0.2`。

这些视觉权重是在 feature-first 训练目标下学到的，直接用于 score-first logit 加权是待验证假设，而不是理论等价变换。

### 4.4 GASP-6：Geometry-Anchored Six Prompt

实现状态：代码完成，正式实验未运行。GASP-6 保留 Static Six Prompt 的 PointBERT、四层 feature-first adapter、Prompt 结构、Static loss、patch-to-point 和 top-20% 对象聚合，只用 source 异常 patch 的无参数局部几何模式约束六个异常 Prompt 分工。

明确不使用：HS6P level token、score-first、assignment entropy、balance loss、normal margin、prototype calibration、可学习 geometry router、geometry gate、几何异常分数和 target memory bank。

主要文件：

```text
models/geometry_descriptor.py
models/gasp6.py
utils/gasp6_config.py
utils/artifact_integrity.py
scripts/build_source_geometry_clusters.py
scripts/run_selected_disjoint_two_rest_gasp6_v1.sh
scripts/smoke_test_gasp6.py
configs/two_rest_gasp6_v1.yaml
configs/ablations/gasp6_a0_static.yaml
configs/ablations/gasp6_a1_mode_loss.yaml
configs/ablations/gasp6_a2_geo_prior.yaml
configs/ablations/gasp6_a3_full.yaml
configs/ablations/gasp6_a4_hard_assignment.yaml
tests/test_geometry_descriptor.py
tests/test_gasp6.py
```

#### 4.4.1 34 维 SE(3) 不变描述符

每个 patch 用 `patch_idx` 从原始点云前三维取得坐标，以 patch 均值为中心，并默认除以最大中心半径。描述符不使用绝对 xyz 或坐标轴方向，维度由 bin 数自动计算：

| 维度 | 数量 | 含义 |
| --- | ---: | --- |
| 0–2 | 3 | 归一化协方差特征值 $\hat\lambda_1,\hat\lambda_2,\hat\lambda_3$ |
| 3–9 | 7 | linearity、planarity、scattering、curvature、anisotropy、omnivariance、eigenentropy |
| 10–17 | 8 | 归一化中心距离概率直方图 |
| 18–25 | 8 | 点对距离概率直方图；尺度归一化时固定区间为 $[0,2]$ |
| 26–33 | 8 | 中心距离均值/标准差、最近邻距离均值/标准差、最大半径、25%/50%/75% 分位数 |

实现全部使用 PyTorch；`torch.linalg.eigvalsh` 对所有 patch 批量计算，不逐 patch 做 Python 特征分解。默认使用完整点对距离；描述符在 `torch.no_grad()` 下计算，退化 patch 通过 clamp 与 `nan_to_num` 保证有限值。

#### 4.4.2 source-only 六簇

`build_source_geometry_clusters.py` 只构造 source 类别 Dataset，并用 target 类别列表做禁止泄漏审计。流程为：验证并加载当前 pair 的 Static Stage 1 checkpoint，提取 PointBERT `patch_idx`，按现有 `patch_anomaly_ratio > 0.05` 规则保留 source 异常 patch，拟合 source mean/std，再用 sklearn KMeans 建立六簇。

固定设置：`random_state=0`、`n_init=20`、最多确定性采样 100000 个异常 patch。少于 6 个异常 patch时直接报错。cluster id 不重排：cluster 0–5 分别对应异常 Prompt 0–5。

每个 pair 只构建一次：

```text
outputs/selected_two_rest_gasp6_v1/<dataset>/<source_pair>/geometry_clusters.npz
outputs/selected_two_rest_gasp6_v1/<dataset>/<source_pair>/geometry_clusters.json
```

NPZ 保存 mean、std、centroids、cluster counts 和 inertia；JSON 额外保存 source 类别、描述符定义、source patch 数和 cluster-by-source 计数。

#### 4.4.3 mode supervision 与推理先验

Stage 2 训练仍以 Static uniform log-mean-exp 产生 focal、dice、object BCE 和 diversity loss。只对 source 异常 patch 增加：

$$
L_{mode}=\operatorname{CE}(s^A_g/\tau_{mode},y_g^{mode}),
\qquad
L=L_{Static}+\lambda_{geometry}L_{mode}.
$$

默认 `tau_mode=0.1`、`lambda_geometry_mode=0.1`。空异常 batch 返回与 Prompt similarity 计算图相连的零张量。

推理只读取 source mean/std/centroids：

$$
q^{geo}_{gk}=\operatorname{softmax}(-\|\hat h_g-c_k\|^2/\tau_{geo}),
$$

$$
q^{final}_{gk}=\frac{1-\alpha}{6}+\alpha q^{geo}_{gk},
$$

$$
\bar s^A_g=\tau_p\operatorname{logsumexp}_k
\left(s^A_{gk}/\tau_p+\log(q^{final}_{gk}+\epsilon)\right).
$$

默认 `tau_geo=1.0`、`geometry_prior_strength=0.5`。先验只改变六个异常 Prompt 的相对权重，不生成额外异常分数。强度为 0 时直接调用原 Static 打分函数；单元测试确认 patch logits 和 top-20% object score逐值相等。A4 的硬分配强制 `geometry_prior_strength=1`。

#### 4.4.4 参数量与诊断

| 方法 | Stage 2 可训练参数 |
| --- | ---: |
| Static Six Prompt | 35,840 |
| GASP-6 | 35,840 |
| 新增可训练参数 | 0 |

GASP checkpoint 同时保存 source geometry state；`gasp6_diagnostics.json` 记录 cluster counts/inertia、source 与 target mode-Prompt 对齐、混淆矩阵、Prompt usage、每簇 mode loss、先验熵/最大权重、文本相似度、target 到最近 source centroid 距离、uniform 与 geometry 指标及差值。`sample_scores.npz` 保存 descriptor、六中心距离、先验、六 Prompt 相似度及 prior 前后的 patch/point 分数。

#### 4.4.5 实现差异与 smoke 限制

- 现有 parser 使用 flat argparse；YAML 保留所要求的嵌套 `geometry_anchor`/`object_pooling` 结构，在加载后由 `utils/gasp6_config.py` 映射为 flat namespace。`_base_` 继承已改为递归合并。
- $c_g$ 实际取 patch 邻域坐标均值；默认完整计算 patch 内点对距离。
- 当前仓库没有 `STATIC_SIX_PROMPT_METHOD.md`：该文件已按此前“只留一个 Markdown”的要求合并并删除。本节与本文第 3 节承担实现差异记录，不另建 README。
- 正式 PointBERT smoke 在当前机器失败：没有可用 NVIDIA 驱动，安装的 PointNet2 FPS 扩展明确不支持 CPU。失败发生在 `furthest_point_sampling`，尚未进入 GASP 描述符或 loss。
- 最小兼容 smoke 使用真实 Real3D `car+chicken` source 点云、真实 `airplane` target 点云和真实标签，但将 PointBERT patch/token 提取替换为确定性局部 KNN patch 与冻结的轻量 feature-first token。它验证工程链路，不是精度实验，数值不得进入方法对比表。
- 兼容 smoke 的六簇计数为 `[1,3,3,6,5,1]`；多个 cluster 被单个 source 类别完全主导，Prompt usage 也集中到单个 Prompt。该现象受极小样本和替代 token 影响，但说明正式实验必须重点检查 cluster balance 和 Prompt 对应是否真正形成。

## 5. 当前 two-rest 实验结果

### 5.1 实验状态

| 实验 | 状态 | 证据级别 |
| --- | --- | --- |
| Static Six Prompt | 曾运行六组；部分产物变为 0 字节，目前只能恢复 3 组 | 部分可核验 |
| GASP-6 | 代码、配置、49 项测试和最小兼容 smoke 完成；正式 PointBERT smoke 受当前机器 CUDA/PointNet2 限制，六组正式实验未启动 | 工程可核验/精度未验证 |
| HS6P V1 | `car+chicken` 完成；`candybar+fish` 第 17 epoch 后中断 | 部分可核验 |
| A1–A4 top-0.2 | 脚本存在，当前输出目录不存在 | 未完成/无结果 |
| A5 | 配置存在，未发现输出 | 未完成 |
| Hierarchical Visual Stage 1 | 六个组合全部完成 | 可核验 |
| HS6P V2 | 当前输出目录不存在 | 一组历史会话数值待复核 |
| HS6P V3 | `car+chicken` 完成；`candybar+fish` 只到第 5 epoch | 部分可核验 |

### 5.2 Real3D `car+chicken` 可比结果

箭头后的数值为相对 Static 的绝对变化。

| 方法 | Object AUROC | Object AP | Point AUROC | Point AP | AUPRO | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Static | **0.7176** | **0.7401** | **0.8544** | **0.2113** | **0.6265** | 可核验 |
| HS6P V1，校准前 | 0.6869 (-0.0307) | 0.7087 (-0.0313) | 0.8490 (-0.0054) | 0.1854 (-0.0259) | 0.6138 (-0.0127) | 可核验 |
| HS6P V1，校准后 | 0.6866 (-0.0310) | 0.7077 (-0.0324) | 0.8509 (-0.0035) | 0.1850 (-0.0263) | 0.6107 (-0.0158) | 可核验 |
| HS6P V2 | 0.6869 (-0.0307) | 0.7142 (-0.0259) | 0.8418 (-0.0125) | 0.1613 (-0.0500) | 0.6001 (-0.0264) | 待复核 |
| HS6P V3 | 0.6814 (-0.0362) | 0.7127 (-0.0273) | 0.8307 (-0.0237) | 0.1654 (-0.0459) | 0.5887 (-0.0378) | 可核验 |

可核验来源：

```text
outputs/selected_two_rest_ablation_static_six_prompt_v7_uniform_scoring/Real3D/car+chicken/mean_metrics.json
outputs/selected_two_rest_hs6p_v1/Real3D/car+chicken/mean_metrics.json
outputs/selected_two_rest_hs6p_v1/Real3D/car+chicken/hs6p_diagnostics.json
outputs/selected_two_rest_hs6p_v3_visual_weighted/Real3D/car+chicken/mean_metrics.json
outputs/selected_two_rest_hs6p_v3_visual_weighted/Real3D/car+chicken/hs6p_diagnostics.json
```

V2 数值只来自此前会话对原目录的读取记录；当前目录不存在，重跑前不能进入最终论文表格。

### 5.3 当前可恢复的 Static 结果

| 数据集 | source 组合 | Object AUROC | Object AP | Point AUROC | Point AP | AUPRO | 来源 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Real3D | car+chicken | 0.7176 | 0.7401 | 0.8544 | 0.2113 | 0.6265 | 非空 `mean_metrics.json` |
| AnomalyShapeNet | bottle0+tap0 | 0.7308 | 0.7917 | 0.8994 | 0.3231 | 0.6777 | 非空 `mean_metrics.json` |
| AnomalyShapeNet | bowl5+cup0 | 0.7418 | 0.8047 | 0.8901 | 0.3154 | 0.6553 | 从非空 `test.log` 末行恢复 |

无法恢复：Real3D `candybar+fish`、Real3D `duck+starfish`、AnomalyShapeNet `helmet0+jar0`。当前两组 AnomalyShapeNet 的临时算术平均为 Object AUROC 0.7363、Object AP 0.7982、Point AUROC 0.8947、Point AP 0.3193、AUPRO 0.6665；这不是正式三组合均值。

### 5.4 Hierarchical Visual Stage 1

`best loss` 是 source 训练损失，不是目标测试精度。权重顺序为 PointBERT `[2,5,8,11]`。

| 数据集 | source 组合 | best loss | adapter 层权重 |
| --- | --- | ---: | --- |
| Real3D | car+chicken | 1.9934 | [0.0990, 0.2797, 0.2460, 0.3754] |
| Real3D | candybar+fish | 1.8958 | [0.1270, 0.2235, 0.2799, 0.3696] |
| Real3D | duck+starfish | 1.9168 | [0.0934, 0.2382, 0.3590, 0.3094] |
| AnomalyShapeNet | bowl5+cup0 | 1.8854 | [0.1756, 0.2992, 0.2413, 0.2839] |
| AnomalyShapeNet | bottle0+tap0 | 1.9573 | [0.1782, 0.2767, 0.3001, 0.2450] |
| AnomalyShapeNet | helmet0+jar0 | 1.8185 | [0.1667, 0.2662, 0.2967, 0.2704] |

输出：`outputs/selected_two_rest_hierarchical_visual_adapter_v1/`。

### 5.5 诊断结论

V1 `car+chicken` 的六 Prompt 使用比例为：

```text
[0.1680, 0.1764, 0.1585, 0.1544, 0.1709, 0.1719]
```

归一化 assignment entropy 为 0.9983，接近最大值 1，说明六个 Prompt 仍近似均匀响应，没有学出清晰分工。

V1 校准后相对校准前：Object AUROC -0.0003、Object AP -0.0011、Point AUROC +0.0019、Point AP -0.0004、AUPRO -0.0031。只有 Point AUROC 略升，因此单样本正常原型不能视为有效模块。

V3 `car+chicken` 的冻结权重为 `[0.0990,0.2797,0.2459,0.3753]`，但第 2/5/8/11 层异常 logit 的 mean/std 分别为 `-7.0620/6.5691`、`-7.6138/5.9329`、`-8.2453/7.5091`、`-2.4893/9.9966`。层间尺度不一致，visual feature weight 不能直接当作 score weight；V3 五项指标全部下降，验证了该迁移假设失败。

## 6. 历史基线：V7 Uniform Scoring

V7 属于早期 Real3D one-rest 主线，不是当前 Static Six Prompt 方法。它采用两阶段训练：

1. 冻结 PointBERT 和 text encoder，训练第 2/5/8/11 层 visual adapter 与层融合权重；
2. 冻结视觉侧，训练 SE(3)-invariant patch graph、六个 geometric modes、normal/abnormal Prompt、mode residual 和 geometry abnormal gate。

六个异常 mode 使用 uniform log-sum-exp，router probability 不作为 score prior。Point score 是融合 patch raw logits 的 patch-to-point 平均；Object score 为 top-20% patch sigmoid mean，`global_alpha=0.0`。

历史正式结果：

| 指标 | V7 |
| --- | ---: |
| mean Object AUROC | 0.6773892412 |
| mean Object AP | 0.7022508030 |
| mean Point AUROC | 0.8421919541 |
| mean Point AP | 0.1770443751 |
| mean Point AUPRO | 0.5824100359 |
| top-3 source Object AUROC | 0.7163495964 |

当前工作区正在向 Static Six Prompt 主线收敛，V7 的 geometric 配置、模块和脚本已删除或保存在 `archive_deprecated/` 中；其数值保留为历史参照，不代表当前活动代码可以直接复现。

## 7. 历史 one-rest 实验 V8–V22

### 7.1 V8–V20 总表

除明确标注“部分”外，均为 Real3D 12-source 正式均值；Point AUROC 多数候选复用 V7 point branch。

| 实验 | 方法 | 范围 | Object AUROC | Point AUROC | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| V7 | uniform top-p | 全量 | 0.677389 | 0.842192 | 历史基线 |
| V8 | hard-normal / MIL / object ranking | airplane/部分 | 最高约 0.6215（airplane） | 多数下降 | source hard negative 不稳定，MIL 更差 |
| V9 | fixed spatial component mean | 全量 | 0.657309 | 0.842192 | 空间连通无法区分 normal structural tail |
| Fixed pooling search | 固定 pooling | 全量 | 0.685175 | 0.842192 | +0.007786，最接近但未达到 +0.01 |
| Controlled joint | 联合 object head | 全量 | 0.653390 | 0.842192 | source overfitting |
| Object linear | 线性 object head | 全量 | 0.647674 | 0.842192 | source overfitting |
| Object MLP | MLP object head | 全量 | 0.640741 | 0.842192 | source overfitting |
| V12 fixed | 固定 layerwise evidence | 全量 | 0.671150 | 保持 V7 | 最佳层随类别变化 |
| V12 trainable | layerwise uncertainty calibrator | 全量 | 0.673040 | 保持 V7 | 学死 source 层偏好 |
| V13 | global object feature | airplane | 0.580470 | 保持 V7 | global feature 稀释小缺陷 |
| V14 | decoupled object geometric Prompt | 全量 | 0.665565 | 0.842192 | point 被保护，object 仍跨类退化 |
| ZUMA-style | training-free layer evidence | 部分 | 无可靠全量 | 保持 V7 | 固定层不稳定；Mahalanobis 近似反证据 |
| V16 | tail-calibrated localized alarm | 全量 | 0.602511 | 0.842192 | 强 tail suppression 明显失败 |
| V17 | sparse anomaly query | 全量 | 0.653757 | 0.842192 | query 单独失败 |
| V17 fusion 0.25 | V7 + query | 全量 | 0.682503 | 0.842192 | +0.005114，仍低于保留线 |
| V18 | query-guided evidence | 全量 | 0.578898 | 0.842192 | attention 关系不跨类 |
| V19 A4 | synthetic tiny-defect representation | 全量 | 0.676900 | 0.843984 | point +0.001792，object -0.000489 |
| V20 | local multiview CLIP fusion 0.25 | 全量 | 0.643941 | 0.842192 | local verifier source overfitting |

关键补充：

- V9 证明“高分 patch 空间连通”不是异常的充分条件，normal 复杂结构也会形成连通高分区域。
- airplane 的 object 最佳单层曾是 layer 8（0.638296），但 point 仍以四层融合最好；不同 source 的最佳层不一致。
- V14 在 airplane/diamond/fish/toffees 提升，但 car 下降 0.127142，暴露明显 source 依赖。
- V16 只在 2%–5% 异常面积桶改善，tiny defect 与大缺陷桶均下降。
- V17 attention AUROC 为 0.711447，但 object score 仍失败，说明“能定位”不等于“能可靠聚合成对象分数”。
- V19 的 synthetic anomaly 与 normal 特征距离远小于 real anomaly 与 normal 的距离，合成缺陷没有逼近真实跨类分布。
- V20 candidate recall@1/@4/@8 为 0.448945/0.689797/0.809411，trained candidate AUROC 为 0.614801；source validation Object AUROC 0.705076，但 target trained-CLIP 只有 0.562786，主要失败原因是 source overfitting，而不是候选完全漏检。

历史代码主要位于：

```text
archive_deprecated/failed_object_auroc_experiments/
archive_deprecated/v16_tail_calibrated_alarm/
archive_deprecated/v19_tiny_defect_representation/
archive_deprecated/v20_local_multiview_clip/
archive_deprecated/experiments_pre_v21_cleanup/active_files/
```

### 7.2 V21 预注册阶梯

V21 按 A → B → C 的条件式阶梯执行；A 失败才进入 B，A/B 都失败才进入 C。每个变体完成 12/12 source，未按 target 搜索 lambda、gamma、top-p、阈值或分数方向。

| 方法 | Object AUROC | Object AP | Point AUROC | Point AP | AUPRO | 相对 V7 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| V7 | 0.677389 | 0.702251 | 0.842192 | 0.177044 | 0.582410 | — | 基线 |
| V21-A Context Normality Veto | 0.677389 | 0.702251 | 0.842192 | 0.177044 | 0.582410 | Object 0；Point 约 0 | FAIL |
| V21-B Soft Gate + Veto | 0.678958 | 0.702819 | 0.838860 | 0.173022 | 0.576445 | Object +0.001568；Point -0.003332 | FAIL |
| V21-C Overlap Consensus | 0.668087 | 0.694482 | 0.834343 | 0.165641 | 0.568929 | Object -0.009302；Point -0.007849 | FAIL |

方法与失败原因：

- V21-A 用周围低重叠 patch 的视觉/SE(3) context 预测中心特征，只对正 V7 logit 做 one-sided veto。target residual 很少落入 source-normal calibration 的低半区，`q_normal` 基本为 0，最终精确回退 V7。
- V21-B 用 $y=1-\exp(-r/0.02)$ 替换 geometry gate 的二值 patch target。10/12 source 的 object 上升，但平均增益太小且 point 下降；normal high-score tail 没有降低。
- V21-C 用覆盖同一点的 patch logit 均值减 0.5 倍 MAD。它降低 normal tail，但 anomaly score retention 只有 0.697908，12/12 source 的 point 都下降。

V21 保留条件为 Object-primary（Object ≥ 0.687389 且 Point ≥ 0.841192）或 Point-primary（Point ≥ 0.845192 且 Object ≥ 0.675389）；三者均未通过，不做 clean retrain。

### 7.3 V22 Relative Patch Evidence Selector

V22 冻结 V7 的 PointBERT、OpenCLIP、visual adapter、Prompt、SE(3) graph、router、gate 和 modulator，只训练低容量 object-only selector。Point branch 必须逐位保持 V7。

- V22-A：relative patch evidence selector；
- V22-B：增加 object-internal normal prototype distance；
- V22-C：在 B 上增加低权重 object loss。

| Variant | Object AUROC | Gain vs V7 | Point AUROC | improved/declined | Worst drop | 95% CI | Passed |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| V22-A | 0.6785681192 | +0.0011788780 | 0.8421919541 | 10/2 | -0.0002261590 | [+0.0006121137,+0.0018251835] | False |
| V22-B | 0.6784576454 | +0.0010684041 | 0.8421919541 | 10/2 | -0.0002017612 | [+0.0004836741,+0.0017716508] | False |
| V22-C | 0.6785243557 | +0.0011351145 | 0.8421919541 | 11/1 | -0.0001213481 | [+0.0005750536,+0.0017946518] | False |

保留阈值为 mean Object AUROC ≥ 0.687389，并要求 Point 与 V7 一致、至少 8/12 source 不下降、worst-source drop ≤ 0.03。A/B/C 虽有稳定小幅重排收益且 36/36 source 的 point score 与 V7 完全一致，但都只有约 +0.0011，未达到 +0.01，因此不做 clean retrain，不增加 V22-D。

V22 归档：

```text
archive_deprecated/experiments_pre_v22_cleanup/v22_relative_patch_selector/
```

## 8. 跨实验稳定结论

1. **当前主要瓶颈是对象级证据选择，不是完全缺少点级定位。** 历史 V7 Point AUROC 约 0.842，而 Object AUROC 约 0.677；多次保持 point 不变的 object-side 改动仍失败。
2. **可训练 object head 容易 source overfitting。** Object linear/MLP、V12 calibrator、V14 object Prompt、V18 evidence learner、V20 verifier 都重复出现 source 有效、target 退化。
3. **固定 pooling 有信号但上限有限。** 历史最好为 0.685175（+0.007786），仍未过预注册 +0.01，继续基于 target 搜阈值会造成泄漏。
4. **最佳层随类别变化。** feature-first 学到的层权重也不等价于 score-first 权重，V3 再次验证了这一点。
5. **normal structural tail 与 tiny anomaly 高度重叠。** 强 suppression、hard-normal、margin、context veto、overlap MAD 都会同时削弱真实异常。
6. **高召回候选不等于可靠验证。** V17/V20 能覆盖较多异常区域，但局部 verifier 或 attention 到对象分数的跨类映射不稳定。
7. **当前 Static 的简化是有实验依据的。** SE(3)/几何分支、HS6P specialization、prototype calibration 和视觉权重 score 融合均未显示稳定增益。

## 9. 当前运行与产物

### 9.1 运行命令

Static 六组：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 \
./scripts/run_selected_disjoint_two_rest_static_six_prompt_v1.sh
```

GASP-6 六组（只在准备开始正式实验时运行；本次未启动）：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 \
RESUME=1 \
FORCE_REBUILD_CLUSTERS=0 \
./scripts/run_selected_disjoint_two_rest_gasp6_v1.sh
```

仅运行 Real3D `car+chicken` 可用：

```bash
RUN_ONLY=Real3D:car+chicken \
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 FORCE_REBUILD_CLUSTERS=0 \
./scripts/run_selected_disjoint_two_rest_gasp6_v1.sh
```

当前无可用 PointNet2 CUDA 环境时的最小兼容 smoke：

```bash
/home/objectdec/anaconda3/envs/B/bin/python scripts/smoke_test_gasp6.py \
  --data-root /home/objectdec/data/Point3D/Real3D-AD-PCD_btp_fps2048 \
  --output-dir outputs/selected_two_rest_gasp6_v1_smoke/Real3D/car+chicken \
  --source-classes car chicken --target-class airplane \
  --source-samples-per-class 2 --target-samples 2
```

HS6P V1 六组：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 \
./scripts/run_selected_disjoint_two_rest_hs6p_v1.sh
```

A1–A4 `car+chicken`：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 \
./scripts/run_car_chicken_hs6p_a1_a4_top02.sh
```

HS6P V2：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 \
./scripts/run_selected_disjoint_two_rest_hs6p_v2_hierarchical_visual.sh
```

HS6P V3：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 \
./scripts/run_selected_disjoint_two_rest_hs6p_v3_visual_weighted.sh
```

测试：

```bash
/home/objectdec/anaconda3/envs/B/bin/python -m pytest -q
```

### 9.2 完整测试目录应包含

```text
mean_metrics.json           组合宏平均，含完整对象级和点级指标
per_category_metrics.json   各测试类别指标
metrics.json                精简诊断，不保证含所有点级字段
sample_scores.npz           逐样本 patch/point 分数与标签
hs6p_diagnostics.json       HS6P 层统计、Prompt 使用率、校准前后指标
gasp6_diagnostics.json      GASP cluster、mode/Prompt 对齐、先验和前后指标
geometry_clusters.npz       source descriptor mean/std、六个 centroid 与计数
geometry_clusters.json      聚类配置、source 类别、inertia 和可读诊断
training_complete.yaml      完成状态、最佳 loss、配置摘要
train.log / test.log        训练与测试日志
```

若 `metrics.json` 没有点级字段，应查看 `mean_metrics.json` 或 `per_category_metrics.json`。
GASP 的 `sample_scores.npz` 还包含 descriptor、六中心距离、几何先验、六 Prompt 相似度、uniform/geometry patch 与 point score 和 labels，可离线审计先验作用。
评估脚本不再自动生成独立 Markdown 报告，结构化诊断保存在 JSON/NPZ；pytest 也已关闭缓存插件，避免重新创建缓存 README。

### 9.3 当前产物完整性风险

1. Static 多个 checkpoint、JSON/CSV 和辅助文件为 0 字节。
2. HS6P V1 `car+chicken` 的主要指标、诊断、日志和 checkpoint 非空，但部分辅助文件为 0 字节。
3. `outputs/hs6p_ablations_top02` 与 `outputs/selected_two_rest_hs6p_v2_hierarchical_visual` 不存在。
4. V3 `candybar+fish` 只有前 5 个 epoch，没有完成标记或测试指标。
5. V3 诊断中的 `visual_adapter_training_mode=fused_only` 与 checkpoint 来源/Stage 1 标记的 `hierarchical_deep_supervision` 不一致，最终归档前必须修正或解释。
6. `RESUME=1` 前必须检查 checkpoint 与完成标记是否非空有效；仅判断文件存在可能错误跳过损坏运行。
7. GASP 和更新后的 Static 脚本通过 `utils/artifact_integrity.py` 检查非空、JSON/YAML 可解析、checkpoint 可 `torch.load`、NPZ 字段完整及所需指标存在；损坏产物会重命名为 `.corrupt.<UTC时间戳>` 后重跑，不删除其他结果。

## 10. 实验追加模板

以后新增实验时复制本节，追加到“实验日志”最上方；不要另建 Markdown。

```markdown
### YYYY-MM-DD：实验名 / 版本

- 状态：计划中 / 运行中 / 完成 / 中断 / 归档
- 协议：two-rest / one-rest / 其他（写清 source 与 target）
- 基线：名称、配置、checkpoint、基线指标
- 假设：为什么预期有效
- 唯一改动：相对基线具体改了什么
- 无泄漏约束：训练、验证、校准分别使用哪些数据
- 配置：`path/to/config.yaml`
- 命令：`...`
- 输出：`path/to/output/`
- 完整性：完成组合数、非空 checkpoint/JSON/log

| 指标 | Baseline | Candidate | Gain |
| --- | ---: | ---: | ---: |
| Object AUROC |  |  |  |
| Object AP |  |  |  |
| Point AUROC |  |  |  |
| Point AP |  |  |  |
| AUPRO |  |  |  |

- 诊断：按 source、面积桶、normal tail、Prompt/layer 使用率等记录
- 结论：保留 / 继续验证 / 失败归档
- 后续动作：下一步以及禁止重复的失败方向
```

## 11. 实验日志

### 2026-07-15：GASP-6 `car+chicken` 三随机种子稳定性编排（未执行实验）

- 新增 `scripts/run_car_chicken_gasp6_three_seed_stability.sh`，固定种子为 `111/222/333`；按种子串行执行 A0 Static → A2 geometry prior only → A3 full，不运行 A1、A4 或其他 source 组合。
- 三个种子固定复用同一 Stage 1 checkpoint 与同一 source-only geometry cluster；因此该实验测量的是 Stage 2 Prompt 训练及 GASP 模块的随机稳定性，不是 Stage 1+Stage 2 端到端稳定性。
- `EXPERIMENT_SEED` 通过公共 runner 传给训练和测试入口；公共可复现工具统一设置 Python、NumPy、PyTorch CPU/CUDA 和 DataLoader generator/worker seed，并将实际 seed 报告写入日志、checkpoint、`training_complete.yaml` 和指标 JSON。
- A2 严格加载当前种子的 A0 `best.pth`，恢复检查同时核对 checkpoint 绝对路径与 seed；固定 geometry cluster 只在缺失/损坏且显式设置 `FORCE_REBUILD_CLUSTERS=1` 时构建一次。
- 新增 `scripts/summarize_gasp6_three_seed.py`，从有效产物生成逐种子指标、三种子 mean/sample std/min/max、同种子配对 delta、稳定性判断和诊断均值/标准差；A2/A3 必须具有独立 uniform/geometry 字段，历史指标不硬编码。
- 本次没有启动 A0/A2/A3、没有运行 PointBERT 训练或测试，也没有生成新实验指标；仅进行 Bash/Python 语法、帮助信息及轻量 seed/汇总逻辑测试。

### 2026-07-15：新增 `car+chicken` GASP-6 串行消融编排（未执行实验）

- 新增 `scripts/run_car_chicken_gasp6_ablation_sequence.sh`，固定串行顺序为 A0 Static Stage 2 重训 → A2 geometry prior only → A1 mode supervision only → A3 full；支持分别用 `RUN_A*=0` 跳过。
- 脚本始终复用并校验指定 Stage 1，不允许该序列重训 Stage 1；正式 geometry cluster 只检查/构建一次，完整时直接复用，损坏且 `FORCE_REBUILD_CLUSTERS=0` 时退出。
- `RESUME=1` 使用 `utils/artifact_integrity.py` 检查非空、解析、checkpoint 和所需字段；训练产物有效但测试不完整时只补测试，训练本身无效时先归档旧实验目录再从对应合法起点重跑。`RESUME=0` 也先将旧目录改名为 `.previous.<时间戳>`，不删除旧实验。
- A2 是纯测试路径，`STATIC_PROMPT_CHECKPOINT` 明确指向本序列生成/验证的 A0 `best.pth`，不会训练 Prompt。
- 新增 `scripts/summarize_car_chicken_gasp6_ablation.py`，完成后从各自 `mean_metrics.json` 读取 Static A0、A2、A1、A3-uniform、A3-geometry，输出 JSON/CSV/TXT 和相对 A0 的绝对变化；不硬编码历史指标。
- Static/GASP 公共 runner 只做环境变量入口的最小兼容：`RUN_ONLY`、`SKIP_STAGE1`、`STAGE1_CHECKPOINT`、`EXPERIMENT_CONFIG`、独立 `OUTPUT_ROOT` 和共享 cluster。
- 本次没有运行 A0/A2/A1/A3，没有调用 PointBERT 训练或测试，也没有生成新指标。仅执行新脚本 `bash -n`、`--help` 及只读配置/路径解析，均通过。

### 2026-07-15：GASP-6 `car+chicken` 正式消融前置检查（被 Static Prompt checkpoint 阻塞）

- 按顺序先检查 Static A0。现有 `mean_metrics.json` 非空且可解析，重新读取的五项指标为 Object AUROC `0.7175707051`、Object AP `0.7400684173`、Point AUROC `0.8543738086`、Point AP `0.2112614305`、AUPRO `0.6265114923`。
- 同一目录的 Static `best.pth`、`best_loss.pth`、`per_category_metrics.json`、`sample_scores.npz`、`test.log`、`training_complete.yaml` 等文件为 0 字节。完整性工具已将它们重命名为 `.corrupt.20260715-*`；未覆盖或重训 A0。对项目及 `/home/objectdec` 的非空 `.pth` 做了只读检索，未找到可兼容的 Static Six Prompt `car+chicken` checkpoint。HS6P 和 CPU smoke checkpoint 没有被冒充为 Static。
- Stage 1 adapter 有效：`outputs/selected_two_rest_visual_baseline_v7/Real3D/car+chicken/best.pth`，大小 `7,932,075` bytes，SHA256 `d619795ae873f25be6779ce3881b57d03b008d634269f8e809ca9a4e3fed6bf1`；source 为 `car,chicken`，PointBERT layers 为 `[2,5,8,11]`。
- 当前 CUDA 环境已恢复可用：NVIDIA driver `570.169`、CUDA `12.8`、PyTorch `2.8.0+cu128`、RTX 5090 D（SM 12.0）；`pointnet2_ops.furthest_point_sample` CUDA 实测通过。
- 使用有效 Stage 1 和全部 source 数据正式建立 `outputs/gasp6_car_chicken_ablation/geometry_clusters.npz/json`，未读取 target。共使用 `4,269` 个 source 异常 patch，六簇计数 `[670,245,1043,864,1070,377]`，inertia `62731.671875`，最大/最小簇比 `4.3673`。
- cluster 1 的 chicken 占比 `0.8245`，按 `>0.8` 规则标记为 source-dominated；其余五簇未超过阈值。car/chicken 总异常 patch 数分别为 `1,208/3,061`。
- 阻塞结论：A2 必须加载有效 Static Stage 2 Prompt checkpoint，且运行顺序要求 A2 在 A1/A3 之前。缺少该 checkpoint 时不能合法生成 A2，也不能越过 A2 启动 A1/A3。本轮没有生成 A1/A2/A3 正式指标。
- 运行脚本已最小支持 `EXPERIMENT_CONFIG`：A2 为纯评估路径，不训练 Prompt；A1/A2/A3 使用独立输出目录和共享正式 cluster。

### 2026-07-15：GASP-6 V1 实现与工程验证

- 状态：代码完成，正式六组实验未启动；正式 PointBERT smoke 被当前机器缺少可用 NVIDIA 驱动及 PointNet2 CPU 不支持阻断。
- 基线：Static Six Prompt；保持同一 Stage 1、feature-first adapter、Prompt 参数、Static loss、patch-to-point、top-20% 对象聚合和 two-rest source 组合。
- 唯一方法改动：用 source 异常 patch 的 34 维 SE(3) 不变描述符建立固定六簇；Stage 2 可选 mode CE；推理可选 source-centroid Prompt 相对权重。不增加 router、gate、异常几何分数或可学习参数。
- 配置：`configs/two_rest_gasp6_v1.yaml`；消融为 `configs/ablations/gasp6_a0_static.yaml` 至 `gasp6_a4_hard_assignment.yaml`。
- 测试：完整 pytest 为 49 passed；描述符、聚类、mode loss、baseline 恢复、checkpoint 和损坏 RESUME 均覆盖。
- 兼容 smoke：真实 `car+chicken` source 和 `airplane` target 数据，确定性 KNN patch/冻结轻量 token 替代不可用的 PointBERT CUDA 部分；训练/反传/checkpoint/测试/指标/完整性检查均通过。该 smoke 只有 2 个 target 对象，指标无统计意义，不进入结果表。
- 风险信号：六簇计数 `[1,3,3,6,5,1]`，`car/chicken` 异常 patch 数为 `8/11`，多个簇由单类主导；mode prediction accuracy `0.1579`，Prompt 使用集中于 Prompt 2；先验平均熵 `1.3603`、最大权重均值 `0.5740`。这些受极小样本和替代 token 影响，只作为正式实验必须复核的诊断清单。
- 输出：`outputs/selected_two_rest_gasp6_v1_smoke/Real3D/car+chicken/`；完整性工具对 training、clusters 和 evaluation 三类产物均判定通过。

### 2026-07-15：文档合并与状态核对

- 将当前 Static/HS6P 方法、two-rest 结果、历史 V7–V22 实验与归档结论合并到本文。
- 删除其余独立方法说明、历史报告、清理清单和自动生成的 Markdown，JSON/CSV/NPZ、日志、checkpoint、代码与配置不受影响。
- 当前推荐动作：先修复/重跑 Static 六组并建立完整可审计基线，再决定是否继续新的方法实验。

## 12. NCRP：Normal-Centered Residual Prompting

### 12.1 状态与动机

当前维护状态（2026-07-17）：**NCRP-K1 已提升为唯一活动 NCRP 主方法。** 活动配置只有 `configs/ncrp_k1.yaml`。旧 K=6 A1/A2/A3/A4 与 A1-Q/A2-Q/A3-Q 的以下段落保留为历史实验记录，但对应 YAML、专用串行/汇总脚本以及 QR、adaptive、cover/reject、dual-basis 模型代码均已删除；已有输出不删除、不覆盖。Static A0 继续作为未发表的内部强基线只读保留。

Static Six Prompt 为每个异常 Prompt 学习一组完整异常 token，正常 token 同时出现在正常和异常 Prompt 上。NCRP 改为只保留原正常 Prompt token，由冻结文本编码器得到类别相关正常锚点 $t_c^N$；异常侧不再存在 `abnormal_tokens`，而是在 1280 维文本嵌入空间学习 6 个跨类别共享的残差基。它也不同于 CAP-style 的完整正常/异常上下文 Prompt：NCRP 的异常语义被明确限制为“相对当前类别正常锚点的正交偏离”。

NCRP 完整保留冻结 ULIP-2 PointBERT、第 2/5/8/11 层、feature-first adapter、冻结 Stage 1 adapter、冻结 OpenCLIP、原 patch-to-point、source patch label 聚合、focal/dice/object BCE 接口和 top-20% object mean。它不读取 geometry cluster，不使用 GASP prior/mode loss、HS6P、动态文本网络或 2D 投影。

三随机种子 Static A0 不重训，基线只从以下已有有效产物读取：

```text
outputs/gasp6_car_chicken_three_seed/seed_111/a0_static/
outputs/gasp6_car_chicken_three_seed/seed_222/a0_static/
outputs/gasp6_car_chicken_three_seed/seed_333/a0_static/
```

所有 NCRP 实验固定复用：

```text
outputs/selected_two_rest_visual_baseline_v7/Real3D/car+chicken/best.pth
```

因此该三随机种子实验测量 Stage 2 Prompt/NCRP 的随机稳定性，不是 Stage 1+Stage 2 的端到端随机稳定性。

### 12.2 共享正常锚点与正交残差基

正常 Prompt 后缀仍为 `a point cloud patch of a {category}`。冻结文本编码器输出并归一化 $t_c^N\in\mathbb{R}^{D}$，当前 $D=1280$。共享残差基为 $B\in\mathbb{R}^{6\times D}$，按行使用标准差 0.02 的正态分布初始化并归一化；初始化使用当前实验 seed。

对每个类别分别投影：

$$
b_{c,k}^{\perp}=b_k-(b_k^\top t_c^N)t_c^N,
\qquad d_{c,k}=\operatorname{Normalize}(b_{c,k}^{\perp}).
$$

计算使用 float32 范数、`eps=1e-6` 和 `nan_to_num`。若投影范数接近零，代码从正常锚点绝对值最小的坐标构造确定性的单位正交回退方向，避免 float16/float32 下的 NaN/Inf，同时保持 $d_{c,k}^\top t_c^N\approx0$。

### 12.3 A1：正常中心残差原型

配置：`configs/ablations/ncrp_a1_residual_uniform.yaml`。

$$
t_{c,k}^A=\operatorname{Normalize}(t_c^N+\gamma d_{c,k}),\qquad\gamma=1.
$$

$$
\bar s_g^A=\tau_p\left[\log\sum_k\exp(s_{g,k}^A/\tau_p)-\log 6\right],
\qquad \ell_g=(\bar s_g^A-s_g^N)/\tau.
$$

A1 不做 patch assignment，不使用 cover/reject；只训练正常 token 和一套 residual basis。原 Prompt diversity loss 被投影后残差方向 Gram 的非对角平方均值替代：

$$
L_{basis}=\operatorname{mean}_{c,i\ne j}(d_{c,i}^\top d_{c,j})^2,
\qquad\lambda_{basis}=0.01.
$$

### 12.4 A2：patch 自适应组合

配置：`configs/ablations/ncrp_a2_adaptive_basis.yaml`。

$$
r_g^{raw}=z_g-(z_g^\top t_c^N)t_c^N,
\qquad r_g=\operatorname{Normalize}(r_g^{raw}),
$$

$$
u_{g,k}=r_g^\top d_{c,k},\qquad
a_g=\operatorname{softmax}(u_g/\tau_{basis}),\quad\tau_{basis}=0.1,
$$

$$
d_g=\operatorname{Normalize}\left(\sum_k a_{g,k}d_{c,k}\right),
\quad t_g^A=\operatorname{Normalize}(t_c^N+\gamma d_g),
\quad \ell_g=((z_g^\top t_g^A)-(z_g^\top t_c^N))/\tau.
$$

零残差 patch 返回安全零方向并标记 invalid；assignment 仍为归一化的普通 Softmax。A2 不增加 cover/reject，只验证自适应组合相对 A1 uniform prototypes 的作用。

### 12.5 A3：异常覆盖与正常排斥

配置：`configs/ablations/ncrp_a3_cover_reject.yaml`。

cover/reject 使用 detach 的视觉 patch 和正常锚点重新计算残差与投影方向，梯度只进入 residual basis，不能通过移动正常锚点投机降低辅助损失：

$$
L_{cover}=\operatorname{mean}_{g\in abnormal}(1-\cos(\operatorname{sg}(r_g),\hat d_g)),
$$

$$
L_{reject}=\operatorname{mean}_{g\in normal}\operatorname{ReLU}
\left(\max_k\cos(\operatorname{sg}(r_g),d_{c,k})-0.2\right).
$$

$$
L=L_{focal}+L_{dice}+0.5L_{object}
+0.1L_{cover}+0.05L_{reject}+0.01L_{basis}.
$$

没有有效异常/正常 patch 时，相应损失使用 `basis.sum()*0` 返回可反向的数值零。每个 epoch 记录分项 loss、coverage cosine、normal max alignment、Gram 非对角均值、basis usage、assignment entropy 和正常/异常 residual norm。

### 12.6 A4：local/object 双残差基

配置：`configs/ablations/ncrp_a4_dual_basis.yaml`。

A4 使用 $B_{local},B_{object}\in\mathbb{R}^{6\times1280}$。Local branch 独占 focal、dice、local cover/reject/basis，并产生最终 point map、Point AUROC/AP 和 AUPRO。Object branch 使用相同的冻结视觉 patch，但通过独立 object basis 产生 patch logit，最终对象分数仍为 top-20% patch 概率均值；它只接收 object BCE 和 object basis loss，默认关闭 object cover。

Object branch 使用 $t_{object}^N=\operatorname{stopgrad}(t_{local}^N)$，因此 object BCE 不更新 normal Prompt 或 local basis；local focal/dice 也不更新 object basis。输出额外保存 local object score 与 object-basis object score，以检查两套语义是否真正分化。

### 12.7 参数量、实现入口和产物（历史与当前）

OpenCLIP token/文本维度当前均为 1280：

| 方法 | 正常 token | local residual basis | object residual basis | Stage 2 可训练参数 |
| --- | ---: | ---: | ---: | ---: |
| Static Six Prompt | 5,120 | 不适用 | 不适用 | 35,840 |
| NCRP A1/A2/A3 | 5,120 | 7,680 | 0 | 12,800 |
| NCRP A4 | 5,120 | 7,680 | 7,680 | 20,480 |
| **NCRP-K1（当前主方法）** | **5,120** | **1,280** | **0** | **6,400** |

主要实现：

```text
models/normal_centered_residual_prompt.py
utils/residual_prompt_config.py
configs/two_rest_ncrp_v1.yaml
configs/ncrp_k1.yaml
tests/test_normal_centered_residual_prompt.py
scripts/run_ncrp_k1_real3d_seed111.sh
scripts/run_ncrp_k1_real3d_one_rest_seed111.sh
```

其中 `configs/two_rest_ncrp_v1.yaml` 是历史入口，现已删除；当前唯一有效入口是 `configs/ncrp_k1.yaml`。

训练保存 `best.pth`、`last.pth`、optimizer/scheduler/epoch、`training_complete.yaml` 和完整配置；中断时只从可读取、seed 一致且含 NCRP/optimizer/scheduler 状态的 `last.pth` 恢复。正式输出目录为：

```text
outputs/ncrp_car_chicken_three_seed/seed_<seed>/
  a1_residual_uniform/Real3D/car+chicken/
  a2_adaptive_basis/Real3D/car+chicken/
  a3_cover_reject/Real3D/car+chicken/
  a4_dual_basis/Real3D/car+chicken/
```

`sample_scores.npz` 不保存大型 PointBERT feature，只保存 labels/object labels、local patch/point/object scores、residual norm、basis assignment/similarity 和组合方向范数；A4 额外保存 object patch/object score。`ncrp_diagnostics.json` 保存 usage/entropy/Gram、正常与异常统计、目标类别 usage、资源记录和 local/object 相关性。

现有工程把 `use_static_prompt=true` 同时作为“需要加载冻结 OpenCLIP 并进入 Stage 2 Prompt 框架”的公共开关。NCRP 为最小兼容而保留该开关，但模型构建首先检查 `residual_prompt_enabled`，实例化 NCRP learner，checkpoint 权重写入独立的 `residual_prompt` 键，绝不构造或保存 Static `abnormal_tokens`；`static_prompt_version` 仅沿用为现有 checkpoint/配置的版本标识字段。嵌套 `residual_prompt` YAML 由 `utils/residual_prompt_config.py` 映射到现有 flat argparse namespace，旧 Static/GASP YAML 不含该节时不会增加或改变任何默认分支。

### 12.8 运行与预期验证问题

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 SEED=111 \
bash scripts/run_ncrp_k1_real3d_seed111.sh
```

one-rest 的 `seahorse`、`shell`、`starfish` 入口为 `scripts/run_ncrp_k1_real3d_one_rest_seed111.sh`。这两个脚本只调用 NCRP-K1 主配置，不再提供旧 NCRP 消融选择。

当前明确风险：直接学习文本嵌入 residual 缺少自然语言可解释性；adaptive combination 可能同步抬高正常 patch 异常相似度；cover 可能被少数强异常主导；reject 可能压制细微异常；六个 basis 仍可能使用不均或长期闲置；A4 object basis 可能过拟合 source object label；正交约束可能限制表达；A1–A4 指标不保证单调。

### 2026-07-15：NCRP A1–A4 实现（未执行正式实验）

- 状态：已实现，未进行正式训练。
- 固定基线：读取既有三随机种子 Static A0，不重训、不覆盖。
- 固定视觉侧：三个 seed 和四个消融共用同一 Stage 1 adapter。
- 无泄漏：训练仅使用 source `car+chicken`；target 标签只允许用于最终评估和事后诊断，不用于参数选择。
- 本次检查范围：Python/Bash 语法、单元测试、合成张量前向/反向/checkpoint/inference smoke、帮助信息和 Markdown；没有启动 PointBERT 正式训练或测试。

### 2026-07-15：旧 NCRP seed 111 结果、方向塌缩与 QR 决策

状态：**旧 NCRP 完成 seed 111，新 QR 消融已实现但未训练；seed 222/333 暂停。** 下表全部从当前工作区的有效 `mean_metrics.json` 读取，括号为相对同 seed Static A0 的绝对变化，没有使用历史指标硬编码：

| 方法 | Object AUROC | Object AP | Point AUROC | Point AP | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static A0 | 0.7181 | 0.7404 | 0.8540 | 0.2111 | 0.6265 |
| NCRP A1 | 0.7174 (-0.0006) | 0.7391 (-0.0013) | 0.8530 (-0.0010) | 0.2059 (-0.0052) | 0.6239 (-0.0026) |
| NCRP A2 | 0.7045 (-0.0136) | 0.7231 (-0.0173) | 0.8514 (-0.0026) | 0.2074 (-0.0037) | 0.6229 (-0.0036) |
| NCRP A3 | 0.7072 (-0.0108) | 0.7290 (-0.0114) | 0.8519 (-0.0021) | 0.2031 (-0.0080) | 0.6237 (-0.0029) |
| NCRP A4 | 0.6905 (-0.0276) | 0.7172 (-0.0232) | 0.8576 (+0.0036) | 0.2054 (-0.0057) | 0.6350 (+0.0085) |

关键诊断与结论：

- A1 与 Static 的异常参数化不同：A1 不存在 `abnormal_tokens`，而是学习文本嵌入残差基；但 A1 的六个投影 residual direction 出现明显方向塌缩，basis Gram 非对角绝对均值约为 **0.871**。A1 的固定 uniform usage 不能证明 basis 形成分工。
- A2、A3 的 assignment normalized entropy 分别约为 **0.963**、**0.970**，仍接近 1；没有单 basis 使用率超过 0.8，但 patch 自适应分配整体接近均匀，六个方向没有形成有效异常分工。
- A3 在 source 训练末期的 abnormal coverage cosine 约为正值 0.089，但 target 诊断为 **-0.044**，说明 source 学到的覆盖方向没有良好迁移到未见类别。
- A4 提高 Point AUROC 和 AUPRO，但独立 object basis 把 Object AUROC/AP 降至 0.6905/0.7172；同一 A4 checkpoint 的 local branch object 诊断为 0.7096/0.7299，下降主要来自独立 object branch。因此本轮不继续 A4-Q。
- 暂停 seed 222、333，也不扩展其他 source 组合。下一步先用显式正交参数化验证“单分支 residual space”本身是否成立：在关于正常锚点的正交补中对投影残差基执行 reduced QR，构造 A1-Q、A2-Q、A3-Q。
- QR 只强制方向线性独立，不保证异常语义或 assignment 分工。只有 A1-Q/A2-Q/A3-Q 至少一个在 seed 111 明确超过 Static，才考虑继续 seed 222/333；只有多 seed 稳定成立后，才重新考虑双分支设计。

### 12.10 显式 QR 正交消融（已实现，未正式训练）

> 历史状态说明（2026-07-17）：本节记录当时的设计与检查结果；QR 实现、配置和运行/汇总脚本现已删除，不再是可执行入口。

旧 `soft` 路径保持原实现不变。只有新配置显式设置以下字段时才进入 QR：

```yaml
residual_prompt:
  orthogonalization:
    mode: qr
    qr_eps: 1.0e-6
    canonicalize_sign: true
    compute_in_float32: true
```

对每个 batch 的类别正常锚点 $t_c^N\in\mathbb{R}^D$，先把共享 raw basis $B\in\mathbb{R}^{K\times D}$ 投影到其正交补：

$$
V_{c,k}=B_k-(B_k^\top t_c^N)t_c^N.
$$

随后对 $V_c^\top\in\mathbb{R}^{D\times K}$ 批量执行 reduced QR：

$$
Q_c,R_c=\operatorname{qr}(V_c^\top),\qquad D_c^{res}=Q_c^\top.
$$

实现不逐样本使用 Python 循环，输出为 `[batch,K,D]`，满足 $D_c^{res}(D_c^{res})^\top\approx I$ 和 $D_c^{res}t_c^N\approx0$。float16/bfloat16 默认临时转 float32 做 QR，再转回输入公共 dtype。R 对角的负号通过 $Q\leftarrow Q\operatorname{diag}(\operatorname{sign}(\operatorname{diag}R))$、$R\leftarrow\operatorname{diag}(\operatorname{sign}(\operatorname{diag}R))R$ 规范化；零符号按 1 处理。

为处理重复 basis、basis 与 normal 平行等退化输入，代码首先按 basis 内容进行确定性排序，使 raw basis 行置换后进入相同 QR 顺序；然后选择避开 normal anchor 最大绝对坐标的 $K$ 个确定性坐标方向，将其投影到 normal 正交补，以 `qr_eps` 幅值加入 $V$。加扰动前后各再次显式投影，避免极小浮点残差被 QR 放大。输出记录 R 最小绝对对角、基于 R 奇异值的条件数估计、rank warning、Gram 非对角均值和 residual direction 与 normal anchor 的最大绝对内积。发现近似秩亏时发出明确 `RuntimeWarning`，但不产生 NaN/Inf。

新消融保持 raw 参数量不变，均只训练 5,120 个 normal token 参数和 $6\times1280=7,680$ 个 raw residual basis 参数，总计 12,800：

- A1-Q：`configs/ablations/ncrp_a1q_qr_uniform.yaml`。QR 后构造六个正常中心异常原型，继续使用 uniform log-mean-exp；关闭 adaptive、cover、reject 和 basis loss。Gram loss 只计算诊断，不进入总损失。
- A2-Q：`configs/ablations/ncrp_a2q_qr_adaptive.yaml`。在 A1-Q 上启用原 `tau_basis=0.1` 的 patch Softmax adaptive combination；不使用 cover/reject/basis loss。
- A3-Q：`configs/ablations/ncrp_a3q_qr_cover_reject.yaml`。在 A2-Q 上启用原 cover/reject，保持 $\lambda_{cover}=0.1$、$\lambda_{reject}=0.05$、margin 0.2；视觉 residual 和辅助 normal anchor 的 detach 规则不变，basis loss 关闭。

本轮不实现 A4-Q。旧 A4 的独立 object branch 已明显降低 Object AUROC/AP，应先验证单分支 QR residual space；只有 A1-Q/A2-Q/A3-Q 至少一个在 seed 111 明确超过 Static，才考虑新的双分支设计。

用户手动运行脚本只处理 seed 111，顺序为 A1-Q、A2-Q、A3-Q、只读汇总：

```bash
PYTHON_BIN=/home/objectdec/anaconda3/envs/B/bin/python \
BATCH_SIZE=4 RESUME=1 \
bash scripts/run_car_chicken_ncrp_qr_seed111.sh
```

输出目录：

```text
outputs/ncrp_qr_car_chicken_seed111/
  a1q_qr_uniform/Real3D/car+chicken/
  a2q_qr_adaptive/Real3D/car+chicken/
  a3q_qr_cover_reject/Real3D/car+chicken/
  summary.json
  summary.csv
  summary.txt
```

`scripts/summarize_ncrp_qr_seed111.py` 读取同 seed A0、旧 A1/A2/A3 和新 A1-Q/A2-Q/A3-Q，报告五项指标、相对 A0 绝对变化、Gram、normal-anchor 内积、assignment entropy、最大 usage、target coverage、normal alignment、QR 最小 R 对角、rank warning 和训练/测试时间。它不搜索或修改超参数。

当前风险：QR 正交只保证方向不同，不保证每个方向具有异常语义；QR 对输入顺序天然敏感，本实现使用确定性内容排序并测试最终集合分数不变，但排序交叉处仍可能带来梯度不连续；rank deficiency 可能导致梯度不稳定；强制正交可能降低表达自由度；assignment 仍可能接近均匀；cover 仍可能无法迁移到 target；A1-Q 到 A3-Q 的结果可能不单调。只有 seed 111 出现明确改善，才继续 seed 222/333。

### 12.11 NCRP-K1 / A1-K1 单异常残差向量（seed 111 三组正式结果）

状态：**2026-07-16 已完成 Real3D 三个 two-rest source pair 的 seed 111 正式训练、测试、完整性检查和汇总。** BTP sampled-2048 是公开发表的外部 baseline；Static Six Prompt 是本项目构建但尚未发表的内部强基线；旧 NCRP A1 K=6 是直接消融。超过 BTP 不等于全面超过内部 Static 主干。

动机是检验旧 A1 的六个 residual vectors 在投影后高度相关（`car+chicken` Gram 非对角绝对均值约 0.871）时，六方向是否实际上退化为一个有效方向。A1-K1 只学习共享 raw residual $b\in\mathbb{R}^{1280}$。对类别正常锚点 $t_c^N$：

$$
d_c=\operatorname{Normalize}\left(b-(b^\top t_c^N)t_c^N\right),\qquad
t_c^A=\operatorname{Normalize}(t_c^N+\gamma d_c),\quad \gamma=1.
$$

对 patch 特征 $z_g$ 直接计算

$$
l_g=\frac{z_g^\top t_c^A-z_g^\top t_c^N}{\tau}.
$$

K=1 不使用 QR、adaptive assignment、cover、reject 或 basis decorrelation loss，也不存在 basis 之间的竞争或 diversity。实现保留 `[1,D]`、`[B,1,D]` 和 `[B,G,1]` 形状；单项 uniform 聚合走严格等价的直接取值路径。空 Gram 非对角集合和 K=1 basis loss 返回数值零；assignment entropy 标为 `not_applicable_single_basis`，usage `[1.0]` 标为结构决定而非学得分工。

协议固定 seed 111、batch size 4、旧 A1 的 100 epochs/optimizer/scheduler/学习率、top-percent 0.2 和 `global_alpha=0`。三个 source pair 分别为 `car+chicken`、`candybar+fish`、`duck+starfish`，只使用各自已有 Stage 1 adapter，不重训 adapter，不读取 target 数据训练，也不按 target 指标调参。输出位于 `outputs/ncrp_k1_real3d_seed111/`。

#### 正式五项指标

| Source pair | 方法 | Object AUROC | Object AP | Point AUROC | Point AP | AUPRO |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| car+chicken | BTP sampled-2048 | 0.676700 | 0.716800 | 0.831900 | 0.207800 | 0.532100 |
| car+chicken | Static Six Prompt（内部） | 0.718050 | 0.740425 | 0.854012 | 0.211100 | 0.626519 |
| car+chicken | 旧 NCRP A1 K=6 | 0.717411 | 0.739122 | 0.853032 | 0.205934 | 0.623947 |
| car+chicken | NCRP A1-K1 | 0.717370 | 0.739108 | 0.852515 | 0.205251 | 0.623532 |
| candybar+fish | BTP sampled-2048 | 0.723900 | 0.750300 | 0.871300 | 0.193700 | 0.632300 |
| candybar+fish | Static Six Prompt（内部） | unavailable | unavailable | unavailable | unavailable | unavailable |
| candybar+fish | 旧 NCRP A1 K=6 | 0.752084 | 0.775595 | 0.850845 | 0.175865 | 0.605439 |
| candybar+fish | NCRP A1-K1 | 0.752368 | 0.775555 | 0.851028 | 0.174923 | 0.605077 |
| duck+starfish | BTP sampled-2048 | 0.680500 | 0.720200 | 0.819900 | 0.224500 | 0.493200 |
| duck+starfish | Static Six Prompt（内部） | unavailable | unavailable | unavailable | unavailable | unavailable |
| duck+starfish | 旧 NCRP A1 K=6 | 0.709068 | 0.741331 | 0.868748 | 0.212039 | 0.610338 |
| duck+starfish | NCRP A1-K1 | 0.709380 | 0.741402 | 0.868602 | 0.211340 | 0.610156 |

Static 历史目录中 `candybar+fish` 和 `duck+starfish` 的必要结果存在缺失或 0 字节产物，因此严格标记为 unavailable，没有填 0、手工推测或用历史文字替代。Static 宏平均只有一组，不能与三组宏平均作全面结论。

#### 三组宏平均与配对差值

| 方法 | 有效 pair 数 | Object AUROC | Object AP | Point AUROC | Point AP | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTP sampled-2048 | 3 | 0.693700 | 0.729100 | 0.841033 | 0.208667 | 0.552533 |
| 旧 NCRP A1 K=6 | 3 | 0.726188 | 0.752016 | 0.857541 | 0.197946 | 0.613241 |
| NCRP A1-K1 | 3 | 0.726373 | 0.752022 | 0.857382 | 0.197171 | 0.612922 |

| 配对变化 | Object AUROC | Object AP | Point AUROC | Point AP | AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: |
| K1 − BTP（三组） | +0.032673 | +0.022922 | +0.016348 | -0.011496 | +0.060389 |
| K6 − BTP（三组） | +0.032488 | +0.022916 | +0.016508 | -0.010720 | +0.060708 |
| K1 − K6（三组） | +0.000185 | +0.000006 | -0.000160 | -0.000775 | -0.000319 |
| K1 − Static（仅 car+chicken） | -0.000680 | -0.001318 | -0.001497 | -0.005849 | -0.002987 |

按预先规定的 `|delta| < 0.001` 近似持平阈值，K1 与 K6 的三组宏平均五项指标全部近似持平；每个 pair 的 K1−K6 五项变化也都小于 0.001。一个方向因此足以替代旧 A1 中六个高度相关方向，但证据是“保持性能并减参”，不是“K1 明确提高性能”。K1 在 BTP 三组宏平均上提高 Object AUROC、Object AP、Point AUROC 和 AUPRO，Point AP 则下降 0.011496，未解决 Point AP 短板。唯一有效的同 seed Static 配对中，K1 未超过内部 Static，尤其 Point AP 低 0.005849。

现场 checkpoint/diagnostics 统计的 Prompt 可训练参数为：Static 35,840；旧 A1 K=6 为 12,800；A1-K1 为 6,400（normal tokens 5,120 + residual 1,280）。K1 相对 K6 减少 50%，相对 Static 减少 82.14%。这只表示 **Prompt trainable parameter efficiency**，不代表冻结 PointBERT 总参数同比减少。

结论：A1-K1 当前更适合作为参数效率和“六方向退化”机制消融，而不是最终主方法。下一步应先补 seed 222/333 验证 K1≈K6 是否稳定；在多 seed 证实之前不扩展论文主张。由于 K1 已复现 K6，当前没有优先补 K=2/K=4 的必要；只有多 seed 出现 K1 不稳定、而 K6 稳定时再补中间 K。投稿上，该结果能强化简洁性与消融完整性，但单独不足以支撑“全面优于内部强基线”的主结论。

### 12.12 NCRP-K1 主方法代码收敛（2026-07-17）

NCRP-K1 现作为唯一维护的 NCRP 实现，方法名不再带 A1 消融前缀。主配置为：

```text
configs/ncrp_k1.yaml
```

活动模型只包含四个 normal context tokens 和一个 `[1,D]` residual vector。对每个类别的正常锚点执行一次正交投影并构造单一异常原型；异常相似度直接取该原型的 cosine，不经过多 Prompt log-mean-exp。训练损失只保留 Static Stage 2 接口中的 focal、dice 和 object BCE，不再包含 Prompt diversity、basis decorrelation、assignment、cover/reject 或 mode loss。

已从活动代码删除：K=6 soft basis、patch adaptive assignment、assignment entropy、cover/reject/basis loss、local/object dual basis、QR 正交化，以及这些版本的 YAML、专用运行脚本和汇总脚本。旧 checkpoint 的权重键 `prompt_bank.local_residual_basis` 和旧版本标识 `ncrp_a1_k1_single_residual` 继续兼容，从而不会破坏已完成 NCRP-K1 结果的读取与恢复。

当前 Prompt 可训练参数为 6,400：normal tokens 5,120，加单 residual vector 1,280。`basis_usage=[1.0]` 只是单向量结构事实；assignment entropy 明确记为 `not_applicable_single_basis`，不能解释为学习到的分工。

### 12.13 活动方法与配置最终收敛（2026-07-18）

当前仓库只维护两条方法路径：内部强基线 Static Six Prompt，以及主方法 NCRP-K1。活动 YAML 严格只剩：

```text
configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml
configs/ncrp_k1.yaml
```

GASP-6、HS6P、旧多基 NCRP、QR、adaptive、cover/reject 和 dual-basis 的方法配置、模型、运行/汇总脚本及对应单元测试均已删除。旧实验输出没有删除或覆盖；本文前文继续保留这些实验的历史公式、结果和失败结论，但其中列出的旧运行命令不再是当前可执行入口。

活动代码由 `models/static_prompt.py`、`models/normal_centered_residual_prompt.py`、`models/trainable_baseline.py`、`train.py` 和 `test_standard_aupro.py` 构成。PointBERT/OpenCLIP 编码器、数据加载、指标、复现种子和 artifact integrity 属于两条方法共享的基础设施，继续保留。已有早期 Static checkpoint 可能使用 `geometric_mode_prompt` 等旧权重键；训练/测试入口只保留读取这些字段的兼容映射，不包含旧几何方法的前向、损失或配置分支。

保留运行入口：

```text
scripts/run_selected_disjoint_two_rest_static_six_prompt_v1.sh
scripts/run_ncrp_k1_real3d_seed111.sh
scripts/run_ncrp_k1_real3d_one_rest_seed111.sh
scripts/summarize_selected_two_rest.py
```
