# 蛋白质指令数据集分析文档

本文档基于以下两类信息整理：

- [README.md](C:\Users\12240\Desktop\ssh\protdata\README.md) 中给出的官方说明
- 对下列 7 个 JSON 文件做的本地流式统计与样本抽样

分析范围：

- `PDD_replaced.json`
- `PFUD_replaced.json`
- `PFUD_replaced_with_neighbors.json`
- `PFUD_replaced_with_neighbors_filtered.json`
- `PFUD_replaced_with_neighbors_filtered_top1ge20.json`
- `PSAD_replaced.json`
- `PSPD_replaced.json`

## 一、结论摘要

这 7 个文件可以分成两类：

- 4 个基础任务数据集：`PDD`、`PFUD`、`PSAD`、`PSPD`
- 3 个 `PFUD` 的派生增强版本：在原始 `PFUD` 的基础上加入近邻参考蛋白信息，并逐步过滤样本

从建模角度看，最重要的结论有 5 点：

1. `PDD`、`PSPD`、大部分 `PFUD` 和绝大多数 `PDD/PSPD` 样本都是“一条蛋白对应一条样本”，但 `PSAD` 存在明显重复 accession，同一蛋白平均约对应 `1.22` 条样本。
2. `PFUD_replaced_with_neighbors*.json` 不是简单加字段，而是把原始多轮对话形态改造成了“单轮输入 + 指令中注入邻居信息”的新格式。
3. `PFUD` 邻居增强版中，很多样本的 `input` 不再包含当前目标蛋白的序列/结构，只提供任务问题，模型需要依赖指令中的邻居参考信息。
4. `PFUD` 邻居增强版引入了新的标签体系：`< aa sequence>` 和 `< 3Di sequence>`，不能只按原始 `PFUD` 的 `< protein sequence>` / `< protein structure>` 规则做解析。
5. `PDD` 和 `PSAD` 存在 `task` / `test` 字段混用，`PSPD` 存在少量 `split="validation"`；训练前建议统一规范化。

## 二、总体对比

| 文件 | 大小 | 样本数 | 唯一 accession | 唯一 instruction | 主任务/角色 |
|---|---:|---:|---:|---:|---|
| `PDD_replaced.json` | 176.6 MB | 107,980 | 107,980 | 12 | 约束驱动的蛋白设计 |
| `PFUD_replaced.json` | 816.9 MB | 426,915 | 426,915 | 108 | 蛋白功能理解（原始版） |
| `PFUD_replaced_with_neighbors.json` | 1,447.5 MB | 426,915 | 426,915 | 414,525 | `PFUD` 邻居增强版（未过滤） |
| `PFUD_replaced_with_neighbors_filtered.json` | 1,435.0 MB | 419,329 | 419,329 | 411,518 | `PFUD` 邻居增强版（过滤后） |
| `PFUD_replaced_with_neighbors_filtered_top1ge20.json` | 1,371.3 MB | 399,339 | 399,339 | 391,820 | `PFUD` 邻居增强版（更严格过滤） |
| `PSAD_replaced.json` | 345.2 MB | 250,469 | 205,371 | 1 | 亚基组成分析 + 结构预测 |
| `PSPD_replaced.json` | 260.9 MB | 264,486 | 264,486 | 201,987 | 序列到结构预测 |

补充观察：

- `PDD`、`PFUD`、`PFUD` 三个变体、`PSPD` 都基本是“一条 accession 对应一条样本”。
- `PSAD` 的 accession 少于样本数，说明同一蛋白会以多条样本重复出现。
- `PFUD` 原始版只有 108 类指令模板，但邻居增强版因为把邻居内容拼进 `instruction`，导致 `instruction` 几乎变成“接近样本级唯一”。

## 三、各数据集详细分析

### 1. PDD：Protein Design Dataset

定位：

- 任务是根据自然语言约束生成蛋白序列，部分样本还要求输出结构。

结构特征：

- 字段：`instruction`、`input`、`output`、`accesion`、`split`，训练集额外包含空的 `history`
- `output` 中 `107,980 / 107,980` 样本都包含 `< protein sequence>`
- `96,609 / 107,980` 样本同时包含 `< protein structure>`
- `input` 始终非空
- `history` 仅出现在训练集且恒为 `[]`

分片分布：

- `train`: 98,620
- `valid`: 5,455
- `test`: 3,905

工程含义：

- 这是一个标准的条件生成数据集，输入是约束文本，输出是带标签的序列/结构文本。
- 如果做多任务训练，`PDD` 会偏向“生成式设计”而不是“理解式描述”。

### 2. PFUD：Protein Function Understanding Dataset（原始版）

定位：

- 核心任务是从蛋白序列/结构推断功能、定位、过程、模体等描述性信息。

结构特征：

- 字段：`instruction`、`input`、`output`、`history`、`accesion`、`split`、`task`
- `output` 是自然语言说明，不带标准化结构标签
- 存在两种主要样式：
  - 单轮样式：`input` 非空，直接包含 `< protein sequence>` 和 `< protein structure>`
  - 多轮样式：`input` 为空，蛋白信息和前序问答放在 `history`

关键统计：

- `input` 为空：389,318
- `input` 非空：37,597
- `history` 非空：389,318
- `history` 长度分布：
  - 长度 0：37,597
  - 长度 1：159,930
  - 长度 2：194,245
  - 长度 3：35,143

分片分布：

- `train`: 391,775
- `valid`: 27,438
- `test`: 7,702

工程含义：

- 原始 `PFUD` 是一个典型的“混合单轮/多轮”理解数据集。
- 如果直接喂给单轮 instruction-tuning 模型，往往需要先把 `history` 展开或重写。

### 3. PFUD 邻居增强版：PFUD_replaced_with_neighbors.json

定位：

- 在原始 `PFUD` 基础上加入近邻参考蛋白，把任务改造成带检索参考的功能理解。

结构特征：

- 字段收敛为：`instruction`、`input`、`output`、`accesion`、`split`、`task`
- 不再包含 `history`
- 所有样本的 `instruction` 都包含：
  - `Reference proteins (top-2 nearest)`
  - `Neighbor 1`
  - `Neighbor 2`
  - `< aa sequence>`
  - `< 3Di sequence>`

关键统计：

- 样本数与原始 `PFUD` 完全一致：426,915
- `input` 全部非空
- `input` 中仍包含当前目标蛋白 `< protein sequence>` / `< protein structure>` 的样本：197,527（46.27%）
- `input` 不再包含当前目标蛋白显式序列/结构的样本：229,388（53.73%）

推断（基于计数和样本对照）：

- 这不是简单把邻居附加到原始样本后面，而是做了“多轮展平”。
- 原始 `PFUD` 中：
  - `history` 长度为 `0 或 1` 的样本一共 `37,597 + 159,930 = 197,527`
  - 这个数字与邻居增强版中“`input` 仍含目标蛋白”的样本数完全一致
- 原始 `PFUD` 中：
  - `history` 长度为 `2 或 3` 的样本一共 `194,245 + 35,143 = 229,388`
  - 这个数字与邻居增强版中“`input` 不再含目标蛋白”的样本数完全一致

因此可以高置信度推断：

- 原始多轮对话被改写成单轮样本
- 后续轮次的问题被提取到 `input`
- 邻居蛋白信息被注入到 `instruction`
- 当原始样本有较长 `history` 时，当前目标蛋白本身可能不再直接出现在 `input`

工程含义：

- 该版本更适合检索增强或带参考上下文的训练。
- 但它不再是“纯目标蛋白分析”，因为相当一部分样本主要依赖邻居信息。
- 如果希望模型严格基于当前目标蛋白做判断，这个版本会引入偏置。

### 4. PFUD 邻居增强过滤版：PFUD_replaced_with_neighbors_filtered.json

定位：

- 在邻居增强版基础上做了一轮样本过滤，目标应当是提高参考邻居质量。

关键统计：

- 样本数：419,329
- 相比未过滤版减少：7,586（保留率 `98.22%`）
- `input` 中包含目标蛋白序列/结构：190,743（45.49%）
- `input` 中不包含目标蛋白显式序列/结构：228,586（54.51%）

结构特征：

- 字段结构与未过滤版一致
- 邻居仍然固定写入 `instruction`
- 仍全部使用 `< aa sequence>` 和 `< 3Di sequence>`

工程含义：

- 这是对未过滤邻居版的温和清洗。
- 如果想保留绝大多数样本，同时减少部分低质量近邻带来的噪声，这是更稳妥的选择。

### 5. PFUD 邻居增强严格过滤版：PFUD_replaced_with_neighbors_filtered_top1ge20.json

定位：

- 从命名看，这是在过滤版基础上继续施加更严格的近邻筛选条件（文件名暗示与 `top1 >= 20` 有关）。

关键统计：

- 样本数：399,339
- 相比未过滤版减少：27,576（保留率 `93.54%`）
- 相比普通过滤版再减少：19,990
- `input` 中包含目标蛋白序列/结构：175,170（43.86%）
- `input` 中不包含目标蛋白显式序列/结构：224,169（56.14%）

工程含义：

- 这个版本更偏向“牺牲覆盖率，换更强邻居质量约束”。
- 如果你更在乎参考样本可信度，而不是最大样本量，这个版本通常更适合作为高质量训练子集。

### 6. PSAD：Protein Subunit Analysis + Design-like Structure Output

定位：

- 输入蛋白序列，输出亚基组成描述，并附带预测结构。

结构特征：

- 字段：`instruction`、`input`、`output`、`accesion`、`split`
- `input` 的 `250,469 / 250,469` 样本都包含 `< protein sequence>`
- `output` 的 `250,469 / 250,469` 样本都包含 `< protein structure>`
- `instruction` 模板高度统一，实际只有 `1` 个唯一模板

分片分布：

- `train`: 230,575
- `valid`: 7,088
- `test`: 12,806

补充观察：

- 唯一 accession 为 `205,371`，低于总样本数 `250,469`
- 平均每个 accession 对应约 `1.22` 个样本

工程含义：

- 这是一个非常“模板化”的任务，适合做结构化输出学习。
- 由于指令几乎不变，模型可能更多依赖输入/输出模式，而非自然语言提示多样性。

### 7. PSPD：Protein Structure Prediction Dataset

定位：

- 直接从序列预测结构。

结构特征：

- 字段：`instruction`、`input`、`output`、`accesion`、`split`、`task`
- `input` 恒为空
- `instruction` 中 `264,486 / 264,486` 都直接嵌入 `< protein sequence>`
- `output` 中 `264,486 / 264,486` 都包含 `< protein structure>`

分片分布：

- `train`: 214,855
- `valid`: 24,713
- `test`: 24,895
- `validation`: 23

工程含义：

- 这是一个非常直接的序列到结构映射任务。
- 训练前应将 `validation` 统一并入 `valid`，否则会多出一个极小的异常 split。

## 四、字段与格式兼容性

跨文件统一处理时，建议重点注意以下问题：

1. `accesion` 的拼写就是 `accesion`，不是 `accession`，下游解析要按原字段名读取。
2. `PDD` 和 `PSAD` 中，部分非训练样本使用 `test` 字段表示任务标签，而不是 `task`。
3. `PSPD` 中有 `23` 条样本使用 `split="validation"`，建议统一映射到 `valid`。
4. `PFUD` 原始版主要使用 `< protein sequence>` / `< protein structure>`。
5. `PFUD` 邻居增强版在 `instruction` 中固定使用 `< aa sequence>` / `< 3Di sequence>`。
6. 如果预处理只匹配 `< protein ...>` 标签，会漏掉邻居增强版中的近邻信息。

推荐统一规则：

```python
task_norm = sample.get("task", sample.get("test"))
split_norm = "valid" if sample.get("split") == "validation" else sample.get("split")
```

## 五、建模与使用建议

如果目标是构建一个统一的多任务蛋白模型，建议按用途分层使用：

1. 做基础多任务预训练：
使用 `PDD + PFUD_replaced + PSAD + PSPD`，先保留原始任务定义，减少格式漂移。

2. 做检索增强或 reference-aware 微调：
优先使用 `PFUD_replaced_with_neighbors_filtered.json`，它比未过滤版更干净，同时比 `top1ge20` 保留更多覆盖率。

3. 做高质量、小一些的邻居增强实验：
使用 `PFUD_replaced_with_neighbors_filtered_top1ge20.json`，它更适合强调参考邻居质量。

4. 做严格的“仅基于目标蛋白”评估：
不要直接混用 `PFUD` 邻居增强版；至少要先过滤掉 `input` 中不包含当前目标蛋白序列/结构的样本。

5. 做统一 tokenizer / parser：
必须同时支持以下标签：
`< protein sequence>`、`< protein structure>`、`< aa sequence>`、`< 3Di sequence>`

## 六、推荐的数据选择策略

按常见目标给出一个更直接的选择建议：

- 想保留任务原貌并避免额外偏置：选 `PFUD_replaced.json`
- 想引入检索增强思想但不过度缩小数据：选 `PFUD_replaced_with_neighbors_filtered.json`
- 想做更保守的高质量邻居实验：选 `PFUD_replaced_with_neighbors_filtered_top1ge20.json`
- 想做序列生成：选 `PDD_replaced.json`
- 想做序列到结构预测：选 `PSPD_replaced.json`
- 想做亚基组成 + 结构联合输出：选 `PSAD_replaced.json`

## 七、一句话判断

如果只选一个 `PFUD` 版本用于实际训练，默认推荐 `PFUD_replaced_with_neighbors_filtered.json`：

- 相比未过滤邻居版，它更干净
- 相比 `top1ge20`，它保留了更多样本
- 相比原始版，它更适合做参考增强式 instruction tuning

但如果你的目标是保持“只看目标蛋白本体”的任务定义，仍应优先使用 `PFUD_replaced.json`。
