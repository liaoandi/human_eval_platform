# LLM 评测平台 Human-based Eval Platform

[中文](#中文) | [English](#english)

---

## 中文

Human-based Eval Platform，用于生成评估数据集、并发收集多模型答案、计算胜率矩阵和 Bradley-Terry 排名。

### 设计动机

在公司内部做 AI 产品时，需要回答一个关键问题：我们的模型到底比竞品好多少？传统的自动化评测指标和真实用户感知经常脱节，LLM 评 LLM 又存在同源偏好。最可靠的方法还是让人来评，但人工评测如果没有系统化，就会变成拍脑袋 -- 样本偏、评测员质量参差不齐、结果不可复现。

这个平台把人工评测工程化：从评估集生成到答案收集到排名计算全流程自动化，用 Bradley-Terry 模型替代简单胜率统计，用 Golden Question 机制过滤不认真的评测员，用分层校正消除位置偏差和时间偏差。

### 核心流程

```
评估集生成 -> 答案收集 -> 人工评测 -> 结果分析
```

1. **生成评估集**：从 Query 池中去重、分类、采样，生成有代表性的评估样本
2. **收集答案**：并发调用多个 LLM，收集各模型对评估集的回答
3. **人工评测**：用户进行盲测对比，选择更好的答案
4. **结果分析**：基于对比结果计算胜率矩阵，使用 Bradley-Terry 模型生成排名

---

### 算法详解

#### 1. 评估集生成算法

评估集生成是整个流程的起点，目标是从大量原始 Query 中选出有代表性、无重复、类别均衡的评估样本。整个流程分为三个阶段：预处理、分类、选择。

---

##### Stage 1: 预处理

###### 1.1 精确去重

精确去重基于文本标准化，处理流程为：将文本转为小写、去除标点符号、合并多余空格、去除完全相同的文本，只保留首次出现的 Query。

该方法时间复杂度 O(n)，空间复杂度 O(n)，适合大规模数据的初筛。

###### 1.2 相关性过滤

相关性过滤用于剔除与目标领域无关或无评测价值的 Query。采用两层过滤策略：

**本地快速预过滤**：首先用规则快速剔除明显无效的 Query，减少 API 调用量。规则包括：
- 长度过短，<= 5 字符
- 可读字符比例过低，< 30%

**LLM 相关性判断**：对通过预过滤的 Query，调用 LLM 判断是否与目标领域相关且有评测价值。

判断标准采用严格的准入制：
- **必须拒绝**：与目标领域无关、过于简单，如"怎么改名"、"好玩吗"，语义不完整
- **必须通过**：语义完整、问题具体、具备一定深度的内容

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 5 |
| 输出格式 | JSON Schema，强制输出 Yes/No |

###### 1.3 Query 改写

Query 改写的目的是将口语化、冗余、不规范的 Query 转换为简洁、搜索友好的形式，提升后续去重和分类的质量。

改写原则：
- 保留核心意图，去除冗余表达
- 统一表述风格
- 保留原始文本备份，存入 `raw_query_original` 字段

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 50 |

###### 1.4 语义去重

精确去重无法处理表述不同但含义相同的 Query，如"怎么做红烧肉"和"红烧肉的做法"。语义去重通过 Embedding 向量相似度来识别这类重复。

**Step 1: 向量化**

将所有 Query 转换为 Embedding 向量，采用批量调用以提高效率。

| 参数 | 值 |
|------|-----|
| 模型 | text-embedding-3-small |
| 批大小 | 512 |

**Step 2: 近邻检索**

构建 KNN 索引，K=10，加速相似度检索，避免 O(n^2) 的全量比较。对每个 Query 找出其 K 近邻，计算余弦相似度。

**Step 3: 相似度分级处理**

根据相似度阈值将 Query 对分为两类：

| 相似度范围 | 处理方式 |
|-----------|---------|
| > 0.88，高相似度 | 直接判定为重复，无需 LLM 验证 |
| 0.75 - 0.88，边界相似度 | 需要 LLM 复核确认 |
| < 0.75 | 不视为重复 |

边界相似度的 LLM 复核会批量处理，判断两个 Query 是否询问相同的信息：

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 2048 |
| 批大小 | 60 |

**Step 4: 连通分量与 Canonical 选择**

使用并查集算法找出所有重复 Query 组成的连通分量。对每个连通分量，调用 LLM 选择最具代表性的 Query 作为 Canonical（即标准查询），其他标记为 DUPLICATE。

选择标准：表述更具体、更自然、更完整的 Query 优先。

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 50 |

###### 1.5 类别自动发现

类别体系不是预先定义的，而是根据实际 Query 数据自动发现。这样可以适应不同领域的数据特点。

处理流程：从相关 Query 中随机采样，最多 1000 条，让 LLM 分析这些样本，总结出核心意图分类。

发现原则：
- **覆盖度优先**：分类体系应覆盖 95% 以上的 Query
- **粒度适中**：建议 8-12 个一级分类
- **互斥性**：类别之间应有明确界限
- **兜底类别**：始终包含"其他"类别

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 1024 |
| 采样数量 | 最多 1000 条 |

---

##### Stage 2: 分类

###### 2.1 LLM 分类

将每个 Query 分类到发现的类别中。采用批量并发处理以提高效率。

**并发策略**：
- 外层通过 Semaphore 限制同时进行的 API 调用数，默认 4
- 内层将多个 Query 打包成 batch，每批 20 条，减少调用次数

**容错机制**：
- 单个 Query 分类失败不影响整体流程
- 失败的 Query 会进行最多 3 次重试，采用指数退避策略
- 最终仍失败的标记为 "classification_failed"，后续归入"其他"类别

**模糊匹配**：LLM 输出的类别名可能与预定义类别不完全一致，系统会进行模糊匹配：
- 首先尝试精确匹配
- 然后尝试大小写不敏感匹配
- 最后尝试字符串相似度匹配，阈值 0.66，和子串匹配

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 2048 批量 / 50 单条 |
| 批大小 | 20 |
| 并发数 | 4 |

###### 2.2 类别排除建议

某些类别可能不适合用于模型评测（如账号问题、技术支持类），系统会自动生成排除建议。

LLM 分析已发现的类别列表，识别出不适合评测的类别，输出建议排除的类别名单。

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 400 |
| 输出格式 | JSON Schema，强制输出类别列表 |

###### 2.3 人工选择排除类别

系统进入交互模式，展示所有类别及其样本，由人工决定最终排除哪些类别。

交互流程：
1. 按 Query 数量降序展示所有类别
2. 每个类别展示 3 条随机样本，帮助理解类别内容
3. 用户输入要排除的类别编号，逗号分隔
4. 被排除类别的 Query 标记为 EXCLUDE，不参与后续选择

这一步确保人工对评测集的内容有最终控制权，避免自动化流程引入不合适的内容。

---

##### Stage 3: 选择

###### 3.1 子类别细分

如果某个类别包含过多 Query（超过 50 条阈值），会自动进行子类别细分，避免单一类别在最终评测集中占比过大。

细分原则：
- 每个子类别至少包含 20 条 Query
- 子类别之间语义互斥，不能有重叠
- 子类别名称简洁、通用

处理流程：LLM 分析该类别下的所有 Query，将其分配到更细粒度的子类别中。

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.3，允许更多变化 |
| Max Tokens | 2048 |
| 细分阈值 | 50 条 |
| 最小子类别 | 20 条 |

###### 3.2 代表性打分

对每个 Query 进行多维度评分，用于后续选择时优先挑选高质量、有代表性的 Query。

**评分维度**，每项 0-100 分：

| 维度 | 说明 | 高分示例 | 低分示例 |
|------|------|---------|---------|
| 具体性 specificity | Query 描述的精确程度 | 指向唯一明确的游戏要素 | 模糊不清，无法理解意图 |
| 信息完整性 completeness | Query 包含的上下文信息 | 包含所有必要信息，无需追问 | 信息严重不足 |
| 深度与价值 depth_and_value | Query 的知识含量与评测价值 | 涉及多步推理、策略分析 | 简单的是非题、操作说明 |

**标签判断**，True/False：

| 标签 | 说明 | 示例 |
|------|------|------|
| is_procedural | 询问具体步骤、流程、路线 | "怎么解锁隐藏内容" |
| needs_structured_output | 需要表格、清单、对比形式回答 | "全部内容强度排行" |
| has_heavy_constraints | 包含多个限制条件、否定条件 | "推荐A类型，不要高稀有度" |
| is_trap_unreleased | 询问未上线内容、版本预测 | "下个活动是什么" |

评分采用批量并发处理：

| 参数 | 值 |
|------|-----|
| 模型 | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 2048 |
| 批大小 | 30 |
| 并发数 | 4 |

###### 3.3 分层采样 -- Hamilton 分配法

目标：从 N 个 Query 中选 K 个，保证各类别比例与原始分布一致。

采用 Hamilton 分配法（又称最大余数法），这是议会席位分配的经典算法：

1. 计算每个类别的理论配额：`配额 = 目标数量 x 类别占比`，结果通常是小数
2. 向下取整得到初始分配
3. 计算每个类别的余数，即理论配额减去初始分配
4. 将剩余名额按余数从大到小依次分配给各类别

举例：假设有 A 500 个、B 300 个、C 200 个三个类别，需采样 100 个。理论配额分别为 50、30、20，恰好整除，最终分配即为 50:30:20。若总数变为 1000 个需采样 33 个，理论配额为 16.5、9.9、6.6，初始分配 16、9、6 共 31 个，余数为 0.5、0.9、0.6，剩余 2 个名额分配给 B 和 C，最终为 16:10:7。

在每个类别内部，按代表性打分从高到低选取。

###### 3.4 高风险切片保障

某些类型的 Query 对模型能力区分度更高，需要保证最低数量，称为"切片配额"。这些切片基于 3.2 步骤中识别的标签来确定。

| 切片类型 | 目标比例 | 最低数量 | 作用 |
|---------|---------|---------|------|
| procedural / 步骤类 | 32% | 4 | 测试可执行性 |
| structured / 结构化 | 25% | 3 | 测试结构化输出能力 |
| constraint_heavy / 高约束 | 18% | 2 | 测试复杂指令理解 |
| trap_unreleased / 陷阱题 | 0%，已关闭 | 0 | 测试是否编造信息 |

保障策略采用后置补齐：
1. 先按 Hamilton 分配完成主采样
2. 检查各切片是否达到目标配额
3. 不足的从候选池中按代表性排名补齐
4. 允许 2 个的超额缓冲，避免过度约束

###### 3.5 选后相似度检测

即使经过语义去重，选中的子集中仍可能包含相似 Query，因为去重是全局的，但采样可能恰好选中相似的几个。

检测流程：
1. 对选中的 Query 重新计算 Embedding
2. 构建相似度矩阵
3. 找出相似度超过阈值 0.85 的 pair

替换策略：
- 每个相似 pair 中保留代表性排名更高的，替换另一个
- 从未选中的候选池中找最佳替代，要求同类别、排名最高、与已选中的不相似

| 参数 | 值 |
|------|-----|
| 模型 | text-embedding-3-small |
| 相似度阈值 | 0.85 |

###### 3.6 人工审核与调整

这是评估集生成的最后一道关卡，确保人工对最终结果有完全控制权。

**自动检查报告**：系统首先运行自动化规则检查：
- 检查是否有 DUPLICATE 状态的 Query 被选中
- 检查是否有无效类别（如未分类、分类失败、已排除）的 Query 被选中

**人工逐条审核**：系统展示所有选中的 Query，包括：
- Query ID
- 所属类别
- 选择原因：Hamilton 分配 / 切片补齐 / 替补
- Query 文本

**拒绝与替补**：
- 用户可以输入要拒绝的 Query ID，逗号分隔
- 被拒绝的 Query 标记为 EXCLUDE
- 系统自动从候选池中选择替补，重新运行选择流程
- 循环直到用户满意，按回车确认

这一步是整个流程中最重要的人工干预点，确保最终评测集的质量符合预期。

---

#### 2. 答案收集算法

##### 2.1 并发架构

系统采用三层架构：

**Query 分发层**：维护待处理的 Query 队列，按优先级调度。

**Model Dispatcher 层**：为每个 LLM Provider 维护独立的并发控制。不同 Provider 有不同的速率限制（如 OpenAI 约 10 QPS、Gemini 约 1 QPS），通过独立的 Semaphore 实现分级限流。

**结果聚合层**：收集各模型返回的答案，执行清洗，记录元数据（如延迟、token 数），并写入数据库。

##### 2.2 分级限流

不同 LLM Provider 的 rate limit 差异很大，采用分级并发控制：

| Provider | 并发数 | 原因 |
|----------|--------|------|
| OpenAI/Azure | 10 | TPM 限制较宽松 |
| Gemini | 5 | QPM 限制较严 |
| Doubao | 3 | 自建服务，保守限制 |
| 默认 | 5 | 未知 Provider 的安全值 |

每个 Provider 使用独立的 Semaphore，互不影响。这意味着系统可以同时向 OpenAI 发送 10 个请求、向 Gemini 发送 5 个请求，充分利用各 Provider 的配额。

##### 2.3 智能重试策略

不同类型的错误需要不同的处理策略：

| 错误类型 | 是否重试 | 等待策略 | 原因 |
|---------|---------|---------|------|
| Rate Limit 429 | 是 | 指数退避 + 随机抖动 | 避免同时重试造成再次限流 |
| Timeout | 是 | 短等待，1-3 秒 | 临时网络问题 |
| Server Error 5xx | 是 | 中等等待 | 服务端临时故障 |
| Auth Error 401/403 | 否 | - | 需要人工介入 |
| Invalid Request 400 | 否 | - | 请求本身有问题 |

最大重试次数为 3 次，超过后记录失败并继续处理其他请求。

##### 2.4 答案清洗

LLM 生成的答案常包含不适合评测的内容，需要清洗：

| 清洗项 | 原因 |
|-------|------|
| URL 移除 | 可能暴露模型身份，用户无法点击验证 |
| 引用标记移除 | `[1]`、`【1】` 等在评测场景下无意义 |
| Emoji 移除 | 可能影响用户对答案质量的客观判断 |
| 空白标准化 | 保证视觉呈现一致 |

---

#### 3. Bradley-Terry 排名算法

##### 3.1 模型背景

Bradley-Terry / BT 模型是分析成对比较数据的经典统计模型，1952 年提出，广泛应用于体育排名（即 Elo 评分的理论基础）、搜索结果排序，以及 LLM 评测如 Chatbot Arena。

**核心假设**：每个参与者 i 有一个正的"强度"参数 pi_i，参与者 i 击败 j 的概率仅取决于两者的强度比：

```
P(i beats j) = pi_i / (pi_i + pi_j)
```

这个假设的直观含义是：如果 A 的强度是 B 的两倍，则 A 击败 B 的概率是 2/3。

**模型性质**：
- 概率和为 1：P(i beats j) + P(j beats i) = 1
- 传递性：如果 A 强于 B，B 强于 C，则 A 很可能强于 C
- 可加性：强度的比例决定胜率，与绝对值无关

##### 3.2 参数估计

给定观测数据（即各模型间的对战结果），需要估计每个模型的强度参数。

**最大似然估计（MLE）**：找到一组参数使观测数据出现的概率最大。对于 BT 模型，对数似然函数为：

```
L(pi) = sum_ij [w_ij * log(pi_i) - n_ij * log(pi_i + pi_j)]
```

其中 w_ij 是 i 击败 j 的次数，n_ij 是 i 与 j 的总对局数。

**MM 算法**：直接求解 MLE 较困难，采用 Minorization-Maximization 迭代算法：

1. 初始化所有模型强度为 1
2. 迭代更新：`pi_i(new) = W_i / sum_j[n_ij / (pi_i(old) + pi_j(old))]`，其中 W_i 是模型 i 的总胜场数
3. 每轮迭代后归一化（除以均值），避免数值问题
4. 收敛判断：参数变化小于阈值（如 10^-6）时停止

该算法保证收敛到全局最优解，通常 50-100 轮迭代即可收敛。

##### 3.3 Elo 等效转换

BT 强度是相对值，不够直观。可以转换为更熟悉的 Elo 分数：

```
Elo(i) = 400 * log10(pi_i) + 1500
```

转换后的直观理解：
- Elo 差 100 分 -- 强者胜率约 64%
- Elo 差 200 分 -- 强者胜率约 76%
- Elo 差 400 分 -- 强者胜率约 91%

基准分 1500 是惯例，沿用国际象棋 Elo 系统的中等水平，可根据需要调整。

##### 3.4 置信区间估计

点估计无法反映不确定性，需要计算置信区间。采用 Bootstrap 方法：

1. 从原始对局数据中有放回地重采样，生成 1000 个 Bootstrap 样本
2. 对每个样本独立拟合 BT 模型，得到 1000 组参数估计
3. 对每个模型，取参数分布的 2.5% 和 97.5% 分位数作为 95% 置信区间

置信区间的宽度反映了排名的可靠性：区间越窄，排名越确定；区间重叠的模型，排名差异可能不显著。

---

#### 4. 反作弊系统

##### 4.1 Golden Question 机制

Golden Question（即黄金题）是有已知正确答案的题目，用于检测用户是否认真答题。

**设计原则**：
- **可验证**：有明确正确答案，可以客观判断用户是否答对
- **难度适中**：认真的用户能答对，但不能靠猜测通过
- **不可预测**：用户无法提前知道哪些是黄金题
- **时效性**：定期更新，避免被记忆和传播

**题目结构**：每道黄金题包含一个问题、正确答案、合理但错误的答案。在 A/B 测试中，正确答案随机分配到 A 或 B 位置，避免位置偏好干扰。

##### 4.2 黄金题生成

采用 LLM + Web Search 自动生成：

1. **生成问题**：让 LLM 为特定领域生成事实性问题，要求有明确可验证的答案、不是常识题、答案不会很快过时
2. **获取正确答案**：通过 Web Search 查询权威来源，确保答案准确
3. **生成错误答案**：让 LLM 生成"看起来合理但实际错误"的答案，格式与正确答案一致，避免从形式上就能判断对错
4. **随机分配位置**：将正确和错误答案随机分配到 A/B 位置

生成的黄金题需要人工审核后才能使用，确保质量。

##### 4.3 用户质量评分

基于用户的答题行为计算质量分数，用于识别低质量用户：

**黄金题准确率**：用户答对黄金题的比例。低于 60% 的用户可能是随机点击或不认真。需要至少 3 道黄金题才能计算，样本不足时暂不评判。

**答题时间分析**：统计每道题的答题时长。过快（低于 3 秒）的比例过高（超过 30%）说明用户可能没有认真阅读就随机选择。

**综合判定**：黄金题准确率低于 60%，或快速答题比例过高的用户，标记为"不合格"。不合格用户的答题数据在分析时降权或剔除。

---

### 产出文件

平台运行结束后，会在 `outputs/` 目录下生成以下文件：

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `winrate_{game_id}_{dimension}.png` | 热力图 | 每个 game_id 与评测维度组合的胜率矩阵热力图 |
| `BT_ANALYSIS_REPORT.md` | Markdown 报告 | Bradley-Terry 排名报告，含全局排名、按场次排名、按维度排名、场次与维度交叉排名 |
| `BT_ANALYSIS_REPORT.csv` | CSV 数据 | 与 Markdown 报告内容一致的结构化数据，便于二次分析 |
| `rank_vs_dimension.png` | 热力图 | 模型排名随评测维度变化的对比图，颜色映射 BT 强度值 |
| `winrate_scatter.png` | 散点图 | Raw Winrate 与 Stratified Winrate 的对比散点图，用于检验分层校正效果 |
| `winrate_diff_heatmap_all.png` | 热力图 | Raw 与 Stratified 胜率差异矩阵，用于定位分层校正影响最大的模型对 |
| `winrate_share_stacked.png` | 堆叠柱状图 | 各评测维度对胜率差距的贡献占比，按场次分面展示 |
| `elo_share_stacked.png` | 堆叠柱状图 | 各评测维度对 Elo 差距的贡献占比，按场次分面展示 |

---

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| pipeline | `eval_set_generator_refactored_v2.py` | 评估集生成：去重、分类、采样 |
| pipeline | `answer_collector_concurrent_v2.py` | 答案收集：并发调用、限流、重试 |
| pipeline | `pipeline_common_v2.py` | 公共组件：LLM 封装、数据库工具 |
| utils | `anti_cheat.py` | 反作弊：黄金题生成与验证 |
| analysis | `run_bt_analysis.py` | Bradley-Terry 排名计算，从原始对比数据 |
| analysis | `run_bt_from_matrix.py` | Bradley-Terry 排名计算，从胜率矩阵 |
| analysis | `balanced_bt_resample.py` | Bootstrap 置信区间估计 |
| analysis | `plot_winrate_matrix.py` | 胜率矩阵可视化 |
| analysis | `plot_bt_result.py` | BT 排名结果可视化 |
| analysis | `plot_rank_vs_dimension.py` | 排名与维度关系图 |
| analysis | `analyze_winrate_diff.py` | 胜率差异分析 |

---

### 安装与使用

```bash
# 安装
git clone https://github.com/your-repo/human_eval_platform.git
cd human_eval_platform
pip install -r requirements.txt

# 配置
cp .env.example .env
# 编辑 .env 填入 API keys

# 生成评估集
python -m src.pipeline.eval_set_generator_refactored_v2 --dm_partition 2025-01 --eval_size 100

# 收集答案
python -m src.pipeline.answer_collector_concurrent_v2 --set_id 123 --models gpt-4o,gemini-pro

# 分析排名
python -m src.analysis.run_bt_analysis --input results.csv
```

---

## English

Human-based Eval Platform for generating evaluation datasets, concurrently collecting multi-model answers, computing win-rate matrices, and producing Bradley-Terry rankings.

### Motivation

When building AI products internally, there's a critical question to answer: how much better is our model than the competition? Traditional automated evaluation metrics often diverge from real user perception, and LLM-as-Judge introduces same-source bias. The most reliable approach is still human evaluation, but without systematic engineering, it devolves into guesswork -- biased samples, inconsistent evaluator quality, and non-reproducible results.

This platform turns human evaluation into an engineering discipline: end-to-end automation from eval set generation through answer collection to ranking computation, Bradley-Terry models instead of naive win-rate statistics, Golden Question mechanisms to filter out careless evaluators, and stratified corrections to eliminate position and temporal biases.

### Core Workflow

```
Eval Set Generation -> Answer Collection -> Human Evaluation -> Result Analysis
```

1. **Generate Eval Set**: Deduplicate, classify, and sample from a query pool to produce representative evaluation samples
2. **Collect Answers**: Concurrently call multiple LLMs to collect each model's responses to the eval set
3. **Human Evaluation**: Users perform blind A/B comparisons and select the better answer
4. **Result Analysis**: Compute win-rate matrices from comparison results and generate rankings using the Bradley-Terry model

---

### Algorithm Details

#### 1. Eval Set Generation Algorithm

Eval set generation is the starting point of the entire workflow. The goal is to select representative, deduplicated, and category-balanced evaluation samples from a large pool of raw queries. The process is divided into three stages: Preprocessing, Classification, and Selection.

---

##### Stage 1: Preprocessing

###### 1.1 Exact Deduplication

Exact deduplication is based on text normalization: convert text to lowercase, remove punctuation, merge redundant whitespace, and remove identical texts -- keeping only the first occurrence of each query.

Time complexity is O(n) and space complexity is O(n), making it suitable for initial screening of large-scale data.

###### 1.2 Relevance Filtering

Relevance filtering removes queries that are unrelated to the target domain or lack evaluation value. A two-layer filtering strategy is used:

**Local Fast Pre-filtering**: Rule-based quick elimination of obviously invalid queries to reduce API calls. Rules include:
- Length too short, <= 5 characters
- Readable character ratio too low, < 30%

**LLM Relevance Judgment**: For queries passing pre-filtering, an LLM determines whether they are relevant to the target domain and have evaluation value.

Judgment criteria follow a strict admission policy:
- **Must reject**: Unrelated to target domain, too simple, e.g. "how to change name", "is it fun", semantically incomplete
- **Must pass**: Semantically complete, specific questions, sufficient depth

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 5 |
| Output Format | JSON Schema, forced Yes/No output |

###### 1.3 Query Rewriting

Query rewriting converts colloquial, redundant, or non-standard queries into concise, search-friendly forms, improving the quality of subsequent deduplication and classification.

Rewriting principles:
- Preserve core intent, remove redundant expressions
- Unify expression style
- Keep original text backup, stored in the `raw_query_original` field

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 50 |

###### 1.4 Semantic Deduplication

Exact deduplication cannot handle queries with different wording but the same meaning, e.g. "how to make braised pork" vs. "braised pork recipe". Semantic deduplication uses embedding vector similarity to identify such duplicates.

**Step 1: Vectorization**

Convert all queries into embedding vectors using batch API calls for efficiency.

| Parameter | Value |
|-----------|-------|
| Model | text-embedding-3-small |
| Batch Size | 512 |

**Step 2: Nearest Neighbor Retrieval**

Build a KNN index, K=10, to accelerate similarity retrieval, avoiding O(n^2) brute-force comparison. For each query, find its K nearest neighbors and compute cosine similarity.

**Step 3: Similarity-Tiered Processing**

Classify query pairs into tiers based on similarity thresholds:

| Similarity Range | Handling |
|-----------------|----------|
| > 0.88, high similarity | Directly marked as duplicate, no LLM verification needed |
| 0.75 - 0.88, borderline | Requires LLM review for confirmation |
| < 0.75 | Not considered duplicate |

Borderline similarity LLM reviews are processed in batches, determining whether two queries are asking for the same information:

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 2048 |
| Batch Size | 60 |

**Step 4: Connected Components and Canonical Selection**

Use a Union-Find algorithm to identify all connected components formed by duplicate queries. For each connected component, the LLM selects the most representative query as the Canonical (i.e. the standard query), while others are marked as DUPLICATE.

Selection criteria: Queries that are more specific, more natural, and more complete are preferred.

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 50 |

###### 1.5 Automatic Category Discovery

The category system is not predefined but automatically discovered from the actual query data, adapting to the characteristics of different domains.

Process: Randomly sample from relevant queries, up to 1000, then have the LLM analyze these samples and summarize core intent categories.

Discovery principles:
- **Coverage first**: The category system should cover 95%+ of queries
- **Moderate granularity**: Recommended 8-12 top-level categories
- **Mutual exclusivity**: Clear boundaries between categories
- **Catch-all category**: Always include an "Other" category

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 1024 |
| Sample Size | Up to 1000 |

---

##### Stage 2: Classification

###### 2.1 LLM Classification

Classify each query into the discovered categories using batch concurrent processing for efficiency.

**Concurrency strategy**:
- Outer layer: Semaphore limits concurrent API calls, default 4
- Inner layer: Multiple queries packed into batches, 20 per batch, to reduce call count

**Fault tolerance**:
- A single query classification failure does not affect the overall process
- Failed queries are retried up to 3 times with exponential backoff
- Queries still failing are marked as "classification_failed" and later assigned to "Other"

**Fuzzy matching**: LLM-output category names may not exactly match predefined categories; the system performs fuzzy matching:
- First attempt exact match
- Then case-insensitive match
- Finally string similarity match, threshold 0.66, and substring match

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 2048 batch / 50 single |
| Batch Size | 20 |
| Concurrency | 4 |

###### 2.2 Category Exclusion Suggestions

Some categories may be unsuitable for model evaluation (e.g. account issues or technical support), so the system automatically generates exclusion suggestions.

The LLM analyzes the discovered category list, identifies categories unsuitable for evaluation, and outputs a recommended exclusion list.

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 400 |
| Output Format | JSON Schema, forced category list output |

###### 2.3 Manual Category Exclusion

The system enters interactive mode, displaying all categories with their samples for human decision on which to exclude.

Interactive flow:
1. Display all categories sorted by query count, descending
2. Show 3 random samples per category to aid understanding
3. User inputs category numbers to exclude, comma-separated
4. Excluded categories' queries are marked EXCLUDE and removed from subsequent selection

This step ensures human control over eval set content, preventing the automated process from introducing unsuitable material.

---

##### Stage 3: Selection

###### 3.1 Subcategory Refinement

If a category contains too many queries, exceeding the threshold of 50, automatic subcategory refinement is performed to prevent any single category from dominating the final eval set.

Refinement principles:
- Each subcategory must contain at least 20 queries
- Subcategories must be semantically mutually exclusive with no overlap
- Subcategory names should be concise and general

Process: The LLM analyzes all queries under the category and assigns them to finer-grained subcategories.

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.3, allowing more variation |
| Max Tokens | 2048 |
| Refinement Threshold | 50 queries |
| Minimum Subcategory | 20 queries |

###### 3.2 Representativeness Scoring

Each query is scored across multiple dimensions to prioritize high-quality, representative queries during selection.

**Scoring dimensions**, 0-100 each:

| Dimension | Description | High Score Example | Low Score Example |
|-----------|-------------|-------------------|------------------|
| Specificity | Precision of query description | Points to a unique, clear element | Vague, intent unclear |
| Completeness | Contextual information included | All necessary info present, no follow-up needed | Severely insufficient info |
| Depth and Value | Knowledge content and evaluation value | Multi-step reasoning, strategy analysis | Simple yes/no, basic instructions |

**Tag judgments**, True/False:

| Tag | Description | Example |
|-----|-------------|---------|
| is_procedural | Asks about specific steps, processes, routes | "How to unlock hidden content" |
| needs_structured_output | Requires tables, lists, or comparison format | "Full content tier list" |
| has_heavy_constraints | Contains multiple constraints or negations | "Recommend type A, no high rarity" |
| is_trap_unreleased | Asks about unreleased content or version predictions | "What's the next event" |

Scoring uses batch concurrent processing:

| Parameter | Value |
|-----------|-------|
| Model | gpt-4o |
| Temperature | 0.1 |
| Max Tokens | 2048 |
| Batch Size | 30 |
| Concurrency | 4 |

###### 3.3 Stratified Sampling -- Hamilton Allocation

Goal: Select K queries from N total while maintaining category proportions consistent with the original distribution.

Uses the Hamilton method, also known as the Largest Remainder Method, a classic algorithm for parliamentary seat allocation:

1. Compute each category's theoretical quota: `quota = target_count x category_proportion` -- typically a decimal
2. Floor to get initial allocation
3. Compute each category's remainder -- theoretical quota minus initial allocation
4. Distribute remaining slots to categories in descending order of remainder

Example: Given categories A with 500, B with 300, C with 200 and a target of 100 samples -- theoretical quotas are 50, 30, 20, which divide evenly, yielding 50:30:20. If the target were 33 from 1000, quotas would be 16.5, 9.9, 6.6; initial allocation 16, 9, 6, total 31; remainders 0.5, 0.9, 0.6; the 2 remaining slots go to B and C, yielding 16:10:7.

Within each category, queries are selected in descending order of representativeness score.

###### 3.4 High-Risk Slice Guarantees

Certain query types have higher discriminative power for model capabilities and require minimum quotas, called "slice quotas." These slices are determined by the tags identified in Step 3.2.

| Slice Type | Target Proportion | Minimum Count | Purpose |
|-----------|-------------------|---------------|---------|
| procedural / step-by-step | 32% | 4 | Test executability |
| structured / structured output | 25% | 3 | Test structured output capability |
| constraint_heavy / high constraint | 18% | 2 | Test complex instruction understanding |
| trap_unreleased / trap questions | 0%, disabled | 0 | Test whether model fabricates info |

Guarantee strategy uses post-hoc backfilling:
1. Complete main sampling via Hamilton allocation first
2. Check whether each slice meets its target quota
3. Backfill shortfalls from the candidate pool by representativeness ranking
4. Allow a buffer of 2 extra slots to avoid over-constraining

###### 3.5 Post-Selection Similarity Detection

Even after semantic deduplication, the selected subset may still contain similar queries, since deduplication is global, but sampling may happen to select similar ones.

Detection process:
1. Recompute embeddings for selected queries
2. Build a similarity matrix
3. Identify pairs exceeding the threshold of 0.85

Replacement strategy:
- In each similar pair, keep the one with higher representativeness ranking; replace the other
- Find the best replacement from the unselected candidate pool -- same category, highest ranking, not similar to already-selected queries

| Parameter | Value |
|-----------|-------|
| Model | text-embedding-3-small |
| Similarity Threshold | 0.85 |

###### 3.6 Manual Review and Adjustment

This is the final checkpoint for eval set generation, ensuring full human control over the outcome.

**Automated Check Report**: The system first runs automated rule checks:
- Check for queries with DUPLICATE status that were selected
- Check for queries in invalid categories (unclassified, classification failed, or excluded) that were selected

**Manual Item-by-Item Review**: The system displays all selected queries, including:
- Query ID
- Category
- Selection reason: Hamilton allocation / slice backfill / replacement
- Query text

**Rejection and Replacement**:
- Users can input query IDs to reject, comma-separated
- Rejected queries are marked EXCLUDE
- The system automatically selects replacements from the candidate pool and reruns the selection process
- The loop continues until the user is satisfied, confirming with Enter

This step is the most critical human intervention point in the entire process, ensuring the final eval set meets quality expectations.

---

#### 2. Answer Collection Algorithm

##### 2.1 Concurrent Architecture

The system uses a three-layer architecture:

**Query Dispatch Layer**: Maintains a queue of pending queries, scheduled by priority.

**Model Dispatcher Layer**: Maintains independent concurrency controls for each LLM provider. Different providers have different rate limits (e.g. OpenAI ~10 QPS, Gemini ~1 QPS), implemented via independent Semaphores for tiered rate limiting.

**Result Aggregation Layer**: Collects answers from each model, performs cleaning, records metadata (such as latency and token count), and writes to the database.

##### 2.2 Tiered Rate Limiting

Different LLM providers have vastly different rate limits, requiring tiered concurrency control:

| Provider | Concurrency | Reason |
|----------|-------------|--------|
| OpenAI/Azure | 10 | Generous TPM limits |
| Gemini | 5 | Stricter QPM limits |
| Doubao | 3 | Self-hosted service, conservative limits |
| Default | 5 | Safe default for unknown providers |

Each provider uses an independent Semaphore with no cross-interference. This means the system can simultaneously send 10 requests to OpenAI and 5 to Gemini, fully utilizing each provider's quota.

##### 2.3 Intelligent Retry Strategy

Different error types require different handling strategies:

| Error Type | Retry? | Wait Strategy | Reason |
|-----------|--------|---------------|--------|
| Rate Limit 429 | Yes | Exponential backoff + random jitter | Avoid simultaneous retries causing further throttling |
| Timeout | Yes | Short wait, 1-3s | Temporary network issue |
| Server Error 5xx | Yes | Medium wait | Temporary server failure |
| Auth Error 401/403 | No | - | Requires manual intervention |
| Invalid Request 400 | No | - | Problem with the request itself |

Maximum retry count is 3; beyond that, the failure is logged and processing continues with other requests.

##### 2.4 Answer Cleaning

LLM-generated answers often contain content unsuitable for evaluation and require cleaning:

| Cleaning Item | Reason |
|--------------|--------|
| URL removal | May reveal model identity; users cannot click to verify |
| Citation marker removal | Markers like `[1]` are meaningless in evaluation context |
| Emoji removal | May affect user objectivity in judging answer quality |
| Whitespace normalization | Ensures consistent visual presentation |

---

#### 3. Bradley-Terry Ranking Algorithm

##### 3.1 Model Background

The Bradley-Terry / BT model is a classic statistical model for analyzing pairwise comparison data, proposed in 1952. It is widely used in sports rankings (as the theoretical basis for Elo ratings), search result ranking, and LLM evaluation such as Chatbot Arena.

**Core assumption**: Each participant i has a positive "strength" parameter pi_i. The probability that participant i beats j depends only on their strength ratio:

```
P(i beats j) = pi_i / (pi_i + pi_j)
```

Intuitive meaning: If A's strength is twice B's, then A beats B with probability 2/3.

**Model properties**:
- Probabilities sum to 1: P(i beats j) + P(j beats i) = 1
- Transitivity: If A is stronger than B and B stronger than C, then A is very likely stronger than C
- Scale invariance: Win rates are determined by strength ratios, not absolute values

##### 3.2 Parameter Estimation

Given observed data (i.e. match results between models), we need to estimate each model's strength parameter.

**Maximum Likelihood Estimation (MLE)**: Find the parameters that maximize the probability of the observed data. For the BT model, the log-likelihood function is:

```
L(pi) = sum_ij [w_ij * log(pi_i) - n_ij * log(pi_i + pi_j)]
```

where w_ij is the number of times i beat j, and n_ij is the total number of matches between i and j.

**MM Algorithm**: Directly solving MLE is difficult; instead, the Minorization-Maximization iterative algorithm is used:

1. Initialize all model strengths to 1
2. Iterative update: `pi_i(new) = W_i / sum_j[n_ij / (pi_i(old) + pi_j(old))]`, where W_i is model i's total win count
3. Normalize after each iteration (divide by mean) to avoid numerical issues
4. Convergence check: Stop when parameter changes fall below a threshold (e.g. 10^-6)

This algorithm guarantees convergence to the global optimum, typically within 50-100 iterations.

##### 3.3 Elo Equivalent Conversion

BT strength values are relative and not intuitive. They can be converted to the more familiar Elo score:

```
Elo(i) = 400 * log10(pi_i) + 1500
```

Intuitive interpretation after conversion:
- 100 Elo difference -- stronger player wins ~64%
- 200 Elo difference -- stronger player wins ~76%
- 400 Elo difference -- stronger player wins ~91%

The baseline of 1500 is conventional, from the chess Elo system's intermediate level, and can be adjusted as needed.

##### 3.4 Confidence Interval Estimation

Point estimates cannot reflect uncertainty; confidence intervals are needed. The Bootstrap method is used:

1. Resample with replacement from the original match data to generate 1000 Bootstrap samples
2. Independently fit the BT model on each sample, yielding 1000 sets of parameter estimates
3. For each model, take the 2.5% and 97.5% quantiles of the parameter distribution as the 95% confidence interval

Confidence interval width reflects ranking reliability: narrower intervals mean more certain rankings; overlapping intervals between models suggest the ranking difference may not be statistically significant.

---

#### 4. Anti-Cheating System

##### 4.1 Golden Question Mechanism

Golden Questions are questions with known correct answers, used to detect whether users are answering seriously.

**Design principles**:
- **Verifiable**: Has a definitive correct answer that can be objectively checked
- **Moderate difficulty**: Serious users can answer correctly, but guessing is unreliable
- **Unpredictable**: Users cannot know in advance which questions are golden
- **Timely updates**: Regularly refreshed to prevent memorization and sharing

**Question structure**: Each golden question contains a question, a correct answer, and a plausible but incorrect answer. In A/B testing, the correct answer is randomly assigned to position A or B to avoid position preference bias.

##### 4.2 Golden Question Generation

Automatically generated using LLM + Web Search:

1. **Generate questions**: Have the LLM generate factual questions for a specific domain, requiring definitive verifiable answers, non-trivial difficulty, and time-stable answers
2. **Obtain correct answers**: Query authoritative sources via Web Search to ensure answer accuracy
3. **Generate incorrect answers**: Have the LLM produce "plausible but actually wrong" answers, matching the format of correct answers to prevent form-based detection
4. **Randomize positions**: Randomly assign correct and incorrect answers to A/B positions

Generated golden questions require manual review before use to ensure quality.

##### 4.3 User Quality Scoring

Quality scores are computed based on user answering behavior to identify low-quality users:

**Golden question accuracy**: The proportion of golden questions answered correctly. Users below 60% may be clicking randomly or not taking the task seriously. At least 3 golden questions are needed for computation; judgment is suspended when sample size is insufficient.

**Response time analysis**: Statistics on response time per question. An excessive proportion (over 30%) of overly fast responses (under 3 seconds) indicates the user may be selecting randomly without reading.

**Composite judgment**: Users with golden question accuracy below 60% or excessive fast-answer rates are flagged as "unqualified." Unqualified users' data is downweighted or excluded during analysis.

---

### Output Files

After the platform finishes running, the following files are generated in the `outputs/` directory:

| Filename | Type | Description |
|----------|------|-------------|
| `winrate_{game_id}_{dimension}.png` | Heatmap | Win-rate matrix heatmap for each game_id and evaluation dimension combination |
| `BT_ANALYSIS_REPORT.md` | Markdown Report | Bradley-Terry ranking report with global ranking, per-game ranking, per-dimension ranking, and game-dimension cross ranking |
| `BT_ANALYSIS_REPORT.csv` | CSV Data | Structured data matching the Markdown report, for secondary analysis |
| `rank_vs_dimension.png` | Heatmap | Model ranking variation across evaluation dimensions, color-mapped to BT strength |
| `winrate_scatter.png` | Scatter Plot | Raw Winrate vs. Stratified Winrate comparison, for verifying stratification correction effects |
| `winrate_diff_heatmap_all.png` | Heatmap | Raw vs. Stratified win-rate difference matrix, for locating model pairs most affected by stratification |
| `winrate_share_stacked.png` | Stacked Bar Chart | Each evaluation dimension's contribution to win-rate gaps, faceted by game |
| `elo_share_stacked.png` | Stacked Bar Chart | Each evaluation dimension's contribution to Elo gaps, faceted by game |

---

### Module Reference

| Module | File | Function |
|--------|------|----------|
| pipeline | `eval_set_generator_refactored_v2.py` | Eval set generation: deduplication, classification, sampling |
| pipeline | `answer_collector_concurrent_v2.py` | Answer collection: concurrent calls, rate limiting, retries |
| pipeline | `pipeline_common_v2.py` | Common components: LLM wrappers, database utilities |
| utils | `anti_cheat.py` | Anti-cheating: golden question generation and validation |
| analysis | `run_bt_analysis.py` | Bradley-Terry ranking computation, from raw comparison data |
| analysis | `run_bt_from_matrix.py` | Bradley-Terry ranking computation, from win-rate matrix |
| analysis | `balanced_bt_resample.py` | Bootstrap confidence interval estimation |
| analysis | `plot_winrate_matrix.py` | Win-rate matrix visualization |
| analysis | `plot_bt_result.py` | BT ranking result visualization |
| analysis | `plot_rank_vs_dimension.py` | Rank vs. dimension chart |
| analysis | `analyze_winrate_diff.py` | Win-rate difference analysis |

---

### Installation and Usage

```bash
# Install
git clone https://github.com/your-repo/human_eval_platform.git
cd human_eval_platform
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env to fill in API keys

# Generate eval set
python -m src.pipeline.eval_set_generator_refactored_v2 --dm_partition 2025-01 --eval_size 100

# Collect answers
python -m src.pipeline.answer_collector_concurrent_v2 --set_id 123 --models gpt-4o,gemini-pro

# Analyze rankings
python -m src.analysis.run_bt_analysis --input results.csv
```
