# -*- coding: utf-8 -*-
"""
Eval Set Generator: Generates evaluation datasets (stages 1-3)
Handles query preprocessing, classification, and selection
python3 eval_set_generator_refactored.py --dm_partition 2025-09 --eval_size 15 --mode test --interactive --overwrite
"""
import argparse
import asyncio
import importlib
import logging
import os
import sys

import re
import json
import time
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import openai
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async
from tqdm.auto import tqdm

# 支持通过环境变量切换 pipeline_common 模块
PIPELINE_COMMON_MODULE = os.getenv("PIPELINE_COMMON_MODULE", "src.pipeline.pipeline_common_v2")
if PIPELINE_COMMON_MODULE != "pipeline_commons":
    try:
        commons_module = importlib.import_module(PIPELINE_COMMON_MODULE)
        sys.modules['pipeline_commons'] = commons_module
        logging.debug(f"Using custom pipeline commons module: {PIPELINE_COMMON_MODULE}")
    except Exception as e:
        if PIPELINE_COMMON_MODULE == "src.pipeline.pipeline_common_v2":
            logging.warning(
                "Failed to import src.pipeline.pipeline_common_v2 (%s); falling back to pipeline_commons", e
            )
        else:
            raise ImportError(
                f"Failed to import module specified by PIPELINE_COMMON_MODULE={PIPELINE_COMMON_MODULE}: {e}"
            )

# === 明确导入所需的函数和常量，避免 import * 的命名空间污染 ===
from pipeline_commons import ( # type: ignore
    # 数据列名常量
    COL_QUERY_ID, COL_QUERY, COL_GAME_NAME, COL_CATEGORY, COL_CANONICAL_QUERY_ID, COL_QUERY_TIME,
    
    # 配置常量
    MODEL_CONFIG_INTERNAL, DEFAULT_TIMESTAMP_FORMAT, SOURCE_TABLE,
    
    # Query status constants
    QUERY_STATUS_RELEVANT, QUERY_STATUS_IRRELEVANT, QUERY_STATUS_DUPLICATE,
    QUERY_STATUS_EXCLUDE,
    
    # Category constants
    CATEGORY_NO_CLASSIFICATION, CATEGORY_CLASSIFICATION_FAILED, CATEGORY_OTHER,
    
    # 表名常量
    QUERY_PREPROCESS_DETAILS_TABLE_NAME, QUERY_ITEM_TABLE_NAME, QUERY_SET_TABLE_NAME,
    
    # 核心工具函数
    sanitize_name, llm_chat_call, insert_dataframe,
    
    # 初始化和设置函数
    initialize_clients, setup_logging, generate_set_id,
    get_table_names,
    
    # 数据读取函数
    read_rows_with_condition,
    
)

# ========================================
# CONFIGURATION
# ========================================

# Pipeline configuration constants
DEFAULT_EVAL_SIZE = 30
LOGGING_LEVEL = logging.INFO

# Temperature settings for different stages
TEMPERATURE_DETERMINISTIC = 0.0      # For schema detection and other deterministic outputs
TEMPERATURE_STRICT = 0.1              # For relevance, classification, scoring (need consistency)
TEMPERATURE_CREATIVE = 0.3            # For duplicate detection, subdivision (allow some variation)

# Batch size settings
DUPLICATION_BATCH_SIZE = 500  # For duplicate detection
FILTER_SELECT_SCORING_BATCH_SIZE = 30  # For scoring queries within categories
CLASSIFICATION_BATCH_SIZE = 20  # Batch size for classification
SUBCATEGORY_SIZE_THRESHOLD = 50  # Threshold for subdividing large categories into subcategories
SUBCATEGORY_MIN_SIZE = 20  # Minimum size below which we stop subdividing

# Max tokens settings for different stages
MAX_TOKENS_TINY = 5                    # For yes/no responses (relevance check)
MAX_TOKENS_SHORT = 50                  # For single selections (classification, canonical selection, schema test)
MAX_TOKENS_MEDIUM = 400                # For JSON schema outputs (exclusion suggestions)
MAX_TOKENS_LONG = 512                  # For text outputs with fallback parsing
MAX_TOKENS_CATEGORY = 1024             # For category discovery
MAX_TOKENS_EXTENDED = 2048             # For complex outputs (scoring, duplication, subdivision)

# Category matching configuration
CATEGORY_SIMILARITY_THRESHOLD = 0.66  # Minimum similarity for fuzzy matching 
MIN_CATEGORY_SIZE = 10  # Minimum queries required for a category to be kept separate

# Embedding-based duplicate detection parameters
EMB_BATCH_SIZE = 512  # 增大批处理大小，减少API调用次数
EMB_TOPK = 10  # Number of nearest neighbors to consider
TAU_STRICT = 0.88  # High similarity threshold for automatic duplicate detection  
TAU_LOW = 0.75  # Lower threshold requiring LLM verification - lowered to catch更多语义重复
LLM_VERIFY_BATCH = 60  # 增大LLM验证批处理大小

AUTO_CAT_SAMPLE_LIMIT = 1000  # Max samples for category discovery
FILTER_SELECT_CATEGORY_SAMPLE_SIZE = 3  # Sample size for category ranking


# API Settings
# 降低并发，避免 Azure 429
CONCURRENT_API_CALL_LIMIT = 4  # Used for both classification and answer collection

# Model capabilities - set to True if the model supports strict JSON schema
SUPPORTS_STRICT_JSON_SCHEMA = True  # gpt-4o and later support this

# Target Games (empty = process all)
TARGET_GAMES = [
    "心动小镇", "原神","三角洲行动"
    # "大侠立志传","泰拉瑞亚","香肠派对"
]

# Default categories when LLM discovery fails
DEFAULT_GAME_CATEGORIES = [
    "游戏玩法与攻略",
    "游戏内容与功能",
    "账号与登录问题", 
    "技术问题与bug",
    "活动与版本更新",
    "充值、兑换与商城",
    "社交与好友"
]

# ========================================
# EVAL SET SLICE & DIVERSITY CONFIG
# ========================================
# 目的：在评测集中覆盖更容易出错的查询类型，同时避免选不满或过度集中
ENABLE_SLICE_QUOTA_SELECTION = True

# 每类目标按 eval_size*ratio 取整上限，同时至少 min；允许轻微超额（见 SLICE_OVERAGE_ALLOWANCE）
SLICE_QUOTAS_DEFAULT = {
    "trap_unreleased": {"ratio": 0.0, "min": 0},   # 暂时关闭：未上线/爆料/未来版本，用户认为太简单或不相关
    "procedural": {"ratio": 0.32, "min": 4},        # 增加比重：要步骤/路径/按键，考验可执行性
    "structured": {"ratio": 0.25, "min": 3},        # 增加比重：要表格/清单/对比，考验结构化输出
    "constraint_heavy": {"ratio": 0.18, "min": 2},  # 增加比重：多条件/反向约束，考验相关性理解
}

# 允许的超额缓冲（避免频繁换人导致选不满）
SLICE_OVERAGE_ALLOWANCE = 2

# rank_category 上限：默认不限；可通过环境变量配置
RANK_CATEGORY_CAP_ENV = int(os.getenv("MAX_PER_RANK_CATEGORY_IN_EVAL", "0"))

# 可选：环境变量覆盖切片配额
_EVAL_SLICE_QUOTAS_JSON = os.getenv("EVAL_SLICE_QUOTAS_JSON", "").strip()
if _EVAL_SLICE_QUOTAS_JSON:
    try:
        _q = json.loads(_EVAL_SLICE_QUOTAS_JSON)
        if isinstance(_q, dict) and _q:
            SLICE_QUOTAS_DEFAULT = _q
            logging.info(f"[Config] Overriding SLICE_QUOTAS_DEFAULT via EVAL_SLICE_QUOTAS_JSON: {SLICE_QUOTAS_DEFAULT}")
    except Exception as _exc:  # pylint: disable=broad-except
        logging.warning(f"[Config] Failed to parse EVAL_SLICE_QUOTAS_JSON, using defaults: {_exc}")

# 诊断切片关键词/规则
TRAP_UNRELEASED_KEYWORDS = [
    "未上线", "没上线", "什么时候上线", "何时上线", "上线时间", "会不会出", "会出吗", "爆料", "内鬼", "前瞻",
    "预告", "预计", "未来", "下版本", "下个版本", "下赛季", "新赛季", "赛季", "新角色", "新武器", "新地图",
    "改动", "复刻", "卡池", "up", "UP", "测试服", "体验服"
]
PROCEDURAL_KEYWORDS = [
    "步骤", "流程", "路线", "怎么走", "怎么开", "怎么解锁", "解锁", "开启", "开通", "键位", "按键", "设置",
    "操作", "教程", "指令", "配置", "规划", "跑图", "刷", "怎么刷", "刷取", "获取路线", "材料路线", "路线图",
    "在哪", "位置", "怎么去", "怎么打", "打法"
]
STRUCTURED_KEYWORDS = [
    "表格", "一览", "汇总", "清单", "列表", "对比", "比较", "时间线", "分区", "材料表", "掉落表",
    "敌人分布", "地图标点", "全收集", "全材料", "成本", "概率", "排行", "排名", "推荐清单"
]
CONSTRAINT_KEYWORDS = [
    "同时", "并且", "但是", "另外", "还要", "并", "而且", "不要", "别", "避免", "仅", "只", "必须",
    "优先", "尽量", "最好", "预算", "限制", "条件", "不超过", "至少", "最多"
]

VERSION_PATTERN = re.compile(r"\b\d+(?:\.\d+){1,2}\b")  # e.g. 5.8 / 1.2.3
SEASON_PATTERN = re.compile(r"(?i)\bS\d+\b|赛季\d+|第\d+赛季")

def compute_slice_targets(eval_size: int, quotas: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    targets: Dict[str, int] = {}
    if eval_size <= 0:
        return targets
    for name, cfg in (quotas or {}).items():
        ratio = float(cfg.get("ratio", 0) or 0)
        min_n = int(cfg.get("min", 0) or 0)
        n = max(min_n, int(math.ceil(ratio * eval_size)))
        targets[str(name)] = max(0, min(int(n), int(eval_size)))
    return targets

def _norm_query_text(q: Any) -> str:
    return str(q).strip() if q is not None else ""

def is_trap_unreleased_query(q: Any) -> bool:
    s = _norm_query_text(q)
    if not s:
        return False
    if VERSION_PATTERN.search(s) or SEASON_PATTERN.search(s):
        return True
    return any(kw in s for kw in TRAP_UNRELEASED_KEYWORDS)

def is_procedural_query(q: Any) -> bool:
    s = _norm_query_text(q)
    if not s:
        return False
    if any(x in s for x in ["怎么", "如何", "教程", "步骤", "流程", "路线", "键位", "按键", "设置", "解锁", "开启"]):
        return True
    return any(kw in s for kw in PROCEDURAL_KEYWORDS)

def needs_structured_output_query(q: Any) -> bool:
    s = _norm_query_text(q)
    if not s:
        return False
    return any(kw in s for kw in STRUCTURED_KEYWORDS)

def is_constraint_heavy_query(q: Any) -> bool:
    s = _norm_query_text(q)
    if not s:
        return False
    markers = sum(1 for kw in CONSTRAINT_KEYWORDS if kw in s)
    sep = s.count("，") + s.count(",") + s.count("；") + s.count(";") + s.count("、")
    if markers >= 2:
        return True
    if markers >= 1 and sep >= 2:
        return True
    if ("不要" in s or "别" in s or "避免" in s) and any(x in s for x in ["推荐", "怎么", "如何", "哪个", "选择"]):
        return True
    return False

# ========================================
# PROMPT TEMPLATES
# ========================================

RELEVANCE_CHECK_SYSTEM_PROMPT_TEMPLATE = """
你是一个游戏内容相关性判断专家。你的任务是判断用户查询是否与指定的游戏紧密相关且完整。
我们的目标是构建一个【高价值、有一定深度】的游戏攻略评测集，因此过于简单或外围的问题需要被过滤。

【必须判 No 的情况】
1. **不相关/外围**：
   - 问的是不相关的游戏
   - 询问游戏下载、安装、更新、启动、账号注册、登录、实名认证、充值退款、设备兼容等
   - 纯粹的社区八卦、吐槽、表情包、晒图

2. **过于简单/缺乏深度**：
   - **简单的上线时间/版本时间询问**（如：xx什么时候上线？xx版本几号更新？国际服什么时候出？） 
   - **极简的基础设置操作**（如：怎么改名？怎么删除角色？怎么退出游戏？怎么删除蓝图？怎么开麦？画质怎么调？）
   - **是非问答/简单确认**（如：xx好玩吗？xx是免费的吗？）

3. **不完整/无效**：
   - 仅有游戏名或其截断词
   - 仅包含角色名/道具名/地名，未表达问题
   - 语法明显不完整或过于宽泛

【必须判 Yes 的必要条件（缺一不可）】
- 语义完整、问题具体
- 必须是**具备一定深度的游戏内玩法、攻略、机制、数据、剧情、搭配**等内容
- 需要模型进行一定的推理、总结或查询具体知识才能回答

跟当前游戏相关、意图完整且有评测价值的query，请回答 'Yes'。
否则，请回答 'No'。

游戏: {game_name}
输出JSON格式: {{"relevant": "Yes" 或 "No"}}
"""


DUPLICATION_CHECK_SYSTEM_PROMPT_TEMPLATE = """
你将看到一个由 'ID ::: Query文本' 组成的查询列表。
找出语义相同或高度相似的重复查询，选择更具体且表达更自然的作为标准查询，将重复的指向标准查询。

【重要原则】：
只有当两个查询询问的是**相同的信息或功能**时，才能判定为重复。
仅仅有相同的前缀或涉及相同的游戏内容，但询问不同方面的查询，不是重复！

【判定为重复的情况】：
1. 完全相同或几乎相同的查询，包含拼写错误和错别字
2. 同一问题的不同表述，只是标点、语气词不同、语序不同但意思相同
3. 核心意图相同（基础问题和添加了细节的版本）归为一组，即使有细节差异

输出格式：严格按照每行 'DUPLICATE_ID ::: CANONICAL_ID' 的格式输出。
- 只输出发现的重复项
- 每个ID只能出现在一个映射中
- 不要包含任何解释文本
"""


CATEGORY_EXCLUSION_SYSTEM_PROMPT_BASE = """
你是一位专业的游戏内容策略师。

任务：从给定类别列表中选择【需要排除】的类别。我们只想保留与【游戏内玩法、攻略、内容获取、数值数据】直接相关的类别用于评测。

以下类型的类别【必须排除】：
1. 账号/登录/客户端/安全：包括账号管理、实名认证、登录问题、下载安装、设备兼容、闪退卡顿、技术故障、封号申诉、解封教程、安全信用分等。
2. 社区/社交/评价：包括玩家评价、社区讨论、热度对比、好友社交、互动功能、晒欧气、吐槽等。
3. 运营/商务/未来：包括版本前瞻、未来计划、运营状态（跑路/关服）、退款政策、客服联系、商务合作、联动预告（非实装）等。
4. 福利/充值/外围：包括兑换码、礼包码、充值价格、商城促销、未成年保护、防沉迷等。
5. 无效/其他：名为"其他"的类别，或明显与游戏内容无关的类别。

要求：
1) 只能从提供的候选列表中选择，名称必须与候选列表完全一致；
2) 若没有需要排除的，返回空数组；
"""

CATEGORY_EXCLUSION_USER_PROMPT_TEMPLATE = """
游戏：{game_name}
候选类别（逐项）：
{categories_list}

请基于上面的规则，仅从候选项中选出【需要排除】的类别。
"""

DUPLICATE_CANONICAL_SELECTION_PROMPT = """
你是一个query质量评估专家。以下是一组语义相同或高度相似的重复查询。
请从中选择一个最适合作为代表的查询（canonical query）。

选择标准：
1. 表述最自然、最符合真实用户的搜索习惯
2. 意图清晰明确，不产生歧义  
3. 不过于冗长或过于简短
4. 拼写正确，没有明显的错别字或语法错误

查询列表（格式: ID ::: 查询文本）：
{queries_list}

【重要】请只输出被选中查询的ID，不要输出冒号、查询文本或任何解释。
示例输出: Q001 或 123456789

你的选择："""

CLASSIFIER_SYSTEM_PROMPT_TEMPLATE = (
"你是一位专业的游戏社区分析师。"
"你的任务是阅读关于手机游戏《{game_name}》（或相关游戏）的用户搜索查询，"
"并判断它们具体属于以下 {num_categories} 个分类中的哪一个。\n\n"
"请仅从给定的分类列表中精确选择一个分类。\n\n"
"{categories_text}"
)

CLASSIFIER_USER_TEMPLATE = (
"查询: {query}\n"
"分类 (请从上面给出的分类列表中精确选择一个):"
)

CLASSIFIER_BATCH_USER_TEMPLATE = """
待分类查询列表（ID ::: 查询内容）：
{queries_text}

请对以上每条查询进行分类，返回JSON格式。
包含一个 "classifications" 列表，每项包含 "id" 和 "category"。
【必须确保】每一个输入的ID都有对应的分类结果。
"""

REWRITE_SYSTEM_PROMPT = """
你是一个搜索查询规范化助手。请将输入的口语化/冗余/不完整的查询，改写为简洁、可执行的搜索问题：
- 保留核心意图、关键实体、条件
- 去掉赘述、口头语、感叹、重复
- 如果已是清晰问题，尽量保持原意和长度
- 输出一条简洁的中文问句，不要解释
"""

REWRITE_USER_TEMPLATE = "原始查询：{query}\n请输出规范化后的搜索问句："

SCORE_SYSTEM_PROMPT = """
你是专业的游戏query质量评估专家。请对每个Query进行严格评分，并判断其所属的类型标签。

【评分维度】每项0-100分：

1. 具体性 (specificity)：Query描述的精确程度
   - 90-100：极其具体，指向唯一明确的游戏要素
   - 70-89：比较具体，有明确目标但可能有多个答案
   - 50-69：一般具体，目标较宽泛
   - 30-49：模糊不清
   - 0-29：极其模糊，无法理解意图

2. 信息完整性 (completeness)：Query包含的上下文信息
   - 90-100：包含所有必要信息，无需追问
   - 70-89：信息较完整，略有缺失
   - 50-69：缺少关键信息
   - 30-49：信息严重不足, 仅仅包含“游戏名+实体名
   - 0-29：几乎无信息

3. 游戏深度与价值 (depth_and_value)：Query的知识含量与评测价值
   - 90-100：高价值/高深度。涉及多步推理、数据计算、策略分析、复杂机制详解、数值收益对比。
   - 70-89：中高价值。涉及具体知识点查询（如材料分布、任务流程），需要游戏内知识才能回答。
   - 50-69：中等价值。单一事实性问题，比较直观。
   - 30-49：低价值。说明书式问题、简单的上线时间、极简UI操作（如改名/退出）、是非题（好玩吗）。
   - 0-29：无价值。无意义闲聊或极度简单的操作。

【标签判断】请基于语义判断Query是否符合以下特征（True/False）：

- is_procedural (步骤/教程): 用户询问具体的执行步骤、操作流程、路线规划、解谜方法、按键配置等。
  - Yes: "怎么解锁隐藏地图", "深渊第10层打法流程", "改键位教程"
  - No: "好玩吗", "这是什么"

- needs_structured_output (结构化/表格): 用户明确需要或Query最适合以表格、清单、列表、对比图的形式回答。
  - Yes: "全角色强度排行", "五星武器属性对比表", "突破材料清单汇总"
  - No: "怎么获得安柏"

- has_heavy_constraints (复杂约束): Query包含多个限制条件、否定条件("不要")、排序要求("优先")或复合目标。
  - Yes: "推荐个火系主C，不要五星，适合平民的", "同时满足高爆发和生存能力的配队"
  - No: "推荐个火系角色"

- is_trap_unreleased (未上线/时效陷阱): 询问未上线内容、版本更新时间、内鬼爆料、未来预测等。
  - Yes: "5.0版本什么时候更新", "下个卡池是谁", "复刻时间表"

【重要规则】
- 评分要拉开差距，对于"depth_and_value"维度，凡是简单的“什么时候上线”、“怎么删除蓝图”、“怎么改名”等问题，一律给50分以下。
- 高分段(90+)必须留给真正有深度的攻略类、分析类问题。

返回JSON: {"scores":[{"id":"Q001","specificity":...,"completeness":...,"depth_and_value":...,"is_procedural":true/false, ...}, ...]}
"""

# Category discovery prompt template
CATEGORY_DISCOVERY_SYSTEM_PROMPT_TEMPLATE = """
你将看到一批关于手游《{game_name}》的玩家搜索 Query，共 {sample_count} 条。
请通读后，总结出这批查询背后代表的核心玩家意图分类。

【分类原则】
1. **覆盖度优先**：分类体系应覆盖 95% 以上的查询，尽量减少"其他"类的占比。
2. **粒度适中**：建议生成 8-12 个一级分类，避免分类过粗（如"游戏攻略"太宽泛）或过细。
3. **互斥性**：类别之间应有明确界限，避免重叠。
4. **强制区分**：以下几类属于完全不同的用户意图，必须拆分为独立的一级分类（如果数据中存在）：
   - 游戏内容/攻略（角色、装备、材料、任务）
   - 游戏配置分享（改枪码、灵敏度代码、捏脸码、键位设置）
   - 福利码获取（兑换码、礼包码，仅限官方奖励兑换）
   - 账号与安全（登录、绑定、找回、封号、解封、实名）

请根据 Query 的实际分布生成最合适的分类列表，不要局限于上述示例。
每行输出一个分类名称，不要编号：
"""


# Category subdivision prompt template
CATEGORY_SUBDIVISION_SYSTEM_PROMPT_TEMPLATE = """
你是一个游戏query分类专家。现在有一个类别 "{category_name}" 包含太多查询（{query_count} 个）。
请将这些查询进一步细分为【至少两个】更具体的子类别。每个子类别应该：
1. 包含相似或相关的查询
2. 数量不超过 {max_size} 个查询
3. 【严格要求】每个子类别必须至少包含 {min_size} 个查询
4. 子类别名称应该简洁、通用且描述性
5. 【严格要求】子类别之间应该相互排斥，语义上没有重叠。例如，如果有了"任务攻略"，就不应该再有包含任务攻略的"综合攻略"。

子类别命名原则：
- 使用通用的类别名称，避免过于具体
- 不要在子类别名称中重复游戏名称
- 确保子类别在语义上相互排斥

输出格式：严格按照每行 'QUERY_ID ::: 子类别名称' 的格式输出。
"""

# ========================================
# CORE FUNCTIONS
# ========================================

async def _generate_embeddings(
    client: openai.AsyncOpenAI,
    texts: List[str],
    *,
    model_name: str,
    batch_size: int = EMB_BATCH_SIZE,
    noise_scale: float = 0.001,
    context: str = ""
) -> np.ndarray:
    """Shared embedding helper with consistent retry/noise behaviour"""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vecs: List[Optional[List[float]]] = []
    embedding_dim: Optional[int] = None
    failed_indices: List[int] = []

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]

        try:
            resp = await client.embeddings.create(model=model_name, input=batch)
            for data_item in resp.data:
                if isinstance(data_item, dict):
                    embedding = data_item["embedding"]
                else:
                    embedding = data_item.embedding

                if embedding_dim is None:
                    embedding_dim = len(embedding)
                    logging.info(
                        f"[Embedding]{f' {context}' if context else ''} detected dimension: {embedding_dim}"
                    )

                vecs.append(embedding)

        except Exception as exc:  # pylint: disable=broad-except
            logging.error(
                f"[Embedding]{f' {context}' if context else ''} batch {batch_start}-{batch_start + len(batch)} failed: {exc}"
            )

            if embedding_dim is None:
                logging.warning(
                    f"[Embedding]{f' {context}' if context else ''} embedding dim unknown, delaying noise fallback"
                )
                vecs.extend([None] * len(batch))
                failed_indices.extend(range(len(vecs) - len(batch), len(vecs)))
            else:
                for _ in batch:
                    random_vec = np.random.uniform(-noise_scale, noise_scale, embedding_dim).tolist()
                    vecs.append(random_vec)
                failed_indices.extend(range(len(vecs) - len(batch), len(vecs)))

    if embedding_dim is None:
        raise RuntimeError(
            f"[Embedding]{f' {context}' if context else ''} all embedding requests failed"
        )

    for idx in range(len(vecs)):
        if vecs[idx] is None:
            vecs[idx] = np.random.uniform(-noise_scale, noise_scale, embedding_dim).tolist()

    if failed_indices:
        logging.warning(
            f"[Embedding]{f' {context}' if context else ''} replaced {len(failed_indices)} embeddings with random noise"
        )

    return np.asarray(vecs, dtype=np.float32)


class QueryProcessor:
    """Handles query processing operations"""
    
    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client
        self.semaphore = asyncio.Semaphore(CONCURRENT_API_CALL_LIMIT)

    async def rewrite_queries(self, df: pd.DataFrame, game_name: str) -> pd.DataFrame:
        """Rewrite queries to concise, search-intent-friendly form."""
        if df.empty or COL_QUERY not in df.columns:
            return df

        # 保留原始文本
        if 'raw_query_original' not in df.columns:
            df['raw_query_original'] = df[COL_QUERY]

        mask = df['query_status'] == QUERY_STATUS_RELEVANT
        if not mask.any():
            return df

        rows = df[mask].to_dict('records')
        semaphore = asyncio.Semaphore(CONCURRENT_API_CALL_LIMIT)

        async def _rewrite_single(row: Dict) -> str:
            async with semaphore:
                q = str(row.get(COL_QUERY, "")).strip()
                if not q:
                    return q
                try:
                    completion = await self.client.chat.completions.create(
                        model=MODEL_CONFIG_INTERNAL["discovery"],
                        messages=[
                            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                            {"role": "user", "content": REWRITE_USER_TEMPLATE.format(query=q)}
                        ],
                        temperature=TEMPERATURE_STRICT,
                        max_tokens=MAX_TOKENS_SHORT
                    )
                    return completion.choices[0].message.content.strip()
                except Exception as e:
                    logging.warning(f"Rewrite failed: {e}")
                    return q

        rewrites = await asyncio.gather(*[_rewrite_single(r) for r in rows])
        df.loc[mask, COL_QUERY] = rewrites
        return df

    async def check_relevance(self, df: pd.DataFrame, game_name: str) -> pd.DataFrame:
        """Check query relevance and mark irrelevant queries"""
        # 提前过滤：先做本地快速检查，减少API调用
        queries = df[COL_QUERY].dropna().astype(str).str.strip()
        
        # 本地快速预过滤
        def is_obviously_invalid(query: str) -> bool:
            """快速本地检查，不需要API调用"""
            if len(query) <= 5:
                return True
            # 检查可读字符比例
            readable_chars = sum(1 for c in query if c.isalnum() or '\u4e00' <= c <= '\u9fff')
            if readable_chars / len(query) < 0.3:
                return True
            return False
        
        # 标记明显无效的查询
        obviously_invalid = queries.apply(is_obviously_invalid)
        df.loc[obviously_invalid[obviously_invalid].index, 'query_status'] = QUERY_STATUS_IRRELEVANT
        
        # 只对可能有效的查询调用API
        valid_queries = queries[~obviously_invalid]
        if len(valid_queries) > 0:
            tasks = [
                self._check_single_relevance(query, game_name) 
                for query in valid_queries
            ]
            
            results = await tqdm_async.gather(*tasks, desc="Checking relevance")
            is_relevant = pd.Series(results, index=valid_queries.index)
            
            irrelevant_indices = is_relevant[~is_relevant].index
            df.loc[irrelevant_indices, 'query_status'] = QUERY_STATUS_IRRELEVANT
        
        logging.info(f"[Relevance] Filtered {obviously_invalid.sum()} obviously invalid, "
                    f"checked {len(valid_queries)} via API")
        return df
    
    async def _check_single_relevance(self, query: str, game_name: str) -> bool:
        """Check if a single query is relevant"""
        async with self.semaphore:
            if not query:
                return False
            
            try:
                system_prompt = RELEVANCE_CHECK_SYSTEM_PROMPT_TEMPLATE.format(game_name=game_name)
                user_prompt = f"Query: {query}"
                
                # 使用response_format强制输出Yes/No
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "relevance_check",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "relevant": {
                                    "type": "string",
                                    "enum": ["Yes", "No"]
                                }
                            },
                            "required": ["relevant"],
                            "additionalProperties": False
                        }
                    }
                }
                
                resp = await self.client.chat.completions.create(
                    model=MODEL_CONFIG_INTERNAL["discovery"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=response_format,
                    temperature=TEMPERATURE_STRICT,
                    max_tokens=MAX_TOKENS_TINY
                )
                
                # 解析JSON响应
                try:
                    result = json.loads(resp.choices[0].message.content)
                    return result.get("relevant", "No") == "Yes"
                except (json.JSONDecodeError, KeyError):
                    # 如果解析失败，回退到原始判断逻辑
                    content = resp.choices[0].message.content.strip().lower()
                    # 更严格的判断：必须明确包含yes且不包含no
                    return 'yes' in content and 'no' not in content
            except Exception as e:
                logging.warning(f"Relevance check failed: {e}")
                return True  # Default to relevant (conservative approach)
    
    async def detect_duplicates(self, df: pd.DataFrame, game_name: str) -> pd.DataFrame:
        """基于向量相似度的智能去重
        
        Algorithm:
        1. 使用embedding向量化所有relevant queries
        2. 通过ANN找到相似query pairs
        3. 高相似度直接判定为重复，中等相似度用LLM复核
        4. 使用并查集找连通分量，选择canonical query
        """
        
        relevant_mask = df['query_status'] == QUERY_STATUS_RELEVANT
        if not relevant_mask.any():
            return df
        
        relevant_df = df[relevant_mask]  
        # 缓存字符串转换结果，避免重复转换
        relevant_df_str_id = relevant_df[COL_QUERY_ID].astype(str)
        ids = relevant_df_str_id.tolist()
        texts = relevant_df[COL_QUERY].astype(str).tolist()
        original_count = len(ids)
        
        logging.info(f"[Duplicate Detection] Starting for {original_count} queries...")
        
        # 1. 向量化
        X = await self._embed_queries(texts)
        
        # 2. ANN检索找相似pairs
        nn = self._build_ann_index(X)
        strict_edges, border_edges = self._find_candidate_edges(X, nn, ids)
        
        logging.info(f"[Duplicate Detection] Found {len(strict_edges)} high-similarity pairs (>{TAU_STRICT}), "
                    f"{len(border_edges)} borderline pairs ({TAU_LOW}-{TAU_STRICT})")
        
        # 3. 对边界相似度的pairs用LLM复核
        confirmed_edges = []
        if border_edges:
            confirmed_edges = await self._verify_border_duplicates(border_edges, relevant_df)
            logging.info(f"[Duplicate Detection] LLM confirmed {len(confirmed_edges)}/{len(border_edges)} borderline pairs as duplicates")
        
        # 4. 组图找连通分量
        all_edges = strict_edges + confirmed_edges
        components = self._find_connected_components(all_edges, ids)
        
        # 5. 每个分量选canonical，标记其他为DUPLICATE
        duplicate_map = {}
        for comp in components:
            if len(comp) <= 1:
                continue
            canonical = await self._pick_canonical_query(comp, relevant_df)
            for qid in comp:
                if qid != canonical:
                    duplicate_map[qid] = canonical
        
        # 6. 应用duplicate标记到原DataFrame
        if duplicate_map:
            # 让所有组长也设为canonical_指向自己，方便下游join
            leaders = set(duplicate_map.values())
            leader_self_map = {lid: lid for lid in leaders}
            full_map = {**leader_self_map, **duplicate_map}
            
            dup_df = pd.DataFrame(list(full_map.items()), 
                                 columns=['dup_id', 'canonical_id'])
            # 创建临时列进行merge，确保索引一致性
            df['query_id_str'] = df[COL_QUERY_ID].astype(str)
            df = df.merge(dup_df, left_on='query_id_str', right_on='dup_id', how='left')
            
            # 只有当query_id != canonical_id时才标记为重复
            df.loc[df['dup_id'].notna() & (df['query_id_str'] != df['canonical_id']), 'query_status'] = QUERY_STATUS_DUPLICATE
            df[COL_CANONICAL_QUERY_ID] = df[COL_CANONICAL_QUERY_ID].fillna(df['canonical_id'])
            df = df.drop(columns=['query_id_str', 'dup_id', 'canonical_id'])
        
        # 统计和报告
        total_duplicates = len(duplicate_map)
        unique_groups = len([c for c in components if len(c) > 1])
        logging.info(f"[Duplicate Detection] Complete: {total_duplicates} duplicates in {unique_groups} groups")
        
        return df
    
    async def _embed_queries(self, texts: List[str]) -> np.ndarray:
        return await _generate_embeddings(
            self.client,
            texts,
            model_name=MODEL_CONFIG_INTERNAL["embedding"],
            batch_size=EMB_BATCH_SIZE,
            noise_scale=0.001,
            context="duplicate"
        )
    
    def _build_ann_index(self, X: np.ndarray):
        """Build an ANN index for efficient similarity search."""
        nn = NearestNeighbors(n_neighbors=min(EMB_TOPK + 1, len(X)), metric="cosine")
        nn.fit(X)
        return nn
    
    def _find_candidate_edges(self, X: np.ndarray, nn, ids: List[str]) -> Tuple[List[Tuple], List[Tuple]]:
        """Find candidate duplicate pairs based on embedding similarity."""
        # Get nearest neighbors
        dists, idxs = nn.kneighbors(X, n_neighbors=min(EMB_TOPK + 1, len(X)), return_distance=True)
        
        edges_strict = []
        edges_border = []
        seen_pairs = set()  # Avoid duplicate edges
        
        for i, (dist_row, idx_row) in enumerate(zip(dists, idxs)):
            for dist, j in zip(dist_row[1:], idx_row[1:]):  # Skip self
                if i >= j:  # Avoid duplicate pairs (i,j) and (j,i)
                    continue
                    
                sim = 1.0 - float(dist)
                a, b = ids[i], ids[j]
                pair = tuple(sorted([a, b]))
                
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    if sim >= TAU_STRICT:
                        edges_strict.append((a, b, sim))
                    elif sim >= TAU_LOW:
                        edges_border.append((a, b, sim))
        
        return edges_strict, edges_border
    
    async def _verify_border_duplicates(self, border_edges: List[Tuple], relevant_df: pd.DataFrame) -> List[Tuple]:
        """Use LLM to verify borderline duplicate candidates."""
        confirmed = []
        
        # 优化：使用向量化操作构建字典，避免iterrows
        id_to_query = dict(zip(
            relevant_df[COL_QUERY_ID].astype(str),
            relevant_df[COL_QUERY].astype(str)
        ))
        
        # Process in batches
        for i in range(0, len(border_edges), LLM_VERIFY_BATCH):
            batch_edges = border_edges[i:i + LLM_VERIFY_BATCH]
            
            # Build prompt with unique queries from this batch
            unique_ids = set()
            for a, b, _ in batch_edges:
                unique_ids.add(a)
                unique_ids.add(b)
            
            prompt_lines = [f"{qid} ::: {id_to_query[qid]}" for qid in unique_ids]
            user_prompt = "\n".join(prompt_lines)
            
            try:
                response = await llm_chat_call(
                    self.client,
                    DUPLICATION_CHECK_SYSTEM_PROMPT_TEMPLATE,
                    user_prompt,
                    MODEL_CONFIG_INTERNAL["discovery"],
                    temperature=TEMPERATURE_STRICT,
                    max_tokens=MAX_TOKENS_EXTENDED
                )
                
                # Parse LLM response
                verified_pairs = set()
                for line in response.splitlines():
                    if ":::" in line:
                        parts = line.split(":::", 1)
                        if len(parts) == 2:
                            dup_id = parts[0].strip()
                            canonical_id = parts[1].strip()
                            pair = tuple(sorted([dup_id, canonical_id]))
                            verified_pairs.add(pair)
                
                # Check which edges were confirmed
                for a, b, sim in batch_edges:
                    pair = tuple(sorted([a, b]))
                    if pair in verified_pairs:
                        confirmed.append((a, b, sim))
                        
            except Exception as e:
                logging.error(f"LLM verification failed: {e}")
        
        return confirmed
    
    def _find_connected_components(self, edges: List[Tuple], ids: List[str]) -> List[List[str]]:
        """Find connected components using Union-Find algorithm."""
        parent = {x: x for x in ids}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        
        # Union all edges
        for a, b, _ in edges:
            union(a, b)
        
        # Group by root
        components = defaultdict(list)
        for x in ids:
            components[find(x)].append(x)
        
        return list(components.values())
    
    async def _pick_canonical_query(self, comp_ids: List[str], relevant_df: pd.DataFrame) -> str:
        """Select the best query from a duplicate group to be the canonical one using LLM."""
        # If only one query in the group, return it
        if len(comp_ids) == 1:
            return comp_ids[0]
        
        # 优化：一次性获取所有需要的数据，避免重复查询
        relevant_df_with_str_id = relevant_df.assign(query_id_str=relevant_df[COL_QUERY_ID].astype(str))
        comp_df = relevant_df_with_str_id[relevant_df_with_str_id['query_id_str'].isin(comp_ids)]
        id_to_query = dict(zip(comp_df['query_id_str'], comp_df[COL_QUERY].astype(str)))
        
        # For larger groups or non-trivial cases, use LLM to select the best one
        logging.debug(f"Using LLM to select canonical query from {len(comp_ids)} duplicates")
        
        try:
            # Build query list for LLM - 使用已经构建的id_to_query字典
            queries_lines = [f"{qid} ::: {id_to_query[qid]}" for qid in comp_ids]
            
            queries_list = "\n".join(queries_lines)
            user_prompt = DUPLICATE_CANONICAL_SELECTION_PROMPT.format(queries_list=queries_list)
            
            # Call LLM to select the best canonical query
            response = await llm_chat_call(
                self.client,
                "",  # No system prompt needed, all instructions are in user prompt
                user_prompt,
                MODEL_CONFIG_INTERNAL["discovery"],
                temperature=TEMPERATURE_STRICT,
                max_tokens=MAX_TOKENS_SHORT
            )
            
            # Parse the response to get the selected ID
            selected_id = response.strip()
            
            # Clean up common formatting issues
            # Remove any trailing punctuation or explanation
            selected_id = selected_id.split()[0] if selected_id else ""
            selected_id = selected_id.rstrip(".,;:，。；：")
            
            # Validate that the selected ID is in our list
            if selected_id in comp_ids:
                return selected_id
            
            # Try to find a partial match (in case of formatting issues)
            for qid in comp_ids:
                if qid in selected_id or selected_id in qid:
                    return qid
            
            # If still no match, log warning and use fallback
            logging.warning(f"LLM selected invalid ID '{response.strip()}', falling back to length heuristic")
            # Fallback: pick the longest query (usually more complete)
            return max(comp_ids, key=lambda qid: len(id_to_query[qid]))
                
        except Exception as e:
            logging.warning(f"Failed to select canonical query via LLM: {e}, using length heuristic")
            # Fallback: pick the longest query
            return max(comp_ids, key=lambda qid: len(id_to_query[qid]))

class CategoryManager:
    """Manages category discovery and classification"""
    
    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client
    
    async def discover_categories(self, df: pd.DataFrame, game_name: str) -> List[str]:
        """Discover categories from queries"""
        relevant_df = df[df['query_status'] == QUERY_STATUS_RELEVANT]
        if relevant_df.empty:
            return DEFAULT_GAME_CATEGORIES + [CATEGORY_OTHER]
        
        # 使用随机采样而不是head()，避免顺序偏置
        sample_size = min(AUTO_CAT_SAMPLE_LIMIT, len(relevant_df))
        sample = relevant_df.sample(n=sample_size, random_state=42)[COL_QUERY].tolist()
        sys_prompt = CATEGORY_DISCOVERY_SYSTEM_PROMPT_TEMPLATE.format(
            game_name=game_name,
            sample_count=len(sample)
        )
        
        try:
            response = await llm_chat_call(
                self.client, sys_prompt, "\n".join(sample),
                MODEL_CONFIG_INTERNAL["discovery"],
                temperature=TEMPERATURE_STRICT,
                max_tokens=MAX_TOKENS_CATEGORY
            )
            
            discovered_categories = [c.strip() for c in response.splitlines() if c.strip()]
            
            # Clean category names using the same logic as _clean_category for consistency
            cleaned_categories = []
            for cat in discovered_categories:
                cleaned_cat = self._clean_category(cat)
                if cleaned_cat:
                    cleaned_categories.append(cleaned_cat)
            
            final_categories = cleaned_categories
            
            if not final_categories:
                final_categories = DEFAULT_GAME_CATEGORIES
            
            # Always add a fallback category
            if CATEGORY_OTHER not in final_categories:
                final_categories.append(CATEGORY_OTHER)
                
            # If we only have CATEGORY_OTHER, add some default categories for games
            if len(final_categories) == 1 and final_categories[0] == CATEGORY_OTHER:
                final_categories = DEFAULT_GAME_CATEGORIES + [CATEGORY_OTHER]
            
            return final_categories
            
        except Exception as e:
            logging.error(f"Category discovery failed: {e}")
            return DEFAULT_GAME_CATEGORIES + [CATEGORY_OTHER]
    
    def _clean_category(self, raw: str) -> str:
        """Clean category name from LLM output - exactly like integrated_pipeline.py"""
        cleaned_cat = re.sub(r"^\d+\.\s*", "", raw).strip()
        cleaned_cat = re.sub(r"(?i)^(分类|category)[:：]?\s*", "", cleaned_cat).strip()
        return cleaned_cat
    
    def _find_best_matching_category(self, llm_category: str, categories: List[str]) -> Optional[str]:
        """Find the best matching category - exact match first, then simple fuzzy fallback"""
        if not llm_category or not categories:
            return None
        
        # First try exact match
        if llm_category in categories:
            return llm_category
        
        # Then try case-insensitive exact match
        for cat in categories:
            if llm_category.lower() == cat.lower():
                return cat
        
        # Simple string similarity fallback
        best_match = None
        best_ratio = 0
        threshold = CATEGORY_SIMILARITY_THRESHOLD
        
        for cat in categories:
            ratio = SequenceMatcher(None, llm_category.lower(), cat.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = cat
        
        # If fuzzy matching failed, try substring matching as last resort
        if not best_match:
            for cat in categories:
                if cat.lower() in llm_category.lower() or llm_category.lower() in cat.lower():
                    best_match = cat
                    break
        
        # Fuzzy matching completed
        
        return best_match
    
    async def classify_queries(self, df: pd.DataFrame, categories: List[str], game_name: str) -> pd.DataFrame:
        """Classify queries into categories using batch processing with fallback"""
        if df.empty:
            return df
        
        categories_text_for_prompt = "\n".join(f"- {cat}" for cat in categories)
        system_prompt = CLASSIFIER_SYSTEM_PROMPT_TEMPLATE.format(
            game_name=game_name,
            num_categories=len(categories),
            categories_text=categories_text_for_prompt
        )
        
        # 将行转为字典列表
        rows_as_dicts = df.to_dict('records')
        total_rows = len(rows_as_dicts)
        
        # 分批处理
        batches = []
        for i in range(0, total_rows, CLASSIFICATION_BATCH_SIZE):
            batches.append(rows_as_dicts[i:i + CLASSIFICATION_BATCH_SIZE])
            
        semaphore = asyncio.Semaphore(CONCURRENT_API_CALL_LIMIT)
        
        async def process_batch_with_fallback(batch_rows):
            async with semaphore:
                # 尝试Batch处理
                batch_results = await self._classify_batch(batch_rows, system_prompt, categories)
                
                # 检查结果完整性
                if batch_results and len(batch_results) == len(batch_rows):
                    # 验证ID是否匹配
                    batch_ids = set(str(r[COL_QUERY_ID]) for r in batch_rows)
                    result_ids = set(str(r[COL_QUERY_ID]) for r in batch_results)
                    if batch_ids == result_ids:
                        return batch_results
                
                # 如果Batch失败或结果不完整，降级为单条串行处理（在semaphore内，避免并发爆炸）
                # 这里不使用额外的semaphore，因为已经在外层semaphore限制内
                logging.warning(f"[Classification] Batch failed or incomplete for {len(batch_rows)} items, falling back to single processing")
                fallback_results = []
                for row in batch_rows:
                    # 复用 _classify_single 的逻辑，但需要调整参数传递方式
                    # 注意：_classify_single 原本设计为接收 semaphore，这里我们已经在 semaphore 内
                    # 所以我们可以直接调用核心逻辑，或者创建一个假的 semaphore
                    # 为了简化，我们在 _classify_single 中处理 semaphore 为 None 的情况，或者直接在这里实现简化版
                    
                    try:
                        single_res = await self._classify_single_no_lock(row, system_prompt, categories)
                        fallback_results.append(single_res)
                    except Exception as e:
                        logging.error(f"Single classification fallback failed: {e}")
                        row_copy = row.copy()
                        row_copy[COL_CATEGORY] = CATEGORY_CLASSIFICATION_FAILED
                        fallback_results.append(row_copy)
                
                return fallback_results

        # 并发执行所有批次
        tasks = [process_batch_with_fallback(batch) for batch in batches]
        batch_results_list = await tqdm_async.gather(*tasks, desc="Classifying queries")
        
        # 展平结果
        flat_results = []
        for batch_res in batch_results_list:
            if isinstance(batch_res, list):
                flat_results.extend(batch_res)
            else:
                # 理论上不应该走到这里，除非 process_batch_with_fallback 抛出未捕获异常
                logging.error(f"[Classification] Unexpected batch result type: {type(batch_res)}")

        # 创建分类结果DataFrame
        # 重新构建DataFrame以确保顺序和完整性
        # 使用字典映射更新原DataFrame，比直接构造列表更安全
        result_map = {}
        for res in flat_results:
            qid = str(res.get(COL_QUERY_ID))
            cat = res.get(COL_CATEGORY)
            result_map[qid] = cat
            
        df_classified = df.copy()
        df_classified[COL_CATEGORY] = df_classified[COL_QUERY_ID].astype(str).map(result_map)
        
        # 标记未分类的
        failed_count = df_classified[COL_CATEGORY].isna().sum()
        if failed_count > 0:
            logging.warning(f"[Classification] {failed_count} queries failed to classify")
            df_classified[COL_CATEGORY] = df_classified[COL_CATEGORY].fillna(CATEGORY_CLASSIFICATION_FAILED)
            
        return df_classified

    async def _classify_batch(self, batch_rows: List[Dict], system_prompt: str, categories: List[str]) -> Optional[List[Dict]]:
        """Classify a batch of queries"""
        try:
            # 构建 Batch Prompt
            query_lines = []
            id_map = {} # str_id -> original_row
            
            for row in batch_rows:
                qid = str(row[COL_QUERY_ID])
                qtext = str(row.get(COL_QUERY, "")).strip()
                # 截断过长的query以防token溢出
                if len(qtext) > 200:
                    qtext = qtext[:200] + "..."
                query_lines.append(f"{qid} ::: {qtext}")
                id_map[qid] = row
            
            queries_text = "\n".join(query_lines)
            user_prompt = CLASSIFIER_BATCH_USER_TEMPLATE.format(queries_text=queries_text)
            
            # 使用 JSON Schema (如果支持) 或 JSON Object
            response_format = {"type": "json_object"}
            if SUPPORTS_STRICT_JSON_SCHEMA:
                # 动态构建 enum 以启用 Strict Mode
                batch_ids = list(id_map.keys())
                schema = {
                    "type": "object",
                    "properties": {
                        "classifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "enum": batch_ids},
                                    "category": {"type": "string"}
                                },
                                "required": ["id", "category"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["classifications"],
                    "additionalProperties": False
                }
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification_batch",
                        "strict": True,
                        "schema": schema
                    }
                }

            completion = await self.client.chat.completions.create(
                model=MODEL_CONFIG_INTERNAL["classification"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE_STRICT,
                max_tokens=MAX_TOKENS_EXTENDED, # 需要更大的 output token 空间
                response_format=response_format
            )
            
            content = completion.choices[0].message.content
            data = json.loads(content)
            items = data.get("classifications", [])
            
            results = []
            processed_ids = set()
            
            for item in items:
                qid = item.get("id")
                raw_cat = item.get("category")
                
                if qid in id_map:
                    processed_ids.add(qid)
                    cleaned_cat = self._clean_category(raw_cat)
                    matched_category = self._find_best_matching_category(cleaned_cat, categories)
                    
                    row_res = id_map[qid].copy()
                    if matched_category:
                        row_res[COL_CATEGORY] = matched_category
                    else:
                        row_res[COL_CATEGORY] = CATEGORY_CLASSIFICATION_FAILED
                    results.append(row_res)
            
            return results
            
        except Exception as e:
            logging.warning(f"Batch classification failed: {e}")
            return None

    async def _classify_single_no_lock(self, row_data: Dict, system_prompt: str, categories: List[str]) -> Dict:
        """Classify a single query without semaphore (caller handles concurrency)"""
        query = str(row_data.get(COL_QUERY, "")).strip()
        output = row_data.copy()
        
        if not query:
            output[COL_CATEGORY] = CATEGORY_CLASSIFICATION_FAILED
            return output
        
        try:
            user_prompt = CLASSIFIER_USER_TEMPLATE.format(query=query)
            
            completion = await self.client.chat.completions.create(
                model=MODEL_CONFIG_INTERNAL["classification"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE_STRICT,
                max_tokens=MAX_TOKENS_SHORT
            )
            
            raw_cat = completion.choices[0].message.content.strip()
            cleaned_cat = self._clean_category(raw_cat)
            
            matched_category = self._find_best_matching_category(cleaned_cat, categories)
            
            if matched_category:
                output[COL_CATEGORY] = matched_category
            else:
                output[COL_CATEGORY] = CATEGORY_CLASSIFICATION_FAILED
                logging.warning(f"Could not match category '{cleaned_cat}' for query: {query}")
        except Exception as e:
            logging.warning(f"Classification failed: {e}")
            output[COL_CATEGORY] = CATEGORY_CLASSIFICATION_FAILED
        
        return output

    async def _classify_single(self, semaphore, row_data: Dict, system_prompt: str, categories: List[str]) -> Dict:
        """Classify a single query (Legacy wrapper)"""
        async with semaphore:
            return await self._classify_single_no_lock(row_data, system_prompt, categories)
    
    async def get_exclusion_suggestions(self, categories: List[str], game_name: str) -> List[str]:
        """Get LLM suggestions for categories to exclude using JSON Schema with fallback"""
        # Clean and deduplicate categories while preserving order
        cleaned_categories = [self._clean_category(cat) for cat in categories if cat]
        seen = set()
        ordered_categories = []
        for cat in cleaned_categories:
            if cat not in seen:
                seen.add(cat)
                ordered_categories.append(cat)

        if not ordered_categories:
            logging.info("Empty category list for exclusion suggestions")
            return []

        # Prepare prompts using templates
        categories_list = "\n".join(f"- {cat}" for cat in ordered_categories)
        user_prompt = CATEGORY_EXCLUSION_USER_PROMPT_TEMPLATE.format(
            game_name=game_name,
            categories_list=categories_list
        )

        # Build JSON Schema for strict output validation
        schema = {
            "type": "object",
            "properties": {
                "exclude": {
                    "type": "array",
                    "items": {"type": "string", "enum": ordered_categories},
                    "uniqueItems": True
                }
            },
            "required": ["exclude"],
            "additionalProperties": False
        }

        try:
            # Primary approach: JSON Schema mode for deterministic output
            response = await llm_chat_call(
                self.client,
                CATEGORY_EXCLUSION_SYSTEM_PROMPT_BASE,
                user_prompt,
                MODEL_CONFIG_INTERNAL["classification"],
                temperature=TEMPERATURE_STRICT,
                max_tokens=MAX_TOKENS_MEDIUM,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "exclude_categories", "schema": schema}
                }
            )
            
            data = json.loads(response)
            excluded_categories = data.get("exclude", [])
            
            # Double-check that all suggestions are valid categories
            return [cat for cat in excluded_categories if cat in ordered_categories]

        except Exception as e:
            logging.warning(f"Category exclusion JSON schema mode failed: {e}")

            # Fallback approach: Plain text output with flexible parsing
            try:
                # Build fallback system prompt by extending base prompt with output instruction
                fallback_system_prompt = CATEGORY_EXCLUSION_SYSTEM_PROMPT_BASE + "\n请逐行输出需要排除的类别名称（只输出类别名称本身）。"
                fallback_response = await llm_chat_call(
                    self.client,
                    fallback_system_prompt,
                    user_prompt,
                    MODEL_CONFIG_INTERNAL["classification"],
                    temperature=TEMPERATURE_STRICT,
                    max_tokens=MAX_TOKENS_LONG
                )
                
                # Parse response with multiple delimiters and clean format
                parts = re.split(r"[\n,，、;；|]+", fallback_response)
                cleaned_suggestions = [part.strip(" -•\t").strip() for part in parts if part.strip()]
                
                # Return only valid categories from the original list
                return [cat for cat in cleaned_suggestions if cat in ordered_categories]
                
            except Exception as e2:
                logging.warning(f"Category exclusion fallback mode failed: {e2}")
                return []

class SelectionManager:
    """Manages query selection and ranking"""
    
    def __init__(self, client: openai.AsyncOpenAI):
        self.client = client
        self.category_subdivision_map = {}  # Store category->subcategory mapping
        self.rankcat_cap_effective: int = 0
        self.rankcat_cap_relaxed: bool = False
    
    def _canonical_key_from_row(self, row: pd.Series) -> str:
        cid = row.get(COL_CANONICAL_QUERY_ID)
        if cid is None or (isinstance(cid, float) and math.isnan(cid)) or pd.isna(cid):
            cid = row.get(COL_QUERY_ID)
        return str(cid)

    def _rank_category_from_row(self, row: pd.Series) -> str:
        rc = row.get('rank_category')
        if rc is None or (isinstance(rc, float) and math.isnan(rc)) or pd.isna(rc):
            rc = row.get(COL_CATEGORY)
        return str(rc) if rc is not None else "NA"

    def _add_eval_slice_tags(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """Add slice tags using LLM results if available, fallback to regex"""
        if ranked_df is None or ranked_df.empty:
            return ranked_df
        df = ranked_df.copy()
        
        # Check if tags already exist (from LLM scoring)
        # We prefer LLM tags because they are semantically accurate
        
        # Tag: Trap/Unreleased
        if 'tag_trap_unreleased' not in df.columns:
            df['tag_trap_unreleased'] = df[COL_QUERY].apply(is_trap_unreleased_query)
        else:
            # Fill NA with fallback regex
            mask = df['tag_trap_unreleased'].isna()
            if mask.any():
                df.loc[mask, 'tag_trap_unreleased'] = df.loc[mask, COL_QUERY].apply(is_trap_unreleased_query)
        
        # Tag: Procedural
        if 'tag_procedural' not in df.columns:
            df['tag_procedural'] = df[COL_QUERY].apply(is_procedural_query)
        else:
            mask = df['tag_procedural'].isna()
            if mask.any():
                df.loc[mask, 'tag_procedural'] = df.loc[mask, COL_QUERY].apply(is_procedural_query)

        # Tag: Structured
        if 'tag_structured' not in df.columns:
            df['tag_structured'] = df[COL_QUERY].apply(needs_structured_output_query)
        else:
            mask = df['tag_structured'].isna()
            if mask.any():
                df.loc[mask, 'tag_structured'] = df.loc[mask, COL_QUERY].apply(needs_structured_output_query)
        
        # Tag: Constraint Heavy
        if 'tag_constraint_heavy' not in df.columns:
            df['tag_constraint_heavy'] = df[COL_QUERY].apply(is_constraint_heavy_query)
        else:
            mask = df['tag_constraint_heavy'].isna()
            if mask.any():
                df.loc[mask, 'tag_constraint_heavy'] = df.loc[mask, COL_QUERY].apply(is_constraint_heavy_query)
        
        return df

    def _effective_rankcat_cap(self, n_rankcat: int, eval_size: int) -> int:
        if RANK_CATEGORY_CAP_ENV > 0:
            return RANK_CATEGORY_CAP_ENV
        if n_rankcat >= 8:
            return max(2, math.ceil(eval_size / max(8, n_rankcat)))
        return 0

    def _preselect_eval_slices(
        self,
        ranked_df: pd.DataFrame,
        eval_size: int,
        selected_ids: set,
        selected_canon: set,
        rankcat_counts: Counter,
        rankcat_cap: int
    ) -> set:
        """Best-effort预选高风险切片，允许轻微超额，不破坏rank_category上限"""
        if (not ENABLE_SLICE_QUOTA_SELECTION or
                ranked_df is None or ranked_df.empty or eval_size <= 0):
            return set()

        required_cols = {'tag_trap_unreleased', 'tag_procedural', 'tag_structured', 'tag_constraint_heavy'}
        if not required_cols.issubset(set(ranked_df.columns)):
            return set()

        targets = compute_slice_targets(eval_size, SLICE_QUOTAS_DEFAULT)
        if not targets:
            return set()

        newly_selected: set = set()

        def can_pick(row: pd.Series) -> bool:
            qid = str(row[COL_QUERY_ID])
            if qid in selected_ids:
                return False
            ck = self._canonical_key_from_row(row)
            if ck in selected_canon:
                return False
            rk = self._rank_category_from_row(row)
            if rankcat_cap > 0 and rankcat_counts[rk] >= rankcat_cap:
                return False
            return True

        slice_order = [
            ("trap_unreleased", "tag_trap_unreleased"),
            ("procedural", "tag_procedural"),
            ("structured", "tag_structured"),
            ("constraint_heavy", "tag_constraint_heavy"),
        ]

        for slice_name, col in slice_order:
            target = int(targets.get(slice_name, 0) or 0)
            if target <= 0:
                continue

            current = int(
                ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)][col].sum()
            )
            need = max(0, target - current)
            max_allow = target + SLICE_OVERAGE_ALLOWANCE

            if need <= 0 and current >= max_allow:
                continue

            candidates = ranked_df[ranked_df[col] == True].sort_values('represent_rank')
            for _, row in candidates.iterrows():
                if len(selected_ids) >= eval_size:
                    break
                current = int(
                    ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)][col].sum()
                )
                if current >= max_allow:
                    break
                if not can_pick(row):
                    continue
                qid = str(row[COL_QUERY_ID])
                ck = self._canonical_key_from_row(row)
                rk = self._rank_category_from_row(row)
                selected_ids.add(qid)
                selected_canon.add(ck)
                rankcat_counts[rk] += 1
                newly_selected.add(qid)
                
                # 记录入选原因
                if 'selection_reason' not in ranked_df.columns:
                    ranked_df['selection_reason'] = pd.NA
                ranked_df.loc[ranked_df[COL_QUERY_ID] == row[COL_QUERY_ID], 'selection_reason'] = f"Slice: {slice_name}"

        return newly_selected

    def _enforce_slice_quotas(
        self,
        ranked_df: pd.DataFrame,
        selected_ids: set,
        eval_size: int,
        rankcat_cap: int,
        category_floor: Dict[str, int]
    ) -> set:
        """后置补齐切片，尽量不破坏类别配额/小类上限，不强求"""
        if (not ENABLE_SLICE_QUOTA_SELECTION or
                ranked_df is None or ranked_df.empty or eval_size <= 0 or not selected_ids):
            return selected_ids

        required_cols = {'tag_trap_unreleased', 'tag_procedural', 'tag_structured', 'tag_constraint_heavy'}
        if not required_cols.issubset(set(ranked_df.columns)):
            return selected_ids

        targets = compute_slice_targets(eval_size, SLICE_QUOTAS_DEFAULT)
        if not targets:
            return selected_ids

        slice_order = [
            ("trap_unreleased", "tag_trap_unreleased"),
            ("procedural", "tag_procedural"),
            ("structured", "tag_structured"),
            ("constraint_heavy", "tag_constraint_heavy"),
        ]

        selected_ids = set(selected_ids)

        def build_state(sel_ids: set) -> Tuple[pd.DataFrame, set, Counter, Dict[str, int]]:
            df_sel = ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(sel_ids)].copy()
            canon_local = set(df_sel.apply(self._canonical_key_from_row, axis=1).tolist())
            rankcat_local = Counter(df_sel.apply(self._rank_category_from_row, axis=1).tolist())
            cat_counts = df_sel[COL_CATEGORY].value_counts().to_dict()
            return df_sel, canon_local, rankcat_local, cat_counts

        df_sel, selected_canon, rankcat_counts, cat_counts = build_state(selected_ids)

        def counts_for_slice(df_local: pd.DataFrame) -> Dict[str, int]:
            return {name: int(df_local[col].sum()) for name, col in slice_order}

        # Ensure selection_reason column exists
        if 'selection_reason' not in ranked_df.columns:
            ranked_df['selection_reason'] = pd.NA

        for slice_name, col in slice_order:
            target = int(targets.get(slice_name, 0) or 0)
            if target <= 0:
                continue

            df_sel = ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)].copy()
            selected_canon = set(df_sel.apply(self._canonical_key_from_row, axis=1).tolist())
            rankcat_counts = Counter(df_sel.apply(self._rank_category_from_row, axis=1).tolist())
            cat_counts = df_sel[COL_CATEGORY].value_counts().to_dict()
            current = int(df_sel[col].sum())
            if current >= target:
                continue

            need = target - current
            add_pool = ranked_df[
                (ranked_df[col] == True) &
                (~ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids))
            ].sort_values('represent_rank')

            if add_pool.empty:
                logging.warning(f"[Selection][Slices] Not enough candidates for slice '{slice_name}' (need {need})")
                continue

            replaced = 0
            for _, add_row in add_pool.iterrows():
                if replaced >= need:
                    break
                add_qid = str(add_row[COL_QUERY_ID])
                add_ck = self._canonical_key_from_row(add_row)
                add_rk = self._rank_category_from_row(add_row)
                add_cat = add_row[COL_CATEGORY]

                if add_ck in selected_canon:
                    continue
                if rankcat_cap > 0 and rankcat_counts[add_rk] >= rankcat_cap:
                    continue

                removable_df = df_sel[df_sel[col] != True].sort_values('represent_rank', ascending=False)
                if removable_df.empty:
                    break

                remove_row = None
                for _, cand in removable_df.iterrows():
                    cat_name = cand[COL_CATEGORY]
                    if category_floor.get(cat_name, 0) and cat_counts.get(cat_name, 0) - 1 < category_floor[cat_name]:
                        continue
                    remove_row = cand
                    break

                if remove_row is None:
                    break

                remove_qid = str(remove_row[COL_QUERY_ID])
                remove_ck = self._canonical_key_from_row(remove_row)
                remove_rk = self._rank_category_from_row(remove_row)
                remove_cat = remove_row[COL_CATEGORY]

                selected_ids.discard(remove_qid)
                selected_canon.discard(remove_ck)
                rankcat_counts[remove_rk] = max(0, rankcat_counts[remove_rk] - 1)
                cat_counts[remove_cat] = max(0, cat_counts.get(remove_cat, 0) - 1)

                selected_ids.add(add_qid)
                selected_canon.add(add_ck)
                rankcat_counts[add_rk] += 1
                cat_counts[add_cat] = cat_counts.get(add_cat, 0) + 1
                
                # Update selection reason
                ranked_df.loc[ranked_df[COL_QUERY_ID] == add_row[COL_QUERY_ID], 'selection_reason'] = f"Slice Backfill: {slice_name}"
                
                replaced += 1

            if replaced < need:
                logging.warning(f"[Selection][Slices] '{slice_name}' shortfall: need {need}, replaced {replaced}")

        if len(selected_ids) > eval_size:
            ranked_selected = ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)].sort_values('represent_rank')
            selected_ids = set(ranked_selected.head(eval_size)[COL_QUERY_ID].astype(str).tolist())

        return selected_ids
    
    def _create_strict_scoring_schema(self, query_ids: List[str]) -> Dict:
        """创建严格的JSON Schema用于评分响应
        Args: query_ids: 此批次中的所有query ID列表
        Returns: 严格的JSON Schema，包含enum限定ID和固定数组长度
        """
        return {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "minItems": len(query_ids),
                    "maxItems": len(query_ids),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "enum": query_ids  # 严格限定只能是批次中的ID
                            },
                            "specificity": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100
                            },
                            "completeness": {
                                "type": "integer", 
                                "minimum": 0,
                                "maximum": 100
                            },
                            "representativeness": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100
                            },
                            "depth_and_value": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100
                            },
                            "is_procedural": {"type": "boolean"},
                            "needs_structured_output": {"type": "boolean"},
                            "has_heavy_constraints": {"type": "boolean"},
                            "is_trap_unreleased": {"type": "boolean"}
                        },
                        "required": [
                            "id", "specificity", "completeness", "depth_and_value",
                            "is_procedural", "needs_structured_output", 
                            "has_heavy_constraints", "is_trap_unreleased"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["scores"],
            "additionalProperties": False
        }
    
    async def rank_queries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank queries only (no selection)"""
        relevant_mask = df['query_status'] == QUERY_STATUS_RELEVANT

        if not relevant_mask.any():
            return pd.DataFrame()

        relevant_df = df[relevant_mask].copy()

        # Handle relevant queries with no assigned category before ranking
        missing_cat_mask = relevant_df[COL_CATEGORY].isna() | (relevant_df[COL_CATEGORY] == '')
        if missing_cat_mask.any():
            missing_count = missing_cat_mask.sum()
            logging.warning(f"[Ranking] {missing_count} relevant queries have no category, assigning to '{CATEGORY_OTHER}'")

            relevant_df.loc[missing_cat_mask, COL_CATEGORY] = CATEGORY_OTHER

        # Prepare DataFrame with subdivision information (rank_category column)
        ranked_ready_df = await self._subdivide_if_needed(relevant_df)

        # Rank all categories and attach scoring information
        ranked_df = await self._rank_all_categories(ranked_ready_df)
        
        if ranked_df.empty:
            logging.warning("[Ranking] No ranked results produced")
            return pd.DataFrame()
            
        if 'selected_for_eval' not in ranked_df.columns:
            ranked_df['selected_for_eval'] = False
            
        return ranked_df

    async def select_queries(self, ranked_df: pd.DataFrame, eval_size: int, df_original: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Select queries from ranked results"""
        if ranked_df.empty:
            if df_original is None:
                # Should not happen if called correctly
                return pd.DataFrame(), pd.DataFrame()
            return pd.DataFrame(), df_original.assign(represent_rank=-1, selected_for_eval=False)

        # Select queries using Hamilton allocation; selection flag is updated in ranked_df
        selected_ids = await self._select_by_allocation(ranked_df, eval_size)

        # 检查选中集合中的相似度并进行递补
        if selected_ids:
            selected_ids = await self._check_and_replace_similar_queries(
                ranked_df, selected_ids, eval_size
            )

        # Update selected_for_eval flag
        ranked_df['selected_for_eval'] = ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)
        
        # IMPORTANT: Ensure selection_reason is preserved/updated for the final selection set
        # If selection_reason is missing for any selected item, mark it as "Unknown (Selection Logic)"
        if 'selection_reason' not in ranked_df.columns:
            ranked_df['selection_reason'] = pd.NA
            
        mask_selected_no_reason = ranked_df['selected_for_eval'] & ranked_df['selection_reason'].isna()
        if mask_selected_no_reason.any():
             ranked_df.loc[mask_selected_no_reason, 'selection_reason'] = "Unknown (Logic Gap)"

        df_eval = ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)].copy()
        
        # If original df is provided, create full details view
        if df_original is not None:
            df_details = self._create_selection_details(df_original, ranked_df)
        else:
            # Fallback if original df not provided (mostly for testing or partial updates)
            df_details = ranked_df.copy()
            
        return df_eval.reset_index(drop=True), df_details

    async def rank_and_select(self, df: pd.DataFrame, eval_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Rank queries and select evaluation set (Legacy wrapper)"""
        ranked_df = await self.rank_queries(df)
        return await self.select_queries(ranked_df, eval_size, df)
    
    async def _embed_queries(self, texts: List[str]) -> np.ndarray:
        return await _generate_embeddings(
            self.client,
            texts,
            model_name=MODEL_CONFIG_INTERNAL["embedding"],
            batch_size=EMB_BATCH_SIZE,
            noise_scale=0.001,
            context="selection"
        )
    
    async def _check_and_replace_similar_queries(self, ranked_df: pd.DataFrame, selected_ids: set, eval_size: int) -> set:
        """检查选中的queries是否有高度相似的，如果有则递补"""
        if not selected_ids:
            return selected_ids

        ranked_df = ranked_df.copy()
        selected_mask = ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)
        selected_df = ranked_df[selected_mask].copy()

        logging.info(f"[Selection] Checking similarity in {len(selected_df)} selected queries")

        try:
            X = await self._embed_queries(selected_df[COL_QUERY].astype(str).tolist())
            similarity_matrix = cosine_similarity(X)

            high_similarity_threshold = 0.85
            similar_pairs = []
            for i in range(len(selected_df)):
                for j in range(i + 1, len(selected_df)):
                    if similarity_matrix[i][j] > high_similarity_threshold:
                        similar_pairs.append((i, j, similarity_matrix[i][j]))
                        logging.warning(f"[Selection] Found similar pair (similarity={similarity_matrix[i][j]:.3f})")
                        logging.warning(f"  Query1: {selected_df.iloc[i][COL_QUERY][:100]}")
                        logging.warning(f"  Query2: {selected_df.iloc[j][COL_QUERY][:100]}")

            if similar_pairs:
                to_replace_ids = set()
                for i, j, _ in similar_pairs:
                    row_i = selected_df.iloc[i]
                    row_j = selected_df.iloc[j]
                    rank_i = row_i.get('represent_rank', float('inf'))
                    rank_j = row_j.get('represent_rank', float('inf'))
                    if rank_i <= rank_j:
                        to_replace_ids.add(str(row_j[COL_QUERY_ID]))
                    else:
                        to_replace_ids.add(str(row_i[COL_QUERY_ID]))

                if to_replace_ids:
                    selected_ids = self._replace_queries_with_candidates(
                        selected_ids, list(to_replace_ids), ranked_df, "high similarity detected"
                    )

        except Exception as e:
            logging.error(f"[Selection] Failed to check similarity: {e}, keeping original selection")

        selected_df = ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)].copy()
        selected_df['canonical_key'] = selected_df[COL_CANONICAL_QUERY_ID].fillna(selected_df[COL_QUERY_ID]).astype(str)

        duplicate_ids = []
        for _, group in selected_df.groupby('canonical_key'):
            if len(group) <= 1:
                continue
            group_sorted = group.sort_values('represent_rank')
            duplicate_ids.extend(group_sorted.iloc[1:][COL_QUERY_ID].astype(str).tolist())

        if duplicate_ids:
            selected_ids = self._replace_queries_with_candidates(
                selected_ids, duplicate_ids, ranked_df, "enforcing canonical uniqueness"
            )

        # 切片后置补齐（温和约束）
        if ENABLE_SLICE_QUOTA_SELECTION:
            ranked_df = self._add_eval_slice_tags(ranked_df)
            category_floor = self._hamilton_allocation(Counter(ranked_df[COL_CATEGORY]), min(eval_size, len(ranked_df)))
            rankcat_cap = self.rankcat_cap_effective if not self.rankcat_cap_relaxed else 0
            selected_ids = self._enforce_slice_quotas(ranked_df, selected_ids, eval_size, rankcat_cap, category_floor)

        # Ensure selection size does not exceed eval_size
        if len(selected_ids) > eval_size:
            ranked_selected = ranked_df[ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)]
            ranked_selected = ranked_selected.sort_values('represent_rank')
            trimmed_ids = set(ranked_selected.head(eval_size)[COL_QUERY_ID].astype(str))
            selected_ids = trimmed_ids

        return selected_ids
    
    def _replace_queries_with_candidates(self, selected_ids: set, to_replace_ids: list, 
                                        ranked_df: pd.DataFrame, reason: str) -> set:
        """通用的query替换逻辑
        
        Args:
            selected_ids: Current set of selected query IDs
            to_replace_ids: List of query IDs to replace
            ranked_df: DataFrame with all ranked queries
            reason: Reason for replacement (for logging)
        
        Returns:
            Updated set of selected query IDs
        """
        if not to_replace_ids:
            return selected_ids
        
        logging.info(f"[Selection] {reason}: replacing {len(to_replace_ids)} queries")
        
        # Get candidates sorted by rank
        candidates_df = ranked_df[~ranked_df[COL_QUERY_ID].astype(str).isin(selected_ids)]
        candidates_df = candidates_df.sort_values('represent_rank')
        replacement_iter = candidates_df.iterrows()
        
        replaced = 0
        for remove_id in to_replace_ids:
            try:
                _, new_row = next(replacement_iter)
            except StopIteration:
                logging.warning(f"[Selection] No more candidates for replacement ({reason})")
                break
            
            selected_ids.discard(str(remove_id))
            new_id = str(new_row[COL_QUERY_ID])
            selected_ids.add(new_id)
            
            # Record reason for the replacement (side-effect on ranked_df)
            if 'selection_reason' not in ranked_df.columns:
                ranked_df['selection_reason'] = pd.NA
            ranked_df.loc[ranked_df[COL_QUERY_ID].astype(str) == new_id, 'selection_reason'] = reason
            
            replaced += 1
        
        if replaced:
            logging.info(f"[Selection] Replaced {replaced} queries ({reason})")
        
        return selected_ids
    
    async def _subdivide_if_needed(self, relevant_df: pd.DataFrame) -> pd.DataFrame:
        """Subdivide large categories and return DataFrame with rank_category column"""
        frames = []
        for cat_name, cat_df in relevant_df.groupby(COL_CATEGORY):
            if len(cat_df) > SUBCATEGORY_SIZE_THRESHOLD:
                frames.append(await self._subdivide_category(cat_df, cat_name))
            else:
                tmp = cat_df.copy()
                tmp['rank_category'] = cat_name
                frames.append(tmp)

        if frames:
            return pd.concat(frames, ignore_index=True)

        return pd.DataFrame(columns=relevant_df.columns.tolist() + ['rank_category'])
    
    async def _subdivide_category(self, cat_df: pd.DataFrame, category_name: str) -> pd.DataFrame:
        """Subdivide a single large category and return DataFrame with rank_category"""
        if len(cat_df) <= SUBCATEGORY_MIN_SIZE:
            self.category_subdivision_map[category_name] = {
                "original_count": len(cat_df),
                "subcategories": {category_name: len(cat_df)},
                "subdivision_success": False,
                "reason": "Size below threshold"
            }
            tmp = cat_df.copy()
            tmp['rank_category'] = category_name
            return tmp

        queries_text = [f"{row_id} ::: {query_text}" for row_id, query_text in zip(cat_df[COL_QUERY_ID].astype(str), cat_df[COL_QUERY].astype(str))]
        system_prompt = CATEGORY_SUBDIVISION_SYSTEM_PROMPT_TEMPLATE.format(
            category_name=category_name,
            query_count=len(cat_df),
            max_size=SUBCATEGORY_SIZE_THRESHOLD,
            min_size=SUBCATEGORY_MIN_SIZE
        )

        try:
            response = await llm_chat_call(
                self.client, system_prompt, "\n".join(queries_text),
                MODEL_CONFIG_INTERNAL["discovery"],
                temperature=TEMPERATURE_CREATIVE,
                max_tokens=MAX_TOKENS_EXTENDED
            )

            assignment_map: Dict[str, str] = {}

            for line in response.splitlines():
                if ":::" in line:
                    parts = line.split(":::", 1)
                    if len(parts) == 2:
                        query_id = parts[0].strip()
                        subcat = parts[1].strip()
                        assignment_map[query_id] = subcat

            tmp = cat_df.copy()
            tmp['rank_category'] = tmp[COL_QUERY_ID].astype(str).map(assignment_map)

            # Handle unassigned queries by placing them into the largest assigned subcategory or fallback
            if tmp['rank_category'].isna().any():
                if assignment_map:
                    counts = tmp['rank_category'].value_counts(dropna=True).to_dict()
                    largest_subcat = max(counts, key=counts.get) if counts else category_name
                else:
                    largest_subcat = category_name
                tmp['rank_category'] = tmp['rank_category'].fillna(largest_subcat)

            # Validate subcategory sizes
            valid_mask = tmp['rank_category'].map(tmp['rank_category'].value_counts()) >= SUBCATEGORY_MIN_SIZE
            if not valid_mask.all():
                # Merge small subcategories into the largest valid subcategory
                value_counts = tmp['rank_category'].value_counts().to_dict()
                largest_valid = max(value_counts, key=value_counts.get)
                tmp.loc[~valid_mask, 'rank_category'] = largest_valid

            subcat_sizes = tmp['rank_category'].value_counts().to_dict()
            self.category_subdivision_map[category_name] = {
                "original_count": len(cat_df),
                "subcategories": subcat_sizes,
                "subdivision_success": True
            }

            return tmp

        except Exception as e:
            logging.error(f"Subdivision failed: {e}")
            self.category_subdivision_map[category_name] = {
                "original_count": len(cat_df),
                "subcategories": {category_name: len(cat_df)},
                "subdivision_success": False,
                "reason": f"Exception: {str(e)}"
            }
            tmp = cat_df.copy()
            tmp['rank_category'] = category_name
            return tmp
    
    async def _rank_all_categories(self, ranked_ready_df: pd.DataFrame) -> pd.DataFrame:
        """Rank queries in all categories, returning DataFrame with ranking columns"""
        if ranked_ready_df.empty:
            columns = ranked_ready_df.columns.tolist() if len(ranked_ready_df.columns) > 0 else []
            extra_cols = ['represent_rank', 'scoring_details', 'selected_for_eval']
            return pd.DataFrame(columns=columns + extra_cols)

        ranked_frames = []

        for cat_name, cat_df in tqdm(ranked_ready_df.groupby('rank_category'), desc="Ranking categories"):
            ranked = await self._rank_category(cat_df, cat_name)
            if not ranked:
                continue

            ranks, row_dicts = zip(*ranked)
            ranked_cat_df = pd.DataFrame(list(row_dicts)).copy()
            ranked_cat_df['represent_rank'] = list(ranks)
            ranked_cat_df['rank_category'] = cat_name
            ranked_frames.append(ranked_cat_df)

        if ranked_frames:
            result_df = pd.concat(ranked_frames, ignore_index=True)
        else:
            result_df = ranked_ready_df.copy()
            result_df['represent_rank'] = -1

        return result_df
    
    async def _rank_category(self, cat_df: pd.DataFrame, category_name: str) -> List[Tuple[int, Dict]]:
        """Rank queries within a category"""
        if cat_df.empty:
            return []

        rows = cat_df.to_dict('records')
        return await self._rank_single_batch(rows, category_name)

    
    async def _rank_single_batch(self, rows: List[Dict], batch_name: str) -> List[Tuple[int, Dict]]:
        """使用并发分批评分+本地排序的方式进行排名"""
        if not rows:
            return []
        
        batch_size = FILTER_SELECT_SCORING_BATCH_SIZE
        batches = []
        
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i + batch_size]
            batches.append(batch_rows)
        
        semaphore = asyncio.Semaphore(CONCURRENT_API_CALL_LIMIT)
        
        async def score_single_batch_with_semaphore(batch_rows):
            async with semaphore:
                return await self._score_batch(batch_rows)
        
        batch_results = await asyncio.gather(*[
            score_single_batch_with_semaphore(batch_rows) 
            for batch_rows in batches
        ], return_exceptions=True)
        
        # 整合结果和统计信息
        all_scores = {}
        
        for i, result in enumerate(batch_results):
            # 获取当前批次的原始行
            current_batch = batches[i]
            
            # 先给这一批次所有query设置默认值（兜底）
            for row in current_batch:
                query_id = str(row[COL_QUERY_ID])
                all_scores[query_id] = {
                    "specificity": 30, "completeness": 30, "depth_and_value": 30,
                    "is_procedural": False, "needs_structured_output": False,
                    "has_heavy_constraints": False, "is_trap_unreleased": False
                }
            
            if isinstance(result, Exception):
                logging.error(f"Batch {i} scoring failed: {result}")
                # 已设置默认值，无需额外操作
            else:
                # 使用实际评分结果覆盖默认值
                all_scores.update(result)
        
        # 检查完整性
        input_ids = {str(row[COL_QUERY_ID]) for row in rows}
        scored_ids = set(all_scores.keys())
        
        if input_ids != scored_ids:
            missing_ids = input_ids - scored_ids
            if missing_ids:
                print(f"Warning: {len(missing_ids)} queries missing scores, using defaults")
                for missing_id in missing_ids:
                    all_scores[missing_id] = {
                        "specificity": 50, "completeness": 50, "depth_and_value": 50,
                        "is_procedural": False, "needs_structured_output": False,
                        "has_heavy_constraints": False, "is_trap_unreleased": False
                    }
        
        # 本地排序 - 按用户指定的权重计算总分
        weighted_scores = []
        for row in rows:
            query_id = str(row[COL_QUERY_ID])
            scores = all_scores[query_id]
            # 按权重计算总分：具体性30% + 信息完整性30% + 游戏深度与价值40%
            # 提高深度分权重，降低其他项权重，确保选出有价值的Query
            total_score = (scores["specificity"] * 0.30 + 
                          scores["completeness"] * 0.30 + 
                          scores["depth_and_value"] * 0.40)
            
            # 创建评分详情JSON
            scoring_details = {
                "specificity": scores["specificity"],
                "completeness": scores["completeness"],
                "depth_and_value": scores["depth_and_value"],
                "total_score": round(total_score, 2),
                "weights": {"specificity": 0.30, "completeness": 0.30, "depth_and_value": 0.40},
                "tags": {
                    "is_procedural": scores.get("is_procedural", False),
                    "needs_structured_output": scores.get("needs_structured_output", False),
                    "has_heavy_constraints": scores.get("has_heavy_constraints", False),
                    "is_trap_unreleased": scores.get("is_trap_unreleased", False)
                }
            }
            
            # 将评分详情添加到行数据中
            row['scoring_details'] = json.dumps(scoring_details, ensure_ascii=False)
            
            # 将Tag直接添加到Row中，供后续切片筛选使用
            row['tag_procedural'] = scores.get("is_procedural", False)
            row['tag_structured'] = scores.get("needs_structured_output", False)
            row['tag_constraint_heavy'] = scores.get("has_heavy_constraints", False)
            row['tag_trap_unreleased'] = scores.get("is_trap_unreleased", False)
            
            weighted_scores.append((total_score, row))
        
        # 按分数排序（分数高的排名靠前）
        weighted_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 生成排名结果
        ranked = [(i + 1, row) for i, (_, row) in enumerate(weighted_scores)]
        return ranked

    async def _score_batch(self, batch_rows: List[Dict]) -> Dict[str, Dict]:
        """对一批query进行评分，返回评分结果"""
        if not batch_rows:
            return {}
        
        # 构建短ID映射避免长ID问题
        id_mapping = {}
        short_pairs = []
        
        for i, row in enumerate(batch_rows):
            short_id = f"Q{i+1:03d}"  # Q001, Q002, ...
            original_id = str(row[COL_QUERY_ID])
            id_mapping[short_id] = original_id
            
            # 截断query文本到120字符
            query_text = str(row[COL_QUERY])[:120]
            short_pairs.append((short_id, query_text))
        
        # 构建评分prompt
        score_prompt = self._build_score_prompt(short_pairs)
        
        # 使用配置决定是否使用严格schema
        if SUPPORTS_STRICT_JSON_SCHEMA:
            short_ids = [pair[0] for pair in short_pairs]
            strict_schema = self._create_strict_scoring_schema(short_ids)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "scoring_response",
                    "schema": strict_schema
                }
            }
        else:
            response_format = {"type": "json_object"}
        
        try:
            response = await llm_chat_call(
                self.client,
                SCORE_SYSTEM_PROMPT,
                score_prompt,
                MODEL_CONFIG_INTERNAL["ranking"],
                temperature=TEMPERATURE_STRICT,
                max_tokens=MAX_TOKENS_EXTENDED,
                response_format=response_format
            )
            
            # 解析JSON响应
            data = json.loads(response)
            scores_list = data.get("scores", [])
            
            # 转换回原始ID
            result = {}
            for item in scores_list:
                short_id = item.get("id")
                if short_id in id_mapping:
                    original_id = id_mapping[short_id]
                    result[original_id] = {
                        "specificity": int(item.get("specificity", 50)),
                        "completeness": int(item.get("completeness", 50)),
                        "depth_and_value": int(item.get("depth_and_value", 50)),
                        # Tags
                        "is_procedural": bool(item.get("is_procedural", False)),
                        "needs_structured_output": bool(item.get("needs_structured_output", False)),
                        "has_heavy_constraints": bool(item.get("has_heavy_constraints", False)),
                        "is_trap_unreleased": bool(item.get("is_trap_unreleased", False))
                    }
            
            # 检查是否有缺失的query需要兜底
            all_original_ids = set(id_mapping.values())
            scored_ids = set(result.keys())
            missing_ids = all_original_ids - scored_ids
            
            # 为缺失的query添加默认分数
            if missing_ids:
                logging.warning(f"Scoring batch missing {len(missing_ids)} queries, using default scores")
                for query_id in missing_ids:
                    result[query_id] = {
                        "specificity": 30, "completeness": 30, "depth_and_value": 30,
                        "is_procedural": False, "needs_structured_output": False,
                        "has_heavy_constraints": False, "is_trap_unreleased": False
                    }
            
            return result
            
        except Exception as e:
            logging.error(f"Scoring batch failed: {e}")
            # 返回默认分数
            default_scores = {
                "specificity": 30, "completeness": 30, "depth_and_value": 30,
                "is_procedural": False, "needs_structured_output": False,
                "has_heavy_constraints": False, "is_trap_unreleased": False
            }
            return {str(row[COL_QUERY_ID]): default_scores for row in batch_rows}

    def _build_score_prompt(self, pairs: List[Tuple[str, str]]) -> str:
        """构建评分prompt"""
        lines = [f"{sid} ::: {query_text}" for sid, query_text in pairs]
        return "待打分 (ID ::: 文本) :\n" + "\n".join(lines)


    async def _select_by_allocation(self, ranked_df: pd.DataFrame, eval_size: int) -> set:
        """先切片预选，再Hamilton配额，必要时放宽小类上限，确保凑满"""
        if ranked_df.empty:
            return set()

        df = ranked_df.copy()
        if 'selected_for_eval' not in df.columns:
            df['selected_for_eval'] = False
        
        # 初始化 selection_reason 列
        if 'selection_reason' not in df.columns:
            df['selection_reason'] = pd.NA

        # 计算 rank_category 上限（可被放宽）
        n_rankcat = df['rank_category'].nunique(dropna=True)
        rankcat_cap = self._effective_rankcat_cap(n_rankcat, eval_size)
        self.rankcat_cap_effective = rankcat_cap
        self.rankcat_cap_relaxed = False

        # 预处理：打标签（风险切片）
        if ENABLE_SLICE_QUOTA_SELECTION:
            df = self._add_eval_slice_tags(df)

        # 初始已选集合
        selected_ids: set = set(df.loc[df['selected_for_eval'], COL_QUERY_ID].dropna().astype(str).tolist())
        df_sel_init = df[df[COL_QUERY_ID].astype(str).isin(selected_ids)]
        selected_canon: set = set(df_sel_init.apply(self._canonical_key_from_row, axis=1).tolist())
        rankcat_counts: Counter = Counter(df_sel_init.apply(self._rank_category_from_row, axis=1).tolist())

        # Hamilton 配额
        category_counts = Counter(df[COL_CATEGORY])
        allocation = self._hamilton_allocation(category_counts, eval_size)

        # 0) 切片预选（best-effort）
        # _preselect_eval_slices 内部会修改 df 的 selection_reason
        newly = self._preselect_eval_slices(df, eval_size, selected_ids, selected_canon, rankcat_counts, rankcat_cap)
        if newly:
            df.loc[df[COL_QUERY_ID].astype(str).isin(newly), 'selected_for_eval'] = True

        def can_pick_row(row: pd.Series, cap: int) -> bool:
            qid = str(row[COL_QUERY_ID])
            if qid in selected_ids:
                return False
            ck = self._canonical_key_from_row(row)
            if ck in selected_canon:
                return False
            rk = self._rank_category_from_row(row)
            if cap > 0 and rankcat_counts[rk] >= cap:
                return False
            return True

        # 1) 按类别配额选
        for cat_name, alloc_total in allocation.items():
            if alloc_total <= 0:
                continue

            already_in_cat = int(df[(df['selected_for_eval'] == True) & (df[COL_CATEGORY] == cat_name)].shape[0])
            need = max(0, int(alloc_total) - already_in_cat)
            if need <= 0:
                continue

            cat_df = df[df[COL_CATEGORY] == cat_name]
            if cat_df.empty:
                continue

            subcategory_groups = {
                subcat: group.sort_values('represent_rank')
                for subcat, group in cat_df.groupby('rank_category')
            }
            picked = 0

            # 尽量覆盖子类
            if len(subcategory_groups) > 1 and need >= len(subcategory_groups):
                for _, group in subcategory_groups.items():
                    if picked >= need:
                        break
                    for idx, row in group.iterrows():
                        if not can_pick_row(row, rankcat_cap):
                            continue
                        qid = str(row[COL_QUERY_ID])
                        ck = self._canonical_key_from_row(row)
                        rk = self._rank_category_from_row(row)
                        selected_ids.add(qid)
                        selected_canon.add(ck)
                        rankcat_counts[rk] += 1
                        df.loc[idx, 'selected_for_eval'] = True
                        df.loc[idx, 'selection_reason'] = "Category Allocation (Subdivision)"
                        picked += 1
                        break

            if picked >= need:
                continue

            # 剩余按 rank 补
            all_candidates = pd.concat(list(subcategory_groups.values())).sort_values('represent_rank')
            for idx, row in all_candidates.iterrows():
                if picked >= need:
                    break
                if not can_pick_row(row, rankcat_cap):
                    continue
                qid = str(row[COL_QUERY_ID])
                ck = self._canonical_key_from_row(row)
                rk = self._rank_category_from_row(row)
                selected_ids.add(qid)
                selected_canon.add(ck)
                rankcat_counts[rk] += 1
                df.loc[idx, 'selected_for_eval'] = True
                df.loc[idx, 'selection_reason'] = "Category Allocation (Rank Fill)"
                picked += 1

        # 2) 全局补齐
        if len(selected_ids) < eval_size:
            remaining_needed = eval_size - len(selected_ids)
            candidates = df[~df['selected_for_eval']].copy()
            candidates = candidates.sort_values(by=['represent_rank', COL_CATEGORY])
            filled = 0
            for idx, row in candidates.iterrows():
                if filled >= remaining_needed or len(selected_ids) >= eval_size:
                    break
                if not can_pick_row(row, rankcat_cap):
                    continue
                qid = str(row[COL_QUERY_ID])
                ck = self._canonical_key_from_row(row)
                rk = self._rank_category_from_row(row)
                selected_ids.add(qid)
                selected_canon.add(ck)
                rankcat_counts[rk] += 1
                df.loc[idx, 'selected_for_eval'] = True
                df.loc[idx, 'selection_reason'] = "Global Fill"
                filled += 1

        # 3) 如果因小类上限导致选不满，放宽上限再补一次
        if len(selected_ids) < eval_size and rankcat_cap > 0:
            self.rankcat_cap_relaxed = True
            relaxed_needed = eval_size - len(selected_ids)
            for cap_try in [0]:
                if relaxed_needed <= 0:
                    break
                candidates = df[~df['selected_for_eval']].copy().sort_values(by=['represent_rank', COL_CATEGORY])
                for idx, row in candidates.iterrows():
                    if len(selected_ids) >= eval_size:
                        break
                    if not can_pick_row(row, cap_try):
                        continue
                    qid = str(row[COL_QUERY_ID])
                    ck = self._canonical_key_from_row(row)
                    rk = self._rank_category_from_row(row)
                    selected_ids.add(qid)
                    selected_canon.add(ck)
                    rankcat_counts[rk] += 1
                    df.loc[idx, 'selected_for_eval'] = True
                    df.loc[idx, 'selection_reason'] = "Global Fill (Relaxed Cap)"

        # 返回选中ID（后续步骤会写回）
        df['selected_for_eval'] = df[COL_QUERY_ID].astype(str).isin(selected_ids)
        
        # Update both columns to preserve selection reasons
        cols_to_update = ['selected_for_eval']
        if 'selection_reason' in df.columns:
             # Ensure selection_reason column exists in ranked_df before update
             if 'selection_reason' not in ranked_df.columns:
                 ranked_df['selection_reason'] = pd.NA
             cols_to_update.append('selection_reason')
        
        ranked_df.update(df[cols_to_update])

        return selected_ids
    
    def _hamilton_allocation(self, counts: Counter, total: int) -> Counter:
        """Hamilton method for proportional allocation"""
        alloc = Counter()
        remainders = {}
        assigned = 0
        
        total_count = sum(counts.values())
        for cat, cnt in counts.items():
            share = cnt / total_count * total if total_count > 0 else 0
            floor_val = math.floor(share)
            alloc[cat] = floor_val
            assigned += floor_val
            remainders[cat] = share - floor_val
        
        # Distribute remaining slots
        for cat, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True):
            if assigned >= total:
                break
            alloc[cat] += 1
            assigned += 1
        
        return alloc
    
    
    def _create_selection_details(self, df: pd.DataFrame, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """Create DataFrame with selection details without full DataFrame merge"""
        df_details = df.copy()

        # Initialize columns with defaults
        df_details['represent_rank'] = -1
        df_details['selected_for_eval'] = False
        df_details['rank_category'] = pd.NA
        if 'scoring_details' not in df_details.columns:
            df_details['scoring_details'] = pd.NA

        if ranked_df is not None and not ranked_df.empty:
            ranked_df = ranked_df.copy()
            ranked_df['query_id_str'] = ranked_df[COL_QUERY_ID].astype(str)
            ranked_df = ranked_df.set_index('query_id_str')

            source_ids = df_details[COL_QUERY_ID].astype(str)

            if 'represent_rank' in ranked_df.columns:
                # Fix FutureWarning: Downcasting object dtype arrays on .fillna
                mapped_rank = source_ids.map(ranked_df['represent_rank'])
                df_details['represent_rank'] = mapped_rank.fillna(-1).infer_objects(copy=False)

            if 'selected_for_eval' in ranked_df.columns:
                # Fix FutureWarning
                mapped_selected = source_ids.map(ranked_df['selected_for_eval'])
                df_details['selected_for_eval'] = mapped_selected.fillna(False).infer_objects(copy=False).astype(bool)

            if 'rank_category' in ranked_df.columns:
                df_details['rank_category'] = source_ids.map(ranked_df['rank_category'])

            if 'scoring_details' in ranked_df.columns:
                df_details['scoring_details'] = source_ids.map(ranked_df['scoring_details'])
                
            if 'selection_reason' in ranked_df.columns:
                df_details['selection_reason'] = source_ids.map(ranked_df['selection_reason'])

        df_details['rank_category'] = df_details['rank_category'].astype('object').where(df_details['rank_category'].notna(), pd.NA)
        df_details['scoring_details'] = df_details['scoring_details'].where(df_details['scoring_details'].notna(), pd.NA)
        if 'selection_reason' in df_details.columns:
             df_details['selection_reason'] = df_details['selection_reason'].where(df_details['selection_reason'].notna(), pd.NA)

        return df_details

def save_pipeline_summary(df_details: pd.DataFrame, selection_manager: SelectionManager, 
                          game_name: str, eval_size: int, start_time: float, set_id: str) -> None:
    """保存全流程指标汇总到txt文件"""
    # 生成文件名
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"pipeline_summary_{sanitize_name(game_name)}_{set_id}_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{game_name} 评测集生成完成 - 全流程指标汇总\n")
            f.write("=" * 80 + "\n")
            
            # 1. 各状态计数
            f.write("\nQuery状态分布:\n")
            status_counts = df_details['query_status'].value_counts()
            total_queries = len(df_details)
            for status, count in status_counts.items():
                percentage = count / total_queries * 100
                f.write(f"  {status}: {count:,} ({percentage:.1f}%)\n")
            
            f.write(f"  总计: {total_queries:,} queries\n")
            
            # 2. 分类分配统计
            f.write(f"\n分类分配统计:\n")
            
            # 原始分类分布
            relevant_mask = df_details['query_status'] == QUERY_STATUS_RELEVANT
            relevant_df = df_details[relevant_mask]
            
            if not relevant_df.empty:
                category_counts = relevant_df[COL_CATEGORY].value_counts()
                f.write(f"  原始分类 ({len(category_counts)} 个类别):\n")
                for category, count in category_counts.head(10).items():
                    percentage = count / len(relevant_df) * 100
                    f.write(f"    {category}: {count} ({percentage:.1f}%)\n")
                if len(category_counts) > 10:
                    f.write(f"    ... 还有 {len(category_counts) - 10} 个类别\n")
            
            # 子分类统计
            if hasattr(selection_manager, 'category_subdivision_map') and selection_manager.category_subdivision_map:
                f.write(f"\n  子分类划分统计:\n")
                for parent_cat, subdivision_info in selection_manager.category_subdivision_map.items():
                    if subdivision_info.get("subdivision_success", False):
                        subcats = subdivision_info["subcategories"]
                        f.write(f"    {parent_cat} → {len(subcats)} 个子类别:\n")
                        for subcat, count in subcats.items():
                            f.write(f"      - {subcat}: {count} queries\n")
                    else:
                        reason = subdivision_info.get("reason", "Unknown")
                        f.write(f"    {parent_cat}: 未细分 ({reason})\n")
            
            # 3. 排名与选择统计      
            # 统计每个类别/子类别的选择情况
            selected_mask = df_details['selected_for_eval'] == True
            if selected_mask.any():
                f.write(f"\n  各类别eval选择详情:\n")
                
                # 使用rank_category如果存在，否则使用原始category
                selected_df = df_details[selected_mask]

                # 切片覆盖简报
                if ENABLE_SLICE_QUOTA_SELECTION and COL_QUERY in selected_df.columns:
                    trap_cnt = int(selected_df[COL_QUERY].apply(is_trap_unreleased_query).sum())
                    proc_cnt = int(selected_df[COL_QUERY].apply(is_procedural_query).sum())
                    struct_cnt = int(selected_df[COL_QUERY].apply(needs_structured_output_query).sum())
                    cons_cnt = int(selected_df[COL_QUERY].apply(is_constraint_heavy_query).sum())
                    targets = compute_slice_targets(eval_size, SLICE_QUOTAS_DEFAULT)
                    f.write(f"    诊断切片覆盖(实际/目标，允许+{SLICE_OVERAGE_ALLOWANCE}超额):\n")
                    f.write(f"      - trap_unreleased: {trap_cnt}/{targets.get('trap_unreleased', 0)}\n")
                    f.write(f"      - procedural: {proc_cnt}/{targets.get('procedural', 0)}\n")
                    f.write(f"      - structured: {struct_cnt}/{targets.get('structured', 0)}\n")
                    f.write(f"      - constraint_heavy: {cons_cnt}/{targets.get('constraint_heavy', 0)}\n")
                    cap_note = f"{selection_manager.rankcat_cap_effective}"
                    if selection_manager.rankcat_cap_relaxed:
                        cap_note += " (relaxed to 0 for fill)"
                    f.write(f"      - rank_category cap: {cap_note}\n")
                
                # 入选原因分布
                if 'selection_reason' in selected_df.columns:
                    reason_counts = selected_df['selection_reason'].value_counts()
                    f.write(f"\n    入选原因分布:\n")
                    for reason, count in reason_counts.items():
                        f.write(f"      - {reason}: {count} queries\n")

                # 优先使用rank_category（子分类）统计
                if 'rank_category' in selected_df.columns and selected_df['rank_category'].notna().any():
                    category_selection = selected_df['rank_category'].value_counts().sort_values(ascending=False)
                    f.write(f"    按子类别统计:\n")
                    for rank_cat, count in category_selection.items():
                        if pd.notna(rank_cat):  # 跳过NaN值
                            f.write(f"      - {rank_cat}: {count} queries\n")
                
                # 同时显示原始类别统计作为对比
                if COL_CATEGORY in selected_df.columns:
                    original_category_selection = selected_df[COL_CATEGORY].value_counts().sort_values(ascending=False)
                    f.write(f"    按原始类别统计:\n")
                    for orig_cat, count in original_category_selection.items():
                        if pd.notna(orig_cat):  # 跳过NaN值
                            f.write(f"      - {orig_cat}: {count} queries\n")
            
            
            # 4. 性能统计
            f.write(f"\n性能统计:\n")
            total_time = time.time() - start_time
            f.write(f"  总耗时: {total_time:.1f} 秒\n")
            f.write(f"  处理速度: {total_queries/total_time:.1f} queries/秒\n")
            
            # 粗估 tokens 使用
            relevance_tokens = len(df_details) * 50
            classification_tokens = relevant_df.shape[0] * 100 if not relevant_df.empty else 0
            scoring_tokens = len(relevant_df) * 150  # Estimate based on relevant queries
            dedup_tokens = relevant_df.shape[0] * 30 if not relevant_df.empty else 0
            
            total_estimated_tokens = relevance_tokens + classification_tokens + scoring_tokens + dedup_tokens
            f.write(f"  估算Token使用: {total_estimated_tokens:,} tokens\n")
            f.write(f"    - 相关性检查: {relevance_tokens:,}\n")
            f.write(f"    - 分类: {classification_tokens:,}\n")
            f.write(f"    - 评分: {scoring_tokens:,}\n")
            f.write(f"    - 去重: {dedup_tokens:,}\n")
            
            # 5. 数据质量指标
            f.write(f"\n数据质量指标:\n")
            
            # 重复率
            duplicate_count = (df_details['query_status'] == QUERY_STATUS_DUPLICATE).sum()
            if total_queries > 0:
                duplicate_rate = duplicate_count / total_queries * 100
                f.write(f"  重复率: {duplicate_rate:.1f}% ({duplicate_count:,}/{total_queries:,})\n")
            
            # 相关率
            if total_queries > 0:
                relevant_rate = len(relevant_df) / total_queries * 100
                f.write(f"  相关率: {relevant_rate:.1f}% ({len(relevant_df):,}/{total_queries:,})\n")
            
            # 分类覆盖率
            if not relevant_df.empty:
                classified_count = relevant_df[COL_CATEGORY].notna().sum()
                classification_rate = classified_count / len(relevant_df) * 100
                f.write(f"  分类覆盖率: {classification_rate:.1f}% ({classified_count:,}/{len(relevant_df):,})\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("评测集生成流程已完成！\n")
            f.write("=" * 80 + "\n")
            
        print(f"报告已保存到: {filename}")
        
    except Exception as e:
        logging.error(f"Failed to save pipeline summary: {e}")

class ODPSManager:
    """Manages ODPS operations"""
    
    def __init__(self, writer, table_names: dict = None, overwrite: bool = False):
        self.writer = writer
        self.table_names = table_names or get_table_names("formal")
        self.overwrite = overwrite
    
    def save_results(self, df_details: pd.DataFrame, df_eval: pd.DataFrame,
                    partition: str, set_id: str, game_name: str, **metadata):
        """Save all results to ODPS"""
        if df_details is not None and not df_details.empty:
            self._save_preprocess_details(df_details, partition, set_id)
            self._save_query_items(df_eval, partition, set_id, game_name)
        
        self._save_query_set_info(partition, set_id, game_name, 
                                 len(df_details) if df_details is not None else 0,
                                 len(df_eval) if df_eval is not None else 0,
                                 **metadata)

    def _save_preprocess_details(self, df: pd.DataFrame, partition: str, set_id: str):
        """Save preprocessing details"""
        base_columns = [COL_QUERY_ID, COL_QUERY, COL_GAME_NAME, 'query_status',
                        COL_CANONICAL_QUERY_ID, COL_CATEGORY, 'rank_category', 'represent_rank', 'selected_for_eval']
        optional_columns = [COL_QUERY_TIME, 'scoring_details']
        columns = base_columns + [col for col in optional_columns if col in df.columns]

        # 准备保存的DataFrame
        df_save = df.loc[:, [col for col in columns if col in df.columns]].copy()
        df_save['set_id'] = set_id
        
        # Fill defaults
        df_save['represent_rank'] = df_save['represent_rank'].fillna(-1)
        df_save['selected_for_eval'] = df_save['selected_for_eval'].fillna(False).astype(bool)
        
        # Ensure canonical_query_id has fallback
        if COL_CANONICAL_QUERY_ID in df_save.columns and COL_QUERY_ID in df_save.columns:
            df_save[COL_CANONICAL_QUERY_ID] = df_save[COL_CANONICAL_QUERY_ID].fillna(df_save[COL_QUERY_ID])

        insert_dataframe(
            self.writer,
            df_save,
            self.table_names["QUERY_PREPROCESS_DETAILS_TABLE_NAME"],
            partition=partition,
            overwrite=self.overwrite,
            set_id=set_id
        )
    
    def _save_query_items(self, df: pd.DataFrame, partition: str, set_id: str, game_name: str):
        """Save query items"""
        if df is None or df.empty:
            return

        base_columns = [COL_QUERY_ID, COL_QUERY, COL_CATEGORY]
        columns = base_columns + ([COL_QUERY_TIME] if COL_QUERY_TIME in df.columns else [])

        # 准备保存的DataFrame
        df_save = df.loc[:, [col for col in columns if col in df.columns]].copy()
        df_save['set_id'] = set_id
        df_save['game_name'] = game_name
        df_save['is_golden'] = 0

        insert_dataframe(
            self.writer,
            df_save,
            self.table_names["QUERY_ITEM_TABLE_NAME"],
            partition=partition,
            overwrite=self.overwrite,
            set_id=set_id
        )
    
    def _save_query_set_info(self, partition: str, set_id: str, game_name: str,
                            total_processed: int, eval_size: int, **metadata):
        """Save query set metadata"""
        df_save = pd.DataFrame([{
            "set_id": set_id,
            "game_name": game_name,
            "run_suffix": metadata.get('run_suffix', ''),
            "created_at": int(time.time()),
            "total_queries_processed": total_processed,
            "final_eval_set_size": eval_size
        }])
        
        insert_dataframe(
            self.writer,
            df_save,
            self.table_names["QUERY_SET_TABLE_NAME"],
            partition=partition,
            overwrite=self.overwrite,
            set_id=set_id
        )


# ========================================
# HELPER FUNCTIONS
# ========================================

def ensure_category_fallback(df: pd.DataFrame, query_mask: pd.Series, 
                            stage_name: str = "Processing") -> None:
    """统一处理category兜底逻辑，避免重复代码
    
    Args:
        df: DataFrame to update in place
        query_mask: Boolean mask for queries to check
        stage_name: Stage name for logging
    """
    if not query_mask.any():
        return
    
    # 处理 CATEGORY_CLASSIFICATION_FAILED
    failed_mask = query_mask & (df[COL_CATEGORY] == CATEGORY_CLASSIFICATION_FAILED)
    if failed_mask.any():
        failed_count = failed_mask.sum()
        logging.warning(
            f"[{stage_name}] {failed_count} queries with CATEGORY_CLASSIFICATION_FAILED; assigning '{CATEGORY_OTHER}'"
        )
        df.loc[failed_mask, COL_CATEGORY] = CATEGORY_OTHER
    
    # 处理缺失或空字符串的类别
    categories = df.loc[query_mask, COL_CATEGORY]
    missing_mask = categories.fillna("").astype(str).str.strip() == ""
    if missing_mask.any():
        missing_indices = categories.index[missing_mask]
        logging.warning(
            f"[{stage_name}] {len(missing_indices)} queries lack category assignments; defaulting to '{CATEGORY_OTHER}'"
        )
        df.loc[missing_indices, COL_CATEGORY] = CATEGORY_OTHER


# ========================================
# MAIN PIPELINE
# ========================================

async def process_game(client: openai.AsyncOpenAI, df_game: pd.DataFrame, 
                      game_name: str, eval_size: int, odps_writer, 
                      partition_str: str, non_interactive: bool, 
                      run_suffix: str, table_names: dict = None, overwrite: bool = False) -> None:
    """Process a single game through all stages"""
    
    set_id = generate_set_id()
    start_time = time.time()  # 记录开始时间用于性能统计
    
    print(f"\nProcessing: {game_name} (Set ID: {set_id})")
    
    # Add set_id to all rows
    df_game['set_id'] = set_id
    
    # Initialize processors
    query_processor = QueryProcessor(client)
    category_manager = CategoryManager(client)
    selection_manager = SelectionManager(client)
    odps_manager = ODPSManager(odps_writer, table_names, overwrite)
    
    # Stage 1: Preprocessing
    print("Stage 1: Preprocessing...")
    df_game['query_status'] = QUERY_STATUS_RELEVANT
    df_game[COL_CANONICAL_QUERY_ID] = pd.NA
    
    # Rename dt column to query_time if it exists
    if 'dt' in df_game.columns:
        df_game[COL_QUERY_TIME] = df_game['dt']
        print(f"Preserved query_time (dt) column with {df_game[COL_QUERY_TIME].notna().sum()} non-null values")
    
    # 优化：先移除完全相同的查询文本，减少后续处理量
    original_count = len(df_game)
    df_game = df_game.drop_duplicates(subset=[COL_QUERY], keep='first')
    if len(df_game) < original_count:
        print(f"Removed {original_count - len(df_game)} exact duplicate queries")
    
    df_game = await query_processor.check_relevance(df_game, game_name)
    # 先改写，提升可读性和后续去重/分类质量
    df_game = await query_processor.rewrite_queries(df_game, game_name)
    df_game = await query_processor.detect_duplicates(df_game, game_name)
    
    categories = await category_manager.discover_categories(df_game, game_name)
    
    # Stage 2: Classification
    print("Stage 2: Classification...")
    
    # Initialize category column
    df_game[COL_CATEGORY] = pd.NA
    
    # Get relevant queries mask
    relevant_mask = df_game['query_status'] == QUERY_STATUS_RELEVANT
    
    if not relevant_mask.any():
        print(f"No relevant queries found for {game_name}")
        return
    
    # Classify only relevant queries and directly update main DataFrame
    df_relevant_subset = df_game[relevant_mask]  # No need to copy, classify_queries doesn't modify input
    df_classified_relevant = await category_manager.classify_queries(df_relevant_subset, categories, game_name)
    
    # Directly update categories in main DataFrame using map
    if COL_CATEGORY in df_classified_relevant.columns:
        category_mapping = df_classified_relevant.set_index(COL_QUERY_ID)[COL_CATEGORY]
        df_game.loc[relevant_mask, COL_CATEGORY] = df_game.loc[relevant_mask, COL_QUERY_ID].map(category_mapping)
        
        classified_count = df_game[COL_CATEGORY].notna().sum()
        print(f"Classified {classified_count} queries into categories")

    # 统一处理category兜底逻辑
    relevant_mask = df_game['query_status'] == QUERY_STATUS_RELEVANT
    ensure_category_fallback(df_game, relevant_mask, "Classification")

    # Stage 3: Selection
    print("Stage 3: Selection...")
    
    # Get exclusion suggestions - use relevant queries for category analysis
    relevant_queries = df_game[df_game['query_status'] == QUERY_STATUS_RELEVANT]
    unique_cats = relevant_queries[COL_CATEGORY].dropna().unique().tolist()
    
    exclusions = await category_manager.get_exclusion_suggestions(unique_cats, game_name)
    
    # Interactive selection (or auto if non_interactive)
    if non_interactive:
        excluded_cats = exclusions
    else:
        excluded_cats = await prompt_for_excluded_categories(df_game, exclusions, non_interactive)
    
    
    # Mark excluded categories directly in main DataFrame
    if excluded_cats:
        mask = df_game[COL_CATEGORY].isin(excluded_cats)
        exclude_count = mask.sum()
        
        df_game.loc[mask, 'query_status'] = QUERY_STATUS_EXCLUDE
        # Note: Exclude queries should KEEP their original category, not clear it
    
    # Rank and select
    # Rank first (heavy operation)
    ranked_df = await selection_manager.rank_queries(df_game)
    # Then select
    df_eval_subset, df_details_full = await selection_manager.select_queries(ranked_df, eval_size, df_game)
    
    # === REVIEW STAGE ===
    if not non_interactive:
        while True:
            # 1. Automated Rule Check
            print("\n" + "="*50)
            print("AUTOMATED CHECK REPORT")
            print("="*50)
            
            check_passed = True
            
            # Check 1: No duplicates (should be handled, but verify)
            dups_in_selection = df_details_full[df_details_full['selected_for_eval'] & (df_details_full['query_status'] == QUERY_STATUS_DUPLICATE)]
            if not dups_in_selection.empty:
                print(f"[FAIL] Found {len(dups_in_selection)} queries marked as DUPLICATE in selection!")
                check_passed = False
            else:
                print("[PASS] No duplicate status queries selected.")

            # Check 2: No strange categories (e.g., NO_CLASSIFICATION or excluded)
            bad_cats = [CATEGORY_NO_CLASSIFICATION, CATEGORY_CLASSIFICATION_FAILED] + (excluded_cats if excluded_cats else [])
            bad_cat_mask = df_details_full['selected_for_eval'] & df_details_full[COL_CATEGORY].isin(bad_cats)
            bad_cat_queries = df_details_full[bad_cat_mask]
            
            if not bad_cat_queries.empty:
                print(f"[FAIL] Found {len(bad_cat_queries)} queries with invalid categories in selection:")
                print(bad_cat_queries[[COL_QUERY, COL_CATEGORY]].to_string())
                check_passed = False
            else:
                print("[PASS] No invalid category queries selected.")
            
            print("-" * 50)
            
            # 2. Human Review
            print("\n" + "="*50)
            print(f"HUMAN REVIEW (Selected {len(df_eval_subset)} queries)")
            print("="*50)
            
            # Prepare display dataframe
            display_cols = [COL_QUERY_ID, COL_QUERY, COL_CATEGORY, 'represent_rank', 'selection_reason']
            # handle case where selection_reason might not exist in df_eval_subset if run previously
            if 'selection_reason' not in df_eval_subset.columns:
                 df_eval_subset['selection_reason'] = "Unknown"
                 
            df_display = df_eval_subset[display_cols].copy()
            df_display = df_display.reset_index(drop=True)
            
            # Print row by row for better readability
            for idx, row in df_display.iterrows():
                print(f"[{row[COL_QUERY_ID]}] {row[COL_CATEGORY]}")
                print(f"    Reason: {row['selection_reason']}")
                print(f"    Q: {row[COL_QUERY]}")
                print("-" * 30)
                
            print("\nPlease review the selected queries above.")
            print("If you want to REJECT any queries, enter their IDs (comma separated).")
            print("If all look good, just press Enter to proceed.")
            
            reject_input = input("IDs to reject: ").strip()
            
            if not reject_input:
                print("Review passed. Proceeding...")
                break
                
            # Process rejections
            rejected_ids = [x.strip() for x in reject_input.split(',')]
            valid_rejected_ids = [rid for rid in rejected_ids if rid in df_details_full[COL_QUERY_ID].astype(str).values]
            
            if not valid_rejected_ids:
                print("No valid IDs provided. Proceeding...")
                break
                
            print(f"Rejecting {len(valid_rejected_ids)} queries: {valid_rejected_ids}")
            
            # Mark rejected queries as EXCLUDED manually
            # We use a special status or just EXCLUDE, but need to make sure they aren't picked again
            # Let's map them to QUERY_STATUS_EXCLUDE
            mask_reject = df_game[COL_QUERY_ID].astype(str).isin(valid_rejected_ids)
            df_game.loc[mask_reject, 'query_status'] = QUERY_STATUS_EXCLUDE
            
            # Important: Clear 'selected_for_eval' flag in df_game to force re-selection
            df_game.loc[mask_reject, 'selected_for_eval'] = False
            
            # Also update ranked_df to reflect exclusion so select_queries knows not to pick them
            # ranked_df contains only relevant queries, so we need to filter
            if not ranked_df.empty:
                # Update selected_for_eval to False for rejected
                ranked_df_mask_reject = ranked_df[COL_QUERY_ID].astype(str).isin(valid_rejected_ids)
                ranked_df.loc[ranked_df_mask_reject, 'selected_for_eval'] = False
                
                # We also need to make sure they aren't picked again.
                # _select_by_allocation uses rankcat_counts based on selected_ids (which we just cleared)
                # But it also picks from candidates. We need to ensure rejected ones are NOT candidates.
                # One way is to drop them from ranked_df, or add an 'excluded' flag.
                # Simpler: remove them from ranked_df entirely for the purpose of selection
                ranked_df = ranked_df[~ranked_df[COL_QUERY_ID].astype(str).isin(valid_rejected_ids)].copy()
            
            # Re-run selection (fast, no re-ranking)
            print("Re-running selection to fill gaps (using cached scores)...")
            df_eval_subset, df_details_full = await selection_manager.select_queries(ranked_df, eval_size, df_game)
    
    # df_details_full already contains all queries with correct ranking and selection information
    # since it was generated from df_game which has all the updates
    df_details = df_details_full
    
    
    # Final category cleanup for non-relevant queries
    # Note: EXCLUDE queries should keep their original category, only irrelevant/duplicate get NO_CLASSIFICATION
    final_irrelevant_mask = df_details['query_status'].isin([
        QUERY_STATUS_IRRELEVANT,
        QUERY_STATUS_DUPLICATE
    ])
    df_details.loc[final_irrelevant_mask, COL_CATEGORY] = CATEGORY_NO_CLASSIFICATION

    # For relevant and excluded queries, 确保类别兜底到"其他"
    processable_mask = df_details['query_status'].isin([QUERY_STATUS_RELEVANT, QUERY_STATUS_EXCLUDE])
    ensure_category_fallback(df_details, processable_mask, "Selection")
    
    
    # Save to ODPS
    odps_manager.save_results(
        df_details, df_eval_subset, partition_str, set_id, game_name,
        run_suffix=run_suffix
    )
    
    # Print comprehensive pipeline summary
    save_pipeline_summary(df_details, selection_manager, game_name, eval_size, start_time, set_id)

async def prompt_for_excluded_categories(df: pd.DataFrame, suggestions: List[str], 
                                        non_interactive: bool) -> List[str]:
    """Prompt user to select categories to exclude"""
    if non_interactive:
        return suggestions
    
    print("\n--- Category Selection ---")
    
    df_relevant = df[df['query_status'] == QUERY_STATUS_RELEVANT]
    
    if df_relevant.empty:
        print("No relevant queries found for category selection.")
        return []
    
    # Get unique categories and their counts, sorted by count descending
    category_counts = df_relevant[COL_CATEGORY].value_counts().reset_index()
    category_counts.columns = [COL_CATEGORY, 'count']
    
    if category_counts.empty:
        print("No categories found to select from.")
        return []

    print("Please review the discovered categories and their samples.")
    print("You will be asked which categories to EXCLUDE from the evaluation set.\n")

    # Store categories to map input numbers back to names
    indexed_cats = category_counts[COL_CATEGORY].tolist()
    
    for index, row in category_counts.iterrows():
        category_name = row[COL_CATEGORY]
        count = row['count']
        print(f"[{index + 1}] {category_name} ({count} queries)")
        
        # Get up to 3 random samples from the category
        sample_df = df_relevant[df_relevant[COL_CATEGORY] == category_name].sample(min(FILTER_SELECT_CATEGORY_SAMPLE_SIZE, count))
        for _, sample_row in sample_df.iterrows():
            print(f"    - Sample: \"{sample_row[COL_QUERY]}\"")
        print("-" * 20)
    
    # Convert suggested category names to indices
    auto_excluded_indices = []
    
    # Force exclude "Other" / "无效" related categories
    force_exclude_keywords = ["其他", "other", "无效", "unknown", "none"]
    
    for i, cat_name in enumerate(indexed_cats):
        if cat_name in suggestions:
            auto_excluded_indices.append(i + 1)
        # Check for force exclude keywords
        elif any(k in cat_name.lower() for k in force_exclude_keywords):
             if (i + 1) not in auto_excluded_indices:
                 auto_excluded_indices.append(i + 1)
                 # Add to suggestions list for display/return
                 if cat_name not in suggestions:
                     suggestions.append(cat_name)
    
    auto_excluded_str = ", ".join(map(str, sorted(auto_excluded_indices)))

    if non_interactive:

        excluded_indices = auto_excluded_indices
    else:
        while True:
            try:
                user_input = input(f"\nEnter the NUMBERS of categories to exclude, separated by commas (e.g., 2, 9),\nor press Enter to accept LLM suggestion [{auto_excluded_str}]: ")
                if not user_input.strip():

                    excluded_indices = auto_excluded_indices
                    break
                
                excluded_indices = [int(i.strip()) for i in user_input.split(',')]
                break  # Exit loop if input is valid
            except ValueError:
                pass

    # Process the final list of excluded indices
    excluded_cats = []
    valid_input = True
    for i in excluded_indices:
        idx = i - 1
        if 0 <= idx < len(indexed_cats):
            excluded_cats.append(indexed_cats[idx])
        else:

            valid_input = False
            break
            
    if valid_input:
        print(f"Excluding categories: {excluded_cats}\n")
        return excluded_cats
    else:
        return []

async def generate_eval_set(game_name: Optional[str], eval_size: int, 
                          dm_partition: str, non_interactive: bool = False, 
                          mode: str = "formal", overwrite: bool = False):
    """Main entry point for eval set generation"""
    
    # Setup
    setup_logging()
    client, odps_reader, odps_writer = initialize_clients()
    
    # Get table names based on mode
    table_names = get_table_names(mode)
    logging.info(f"Running in {mode} mode, using tables: {list(table_names.values())}")
    
    run_suffix = "_" + time.strftime(DEFAULT_TIMESTAMP_FORMAT)
    
    # Fetch data
    df = await read_rows_with_condition(
        odps_reader,
        table_name=SOURCE_TABLE,
        partition_spec=f"dm='{dm_partition}'",
        where_clause=None,
        limit=20000
    )
    
    if df is None or df.empty:
        print("No data fetched from ODPS")
        return
    
    # Determine games to process
    if game_name:
        # Command line game_name parameter has highest priority
        df = df[df[COL_GAME_NAME] == game_name]
        games = [game_name] if not df.empty else []

    elif TARGET_GAMES:
        # Use TARGET_GAMES list if no specific game_name provided
        games = [g for g in TARGET_GAMES if g in df[COL_GAME_NAME].unique()]
        df = df[df[COL_GAME_NAME].isin(games)]

    else:
        # Process all games if neither game_name nor TARGET_GAMES specified
        games = df[COL_GAME_NAME].unique().tolist()

    
    print(f"Processing {len(games)} games: {games}")
    
    # Process each game
    for game in games:
        df_game = df[df[COL_GAME_NAME] == game].copy()
        if df_game.empty:
            continue
        
        game_id = df_game['game_id'].iloc[0] if 'game_id' in df_game.columns else -1
        partition_str = f"dm='{dm_partition}',game_id={game_id}"
        
        await process_game(client, df_game, game, eval_size, odps_writer,
                         partition_str, non_interactive, run_suffix, table_names, overwrite)
    
    print("\nEval set generation complete")

# ========================================
# CLI
# ========================================

async def main():
    parser = argparse.ArgumentParser(description="Generate evaluation sets (Stages 1-3)")
    parser.add_argument("--game_name", type=str, help="Specific game to process")
    parser.add_argument("--eval_size", type=int, default=DEFAULT_EVAL_SIZE, help="Target evaluation set size")
    parser.add_argument("--dm_partition", type=str, required=True, help="DM partition (e.g., '2025-06')")
    
    # Interactive mode options
    interactive_group = parser.add_mutually_exclusive_group()
    interactive_group.add_argument("--interactive", action="store_true", help="Run interactively (prompt for user input)")
    interactive_group.add_argument("--non_interactive", action="store_true", help="Run non-interactively (use defaults)")
    
    parser.add_argument("--mode", type=str, choices=["formal", "test"], default="test", help="Run mode: 'formal' for production tables, 'test' for test tables")
    
    # Overwrite mode options  
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true", help="Enable overwrite mode for ODPS tables")
    overwrite_group.add_argument("--insert", action="store_true", help="Use insert mode for ODPS tables (no overwrite)")
    
    args = parser.parse_args()
    
    # Determine interactive mode (default to interactive if neither specified)
    if args.non_interactive:
        non_interactive = True
    elif args.interactive:
        non_interactive = False
    else:
        non_interactive = False  # Default to interactive mode
    
    # Determine overwrite mode (default to insert mode if neither specified)  
    if args.overwrite:
        overwrite = True
    elif args.insert:
        overwrite = False
    else:
        overwrite = False  # Default to insert mode
    
    await generate_eval_set(args.game_name, args.eval_size, args.dm_partition, non_interactive, args.mode, overwrite)

if __name__ == "__main__":
    asyncio.run(main())