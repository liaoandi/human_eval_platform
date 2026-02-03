# -*- coding: utf-8 -*-
"""
Anti-Cheat Question Generator: 整合版

用 **OpenAI GPT-4o（自动搜索功能）** 生成「每题2个答案（1真1假）」的游戏常识反作弊题，
并写入两张 ODPS 表：query & llm_answer。

功能特点：
- GPT-4o 自动搜索游戏的最新信息，确保准确性
- 基于实时搜索结果生成准确的反作弊题目
- 支持打印模式，可预览结果而不写入ODPS
- 支持批量处理多个游戏

题目格式：
- 题干：请仔细阅读以下两个回答，并按照要求做出对应选择。[游戏问题]
- 2个答案：1个正确答案，1个错误答案
- 选择指示随机化：答案可以指向"模型A"、"模型B"
- 正确答案和错误答案必须指向不同的选项

表结构（与pipeline_commons标准表对齐）：
- query_item(query_id BIGINT, set_id BIGINT, game_id BIGINT, game_name STRING, 
            raw_query STRING, category STRING, is_golden BIGINT, query_time STRING) PARTITIONED BY (dm STRING)
- llm_answer(answer_id BIGINT, set_id BIGINT, query_id BIGINT, model_name STRING, 
            game_name STRING, answer_content STRING, generation_metadata STRING, 
            generated_at BIGINT) PARTITIONED BY (dm STRING, game_id BIGINT, model_id BIGINT)
"""

import argparse
import asyncio
import datetime
import importlib
import json
import logging
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional
import pandas as pd

# 默认配置
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.3
DEFAULT_CATEGORY = "game_knowledge"
DEFAULT_IS_GOLDEN = 1
DEFAULT_MODEL_ID = 0
DEFAULT_MODEL_NAME = "golden"

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

# 导入pipeline_commons的功能
from pipeline_commons import (
    initialize_clients, insert_dataframe, setup_logging, get_table_names, read_rows_with_condition
)

# 导入新的answer_id生成函数
from answer_collector_concurrent_v2 import make_answer_id

# ---------------------------
# 游戏信息获取函数 - 从ODPS表中读取真实数据
# ---------------------------

async def get_game_info(odps_reader, table_names: dict, dm_partition: str, game_name: Optional[str] = None) -> List[Dict]:
    """从现有评测集中获取游戏信息

    Args:
        odps_reader: ODPS读取客户端
        table_names: 表名字典
        dm_partition: DM分区
        game_name: 可选的游戏名称过滤

    Returns:
        List[Dict]: 包含game_id, game_name, set_id等信息的记录列表
    """
    # 构建查询条件
    where_clause = None
    if game_name:
        where_clause = f"game_name='{game_name}'"
    
    # 从query_set表中读取游戏元数据
    df_game_data = await read_rows_with_condition(
        odps_reader,
        table_name=table_names["QUERY_SET_TABLE_NAME"],
        partition_spec=f"dm='{dm_partition}'",
        where_clause=where_clause,
        limit=None
    )
    
    if df_game_data.empty:
        if game_name:
            available_msg = f"game_name='{game_name}'"
        else:
            available_msg = "任何游戏"
        raise ValueError(f"未找到 {available_msg} 的数据在分区 dm='{dm_partition}' 中")
    
    # 提取游戏信息
    game_info_list = []
    
    if 'game_id' in df_game_data.columns:
        dedup_cols = ['game_id', 'game_name'] if 'game_name' in df_game_data.columns else ['game_id']

        if 'created_at' in df_game_data.columns:
            df_sorted = df_game_data.sort_values('created_at', ascending=False)
            df_filtered = df_sorted.drop_duplicates(subset=dedup_cols, keep='first')
        else:
            df_filtered = df_game_data.drop_duplicates(subset=dedup_cols)

        for _, row in df_filtered.iterrows():
            game_info_list.append({
                'game_id': row['game_id'],
                'game_name': row['game_name'],
                'set_id': row.get('set_id')
            })
    
    return game_info_list

async def get_max_query_id_from_tables(odps_reader, table_names: dict, dm_partition: str) -> int:
    """返回 query_item 表在指定分区内的最大 query_id，若不存在则返回 0"""
    query_table = table_names["QUERY_ITEM_TABLE_NAME"]
    max_query_sql = f"""
    SELECT MAX(query_id) AS max_query_id
    FROM {query_table}
    WHERE dm = '{dm_partition}'
    """

    instance = odps_reader.execute_sql(max_query_sql)
    instance.wait_for_success()

    with instance.open_reader() as reader:
        record = next(reader)
        max_query_id = getattr(record, "max_query_id", None)

    if max_query_id is None:
        logging.warning("query_item 表未找到现有数据，将从 0 开始分配 query_id")
        return 0

    return int(max_query_id)

# ---------------------------
# 工具函数
# ---------------------------


def _get_month_end_date(dm_partition: str) -> str:
    """根据dm分区计算该月最后一天的日期字符串
    
    Args:
        dm_partition: 格式如 '2025-07'
        
    Returns:
        str: 该月最后一天的日期，格式如 '2025-07-31'
    """
    try:
        # 解析dm分区
        year, month = map(int, dm_partition.split('-'))
        
        # 计算下个月的第一天
        if month == 12:
            next_month = datetime.date(year + 1, 1, 1)
        else:
            next_month = datetime.date(year, month + 1, 1)
        
        # 当月最后一天
        last_day_of_month = next_month - datetime.timedelta(days=1)
        
        # 返回YYYY-MM-DD格式
        return last_day_of_month.strftime('%Y-%m-%d')
    except Exception as e:
        logging.warning(f"解析dm分区 '{dm_partition}' 失败: {e}，使用今天日期")
        return datetime.date.today().strftime('%Y-%m-%d')

def strip_json(text: str) -> str:
    """从文本中提取JSON部分"""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
        t = re.sub(r"\s*```$", "", t)
    fi, la = t.find("{"), t.rfind("}")
    if fi != -1 and la != -1 and la > fi:
        return t[fi:la+1]
    return t


# ---------------------------
# 常量定义
# ---------------------------

STANDARD_QUESTION_TEXT = "请仔细阅读以下两个表述，并按照正确表述指定的模型选项（模型A、模型B）作答。"

# ---------------------------
# LLM：OpenAI gpt-4o
# ---------------------------

class OpenAI2Choice:
    """集成了pipeline_commons的异步OpenAI调用器 - 生成反作弊题目（1真1假格式）"""
    def __init__(self, aclient, model: str = OPENAI_MODEL, temperature: float = OPENAI_TEMPERATURE):
        self.aclient = aclient  # 使用pipeline_commons初始化的异步客户端
        self.model = model
        self.temperature = temperature

    def prompt(self, game_name: str, dm_partition: str) -> str:
        # 解析时间信息
        time_context = ""
        try:
            if dm_partition and len(dm_partition) >= 7:  # 格式如 "2025-06"
                year, month = dm_partition.split('-')
                time_context = f"\n⏰ 时间背景：请基于{year}年{month}月的游戏状态生成题目，确保内容符合该时间点的游戏版本和可用内容。"
        except:
            pass
            
        
        return f"""
你是严谨的出题员。请为《{game_name}》生成若干道"游戏常识反作弊题"。
题目格式要求：
1. 每道题的题干必须且仅为："{STANDARD_QUESTION_TEXT}"。
2. 每道题恰好有 **2 个答案**：1 个正确答案，1 个错误答案。
3. 每个答案都是关于游戏事实的完整独立陈述，【不要包含任何模型选项提示】（不要出现“模型A/模型B/左边/右边”等字样，也不要写“如果…请选择…”）。
4. 脚本会在后处理阶段自动为左右两个表述追加固定的选择提示：左边对应模型A、右边对应模型B，因此输出里不需要也不允许你写选择指示语。{time_context}

质量要求（至关重要）：
- **核心事实原则**：只使用**绝对无争议**的基础事实（如：游戏核心玩法类型、主要货币名称、标志性NPC名字、基础操作方式）。
- **避免主观描述**：绝对不要描述地图的大小（"很大/很小"）、战术风格（"适合近战"）或数值强度，因为这些是主观的。
- **避免特定技能细节**：不要描述具体角色的技能细节（如"透视"、"增加射程"），因为容易产生歧义或版本变动。
- **避免“不存在”陷阱**：尽量避免使用“游戏没有功能X”作为正确/错误答案，除非该功能在同类游戏中极罕见。因为功能可能存在但改了名字（例如将“公会”称为“战队”或“小镇”）。
- **术语准确**：必须使用游戏内的官方中文术语（例如：如果游戏内称为"货币A"，就不要叫"金币"）。

示例范畴：
- 正确：关于游戏内角色关系的无争议事实
- 错误：与游戏世界观冲突的明显错误
- 避免：主观或不准确的描述

**输出严格 JSON，且仅输出 JSON**：
{{
  "items": [
    {{
      "question": "{STANDARD_QUESTION_TEXT}",
      "answers": [
        {{"text": "在《{game_name}》中，[关于游戏某个方面的绝对正确事实]", "correct": true}},
        {{"text": "在《{game_name}》中，[关于游戏某个方面的明显错误事实]", "correct": false}}
      ]
    }}
  ]
}}
""".strip()

    async def generate_batch(self, game_name: str, dm_partition: str, n: int, max_retries: int = 3) -> List[Dict[str, Any]]:
        attempts = 0

        while attempts < max_retries:
            attempts += 1

            system_prompt = (
                "You are a careful test item writer for anti-cheat questions. "
                "Search for current information about the game before generating questions. Only return strict JSON."
            )
            user_prompt = self.prompt(game_name, dm_partition) + (
                f"\n\n请一次性生成 {n} 道题并返回到 JSON 的 items 中。"
            )

            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "max_tokens": 2048
            }

            try:
                resp = await self.aclient.chat.completions.create(**api_params)
                txt = resp.choices[0].message.content
                data = json.loads(strip_json(txt) or "{}")
            except Exception as exc:
                logging.warning(f"生成题目失败（第{attempts}次）: {exc}")
                continue

            items = data.get("items")
            if not isinstance(items, list):
                logging.warning("响应格式不正确: items 不是列表，忽略本次结果")
                continue

            normalized_items: List[Dict[str, Any]] = []
            for idx, it in enumerate(items, start=1):
                question = str(it.get("question", "")).strip()
                answers = it.get("answers")
                
                # 验证题目格式
                if question != STANDARD_QUESTION_TEXT:
                    logging.warning(f"题目 {idx} 被跳过：题干不正确 (期望: '{STANDARD_QUESTION_TEXT}', 实际: '{question[:50]}...')")
                    continue
                
                if not isinstance(answers, list):
                    logging.warning(f"题目 {idx} 被跳过：answers 不是列表")
                    continue
                    
                if len(answers) != 2:
                    logging.warning(f"题目 {idx} 被跳过：答案数量不是2个 (实际: {len(answers)})")
                    continue

                # 验证正确答案数量
                num_correct = sum(1 for ans in answers if bool(ans.get("correct")))
                if num_correct != 1:
                    logging.warning(f"题目 {idx} 被跳过：正确答案数量不是1个 (实际: {num_correct})")
                    continue

                # ✅ 位置绑定：左=模型A，右=模型B；仅随机左右真伪，避免“反选”造成认知冲突
                #    - 左侧表述永远对应模型A文本框
                #    - 右侧表述永远对应模型B文本框
                #    - 仅随机“真/假表述”出现在左/右，避免用户形成“总选左边”的投机策略
                answers_shuffled = list(answers)
                random.shuffle(answers_shuffled)

                # 标准化答案（先剥离模型可能输出的选择提示，再统一按左右位置追加）
                normalized_answers: List[Dict[str, Any]] = []
                for pos, ans in enumerate(answers_shuffled):
                    text = str(ans.get("text", "")).strip()
                    is_correct = bool(ans.get("correct"))

                    if not text:
                        logging.warning(f"题目 {idx} 的某个答案为空，跳过整道题")
                        normalized_answers = []
                        break

                    # 清理：移除模型可能生成的选择指示（旧格式/新格式），统一由后处理追加
                    # 兼容示例：
                    # - “如果本条表述正确，请全部选择**模型A**”
                    # - “如果出现在左边的表述正确，请全部选择**模型A**”
                    text = re.sub(
                        r"(。)?\s*如果.*?表述正确.*?请全部选择\*\*[^*]+\*\*\s*$",
                        "",
                        text
                    ).strip()

                    # 位置绑定：左边=模型A，右边=模型B
                    if pos == 0:
                        model_label = "模型A"
                        side_hint = "左边"
                    else:
                        model_label = "模型B"
                        side_hint = "右边"

                    # 修改：描述文字中不要出现左边/右边，以适应前端可能的swap
                    text = text.rstrip("。，") + f"。如果本条表述正确，请全部选择**{model_label}**"
                    normalized_answers.append({
                        "text": text,
                        "correct": is_correct,
                        "model_choice": model_label,
                        "side": side_hint
                    })

                if len(normalized_answers) == 2:
                    normalized_items.append({
                        "question": STANDARD_QUESTION_TEXT,
                        "answers": normalized_answers
                    })
                    logging.debug(f"题目 {idx} 验证通过")

            if normalized_items:
                valid_count = len(normalized_items)
                requested_count = n
                if valid_count < requested_count:
                    logging.warning(
                        f"题目生成不足：请求 {requested_count} 道题，但只有 {valid_count} 道通过验证。"
                        f"建议检查 GPT 生成质量或放宽验证条件。"
                    )
                else:
                    logging.info(f"成功生成 {valid_count} 道题目（请求: {requested_count}）")
                return normalized_items[:n]

        logging.error(f"生成失败：连续 {max_retries} 次尝试后，没有生成任何有效题目")
        return []

# ---------------------------
# ODPS 表管理函数（使用pipeline_commons）
# ---------------------------

def ensure_anti_cheat_tables(odps_client, query_table: str, answer_table: str):
    """确保标准表存在（与pipeline系统保持一致的schema）"""
    from odps import models, types
    
    # 确保query_item表存在（与pipeline标准schema一致，使用双分区）
    if not odps_client.exist_table(query_table):
        query_schema = models.Schema.from_lists(
            ["query_id","set_id","game_name","raw_query","category","is_golden","query_time"],
            [types.bigint, types.bigint, types.string, types.string, types.string, types.bigint, types.string],
            partition_columns=["dm","game_id"],
            partition_types=[types.string, types.bigint]
        )
        odps_client.create_table(query_table, query_schema)
    
    # 确保llm_answer表存在（与pipeline标准schema一致，使用三重分区）
    if not odps_client.exist_table(answer_table):
        answer_schema = models.Schema.from_lists(
            ["answer_id","set_id","query_id","model_name","game_name","answer_content","generation_metadata","generated_at","query_time"],
            [types.bigint, types.bigint, types.bigint, types.string, types.string, types.string, types.string, types.bigint, types.string],
            partition_columns=["dm","game_id","model_id"],
            partition_types=[types.string, types.bigint, types.bigint]
        )
        odps_client.create_table(answer_table, answer_schema)

# ---------------------------
# CLI 辅助函数
# ---------------------------

async def get_latest_set_ids_per_game(odps_reader, table_names: dict, dm_partition: str) -> Optional[List[int]]:
    """
    获取每个游戏最新的set_id（基于创建时间，不是ID值）
    
    Args:
        odps_reader: ODPS读取客户端
        table_names: 包含表名的字典
        dm_partition: DM分区
    
    Returns:
        最新set_id列表（每个游戏一个）或None（如果未找到）
    """
    try:
        logging.info(f"从分区 dm='{dm_partition}' 查询最新的set_ids")
        
        # 查询 query_set 表（包含 created_at 时间戳）
        df_all = await read_rows_with_condition(
            odps_reader,
            table_name=table_names["QUERY_SET_TABLE_NAME"], 
            partition_spec=f"dm='{dm_partition}'",
            where_clause=None,
            limit=None
        )
        
        if df_all.empty or 'set_id' not in df_all.columns or 'game_name' not in df_all.columns or 'created_at' not in df_all.columns:
            logging.error(f"在分区 dm='{dm_partition}' 中未找到数据或缺少必需的列")
            return None
        
        # 基于 created_at 时间戳找到每个游戏最新的 set_id
        # 获取每个游戏中 created_at 最大值的索引
        latest_indices = df_all.groupby('game_name')['created_at'].idxmax()
        latest_per_game = df_all.loc[latest_indices, ['game_name', 'set_id', 'created_at']].reset_index(drop=True)
        latest_set_ids = sorted(latest_per_game['set_id'].unique())
        
        logging.info(f"找到 {len(latest_set_ids)} 个游戏的最新set_ids（按创建时间）：")
        for row in latest_per_game.itertuples(index=False):
            created_time = pd.to_datetime(row.created_at, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"    {row.game_name}: set_id={row.set_id} (创建时间: {created_time})")
        
        return latest_set_ids
        
    except Exception as e:
        logging.error(f"查询最新set_ids失败: {e}")
        return None

# ---------------------------
# 主流程
# ---------------------------

async def run(
    dm_partition: str,
    mode: str = "test",
    game_name: Optional[str] = None,
    num_questions: int = 3,
    overwrite: bool = False,
    print_only: bool = False,
) -> Dict[str, Any]:
    """
    使用OpenAI gpt-4o生成反作弊题目并写入ODPS表
    
    Args:
        dm_partition: DM分区 (e.g., '2025-06')
        mode: 运行模式 ('test' 或 'formal')
        game_name: 可选的游戏名称过滤，如果提供则只为该游戏生成题目
        num_questions: 生成题目数量
        overwrite: 是否覆盖现有数据
        print_only: 是否只打印结果不写入ODPS（用于预览和测试）
    """
    # 设置日志
    setup_logging()
    
    # 初始化客户端
    openai_client, odps_reader, odps_writer = initialize_clients()
    
    # 获取标准表名
    table_names = get_table_names(mode)
    logging.info(f"运行模式: {mode}, 使用表: {list(table_names.values())}")
    
    game_info_list = await get_game_info(odps_reader, table_names, dm_partition, game_name)
    logging.info(f"共找到 {len(game_info_list)} 个游戏准备生成题目")
    
    # 使用pipeline标准表名
    query_table = table_names["QUERY_ITEM_TABLE_NAME"]
    answer_table = table_names["LLM_ANSWER_TABLE_NAME"]
    
    ensure_anti_cheat_tables(odps_writer, query_table, answer_table)
    
    query_date = _get_month_end_date(dm_partition)
    logging.info(f"query_time 固定为 {query_date}")
    
    max_query_id = await get_max_query_id_from_tables(odps_reader, table_names, dm_partition)
    next_query_id = max_query_id + 1
    
    # 为每个游戏生成反作弊题目
    all_query_data, all_answer_data = [], []
    model_id = DEFAULT_MODEL_ID  # 反作弊专用模型ID，移到循环外以便print模式使用
    llm = OpenAI2Choice(aclient=openai_client)
    collected_set_ids = set()
    
    for game_info in game_info_list:
        current_game_id = game_info["game_id"]  
        current_game_name = game_info["game_name"]
        current_set_id = game_info.get("set_id")
        if current_set_id is not None:
            collected_set_ids.add(current_set_id)
        
        logging.info(f"正在为游戏 '{current_game_name}' (ID: {current_game_id}, set_id: {current_set_id}) 生成 {num_questions} 道反作弊题目")
        logging.info(f"GPT-4o 将自动搜索《{current_game_name}》的最新信息")
        
        # LLM生成（GPT-4o 会根据指令自动搜索）
        items = await llm.generate_batch(
            current_game_name, 
            dm_partition, 
            n=num_questions, 
            max_retries=3 # 增加重试次数
        )
        
        if not items:
            logging.warning(f"为游戏 '{current_game_name}' 生成题目失败，跳过")
            continue
        
        # 3) 准备数据 - 使用递延的query_id
        for i, it in enumerate(items):
            # 使用递延的query_id
            qid = next_query_id
            next_query_id += 1
            
            all_query_data.append({
                "query_id": qid, 
                "set_id": current_set_id,  # 使用当前游戏的set_id
                "game_name": current_game_name, 
                "raw_query": it["question"], 
                "category": DEFAULT_CATEGORY, 
                "is_golden": DEFAULT_IS_GOLDEN,
                "query_time": query_date  # 使用dm分区月末日期字符串
            })
            
            # 为这批答案生成统一的时间戳
            generation_timestamp = int(time.time())
            
            for j, ans in enumerate(it["answers"], start=1):
                # 使用新的make_answer_id函数生成answer_id
                # 对于反作弊题目，每个query有多个答案，我们使用(query_id * 100 + j)作为虚拟query_id
                virtual_query_id = qid * 100 + j
                aid = make_answer_id(current_set_id, model_id, virtual_query_id, generation_timestamp)
                # 增强的元数据：包含正确性和选择指示
                # 从答案文本中提取选择指示（markdown加粗格式）
                text = ans["text"]
                model_choice = ans.get("model_choice")
                if model_choice not in ("模型A", "模型B"):
                    model_choice = "模型A" if ans.get("correct") else "模型B"
                selection_slug = "modelA" if model_choice == "模型A" else "modelB"
                meta = json.dumps({
                    "correct": bool(ans["correct"]),
                    "selection": selection_slug,
                    "model_choice": model_choice,
                    "side": ans.get("side"),
                    "anti_cheat": True
                }, ensure_ascii=False)
                # 匹配标准llm_answer表schema: answer_id, set_id, query_id, model_name, game_name, answer_content, generation_metadata, generated_at, query_time (分区: dm, game_id, model_id)
                all_answer_data.append({
                    "answer_id": aid, 
                    "set_id": current_set_id,  # 使用当前游戏的set_id
                    "query_id": qid, 
                    "model_id": model_id,  # 将在写入前移除，因为是分区字段
                    "model_name": DEFAULT_MODEL_NAME, 
                    "game_name": current_game_name, 
                    "answer_content": ans["text"], 
                    "generation_metadata": meta,
                    "generated_at": generation_timestamp,  # 使用与answer_id相同的时间戳
                    "query_time": query_date  # dm分区月末日期
                })
    
    # 检查是否生成了任何题目
    if not all_query_data:
        raise Exception("未能为任何游戏生成有效题目")
    
    # 4) 处理结果
    df_query = pd.DataFrame(all_query_data)
    df_answer = pd.DataFrame(all_answer_data)
    
    # 确保新生成的query数据的query_time格式也是标准的
    if not df_query.empty:
        df_query['query_time'] = query_date
    
    if print_only:
        # 只打印模式
        print("\n" + "="*80)
        print("反作弊题目生成结果（Print模式 - 不写入ODPS）")
        print("="*80)
        
        print(f"\n生成统计:")
        print(f"  - 游戏数量: {len(game_info_list)}")
        print(f"  - 题目总数: {len(all_query_data)}")
        print(f"  - 答案总数: {len(all_answer_data)}")
        print(f"  - Model ID: {model_id} (反作弊专用)")
        
        # 显示每个游戏的set_id
        unique_set_ids = df_query['set_id'].unique() if 'set_id' in df_query.columns else []
        if len(unique_set_ids) > 0:
            print(f"  - Set IDs: {', '.join(map(str, unique_set_ids))}")
        
        # 显示每个游戏的题目
        for game_info in game_info_list:
            game_id = game_info["game_id"]
            game_name = game_info["game_name"]
            game_queries = df_query[df_query['game_name'] == game_name]
            
            if game_queries.empty:
                continue
                
            print(f"\n{'='*60}")
            print(f"游戏: {game_name} (ID: {game_id})")
            print(f"题目数量: {len(game_queries)}")
            print("-"*60)
            
            # 显示该游戏的所有题目
            for idx, row in game_queries.iterrows():
                query_id = row['query_id']
                question = row['raw_query']
                
                print(f"\n题目 {query_id}:")
                print(f"  {question}")
                
                # 获取对应的答案
                answers = df_answer[df_answer['query_id'] == query_id]
                for _, ans_row in answers.iterrows():
                    answer_text = ans_row['answer_content']
                    metadata = json.loads(ans_row['generation_metadata'])
                    is_correct = metadata.get('correct', False)
                    selection = metadata.get('selection', 'unknown')
                    
                    mark = "✓" if is_correct else "✗"
                    print(f"    [{mark}] {answer_text}")
        
        print("\n" + "="*80)
        print("数据预览（前5条）:")
        print("-"*60)
        print("\nQuery表:")
        print(df_query[['query_id', 'game_name', 'raw_query', 'category']].head(5).to_string())
        print("\nAnswer表:")
        print(df_answer[['answer_id', 'query_id', 'answer_content', 'model_name', 'query_time']].head(5).to_string())
        
        result_info = "PRINT_ONLY_MODE - 未写入ODPS"
        logging.info(f"Print mode: Generated {len(all_query_data)} questions and {len(all_answer_data)} answers for {len(game_info_list)} games")
    else:
        # 正常写入模式
        # 按游戏分组写入数据
        games_written = 0
        
        for game_info in game_info_list:
            current_game_id = game_info["game_id"]
            current_game_name = game_info["game_name"]
            
            # 过滤当前游戏的数据（创建副本以便后续修改）
            game_query_data = df_query[df_query['game_name'] == current_game_name].copy()
            game_answer_data = df_answer[df_answer['game_name'] == current_game_name].copy()
            
            if game_query_data.empty or game_answer_data.empty:
                logging.warning(f"游戏 '{current_game_name}' 没有生成数据，跳过写入")
                continue
                
            # query表分区（需要同时提供dm和game_id分区）
            query_partition_str = f"dm='{dm_partition}',game_id={current_game_id}"
            
            # answer表分区（使用三重分区结构与answer_collector_concurrent_v2.py对齐）
            answer_partition_str = f"dm='{dm_partition}',game_id={current_game_id},model_id={model_id}"
            
            # 移除DataFrame中的分区字段（model_id是answer表的分区字段）
            game_answer_for_insert = game_answer_data.drop('model_id', axis=1) if 'model_id' in game_answer_data.columns else game_answer_data
            
            # 处理 query 表的 overwrite 逻辑
            query_overwrite_mode = False  # 默认不覆盖
            
            if overwrite:
                # overwrite 模式：读取现有数据，保留非 golden 记录，与新 golden 记录合并后覆盖写入
                try:
                    where_clause_for_partition = query_partition_str.replace(",", " AND ")
                    
                    existing_df = await read_rows_with_condition(
                        odps_reader,
                        table_name=query_table,
                        partition_spec=None,  # 不使用 partition_spec
                        where_clause=where_clause_for_partition,  # 使用 where_clause
                        limit=None
                    )
                    
                    if not existing_df.empty and 'is_golden' in existing_df.columns:
                        # 确保 is_golden 是数值类型
                        existing_df['is_golden'] = pd.to_numeric(existing_df['is_golden'], errors='coerce').fillna(0).astype(int)
                        
                        # 保留非 golden 记录
                        preserved_df = existing_df[existing_df['is_golden'] != 1].copy()
                        old_golden_count = len(existing_df[existing_df['is_golden'] == 1])
                        
                        if not preserved_df.empty:
                            # 确保列对齐
                            missing_cols = [col for col in game_query_data.columns if col not in preserved_df.columns]
                            for col in missing_cols:
                                series = game_query_data[col]
                                fill_value = 0 if pd.api.types.is_numeric_dtype(series.dtype) else ""
                                preserved_df[col] = fill_value
                            preserved_df = preserved_df[game_query_data.columns]
                            
                            # 合并保留的非 golden 数据和新的 golden 数据
                            game_query_data = pd.concat([preserved_df, game_query_data], ignore_index=True)
                            logging.info(f"保留 {len(preserved_df)} 条非 golden 记录，删除 {old_golden_count} 条旧 golden 记录，添加 {len(game_query_data) - len(preserved_df)} 条新 golden 记录")
                        else:
                            logging.info(f"分区中没有非 golden 记录，删除 {old_golden_count} 条旧 golden 记录，添加 {len(game_query_data)} 条新 golden 记录")
                        
                        # 使用 overwrite 模式写入合并后的完整数据
                        query_overwrite_mode = True
                    else:
                        logging.info(f"分区为空或缺少 is_golden 字段，直接写入新数据")
                        query_overwrite_mode = False
                        
                except Exception as exc:
                    logging.warning(f"读取现有数据失败: {exc}，将直接使用 insert 模式")
                    query_overwrite_mode = False
            
            # 写入前确保移除任何分区字段（防止意外包含）
            partition_fields = ['dm', 'game_id', 'model_id']
            query_data_clean = game_query_data.drop(columns=[col for col in partition_fields if col in game_query_data.columns], errors='ignore')
            answer_data_clean = game_answer_for_insert.drop(columns=[col for col in partition_fields if col in game_answer_for_insert.columns], errors='ignore')
            
            # 写入当前游戏的数据
            current_game_set_id = game_query_data['set_id'].iloc[0] if not game_query_data.empty else current_set_id
            logging.info(f"写入 {current_game_name}: {len(query_data_clean)} 题 / {len(answer_data_clean)} 答")
            insert_dataframe(odps_writer, query_data_clean, query_table, partition=query_partition_str, overwrite=query_overwrite_mode, set_id=current_game_set_id)
            insert_dataframe(odps_writer, answer_data_clean, answer_table, partition=answer_partition_str, overwrite=overwrite, set_id=current_game_set_id)
            
            games_written += 1
        
        result_info = f"成功写入 {games_written} 个游戏的数据到 {query_table} 和 {answer_table}"
        logging.info(f"Successfully generated {len(all_query_data)} questions and {len(all_answer_data)} answers for {games_written} games")
    
    # 收集所有游戏的set_ids
    unique_set_ids = list(collected_set_ids)
    
    return {
        "queries": len(all_query_data), 
        "answers": len(all_answer_data),
        "games": len(game_info_list),
        "query_table": query_table, 
        "answer_table": answer_table,
        "set_ids": unique_set_ids,  # 返回所有set_ids
        "dm_partition": dm_partition,
        "mode": mode,
        "status": result_info
    }

# 同步wrapper函数，方便非异步环境调用
def run_sync(**kwargs) -> Dict[str, Any]:
    """同步版本的run函数，便于在非异步环境中调用"""
    return asyncio.run(run(**kwargs))

# == SECTION 6: MAIN FUNCTION == #

async def main():
    """Main entry point for anti-cheat question generator."""
    parser = argparse.ArgumentParser(
        description="Generate anti-cheat questions using OpenAI gpt-4o - 从现有的评测集中获取游戏信息",
        epilog="""
使用示例:

为特定游戏生成题目:
  python anti_cheat.py --dm_partition 2025-06 --game_name your_game --mode test

只打印结果不写入ODPS（预览模式）:
  python anti_cheat.py --dm_partition 2025-06 --mode test --print

使用覆盖模式:
  python anti_cheat.py --dm_partition 2025-06 --mode test --overwrite
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dm_partition", type=str, required=True, help="DM partition to process (e.g., '2025-06')")
    parser.add_argument("--mode", type=str, choices=["formal", "test"], default="test", help="Run mode: 'formal' for production tables, 'test' for test tables (default: test)")
    
    # Optional filtering
    parser.add_argument("--game_name", type=str, help="Optional: filter to specific game name")
    
    # Optional arguments
    parser.add_argument("--num_questions", type=int, default=3, help="Number of questions to generate per game (default: 3)")
    
    # Print mode control
    parser.add_argument("--print", action="store_true", help="Print results only, don't write to ODPS (preview mode)")
    
    # ODPS overwrite behavior control (aligned with other scripts)
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true", 
                                 help="Enable overwrite mode: removes existing golden records (is_golden=1) in the same partition, preserves non-golden records, then inserts new golden records")
    overwrite_group.add_argument("--insert", action="store_true", 
                                 help="Use insert mode: adds new golden records without touching existing data (default, safer)")
    
    args = parser.parse_args()
    
    # Handle overwrite mode (default to insert mode if neither specified)
    overwrite_mode = args.overwrite if args.overwrite else False
    # Handle print mode
    print_only = getattr(args, 'print', False)
    
    try:
        result = await run(
            dm_partition=args.dm_partition,
            mode=args.mode,
            game_name=args.game_name,
            num_questions=args.num_questions,
            overwrite=overwrite_mode,
            print_only=print_only  # 添加打印模式参数
        )
        
        print("\n反作弊题目生成完成:")
        if result.get('set_ids'):
            print(f"- Set IDs: {', '.join(map(str, result['set_ids']))}")
        print(f"- 游戏数量: {result['games']}")
        print(f"- 生成查询数: {result['queries']}")
        print(f"- 生成答案数: {result['answers']}")
        print(f"- Query表: {result['query_table']}")
        print(f"- Answer表: {result['answer_table']}")
        print(f"- 分区: dm={result['dm_partition']}")
        print(f"- 模式: {result['mode']}")
        if args.game_name:
            print(f"- 游戏过滤: {args.game_name}")
        
    except Exception as e:
        logging.error(f"Anti-cheat generation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())