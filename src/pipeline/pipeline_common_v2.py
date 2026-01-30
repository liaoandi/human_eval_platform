# -*- coding: utf-8 -*-
"""
Pipeline Commons: 共用配置、函数和LLM调用器
拆分自integrated_pipeline.py，供eval_set_generator.py和answer_collector.py共用
"""
# --- Perplexity Cookies Utilities (fail-fast) ---

import os, json, pathlib, time, logging, subprocess, sys
from datetime import datetime
from typing import Any, Dict, Optional, Union

class PPLXCookieError(RuntimeError):
    pass

def _read_json_file(path: Union[str, pathlib.Path]) -> Any:
    p = pathlib.Path(path)
    if not p.exists():
        raise PPLXCookieError(f"Cookie 文件不存在: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise PPLXCookieError(f"Cookie 文件 JSON 解析失败: {p}, err={e}")

def _normalize_cookie_obj(obj: Any) -> Any:
    """
    保持“原样可传给 helallao 客户端”的结构，但做最少的 sanity check：
    - 允许 dict 或 list[dict] 两类常见导出格式
    - 不强行改形态，直接返回；只要非空即可
    """
    if obj is None:
        raise PPLXCookieError("未提供 Cookie 对象")
    if isinstance(obj, dict):
        if not obj:
            raise PPLXCookieError("Cookie 字典为空")
        return obj
    if isinstance(obj, list):
        if not obj:
            raise PPLXCookieError("Cookie 列表为空")
        # 简单校验元素形态
        if not all(isinstance(x, dict) for x in obj):
            raise PPLXCookieError("Cookie 列表元素必须是 dict")
        return obj
    raise PPLXCookieError(f"不支持的 Cookie 结构类型: {type(obj)}")


def _prepare_cookie_for_perplexity_client(cookie_obj: Any) -> Any:
    """
    perplexity_async.Client 期望 cookies 为 dict/jar 形式。
    若外部传入 list[dict]（常见于 Chrome/脚本导出格式），需要折叠为 {name: value}。
    仅在可安全转换时才改变结构，避免泄漏敏感内容到日志。
    """

    # 某些导出格式是 {"cookies": [...]}
    if isinstance(cookie_obj, dict):
        inner = cookie_obj.get("cookies")
        if isinstance(inner, list):
            converted = _prepare_cookie_for_perplexity_client(inner)
            if converted is not inner:
                return converted
        if isinstance(inner, dict):
            simple_dict = {}
            for key, value in inner.items():
                if value is None:
                    continue
                try:
                    simple_dict[str(key)] = str(value)
                except Exception:
                    continue
            if simple_dict:
                return simple_dict
        return cookie_obj

    if isinstance(cookie_obj, list):
        simple_dict = {}
        for idx, item in enumerate(cookie_obj):
            if not isinstance(item, dict):
                continue

            name = (
                item.get("name")
                or item.get("Name")
                or item.get("key")
                or item.get("Key")
            )
            # value 可能为字符串或可序列化对象
            value = item.get("value") if "value" in item else item.get("Value")

            if not name or value is None:
                continue

            try:
                # 确保最终传入字符串，避免后续 join 报错
                simple_dict[str(name)] = str(value)
            except Exception:
                continue

        if simple_dict:
            return simple_dict

        logging.warning("Perplexity cookies 列表格式无法转换，按原样返回，可能导致调用失败")

    return cookie_obj

def _load_from_env() -> Any:
    """优先从环境变量 PERPLEXITY_COOKIES 读取（JSON 字符串）"""
    raw = os.getenv("PERPLEXITY_COOKIES")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        raise PPLXCookieError(f"PERPLEXITY_COOKIES 不是合法 JSON: {e}")

def _load_from_path_env() -> Any:
    """支持通过 PERPLEXITY_COOKIES_PATH 指定 JSON 文件路径"""
    path = os.getenv("PERPLEXITY_COOKIES_PATH")
    if not path:
        return None
    return _read_json_file(path)

def _load_from_cache(cache_path: Union[str, pathlib.Path], max_age_hours: int) -> Any:
    """
    仅在明确允许的情况下使用缓存；默认不走缓存。
    缓存文件结构示例：
    {
      "timestamp": 1737000000.123,  # time.time()
      "cookies": <dict or list>
    }
    """
    p = pathlib.Path(cache_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = float(data.get("timestamp", 0))
        age_h = (time.time() - ts) / 3600.0
        if age_h > max_age_hours:
            logging.warning(f"Cookie 缓存过期（{age_h:.1f}h > {max_age_hours}h）: {p}")
            return None
        return data.get("cookies")
    except Exception as e:
        logging.warning(f"读取 Cookie 缓存失败，忽略: {p}, err={e}")
        return None


def _check_cookie_cache_age(cache_path: Union[str, pathlib.Path]) -> Optional[float]:
    p = pathlib.Path(cache_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("timestamp")
        if ts is None:
            return None
        try:
            ts_float = float(ts)
            age_h = (time.time() - ts_float) / 3600.0
            return age_h
        except (TypeError, ValueError):
            try:
                dt = datetime.fromisoformat(str(ts))
                age_h = (datetime.now(dt.tzinfo) - dt).total_seconds() / 3600.0
                return age_h
            except Exception:
                return None
    except Exception:
        return None


def _prompt_yes_no(message: str, default: bool = False) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        logging.info("当前环境非交互模式，跳过输入提示，使用默认值 %s", default)
        return default

    prompt_suffix = " [Y/n]" if default else " [y/N]"
    while True:
        try:
            answer = input(f"{message}{prompt_suffix} ").strip().lower()
        except EOFError:
            logging.info("标准输入不可用，按默认值 %s 处理", default)
            return default

        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("请输入 y 或 n")


def _refresh_pplx_cookies(refresh_command: Optional[Union[str, list]] = None) -> bool:
    script_dir = pathlib.Path(__file__).parent.resolve()

    if refresh_command is None:
        env_command = os.getenv("PERPLEXITY_REFRESH_COMMAND")
        if env_command:
            refresh_command = env_command.strip()

    if refresh_command is None:
        script_path = script_dir / "refresh_pplx_cookie_now.sh"
        if script_path.exists():
            refresh_command = ["/bin/bash", str(script_path)]
        else:
            fallback = script_dir / "perplexity_cookie_auto_full.py"
            if fallback.exists():
                refresh_command = [sys.executable, str(fallback)]
            else:
                logging.error("未找到默认的 Perplexity Cookie 刷新脚本")
                return False

    logging.info("开始执行 Perplexity Cookie 刷新命令: %s", refresh_command)

    try:
        if isinstance(refresh_command, str):
            result = subprocess.run(refresh_command, shell=True, cwd=str(script_dir), check=False)
        else:
            result = subprocess.run(refresh_command, cwd=str(script_dir), check=False)
    except Exception as exc:
        logging.error(f"执行刷新命令失败: {exc}")
        return False

    if result.returncode == 0:
        logging.info("Perplexity Cookie 刷新成功")
        return True

    logging.error(f"Perplexity Cookie 刷新命令返回非零状态码: {result.returncode}")
    return False


# Global flag to track if cookie refresh has been checked
_PPLX_REFRESH_CHECKED = False

def _maybe_prompt_refresh(cache_path: Union[str, pathlib.Path], refresh_command: Optional[Union[str, list]] = None) -> None:
    global _PPLX_REFRESH_CHECKED
    
    # 如果已经检测过或者设置了跳过标志，直接返回
    if _PPLX_REFRESH_CHECKED or os.getenv("PERPLEXITY_SKIP_REFRESH_PROMPT", "").lower() == "true":
        return
    
    # 标记为已检测
    _PPLX_REFRESH_CHECKED = True

    auto_refresh = os.getenv("PERPLEXITY_AUTO_REFRESH", "").lower() == "true"

    age = _check_cookie_cache_age(cache_path)
    age_desc = f"约 {age:.1f} 小时" if age is not None else "未知"

    if auto_refresh:
        logging.info("PERPLEXITY_AUTO_REFRESH=true，自动刷新 Cookie（当前缓存年龄%s）", age_desc)
        if not _refresh_pplx_cookies(refresh_command):
            logging.error("自动刷新 Perplexity Cookie 失败，请手动确认")
        return

    if not sys.stdin or not sys.stdin.isatty():
        logging.info("非交互环境，跳过 Cookie 刷新提示（当前缓存年龄%s）", age_desc)
        return

    if age is None:
        message = "未找到 Perplexity Cookie 缓存或无法判断年龄。是否刷新?"
    else:
        message = f"当前 Perplexity Cookie 缓存年龄 {age_desc}。是否刷新?"

    if _prompt_yes_no(message, default=False):
        if not _refresh_pplx_cookies(refresh_command):
            logging.error("用户确认刷新，但刷新命令执行失败。后续将继续尝试使用现有 Cookie")
    else:
        logging.info("用户选择不刷新 Perplexity Cookie（当前缓存年龄%s）", age_desc)

def cache_pplx_cookies(cookies: Any, cache_path: Union[str, pathlib.Path]) -> None:
    """可选：写入缓存文件（只有你明确调用时才会写）"""
    p = pathlib.Path(cache_path)
    payload = {"timestamp": time.time(), "cookies": cookies}
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

def load_pplx_cookies(
    *,
    allow_cache: bool = False,
    cache_path: Union[str, pathlib.Path] = ".perplexity_cookies_cache.json",
    cache_max_age_hours: int = 24,
    refresh_command: Optional[Union[str, list]] = None
) -> Any:
    """
    Fail-fast 加载流程（默认不使用缓存）：
    1) PERPLEXITY_COOKIES (env, JSON)
    2) PERPLEXITY_COOKIES_PATH (env 指向文件)
    3) [可选] cache_path（仅当 allow_cache=True）
    全部失败 -> 直接抛错
    """
    if allow_cache:
        _maybe_prompt_refresh(cache_path, refresh_command=refresh_command)

    # 1) ENV: PERPLEXITY_COOKIES
    obj = _load_from_env()
    if obj is not None:
        return _prepare_cookie_for_perplexity_client(_normalize_cookie_obj(obj))

    # 2) ENV: PERPLEXITY_COOKIES_PATH
    obj = _load_from_path_env()
    if obj is not None:
        return _prepare_cookie_for_perplexity_client(_normalize_cookie_obj(obj))

    # 3) 可选缓存
    if allow_cache:
        obj = _load_from_cache(cache_path, cache_max_age_hours)
        if obj is not None:
            return _prepare_cookie_for_perplexity_client(_normalize_cookie_obj(obj))

    # 全部失败
    raise PPLXCookieError(
        "未找到可用的 Perplexity Cookies；请设置 PERPLEXITY_COOKIES（JSON 字符串）或 "
        "PERPLEXITY_COOKIES_PATH（指向 JSON 文件）。"
    )

# == 明确定义公开接口，防止不必要的导入 == #
__all__ = [
    # 数据列名常量
    'COL_QUERY_ID', 'COL_QUERY', 'COL_GAME_NAME', 'COL_CATEGORY', 'COL_CANONICAL_QUERY_ID', 'COL_QUERY_TIME',
    
    # 配置常量
    'MODEL_CONFIG_INTERNAL', 'MODEL_CONFIGS', 'DEFAULT_TIMESTAMP_FORMAT',
    'SOURCE_TABLE', 'OPENAI_API_KEY',
    
    # Query status constants
    'QUERY_STATUS_RELEVANT', 'QUERY_STATUS_IRRELEVANT', 'QUERY_STATUS_DUPLICATE', 
    'QUERY_STATUS_EXCLUDE',
    
    # Category constants
    'CATEGORY_NO_CLASSIFICATION', 'CATEGORY_CLASSIFICATION_FAILED', 'CATEGORY_OTHER',
    
    # 表名常量
    'QUERY_PREPROCESS_DETAILS_TABLE_NAME', 'QUERY_ITEM_TABLE_NAME', 
    'QUERY_SET_TABLE_NAME', 'LLM_ANSWER_TABLE_NAME',
    
    # 核心工具函数
    'sanitize_name', 'llm_chat_call', 'insert_dataframe',
    
    # LLM调用函数
    'collect_ans_call_openai_async',
    # 'collect_ans_call_perplexity_async',  # unused
    # 'collect_ans_call_deepseek_async',    # unused
    'collect_ans_call_gemini_async',
    'collect_ans_call_tapai_async', 'collect_ans_call_gpt5_async',
    'collect_ans_call_doubao_async', # Added new function
    
    # 初始化和设置函数
    'initialize_clients', 'setup_logging', 'generate_set_id',
    'save_model_configs_to_odps','get_table_names',
    
    # 限流器函数
    'get_semaphore', 'with_limit',
    
    # 重试与工具函数
    'retry_with_backoff', 'rough_token_estimate', 'extract_citations',
    
    # 数据读取函数
    'read_rows_with_condition'
]
import asyncio
import logging
import logging.handlers
import math
import os
import pathlib
import re
import sys
import time
import json
import random
import httpx
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI
import pandas as pd
from odps import ODPS
from dotenv import load_dotenv

# == SECTION 0: CONFIGURATION == #
def _load_env():
    """Load environment variables from a .env file in the script's directory.
    
    .env file has the highest priority and will override system environment variables.
    """
    script_dir = pathlib.Path(__file__).parent.resolve()
    dotenv_path = script_dir / '.env'
    load_dotenv(dotenv_path=dotenv_path, override=True)

_load_env()

# --- ODPS Configuration ---
AK = os.getenv("ALIYUN_AK")
SK = os.getenv("ALIYUN_SK")
SH_AK = os.getenv("SH_ALIYUN_AK")
SH_SK = os.getenv("SH_ALIYUN_SK")

ODPS_PROJECT = os.getenv("ODPS_PROJECT", "your_project")
# Base table names
SOURCE_TABLE = "dwd_ape_eval_query_collect_inc"
_BASE_LLM_MODELS_TABLE = "dwd_ape_eval_llm_models_inc"
_BASE_QUERY_SET_TABLE = "dwd_ape_eval_query_set_inc"
_BASE_QUERY_PREPROCESS_DETAILS_TABLE = "dwd_ape_eval_query_preprocess_details_inc"
_BASE_QUERY_ITEM_TABLE = "dwd_ape_eval_query_item_inc"
_BASE_LLM_ANSWER_TABLE = "dwd_ape_eval_llm_answer_inc"

# Default table names (formal mode)
LLM_MODELS_TABLE_NAME = _BASE_LLM_MODELS_TABLE
QUERY_SET_TABLE_NAME = _BASE_QUERY_SET_TABLE
QUERY_PREPROCESS_DETAILS_TABLE_NAME = _BASE_QUERY_PREPROCESS_DETAILS_TABLE
QUERY_ITEM_TABLE_NAME = _BASE_QUERY_ITEM_TABLE
LLM_ANSWER_TABLE_NAME = _BASE_LLM_ANSWER_TABLE

def get_table_names(mode: str = "formal") -> dict:
    """Get table names based on mode (formal or test)"""
    if mode.lower() == "test":
        return {
            "LLM_MODELS_TABLE_NAME": f"{_BASE_LLM_MODELS_TABLE}_test",
            "QUERY_SET_TABLE_NAME": f"{_BASE_QUERY_SET_TABLE}_test",
            "QUERY_PREPROCESS_DETAILS_TABLE_NAME": f"{_BASE_QUERY_PREPROCESS_DETAILS_TABLE}_test",
            "QUERY_ITEM_TABLE_NAME": f"{_BASE_QUERY_ITEM_TABLE}_test",
            "LLM_ANSWER_TABLE_NAME": f"{_BASE_LLM_ANSWER_TABLE}_test"
        }
    else:  # formal mode
        return {
            "LLM_MODELS_TABLE_NAME": _BASE_LLM_MODELS_TABLE,
            "QUERY_SET_TABLE_NAME": _BASE_QUERY_SET_TABLE,
            "QUERY_PREPROCESS_DETAILS_TABLE_NAME": _BASE_QUERY_PREPROCESS_DETAILS_TABLE,
            "QUERY_ITEM_TABLE_NAME": _BASE_QUERY_ITEM_TABLE,
            "LLM_ANSWER_TABLE_NAME": _BASE_LLM_ANSWER_TABLE
        }

# COLUMN NAMES
COL_QUERY_ID = "query_id"
COL_QUERY = "raw_query"
COL_GAME_NAME = "game_name"
COL_CATEGORY = "category"
COL_CANONICAL_QUERY_ID = "canonical_query_id"
COL_QUERY_TIME = "query_time"  # 提问时间 (renamed from dt)

# Global API and Model Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
AZURE_OPENAI_DEPLOYMENT_GPT4O = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O", "gpt-4o")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
COLLECT_ANS_MAX_TOKENS = 16384  # 调高到16384 tokens，为复杂答案留出更多空间

# Internal model configuration for pipeline stages, category discovery, query classification, and ranking
MODEL_CONFIG_INTERNAL = {
    # Azure 部署名可通过环境变量覆盖；未设置时回落到默认模型名
    "discovery": AZURE_OPENAI_DEPLOYMENT_GPT4O,
    "classification": AZURE_OPENAI_DEPLOYMENT_GPT4O,
    "ranking": AZURE_OPENAI_DEPLOYMENT_GPT4O,
    "embedding": AZURE_OPENAI_EMBEDDING_DEPLOYMENT
}

# External LLM configurations for answer collection
MODEL_CONFIGS = [
    # gpt-4o-search-preview 已弃用，使用 gpt-5 替代
    # {
    #     "id": 1, "model_name": "gpt-4o-search-preview", "base_model": "gpt-4o",
    #     "model_type": "search", "provider": "OpenAI",
    #     "params": json.dumps({
    #         "max_tokens": COLLECT_ANS_MAX_TOKENS  # 统一使用8192 tokens，确保公平性
    #         # temperature 参数不被支持，已在代码中过滤
    #         # search_context_size 和 location 为公平起见已移除
    #     }),
    #     "api_model_name": "gpt-4o-search-preview", "column_name_suffix": "gpt-4o-search-preview",
    #     "call_func_ref": "_collect_ans_call_openai_async"
    # },
    # {
    #     "id": 6, "model_name": "gpt-5", "base_model": "gpt-5",
    #     "model_type": "search", "provider": "OpenAI",
    #     "params": json.dumps({
    #         "reasoning_effort": "high",
    #         "text_verbosity": "medium",
    #         "max_output_tokens": COLLECT_ANS_MAX_TOKENS  # 明确设置为16384 tokens
    #     }),
    #     "api_model_name": "gpt-5", "column_name_suffix": "gpt-5",
    #     "call_func_ref": "collect_ans_call_gpt5_async",
    #     "concurrent_limit": 4  # 优化并发数到4，平衡速度和稳定性
    # },
    {
        "id": 10, "model_name": "gpt-5-pro", "base_model": "gpt-5-pro",
        "model_type": "search", "provider": "Azure",
        "params": json.dumps({
            "reasoning_effort": "high",
            "text_verbosity": "medium",
            "max_output_tokens": COLLECT_ANS_MAX_TOKENS
        }),
        "api_model_name": "gpt-5-pro", "column_name_suffix": "gpt-5-pro",
        "call_func_ref": "collect_ans_call_gpt5_async",
        "concurrent_limit": 1
    },
    {
        "id": 14, "model_name": "gpt-5.1-chat", "base_model": "gpt-5.1",
        "model_type": "search", "provider": "Azure",
        "params": json.dumps({
            "reasoning_effort": "medium",
            "text_verbosity": "medium",
            "max_output_tokens": COLLECT_ANS_MAX_TOKENS
        }),
        "api_model_name": "gpt-5.1-chat", "column_name_suffix": "gpt-5.1-chat",
        "call_func_ref": "collect_ans_call_gpt5_async",
        "concurrent_limit": 2
    },
    # { --cookie参数有问题
    #     "id": 7, "model_name": "perplexity-pro-gpt5thinking", "base_model": "gpt-5-thinking",
    #     "model_type": "search", "provider": "Perplexity",
    #     "params": json.dumps({
    #         "max_tokens": COLLECT_ANS_MAX_TOKENS,  # 统一使用16384 tokens，为复杂答案留出更多空间
    #         "temperature": 0.1,
    #         "mode": "pro",
    #         "model": "gpt-5-thinking",  # 使用 Perplexity 新增的 gpt-5-thinking 模型
    #         "sources": ["web"],
    #         "language": "zh-CN"
    #     }),
    #     "api_model_name": "perplexity-pro-gpt5thinking", "column_name_suffix": "perplexity-pro-gpt5thinking",
    #     "call_func_ref": "collect_ans_call_perplexity_async",
    #     "concurrent_limit": 2  # 降低默认并发以减少空返回概率，可用 PERPLEXITY_CONCURRENT_LIMIT 覆盖
    # },
    # deepseek_r1 已弃用
    # {
    #     "id": 3, "model_name": "deepseek_r1-0528", "base_model": "deepseek_r1",
    #     "model_type": "reasoning", "provider": "DeepSeek",
    #     "params": json.dumps({
    #         "max_tokens": COLLECT_ANS_MAX_TOKENS,  # 统一使用8192 tokens，确保公平性
    #         "temperature": 0.1
    #     }),
    #     "api_model_name": "ep-20250531191335-jlr94", "column_name_suffix": "deepseek_r1-0528",
    #     "call_func_ref": "_collect_ans_call_deepseek_async"
    # },
    {
        "id": 5, "model_name": "tap-ai", "base_model": "tap-ai",
        "model_type": "search", "provider": "TapTap",
        "params": json.dumps({
            "workflow": "SubQueryRAG",
            "source_type": 100,
            "mode": "2",
            "reasoning": True,
            "enable_outer_search": True,
            "forbidden_cache": False,
            "show_new_citation": True
        }),
        "api_model_name": "tap-ai", "column_name_suffix": "tap-ai",
        "call_func_ref": "collect_ans_call_tapai_async",
        "concurrent_limit": 2  # TAP AI specific concurrent limit
    },
    # {
    #     "id": 8, "model_name": "gemini-2.5-pro", "base_model": "gemini-2.5",
    #     "model_type": "reasoning", "provider": "Google",
    #     "params": json.dumps({
    #         "maxOutputTokens": 32768,  # Gemini特殊：包含thinking tokens，所以需要更大值
    #         "temperature": 0.1,
    #         "topP": 1.0,  # 默认为0.95，跟其他家不一样
    #         "candidateCount": 1,
    #         "thinkingBudget": 8192,  # 思考预算8192 tokens；可通过 GEMINI_THINKING_BUDGET 覆盖
    #         "enable_search": True  # 启用Google搜索功能
    #     }),
    #     "api_model_name": "gemini-2.5-pro", "column_name_suffix": "gemini-2.5-pro",
    #     "call_func_ref": "collect_ans_call_gemini_async",
    #     "concurrent_limit": 2  # 下调并发以降低 500 发生概率，可用 GEMINI_CONCURRENT_LIMIT 覆盖
    # }
    {
        "id": 12, "model_name": "gemini-3-pro-preview", "base_model": "gemini-3",
        "model_type": "reasoning", "provider": "Google",
        "params": json.dumps({
            "maxOutputTokens": 32768,
            "temperature": 0.1,
            "topP": 1.0,
            "candidateCount": 1,
            "enable_search": True,
            "thinkingBudget": 16384
        }),
        "api_model_name": "gemini-3-pro-preview", "column_name_suffix": "gemini-3-pro-preview",
        "call_func_ref": "collect_ans_call_gemini_async",
        "concurrent_limit": 2
    },
    {
        "id": 13, "model_name": "doubao-seed-1.6", "base_model": "doubao-seed-1.6",
        "model_type": "reasoning", "provider": "ByteDance",
        "params": json.dumps({
             "max_tokens": COLLECT_ANS_MAX_TOKENS,
             "temperature": 0.1,
             "enable_web_search": True  # Enable search
        }),
        "api_model_name": "ep-20250619161611-qn2gg", "column_name_suffix": "doubao-seed-1.6",
        "call_func_ref": "collect_ans_call_doubao_async",  # Use new dedicated function
        "concurrent_limit": 2
    },
]

# Pipeline configuration constants
DEFAULT_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

# Query status constants
QUERY_STATUS_RELEVANT = 'relevant'
QUERY_STATUS_IRRELEVANT = 'irrelevant'
QUERY_STATUS_DUPLICATE = 'duplicate'
QUERY_STATUS_EXCLUDE = 'exclude'

# Category constants
CATEGORY_NO_CLASSIFICATION = '不参与分类'
CATEGORY_CLASSIFICATION_FAILED = '分类失败'
CATEGORY_OTHER = '其他'

# == SECTION 0.5: CONCURRENCY LIMITER == #

# —— 简易并发限流器 (按 model_name 维度) ——
_model_semaphores = {}

def _get_semaphore(model_name: str, limit: Optional[int]) -> Optional[asyncio.Semaphore]:
    if not limit or limit <= 0:
        return None
    if model_name not in _model_semaphores:
        _model_semaphores[model_name] = asyncio.Semaphore(limit)
    return _model_semaphores[model_name]

async def with_limit(model_name: str, concurrent_limit: Optional[int], coro):
    sem = _get_semaphore(model_name, concurrent_limit)
    if sem:
        async with sem:
            return await coro
    else:
        return await coro

def get_semaphore(model_name: str, concurrent_limit: Optional[int] = None) -> Optional[asyncio.Semaphore]:
    """
    获取指定模型的信号量，用于控制并发数
    
    Args:
        model_name: 模型名称
        concurrent_limit: 并发限制数，如果为None或0则不限制
    
    Returns:
        asyncio.Semaphore或None
    """
    return _get_semaphore(model_name, concurrent_limit)

# == SECTION 0.6: UTILITY HELPERS == #

async def retry_with_backoff(coro_factory, max_retries=2, base_delay=2.0, cap_delay=10.0):
    """
    通用异步重试机制，支持指数退避
    
    Args:
        coro_factory: 返回协程的可调用对象
        max_retries: 最大重试次数
        base_delay: 基础延迟时间
        cap_delay: 最大延迟时间
    
    Returns:
        协程的执行结果
    
    Raises:
        Exception: 如果所有重试都失败
    """
    attempt = 0
    while True:
        try:
            return await coro_factory()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            expo = min(cap_delay, base_delay * (2 ** (attempt - 1)))
            delay = random.random() * expo
            logging.warning(f"Retrying after error: {str(e)[:200]} (attempt {attempt}) sleeping {delay:.2f}s")
            await asyncio.sleep(delay)

def rough_token_estimate(text: str) -> int:
    """
    通用 token 估算器，适用于中英混合文本
    
    Args:
        text: 要估算的文本
        
    Returns:
        估算的 token 数量
    """
    if not text:
        return 0
    import re
    # 中文字符计数
    zh = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 非中文字符计数
    non = len(text) - zh
    # 粗略: 中文 ~1.5 字/Token, 英文 ~4 字符/Token
    return int(zh / 1.5 + non / 4)


def standardize_response(
    answer: str,
    *,
    metadata: Optional[dict] = None,
    usage: Optional[dict] = None,
    prompt_text: Optional[str] = None
) -> SimpleNamespace:
    """标准化第三方 LLM 响应结构，减少重复封装。"""

    content = answer or ""
    metadata_obj = metadata if metadata is not None else {}

    usage_dict: Dict[str, Any] = {}
    if prompt_text is not None:
        prompt_tokens = rough_token_estimate(prompt_text)
        completion_tokens = rough_token_estimate(content)
        usage_dict.update(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )

    if usage:
        usage_dict.update(usage)

    usage_dict.setdefault("prompt_tokens", 0)
    usage_dict.setdefault("completion_tokens", 0)
    usage_dict.setdefault(
        "total_tokens", usage_dict["prompt_tokens"] + usage_dict["completion_tokens"]
    )

    message = SimpleNamespace(content=content, metadata=metadata_obj)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(**usage_dict),
    )


def _get_message_metadata(raw_response) -> dict:
    if hasattr(raw_response, "choices") and raw_response.choices:
        message = getattr(raw_response.choices[0], "message", None)
        if message and hasattr(message, "metadata") and message.metadata:
            return message.metadata or {}
    return {}


def _extract_openai_citations(raw_response) -> list:
    citations = []
    for item in getattr(raw_response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            for ann in getattr(part, "annotations", []) or []:
                if getattr(ann, "type", None) == "url_citation":
                    citations.append({
                        "title": getattr(ann, "title", None),
                        "url": getattr(ann, "url", None)
                    })
    return citations


def _extract_gemini_citations(raw_response) -> list:
    citations = []
    metadata = _get_message_metadata(raw_response)
    for chunk in metadata.get("grounding_chunks", []) or []:
        if not isinstance(chunk, dict):
            continue
        if "web" in chunk and isinstance(chunk["web"], dict):
            citations.append({
                "title": chunk["web"].get("title"),
                "url": chunk["web"].get("uri")
            })
        elif "uri" in chunk or "title" in chunk:
            citations.append({
                "title": chunk.get("title"),
                "url": chunk.get("uri")
            })
    for source in metadata.get("citation_sources", []) or []:
        if isinstance(source, dict):
            citations.append({
                "title": source.get("title"),
                "url": source.get("uri")
            })
    return citations


def _extract_perplexity_citations(raw_response) -> list:
    citations = []
    metadata = _get_message_metadata(raw_response)
    for result in metadata.get("web_results", []) or []:
        if isinstance(result, dict):
            citations.append({
                "title": result.get("title"),
                "url": result.get("url")
            })
    return citations


def _extract_taptap_citations(raw_response) -> list:
    citations = []
    metadata = _get_message_metadata(raw_response)
    for citation in metadata.get("citations", []) or []:
        if isinstance(citation, dict):
            citations.append({
                "title": citation.get("title"),
                "url": citation.get("url")
            })
    return citations


def _extract_doubao_citations(raw_response) -> list:
    """Extract citations from Doubao (Volcengine) response annotations"""
    citations = []
    metadata = _get_message_metadata(raw_response)
    for ann in metadata.get("annotations", []) or []:
        if isinstance(ann, dict):
            citations.append({
                "title": ann.get("title"),
                "url": ann.get("url")
            })
    return citations


_CITATION_PROVIDER_ALIASES = {
    "tap-ai": "taptap",
    "tapai": "taptap",
    "doubao": "doubao",
    "volcengine": "doubao",
    "bytedance": "doubao"
}


_CITATION_HANDLERS = {
    "openai": _extract_openai_citations,
    "gemini": _extract_gemini_citations,
    "perplexity": _extract_perplexity_citations,
    "taptap": _extract_taptap_citations,
    "doubao": _extract_doubao_citations,
}



def extract_citations(provider: str, raw_response) -> dict:
    """
    统一的引用/证据抽取器，标准化不同提供商的引用格式
    """
    citations = []
    handler_key = (provider or "").lower()
    handler_key = _CITATION_PROVIDER_ALIASES.get(handler_key, handler_key)
    handler = _CITATION_HANDLERS.get(handler_key)
    if handler:
        try:
            citations = handler(raw_response) or []
        except Exception:
            citations = []
    return {
        "citations": citations,
        "has_citations": len(citations) > 0
    }

# == SECTION 1: UTILITY FUNCTIONS == #

async def read_rows_with_condition(odps_client: ODPS, 
                                    table_name: str,
                                    partition_spec: Optional[str] = None,
                                    where_clause: Optional[str] = None,
                                    limit: Optional[int] = None) -> pd.DataFrame:
    """
    Read rows from ODPS table with optional conditions and return as pandas DataFrame.
    
    Args:
        odps_client: ODPS client instance
        table_name: Name of the table to read from
        partition_spec: Partition specification (e.g., "dm='2025-06'")
        where_clause: Additional WHERE clause conditions
        limit: Maximum number of rows to fetch
    
    Returns:
        DataFrame with the fetched data
    """
    try:
        # Base query
        if partition_spec:
            query = f"SELECT * FROM {table_name} WHERE {partition_spec}"
        else:
            query = f"SELECT * FROM {table_name}"
        
        # Add additional WHERE conditions
        if where_clause:
            if partition_spec:
                query += f" AND {where_clause}"
            else:
                query += f" WHERE {where_clause}"
        
        # Add LIMIT if specified
        if limit:
            query += f" LIMIT {limit}"
        
        logging.info(f"Executing ODPS query: {query}")
        
        # Execute the query and convert to DataFrame
        instance = odps_client.execute_sql(query)
        instance.wait_for_success()
        
        result = instance.open_reader()
        data = [list(record.values) for record in result]
        columns = [col.name for col in result.schema.columns]
        
        df = pd.DataFrame(data, columns=columns)
        logging.info(f"Successfully fetched {len(df)} rows from {table_name}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error reading from ODPS table {table_name}: {e}")
        raise

def insert_dataframe(odps_client: ODPS, 
                      df: pd.DataFrame, 
                      table_name: str, 
                      partition: Optional[str] = None, 
                      overwrite: bool = False,
                      set_id: Optional[int] = None):
    """
    Insert a pandas DataFrame into an ODPS table with optional partitioning.
    
    Args:
        odps_client: ODPS client instance
        df: DataFrame to insert
        table_name: Target table name
        partition: Partition specification (e.g., "dm='2025-06',game_id=12345,model_id=4")
        overwrite: Whether to overwrite existing data in the partition
        set_id: Optional set_id for logging purposes
    """
    try:
        if df.empty:
            logging.warning(f"DataFrame is empty, skipping insertion to {table_name}")
            return
        
        # Get the table object
        table = odps_client.get_table(table_name)
        
        if partition:
            logging.info(f"Inserting {len(df)} rows to {table_name} partition {partition} (set_id: {set_id}, overwrite: {overwrite})")
            
            # Create the partition if it doesn't exist
            if not table.exist_partition(partition):
                table.create_partition(partition, if_not_exists=True)
                logging.info(f"Created partition {partition}")
            
            # Open the partition for writing
            with table.open_writer(partition=partition, overwrite=overwrite) as writer:
                _write_records_to_table(df, table, writer)
        else:
            logging.info(f"Inserting {len(df)} rows to {table_name} (set_id: {set_id}, overwrite: {overwrite})")
            
            # Open the table for writing
            with table.open_writer(overwrite=overwrite) as writer:
                _write_records_to_table(df, table, writer)
        
        logging.info(f"Successfully inserted data to {table_name}")
        
    except Exception as e:
        logging.error(f"Error inserting data to {table_name}: {e}", exc_info=True)
        raise

def sanitize_name(name: str) -> str:
    """
    Sanitize a name for use in file paths and identifiers.
    
    Replaces all characters except alphanumeric, underscore, and Chinese characters
    with underscores to create safe file and directory names.
    
    Args:
        name: The input string to sanitize
        
    Returns:
        A sanitized string safe for use in file paths
        
    Example:
        >>> sanitize_name("心动小镇@#$")
        "心动小镇___"
    """
    return re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', name)

def _write_records_to_table(df: pd.DataFrame, table, writer):
    """Helper function to write DataFrame records to ODPS table.
    
    This function handles various data types including:
    - NaN values (converted to None)
    - JSON strings (preserved as strings)
    - Pandas NA values (converted to None)
    - Boolean values (preserved)
    - Numeric values (with NaN handling)
    """
    schema_cols = {c.name for c in table.table_schema.columns}  # 预计算
    for row in df.itertuples(index=False):  # 大表性能优化
        record = table.new_record()
        for col_name in df.columns:
            if col_name in schema_cols:
                value = getattr(row, col_name)
                
                # Handle various types of missing/null values
                if pd.isna(value):
                    value = None
                elif isinstance(value, (int, float)) and math.isnan(value):
                    value = None
                elif hasattr(pd, 'NA') and value is pd.NA:
                    value = None
                # Handle string representations of pandas NA
                elif isinstance(value, str) and value.lower() in ['nan', '<na>', 'null']:
                    value = None
                # Preserve JSON strings and other valid string data
                elif isinstance(value, str) and value.strip():
                    # Keep the string as-is (including JSON data)
                    pass
                # Handle boolean values explicitly
                elif isinstance(value, bool):
                    # Keep boolean values as-is
                    pass
                
                record[col_name] = value
        writer.write(record)

# == SECTION 2: LLM CALLING FUNCTIONS == #

async def collect_ans_call_openai_async(aclient: openai.AsyncOpenAI, query_text: str, model_name: str, system_prompt: str, params: dict = None) -> Any:
    # Get parameters from MODEL_CONFIGS or use defaults
    if params is None:
        params = {}
    
    # Build API call parameters
    # 重新启用 system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query_text})
    api_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": params.get("max_tokens", COLLECT_ANS_MAX_TOKENS)
    }
    
    # Add temperature if provided
    if "temperature" in params:
        api_params["temperature"] = params["temperature"]

    try:
        response = await aclient.chat.completions.create(**api_params)
        return response
    except Exception as e:
        # Re-raise all exceptions
        raise

async def collect_ans_call_perplexity_async(aclient: Any, query_text: str, model_name: str, system_prompt: str, params: dict = None) -> Any:
    """
    使用 helallao/perplexity-ai 的非官方实现
    - 仅以 FINAL/FINAL_ANSWER 为准输出
    - SEARCH_RESULTS → metadata['web_results']，供 extract_citations('perplexity', ...) 使用
    - 严格传递 mode/model/sources/language，避免“pro 参数不生效”
    """
    if params is None:
        params = {}

    try:
        import json, re, asyncio
        import perplexity_async
    except ImportError:
        raise ImportError("需要安装: pip install perplexity-api-async")

# 创建 Perplexity 客户端 
    perplexity_cookies = load_pplx_cookies(
        allow_cache=True,
        cache_path=os.getenv("PERPLEXITY_COOKIES_CACHE", ".perplexity_cookies_cache.json")
    )

    # 2) 初始化非官方客户端
    cli = await perplexity_async.Client(perplexity_cookies)

    # 3) 组织查询：把 system_prompt 前置，最大限度贴近网页端组合提示
    full_query = f"{system_prompt}\n\n{query_text}" if system_prompt else query_text

    # 4) 关键参数——确保“pro 模式/模型/语言/来源”真的传下去
    # helallao 的 search 支持：mode / model / sources / language / stream / incognito
    mode = params.get("mode", "pro")  # 你强调的重点：pro
    model = params.get("model", "gpt-5-thinking")
    sources = params.get("sources", ["web"])
    language = params.get("language", "zh-CN")
    incognito = params.get("incognito", False)

    # —— Debug：直接打印要发送的关键参数（你日志里能肉眼确认）——
    logging.info(f"[PPLX helallao] mode={mode}, model={model}, sources={sources}, language={language}, incognito={incognito}")

    try:
        resp = await cli.search(
            full_query,
            mode=mode,
            model=model,
            sources=sources,
            language=language,
            stream=False,
            incognito=incognito
        )
    finally:
        if hasattr(cli, "close"):
            await cli.close()

    # 5) 解析 FINAL/SEARCH_RESULTS（最小化、可维护）
    # 结构示例：resp['text'] 为步骤数组；FINAL/FINAL_ANSWER 包含最终 answer；SEARCH_RESULTS 里有 web_results
    answer = ""
    meta = {}
    if isinstance(resp, dict) and isinstance(resp.get("text"), list):
        steps = resp["text"]

        # 先拿 SEARCH_RESULTS 里的 web_results
        for it in steps:
            if it.get("step_type") == "SEARCH_RESULTS":
                c = it.get("content") or {}
                if isinstance(c, dict) and isinstance(c.get("web_results"), list):
                    meta["web_results"] = c["web_results"]
                    meta["has_citations"] = True

        # 找 FINAL / FINAL_ANSWER（找不到就取最后一个）
        final_it = None
        for it in steps:
            if it.get("step_type") in ("FINAL", "FINAL_ANSWER"):
                final_it = it
                break
        if not final_it and steps:
            final_it = steps[-1]

        # FINAL.content 可能是 dict/str；优先取 dict['answer']
        content = (final_it or {}).get("content", "")
        if isinstance(content, dict):
            answer = content.get("answer", "") or ""
        elif isinstance(content, str):
            answer = content
        else:
            answer = str(content)

    # 极端情况下兜底
    answer = (answer or "").strip()
    
    # 处理JSON格式的答案和Unicode解码
    if answer:
        try:
            # 检查是否是JSON格式的答案
            if answer.startswith('{"answer":') and answer.endswith('}'):
                import json
                answer_data = json.loads(answer)
                answer = answer_data.get('answer', answer)
            
            # 如果答案包含Unicode转义序列，解码它们
            if '\\u' in answer:
                answer = answer.encode().decode('unicode_escape')
        except Exception:
            pass

    # 6) 标准化为 OpenAI 风格，供你的上层共用
    return standardize_response(answer, metadata=meta, prompt_text=full_query)


async def collect_ans_call_deepseek_async(aclient: openai.AsyncOpenAI, query_text: str, model_name: str, system_prompt: str, params: dict = None) -> Any:
    # aclient for deepseek is initialized with DeepSeek's base_url
    if params is None:
        params = {}
    
    # 重新启用 system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query_text})
    api_params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": params.get("max_tokens", COLLECT_ANS_MAX_TOKENS)
    }
    
    # Add DeepSeek parameters (simplified per user adjustment)
    if "temperature" in params:
        api_params["temperature"] = params["temperature"]
    
    response = await aclient.chat.completions.create(**api_params)
    return response

async def collect_ans_call_gemini_async(aclient: Any, query_text: str, model_name: str, system_prompt: str, params: dict = None) -> SimpleNamespace:
    """Handle Google Gemini calls via direct HTTP API with retry/backoff."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # HTTP/2 connection for optimal performance with Gemini API
    try:
        import h2  # noqa
    except ImportError:
        raise ImportError(
            "h2 package is required for optimal Gemini API performance. "
            "Install it with: pip install 'httpx[http2]'"
        )
    params = params or {}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        http2=True
    ) as client:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        # 启用 system prompt，合并到输入文本前部
        full_prompt = f"{system_prompt}\n\n---\n\n{query_text}" if system_prompt else query_text
        if params is None:
            params = {}
        
        # Build generation config with parameters
        generation_config = {
            "maxOutputTokens": params.get("maxOutputTokens", COLLECT_ANS_MAX_TOKENS)
        }
        
        # Add optional Gemini parameters (based on user adjustment)
        if "temperature" in params:
            generation_config["temperature"] = params["temperature"]
        if "topP" in params:
            generation_config["topP"] = params["topP"]  # Now set to 1.0 (default was 0.95)
        # topK removed per user adjustment
        if "candidateCount" in params:
            generation_config["candidateCount"] = params["candidateCount"]
        
        # Configure thinking budget for Gemini 2.5 models
        # Prefer environment override GEMINI_THINKING_BUDGET; fall back to params["thinkingBudget"]
        thinking_budget_env = os.getenv("GEMINI_THINKING_BUDGET")
        thinking_budget_param = params.get("thinkingBudget") if isinstance(params, dict) else None
        thinking_budget_value = None
        try:
            if thinking_budget_env is not None and str(thinking_budget_env).strip() != "":
                thinking_budget_value = int(str(thinking_budget_env).strip())
            elif thinking_budget_param is not None:
                thinking_budget_value = int(thinking_budget_param)
        except Exception:
            thinking_budget_value = None

        # Only attach thinkingConfig when a positive budget is provided
        if isinstance(thinking_budget_value, int) and thinking_budget_value > 0:
            generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget_value}
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": generation_config
        }
        # thinkingConfig must be nested under generationConfig
        
        # 搜索开关与单格式尝试 (HTTPX分支要点)
        enable_search = os.getenv("GEMINI_ENABLE_SEARCH", "true").lower() == "true"
        if enable_search:
            payload["tools"] = [{"google_search": {}}]  # 只用一种格式：失败就失败，不启用搜索
        else:
            payload.pop("tools", None)
        
        # Exponential backoff with full jitter; honor Retry-After when provided
        # 限制总重试次数，保存最后一个错误，避免日志风暴
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "3"))  # 降低重试次数
        base_delay = float(os.getenv("GEMINI_RETRY_BASE", "2.0"))
        cap_delay = float(os.getenv("GEMINI_RETRY_CAP", "30.0"))   # 降低延迟上限

        # 简化重试逻辑
        data = None
        final_error = None
        
        attempt = 0
        while attempt < max_retries:
            try:
                response = await client.post(url, json=payload, timeout=180.0)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get('error', {}).get('message', str(e)[:200])
                except:
                    error_detail = e.response.text[:500] if hasattr(e.response, 'text') else str(e)[:200]
                
                # 对于搜索不支持的错误，关闭搜索重试
                if enable_search and ("Search Grounding is not supported" in error_detail or "not supported" in error_detail):
                    logging.warning(f"Search not supported, retrying without search: {error_detail}")
                    payload.pop("tools", None)  # 移除搜索工具
                    continue
                
                # 对于其他错误（429, 500等），进行重试
                retriable = status in {429, 500, 502, 503, 504}
                if not retriable:
                    final_error = Exception(f"Non-retriable error (status {status}): {error_detail}")
                    break
                
                attempt += 1
                if attempt >= max_retries:
                    final_error = Exception(f"Max retries exceeded (status {status}): {error_detail}")
                    break
                
                # 重试延迟逻辑
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = None
                else:
                    delay = None

                if delay is None:
                    expo = min(cap_delay, base_delay * (2 ** (attempt - 1)))
                    delay = random.random() * expo

                logging.warning(f"Gemini {status} on attempt {attempt}/{max_retries}; sleeping {delay:.2f}s before retry")
                await asyncio.sleep(delay)
                
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectTimeout) as e:
                attempt += 1
                if attempt >= max_retries:
                    final_error = Exception(f"Network error after {attempt} attempts: {str(e)[:200]}")
                    break
                expo = min(cap_delay, base_delay * (2 ** (attempt - 1)))
                delay = random.random() * expo
                logging.warning(f"Gemini network exception, retrying in {delay:.2f}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(delay)
        
        # 如果最终失败，抛出错误
        if data is None:
            error_msg = f"Gemini API call failed after all retries. Final error: {final_error}"
            logging.error(error_msg)
            raise Exception(error_msg)
        
        # Debug logging
        
        # Get usage metadata first
        usage_metadata = data.get("usageMetadata", {})
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        total_tokens = usage_metadata.get("totalTokenCount", 0)
        
        answer = ""
        # Extract answer from response
        if data.get("candidates") and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate.get("content", {}):
                parts = candidate["content"]["parts"]
                if parts and len(parts) > 0 and "text" in parts[0]:
                    answer = parts[0]["text"]
        
        # If still no answer, return empty string
        if not answer:
            answer = ""
        
        # Extract search results and grounding metadata if present
        metadata = {}
        
        # Check for grounding metadata (Gemini's search results)
        if data.get("candidates") and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            
            # Check for grounding metadata in candidate
            if "groundingMetadata" in candidate:
                grounding = candidate["groundingMetadata"]
                
                # Extract search results from grounding chunks
                if "groundingChunks" in grounding:
                    chunks = grounding["groundingChunks"]
                    metadata["grounding_chunks"] = chunks
                    metadata["has_citations"] = True
                
                # Extract web search queries if present
                if "searchQueries" in grounding:
                    metadata["search_queries"] = grounding["searchQueries"]
                
                # Extract retrieval queries if present
                if "retrievalQueries" in grounding:
                    metadata["retrieval_queries"] = grounding["retrievalQueries"]
                    
                # Add support confidence if available
                if "supportConfidence" in grounding:
                    metadata["support_confidence"] = grounding["supportConfidence"]
            
            # Check for citation metadata
            if "citationMetadata" in candidate:
                citations = candidate["citationMetadata"]
                if citations and "citationSources" in citations:
                    metadata["citation_sources"] = citations["citationSources"]
                    metadata["has_citations"] = True
        
        # Standardize the response to match the structure of the OpenAI library's response object.
        message = SimpleNamespace(content=answer)
        
        # 使用统一的引用抽取器，但保留现有的grounding元数据  
        citation_data = extract_citations("gemini", SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(metadata=metadata))]))
        if citation_data["has_citations"]:
            # 合并引用数据和现有元数据
            if metadata:
                metadata.update(citation_data)
            else:
                metadata = citation_data
        
        if metadata:
            message.metadata = metadata
        
        standardized_choice = SimpleNamespace(message=message)
        standardized_usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
        standardized_response = SimpleNamespace(choices=[standardized_choice], usage=standardized_usage)
        
        return standardized_response

async def collect_ans_call_gpt5_async(
    aclient: openai.AsyncOpenAI,
    query_text: str,
    model_name: str,
    system_prompt: str,
    params: dict = None
) -> SimpleNamespace:
    if params is None:
        params = {}

    # 是否启用托管 web_search（可用配置开关控制）
    enable_web = params.get("enable_web_search", True)
    tools = [{"type": "web_search"}] if enable_web else []

    # 统一输出上限，保证公平性
    max_out = params.get("max_output_tokens", COLLECT_ANS_MAX_TOKENS)

    api_args = {
        "model": model_name,
        "input": query_text,
        "tool_choice": "auto",
        "reasoning": {"effort": params.get("reasoning_effort", "high")},
        "text": {"verbosity": params.get("text_verbosity", "medium")},
        "max_output_tokens": max_out
    }
    # 重新启用 instructions 字段
    if system_prompt:
        api_args["instructions"] = system_prompt
    if tools:
        api_args["tools"] = tools
    if "temperature" in params:
        api_args["temperature"] = params["temperature"]

    response = await aclient.responses.create(**api_args)

    # —— 抽取正文 ——
    answer = getattr(response, "output_text", None) or ""
    if not answer:
        try:
            # 兼容拼装 message/part 的输出
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for part in getattr(item, "content", []) or []:
                        if getattr(part, "type", None) == "output_text" and getattr(part, "text", None):
                            answer += part.text
        except Exception:
            pass

    # —— 标准化返回 —— 
    message = SimpleNamespace(content=answer)
    
    # 使用统一的引用抽取器
    citation_data = extract_citations("openai", response)
    if citation_data["has_citations"]:
        message.metadata = citation_data
    else:
        # 即使没有引用，也要创建空的metadata，以便后续添加其他信息
        message.metadata = {}

    usage_raw = getattr(response, "usage", None)
    if usage_raw:
        usage = SimpleNamespace(
            prompt_tokens=getattr(usage_raw, "input_tokens", 0),
            completion_tokens=getattr(usage_raw, "output_tokens", 0),
            total_tokens=getattr(usage_raw, "total_tokens", 0),
        )
    else:
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


async def collect_ans_call_doubao_async(aclient: Any, query_text: str, model_name: str, system_prompt: str, params: dict = None) -> SimpleNamespace:
    """
    Handle Doubao (Volcengine) calls via native SDK responses.create API.
    Supports Web Search, Reasoning, and high-quality Citations.
    """
    if params is None:
        params = {}
    
    # Check if web search is enabled (default to False if not specified, but config sets to True)
    enable_web_search = params.get("enable_web_search", False)
    tools = [{"type": "web_search"}] if enable_web_search else None
    
    # 构造 input：Volcengine SDK 期望 input=[{"role": "user", "content": ...}] 或 text
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query_text})
    
    try:
        # 使用 responses.create 接口 (非流式，简化解析)
        # 注意：aclient 在这里已经是 AsyncArk 实例（或兼容对象）
        response = await aclient.responses.create(
            model=model_name,
            input=messages,
            tools=tools,
            stream=False,
            # temperature 参数可能需要通过 model specific params 传递，或者 SDK 不支持直接传
            # 这里暂时不传 temperature，避免报错
        )
        
        # --- 解析复杂响应结构 ---
        
        # 1. 提取 Reasoning (从 ResponseReasoningItem)
        reasoning_parts = []
        # 2. 提取 Answer (从 ResponseOutputMessage)
        answer_content = ""
        # 3. 提取 Citations (从 ResponseOutputMessage annotations)
        annotations = []
        # 4. 提取 Usage
        usage_info = {}
        
        if hasattr(response, "output"):
            for item in response.output:
                # 提取 Reasoning
                if getattr(item, "type", "") == "reasoning":
                    if hasattr(item, "summary"):
                        for summary in item.summary:
                            if hasattr(summary, "text"):
                                reasoning_parts.append(summary.text)
                                
                # 提取 Answer 和 Citations
                elif getattr(item, "type", "") == "message":
                    if hasattr(item, "content"):
                        for part in item.content:
                            if getattr(part, "type", "") == "output_text":
                                if hasattr(part, "text"):
                                    answer_content += part.text
                                if hasattr(part, "annotations"):
                                    # 将 annotation 对象转换为 dict
                                    for ann in part.annotations:
                                        # 尝试转为 dict，如果不行则手动提取关键字段
                                        try:
                                            ann_dict = ann.__dict__ if hasattr(ann, "__dict__") else {}
                                            # 补充一些非标准字段提取
                                            if not ann_dict and hasattr(ann, "title"):
                                                ann_dict = {
                                                    "title": getattr(ann, "title", None),
                                                    "url": getattr(ann, "url", None),
                                                    "site_name": getattr(ann, "site_name", None),
                                                    "publish_time": getattr(ann, "publish_time", None)
                                                }
                                            annotations.append(ann_dict)
                                        except Exception:
                                            pass

        # 兜底：如果没找到 output_text，尝试从 choices 找 (兼容旧结构)
        if not answer_content and hasattr(response, "choices") and response.choices:
            answer_content = response.choices[0].message.content
            
        # 提取 Usage
        if hasattr(response, "usage"):
            u = response.usage
            usage_info = {
                "prompt_tokens": getattr(u, "input_tokens", 0),
                "completion_tokens": getattr(u, "output_tokens", 0),
                "total_tokens": getattr(u, "total_tokens", 0)
            }
            # 记录工具使用情况
            if hasattr(u, "tool_usage"):
                usage_info["tool_usage"] = str(u.tool_usage)

        # 构造标准化返回
        full_reasoning = "\n\n".join(reasoning_parts)

        metadata = {
            "annotations": annotations,
            "has_citations": len(annotations) > 0,
            "reasoning_content": full_reasoning  # 将思考链存放在 metadata 中
        }
        
        # 提取 citations（保持统一接口）
        citation_data = extract_citations("doubao", SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(metadata=metadata))]))
        if citation_data["has_citations"]:
            metadata.update(citation_data)

        # 构造返回对象
        message = SimpleNamespace(content=answer_content, metadata=metadata)
        usage = SimpleNamespace(**usage_info)
        
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    except Exception as e:
        logging.error(f"Doubao (Volcengine) call failed: {e}")
        raise


async def collect_ans_call_tapai_async(aclient: Any, query_text: str, model_name: str, system_prompt: str, params: dict = None) -> SimpleNamespace:
    """
    Handles API call to TAP AI internal search assistant.
    """
    # Check if TAP AI should be disabled
    if os.getenv("DISABLE_TAP_AI", "false").lower() == "true":
        disabled_msg = "TAP AI is disabled in this environment"
        return standardize_response(
            disabled_msg,
            metadata={},
            usage={"tap_metadata": {"disabled": True}},
        )
    
    # Use longer timeout since TAP AI can be slow
    timeout_seconds = int(os.getenv("TAP_AI_TIMEOUT", "600"))  # 默认 10 分钟，可通过环境变量调整
    connect_timeout = int(os.getenv("TAP_AI_CONNECT_TIMEOUT", "10"))  # 连接超时默认10秒
    logging.info(f"Calling TAP AI with {timeout_seconds}s timeout for query: {query_text[:100]}...")
    
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout),
        verify=False,
        follow_redirects=True
    ) as client:
        if params is None:
            params = {}
        
        # Allow overriding TAP AI URL for tunnel/proxy access
        # TAP_AI_ENV can be 'sh' (default, test environment) or 'bj' (production environment)
        tap_ai_env = os.getenv("TAP_AI_ENV", "sh").lower()
        if tap_ai_env == "bj":
            default_url = "http://search-assistant.bj.internal.tapapis.com/v4/qa"
        else:
            default_url = "http://search-assistant.sh.internal.tapapis.com/v4/qa"
        
        tap_ai_url = os.getenv("TAP_AI_URL", default_url)
        
        url = tap_ai_url
        payload = {
            "workflow": params.get("workflow", "SubQueryRAG"),
            "question": query_text,
            "source_type": params.get("source_type", 100),
            "mode": params.get("mode", "2"),
            "debug": False,
            "user_id": "452747134",
            "reasoning": params.get("reasoning", True),
            "options": {
                "forbidden_cache": params.get("forbidden_cache", False),
                "enable_outer_search": params.get("enable_outer_search", True),
                "mock_llm": False
            },
            "scene_info": {
                "user": {
                    "show_new_citation": params.get("show_new_citation", True)
                }
            },
            "exp_info": {
                "exp_params": {
                    "add_spider_data": params.get("add_spider_data", True)
                }
            }
        }
        
        start_time = time.monotonic()
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            duration = time.monotonic() - start_time
            logging.info(f"TAP AI responded successfully in {duration:.2f}s with status {response.status_code}")
        
        except httpx.TimeoutException as e:
            error_msg = f"TAP AI request timed out after {timeout_seconds}s. Consider increasing timeout or using DISABLE_TAP_AI=true"
            logging.error(error_msg)
            raise Exception(error_msg) from e
        except httpx.ConnectError as e:
            error_msg = f"Cannot connect to TAP AI. If not on internal network, set DISABLE_TAP_AI=true"
            logging.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            logging.error(f"TAP AI request failed: {e}")
            raise
        
        data = response.json()
        
        # Extract answer from TAP AI response structure
        answer = data.get("answer", "")
        if not answer and data.get("data"):
            answer = data["data"].get("answer", "")

        metadata: Dict[str, Any] = {}
        workflow = data.get("workflow")
        if workflow:
            metadata["workflow"] = workflow
        reasoning = data.get("reasoning")
        # TAP AI 返回 False/None/非 bool 值都不需要写入
        if isinstance(reasoning, bool):
            if reasoning:
                metadata["reasoning"] = reasoning
        elif reasoning is not None:
            metadata["reasoning"] = reasoning

        # 临时构建响应对象用于引用抽取
        temp_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(metadata={"citations": data.get("citations", [])}))]
        )
        citation_data = extract_citations("taptap", temp_response)
        if citation_data["has_citations"]:
            metadata.update(citation_data)

        usage = {
            "tap_metadata": {
                "duration_ms": int(duration * 1000)
            }
        }

        return standardize_response(answer, metadata=metadata or None, usage=usage, prompt_text=query_text)

# == SECTION 3: CLIENT INITIALIZATION == #

def initialize_clients():
    """Initialize ODPS and OpenAI clients."""
    try:
        # Initialize OpenAI client (prefer Azure if配置存在)
        if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
            general_openai_client = AsyncAzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                timeout=60.0,
                max_retries=1  # 自己做退避
            )
            logging.info("Initialized Azure OpenAI client")
        else:
            general_openai_client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                timeout=60.0,
                max_retries=2
            )
            logging.info("Initialized OpenAI client (non-Azure)")
        
        # Initialize ODPS clients
        odps_reader = ODPS(
            access_id=AK, 
            secret_access_key=SK, 
            project=ODPS_PROJECT, 
            endpoint="http://service.cn-beijing.maxcompute.aliyun.com/api"
        )
        odps_writer = ODPS(
            access_id=SH_AK or AK, 
            secret_access_key=SH_SK or SK, 
            project=ODPS_PROJECT, 
            endpoint="http://service.cn-shanghai.maxcompute.aliyun.com/api"
        )
        
        logging.info("ODPS clients and OpenAI client initialized successfully.")
        return general_openai_client, odps_reader, odps_writer
        
    except Exception as e:
        logging.critical(f"Failed to initialize clients: {e}", exc_info=True)
        sys.exit(1)

# == SECTION 4: MODEL CONFIGS UTILITIES == #

def save_model_configs_to_odps(odps_writer: ODPS, dm_partition: str, table_names: dict = None):
    """Save MODEL_CONFIGS to ODPS for reference."""
    if table_names is None:
        table_names = get_table_names("formal")
    
    model_records = []
    for config in MODEL_CONFIGS:
        model_records.append({
            'model_id': config['id'],  # 使用 model_id 字段名，与 ODPS 表结构一致
            'model_name': config['model_name'],
            'base_model': config.get('base_model', config['model_name']),
            'model_type': config.get('model_type', 'general'),
            'provider': config['provider'],
            'params': config.get('params', json.dumps({}))
        })
    
    if model_records:
        df_models = pd.DataFrame(model_records)
        models_partition = f"dm='{dm_partition}'"
        # 模型配置表始终使用 overwrite 模式，因为模型配置通常不变，每次覆盖即可
        overwrite_models = True  # 强制设为 True，不依赖环境变量
        insert_dataframe(odps_writer, df_models, table_names["LLM_MODELS_TABLE_NAME"], partition=models_partition, overwrite=overwrite_models)
        logging.info(f"Saved {len(model_records)} model configs to ODPS")

# == SECTION 5: COMMON UTILITY FUNCTIONS == #
def setup_logging(level=logging.INFO):
    """Setup logging configuration with resilient handler management."""

    logger = logging.getLogger()
    logger.setLevel(level)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Remove existing StreamHandlers to prevent duplicate logs
    # Some libraries (like odps) or previous calls might have added one
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Always add our own controlled StreamHandler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_path = os.path.abspath('pipeline.log')
    has_file_handler = False
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and getattr(handler, 'baseFilename', None) == log_path
        ):
            handler.setFormatter(fmt)
            handler.setLevel(level)
            has_file_handler = True
            break

    if not has_file_handler:
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8',
            delay=True,
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def generate_set_id() -> int:
    """Generate a unique set ID for tracking."""
    return int(time.time() * 1000) * 1000 + random.randint(0, 999)

# == SECTION 3: ADDITIONAL LLM HELPER FUNCTIONS == #

async def llm_chat_call(client: openai.AsyncOpenAI, 
                         system: str, 
                         user: str, 
                         model: str, 
                         temperature: float, 
                         max_tokens: int,
                         response_format: Optional[Any] = None) -> str:
    """
    Make a generalized chat completion call to LLM.
    
    Args:
        client: OpenAI async client instance
        system: System prompt content
        user: User prompt content  
        model: Model name to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        response_format: Optional response format specification (e.g., JSON schema)
        
    Returns:
        The trimmed response content from the LLM
        
    Raises:
        Exception: If the API call fails
    """
    params = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if response_format is not None:
        params["response_format"] = response_format
    
    max_attempts = 4
    base_sleep = 0.6

    for attempt in range(max_attempts):
        try:
            resp = await client.chat.completions.create(**params)
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            msg = str(exc)
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if "billing_not_active" in msg:
                logging.error("Billing not active, aborting calls")
                raise
            is_429 = status == 429 or "Too Many Requests" in msg
            if is_429 and attempt < max_attempts - 1:
                sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.2)
                logging.warning(f"429/Rate limited, retry {attempt+1}/{max_attempts} after {sleep_s:.2f}s")
                await asyncio.sleep(sleep_s)
                continue
            raise

# == SECTION 4: ANSWER CLEANING UTILITIES == #

import re
from typing import Tuple

# Sanitization patterns for answer cleaning
_SANITIZE_MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\(\s*(?:https?://|http://|www\.)[^\s)]+\s*\)",
    re.IGNORECASE,
)
_SANITIZE_URL_PATTERN = re.compile(
    r"\b(?:https?://|http://|www\.)\S+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[\w\-.~/?#%&=+]*)?",
    re.IGNORECASE,
)
_SANITIZE_CITATION_BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")
_SANITIZE_CITATION_CONTENT_PATTERN = re.compile(
    r"^\^?(?:\d+|ref|refs?|cite|note\d*|footnote\d*|source\d*)$",
    re.IGNORECASE,
)
_SANITIZE_EMPTY_PARENS_PATTERN = re.compile(r"\(\s*\)")
_SANITIZE_EMPTY_BRACKETS_PATTERN = re.compile(r"\[\s*\]")
_SANITIZE_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "\U0001F004\U0001F0CF"   # mahjong tile, playing card
    "]+",
    flags=re.UNICODE,
)


def sanitize_answer_text(text: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Remove URLs, emoji, citation footnotes etc. from answer text.
    
    This function cleans LLM-generated answers by removing:
    - Markdown links: [text](url) -> text
    - Plain URLs: https://, http://, www.
    - Citation markers: [1], [^2], [ref], [cite], etc.
    - Emoji characters
    - Empty parentheses and brackets
    - Excess whitespace and newlines
    
    Chinese characters and punctuation are fully preserved.
    
    Args:
        text: The text to clean (can be None)
        
    Returns:
        A tuple of (cleaned_text, statistics_dict) where:
        - cleaned_text: The sanitized text
        - statistics_dict: Dict with cleaning stats (removed_markdown_links, 
          removed_plain_urls, removed_citations, removed_emoji, etc.)
    
    Example:
        >>> text = "访问[官网](https://example.com)了解《原神》🎮[1]"
        >>> cleaned, stats = sanitize_answer_text(text)
        >>> print(cleaned)
        访问官网了解《原神》
        >>> print(stats)
        {'removed_markdown_links': 1, 'removed_plain_urls': 0, 
         'removed_citations': 1, 'removed_emoji': 1, 'raw_equals_clean': False}
    """
    if text is None:
        return "", {"raw_equals_clean": True}

    sanitized = str(text)
    stats: Dict[str, Any] = {}
    changed = False

    def _replace_markdown(match: re.Match) -> str:
        nonlocal changed
        changed = True
        stats["removed_markdown_links"] = stats.get("removed_markdown_links", 0) + 1
        return match.group(1).strip()

    sanitized = _SANITIZE_MARKDOWN_LINK_PATTERN.sub(_replace_markdown, sanitized)

    sanitized, url_count = _SANITIZE_URL_PATTERN.subn("", sanitized)
    if url_count:
        changed = True
        stats["removed_plain_urls"] = stats.get("removed_plain_urls", 0) + url_count

    def _replace_citation(match: re.Match) -> str:
        nonlocal changed
        inner = match.group(1).strip()
        if _SANITIZE_CITATION_CONTENT_PATTERN.match(inner):
            changed = True
            stats["removed_citations"] = stats.get("removed_citations", 0) + 1
            return ""
        return match.group(0)

    sanitized = _SANITIZE_CITATION_BRACKET_PATTERN.sub(_replace_citation, sanitized)

    sanitized, emoji_count = _SANITIZE_EMOJI_PATTERN.subn("", sanitized)
    if emoji_count:
        changed = True
        stats["removed_emoji"] = stats.get("removed_emoji", 0) + emoji_count

    sanitized, empty_paren_count = _SANITIZE_EMPTY_PARENS_PATTERN.subn("", sanitized)
    if empty_paren_count:
        changed = True
        stats["removed_empty_parens"] = stats.get("removed_empty_parens", 0) + empty_paren_count

    sanitized, empty_bracket_count = _SANITIZE_EMPTY_BRACKETS_PATTERN.subn("", sanitized)
    if empty_bracket_count:
        changed = True
        stats["removed_empty_brackets"] = stats.get("removed_empty_brackets", 0) + empty_bracket_count

    # Normalize leading bullet symbols (e.g., *, •) into hyphen bullets
    bullet_pattern = re.compile(rf"^([\t ]*)([{re.escape('*•·●▪◦➤▹►')}]+)([ \t]+)(.*)$", re.MULTILINE)

    def _normalize_bullet_prefix(match: re.Match) -> str:
        nonlocal changed
        indent, _bullets, _, remainder = match.groups()
        normalized_remainder = remainder.lstrip()
        changed = True
        stats["normalized_bullet_prefixes"] = stats.get("normalized_bullet_prefixes", 0) + 1
        return f"{indent}- {normalized_remainder}" if normalized_remainder else f"{indent}-"

    sanitized = bullet_pattern.sub(_normalize_bullet_prefix, sanitized)

    # Normalize simple spacing introduced by removals
    sanitized = re.sub(r"[ \t]+\n", "\n", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    sanitized = re.sub(r" {2,}", " ", sanitized)
    sanitized = sanitized.strip()

    stats["raw_equals_clean"] = not changed
    return sanitized, stats
