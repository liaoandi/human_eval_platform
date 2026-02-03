# -*- coding: utf-8 -*-
"""
Answer Collector Concurrent V2: 基于 answer_collector_upgraded.py 的全新并发版

python answer_collector_concurrent_v2.py --set_id 123 --dm_partition 2025-08 --mode formal
"""
import argparse
import asyncio
import re
import logging
import os
import time
import json
import random
import importlib
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import openai
import pandas as pd
from tqdm.asyncio import tqdm as tqdm_async

try:
    from volcenginesdkarkruntime import AsyncArk
except ImportError:
    import logging
    logging.warning("volcenginesdkarkruntime not found. Doubao web search will be disabled.")
    AsyncArk = None

try:
    import numpy as np
except ImportError:
    np = None

# 支持通过环境变量切换 pipeline_common 模块（例如 pipeline_common_test）
PIPELINE_COMMON_MODULE = os.getenv("PIPELINE_COMMON_MODULE", "src.pipeline.pipeline_common_v2")
if PIPELINE_COMMON_MODULE != "pipeline_commons":
    try:
        commons_module = importlib.import_module(PIPELINE_COMMON_MODULE)
        sys.modules['pipeline_commons'] = commons_module
    except Exception as e:
        if PIPELINE_COMMON_MODULE == "src.pipeline.pipeline_common_v2":
            logging.warning(
                "Failed to import src.pipeline.pipeline_common_v2 (%s); falling back to pipeline_commons", e
            )
        else:
            raise ImportError(
                f"Failed to import module specified by PIPELINE_COMMON_MODULE={PIPELINE_COMMON_MODULE}: {e}"
            )

# === Import necessary functions and constants from pipeline_commons ===
from pipeline_commons import (
    # Data column constants
    COL_QUERY_ID, COL_QUERY, COL_GAME_NAME, COL_QUERY_TIME,
    
    # Configuration constants
    OPENAI_API_KEY, MODEL_CONFIGS,  # 导入模型配置，避免重复定义
    
    # Core utility functions
    insert_dataframe,
    
    # LLM call functions
    collect_ans_call_openai_async,
    collect_ans_call_gemini_async,
    collect_ans_call_internal_async,
    collect_ans_call_gpt5_async,
    collect_ans_call_doubao_async,  # Added new Doubao function
    
    # Initialization and setup functions
    initialize_clients, setup_logging,
    get_table_names,
    save_model_configs_to_odps,
    
    # Data reading functions
    read_rows_with_condition,
    
    # Answer cleaning function
    sanitize_answer_text,
)

# == UPGRADED CONCURRENT CONFIGURATION == #

# System prompt template for answer collection

COLLECT_ANS_DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
你是手机游戏内容的专业搜索助手。
请依据你已掌握的游戏知识与检索能力，尽量根据多角度提供详细解答，确保信息丰富，但不要引入无关内容。

【输出规则】
- 回答中避免出现网址、URL、超链接、角标引用（如 [1]、[^2]、[ref] 等）和 emoji。
- 直接产出可执行的完整回答，必要时在答案里说明假设，不要让用户确认或再次提问。
- 如果需要分段或分点，请使用数字序号或「-」开头，而不是其他符号。
- 游戏名使用《游戏名》样式。
- “用户提问时间”为 {query_time}，请注意当前系统时间和用户提问时间的差异，但在答案中**不要**提到用户提问时间或系统时间的信息。
- 搜索的时候可以使用中英文进行，但回答请用中文。
- 回答请尽量简练，篇幅控制在 1000 字以内。
"""

# Max concurrent API calls for answer collection
CONCURRENT_API_CALL_LIMIT = 10

# Retry settings
MAX_RETRIES = 5  # 增加到5次重试，提高对临时500错误的容错性
RETRY_DELAY = 2.0  # 增加到2秒基础延迟
# GPT-5 specific timeout (can be overridden by env var)
# 注意：OpenAI SDK 内部有重试机制，单次请求约7分钟，3次重试需要 ~25分钟
GPT5_DEFAULT_TIMEOUT = 1800  # 30 minutes for GPT-5 (accounting for OpenAI SDK internal retries)

# Global lock for ODPS write operations to ensure thread safety
ODPS_WRITE_LOCK = asyncio.Lock()

# == SECTION 1: HELPER FUNCTIONS == #

# == SECTION 1: HELPER FUNCTIONS == #

class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder that handles numpy/pandas types and other non-serializable objects.
    """
    def default(self, obj):
        # Handle numpy types
        if np:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        
        # Handle pandas types
        if pd:
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj.to_dict(orient="records") if isinstance(obj, pd.DataFrame) else obj.to_dict()
            if pd.isna(obj):
                return None
            
        # Handle bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
            
        # Handle sets (to list)
        if isinstance(obj, set):
            return list(obj)

        # Fallback to string representation for other types
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def _dumps_safe(obj: Any) -> str:
    """
    Safe JSON serialization using the custom SafeJSONEncoder.
    """
    try:
        return json.dumps(obj, cls=SafeJSONEncoder, ensure_ascii=False)
    except Exception as e:
        # Extreme fallback
        return json.dumps({"serialization_error": str(e), "object_str": str(obj)}, ensure_ascii=False)

def _to_native_scalar(value: Any) -> Any:
    """
    Legacy helper kept for compatibility, but reimplemented to be cleaner.
    Used in make_answer_id and failed_file generation.
    """
    if value is None:
        return None
    
    if np:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
            
    if pd and isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_dict(orient="records") if isinstance(value, pd.DataFrame) else value.to_dict()

    return value

def _escape_sql_string(value: str) -> str:
    """
    Basic SQL string escaping to prevent injection.
    Escapes single quotes by doubling them.
    """
    if value is None:
        return "NULL"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"

def _is_meaningful_answer(text: str, min_len: int) -> bool:
    """
    Heuristic check whether an answer is meaningful.
    """
    if text is None:
        return False
    s = str(text).strip()
    if s.lower() in {"", "{}", "[]", "null", "none"}:
        return False
    
    # Remove code fences, citations, and headings for length check
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"\[\d+\]", " ", s)
    s = re.sub(r"关于《.*?》.*?以下是详细解答[:：]", " ", s)
    
    # Keep only word chars and CJK
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return len(cleaned) >= int(min_len)

def _get_model_timeout(model_name: str) -> float:
    """Get timeout configuration for a specific model."""
    if "gpt-5" in model_name.lower():
        return float(os.getenv("GPT5_REQUEST_TIMEOUT", 
                     os.getenv("LLM_REQUEST_TIMEOUT", str(GPT5_DEFAULT_TIMEOUT))))
    return float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

def _get_retry_strategy(error: Exception, model_name: str, attempt: int) -> Tuple[bool, float]:
    """
    Determine if we should retry and how long to wait.
    Returns: (should_retry, delay_seconds)
    """
    should_retry = True
    retry_delay_multiplier = 1.0
    retry_after_hdr = None
    
    # Check for HTTP errors
    http_status = None
    if hasattr(error, 'status_code'):
        http_status = error.status_code
    elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        http_status = error.response.status_code
    elif hasattr(error, 'code') and isinstance(error.code, int):
        http_status = error.code

    # Try to read Retry-After header for 429/5xx cases
    if hasattr(error, 'response') and hasattr(error.response, 'headers'):
        try:
            retry_after_hdr = error.response.headers.get("Retry-After")
        except Exception:
            retry_after_hdr = None
        
    if http_status:
        if http_status in [401, 403]:
            return False, 0
        elif http_status == 429:
            retry_delay_multiplier = 5.0
        elif 500 <= http_status < 600:
            retry_delay_multiplier = 5.0 if "gpt-5" in model_name.lower() else 2.0
            
    # Check error messages
    error_str = str(error).lower()
    if any(p in error_str for p in ['invalid api key', 'unauthorized', 'forbidden', 'authentication']):
        return False, 0
        
    # Calculate delay with exponential backoff and jitter
    # Note: attempt limit is handled by the caller loop
        
    is_gpt5 = "gpt-5" in model_name.lower()

    # Priority: respect Retry-After when present (429/5xx)
    retry_after_val = None
    if retry_after_hdr:
        try:
            retry_after_val = float(retry_after_hdr)
        except Exception:
            retry_after_val = None

    if retry_after_val and retry_after_val > 0:
        base_delay = retry_after_val
    elif is_gpt5:
        base = float(os.getenv("GPT5_RETRY_BASE", str(RETRY_DELAY * 5.0)))
        cap = float(os.getenv("GPT5_RETRY_CAP", "600.0"))
        base_delay = min(cap, base * (2 ** attempt) * retry_delay_multiplier)
    else:
        base_delay = RETRY_DELAY * (2 ** attempt) * retry_delay_multiplier
        
    # Add jitter
    delay = min(base_delay + random.uniform(0.5, 3.0), 240.0 if is_gpt5 else 120.0)
    return True, delay


def _render_system_prompt_safe(template: str, game_name: str, query_time=None) -> str:
    """
    Safe one-time rendering of system prompt template that handles all placeholders.
    Avoids double .format() issues by rendering everything at once.
    """
    # Format query time
    formatted_time = "未知时间"
    if query_time is not None:
        try:
            # Convert to datetime and format
            query_datetime = pd.to_datetime(query_time, unit='s' if isinstance(query_time, (int, float)) else None)
            formatted_time = query_datetime.strftime('%Y年%m月')
        except Exception:
            pass  # Keep default "未知时间"
    
    # Single-pass formatting with all known placeholders
    try:
        return template.format(game_name=game_name, query_time=formatted_time)
    except KeyError as e:
        # Handle missing placeholders gracefully
        logging.warning(f"Missing placeholder in system prompt template: {e}")
        import string
        return string.Template(template).safe_substitute(game_name=game_name, query_time=formatted_time)

async def _close_client_safe(client):
    """
    Safe client closing that handles both aclose() and close() methods.
    Different OpenAI client versions may have different closing methods.
    """
    if client is None:
        return
    
    try:
        # Try aclose() first (newer async clients)
        if hasattr(client, 'aclose'):
            result = client.aclose()
            if hasattr(result, '__await__'):
                await result
        # Fallback to close() (older clients)
        elif hasattr(client, 'close'):
            result = client.close()
            if hasattr(result, '__await__'):
                await result
    except Exception as e:
        logging.warning(f"Error closing client {type(client)}: {e}")

def make_answer_id(set_id: int, model_id: int, query_id: int, timestamp: Optional[int] = None) -> int:
    """
    Generate a unique answer_id using timestamp as the primary component.
    
    Fixed 19-digit format: [timestamp_sec(10)][model_id(2)][set_suffix(3)][query_suffix(4)]
    Example: 1734353400 + 12 + 788 + 5432 = 1734353400127885432
    
    This ensures:
    - All IDs are exactly 19 digits (padding with zeros if needed)
    - Time-ordered IDs for easy sorting
    - Uniqueness through combination of model_id and ID suffixes
    - No hash computation needed
    
    Args:
        set_id: Evaluation set ID
        model_id: Model ID (will use last 2 digits)
        query_id: Query ID (will use last 4 digits)
        timestamp: Unix timestamp in seconds (if None, uses current time)
    
    Returns:
        A fixed 19-digit answer_id
    """
    # Get timestamp in seconds (10 digits until year 2286)
    if timestamp is None:
        timestamp_sec = int(time.time())
    else:
        timestamp_sec = int(timestamp)
    
    # Ensure timestamp is 10 digits (pad with zeros if needed, though unlikely)
    timestamp_sec = timestamp_sec % 10000000000
    
    # Extract fixed-width components
    model_part = model_id % 100  # Last 2 digits (00-99)
    set_suffix = (set_id % 1000)  # Last 3 digits (000-999)
    query_suffix = (query_id % 10000)  # Last 4 digits (0000-9999)
    
    # Combine all parts: timestamp(10) + model(2) + set(3) + query(4) = 19 digits
    answer_id = (
        timestamp_sec * 1000000000 +  # Shift left 9 digits
        model_part * 10000000 +        # Shift left 7 digits
        set_suffix * 10000 +           # Shift left 4 digits
        query_suffix                   # Last 4 digits
    )
    
    # Ensure exactly 19 digits by padding if necessary (though formula above guarantees it)
    return answer_id

def _apply_env_overrides(params: dict, model_name: str = None) -> dict:
    """Apply environment variable overrides to model parameters.
    
    Args:
        params: Original parameters dictionary
        model_name: Model name to check for special handling
    """
    p = dict(params or {})
    
    # Temperature override
    t = os.getenv("LLM_TEMPERATURE_OVERRIDE")
    if t is not None:
        t = float(t)
        if "temperature" in p: 
            p["temperature"] = t
        if "Temperature" in p: 
            p["Temperature"] = t
    
    # Max tokens override - but Gemini needs special handling
    m = os.getenv("LLM_MAX_TOKENS_OVERRIDE")
    if m is not None:
        m = int(m)
        if "max_tokens" in p: 
            p["max_tokens"] = m
        
        # For Gemini, don't override maxOutputTokens from environment
        # because it includes thinking tokens. Keep the model's configured value.
        if "maxOutputTokens" in p:
            # Override for non-Gemini models only
            if not (model_name and "gemini" in model_name.lower()):
                p["maxOutputTokens"] = m
    
    return p


def _resolve_concurrency_limit(model_config: dict, provider: str, model_name: str) -> int:
    """Resolve concurrency limit using model config and environment overrides."""
    try:
        limit = int(model_config.get("concurrent_limit", CONCURRENT_API_CALL_LIMIT))
    except (TypeError, ValueError):
        limit = CONCURRENT_API_CALL_LIMIT

    override_specs = (
        ("LLM_CONCURRENT_LIMIT", True),
        ("GPT5_CONCURRENT_LIMIT", "gpt-5" in (model_name or "").lower()),
        ("GEMINI_CONCURRENT_LIMIT", provider == "Google" or "gemini" in (model_name or "").lower()),
    )

    for env_name, condition in override_specs:
        if condition and (env_value := os.getenv(env_name)):
            try:
                limit = int(env_value)
            except ValueError:
                continue

    return limit


def get_available_models() -> List[str]:
    """Get list of available model names for filtering."""
    return [config["model_name"] for config in MODEL_CONFIGS]

def filter_models_by_name(model_names: List[str]) -> List[Dict]:
    """Filter MODEL_CONFIGS to include only specified models (case-insensitive, supports api_model_name)."""
    if not model_names:
        return MODEL_CONFIGS
    
    # Build case-insensitive index with both model_name and api_model_name
    idx = {}
    for c in MODEL_CONFIGS:
        # Index by model_name (lowercase)
        idx[c["model_name"].lower()] = c
        # Index by api_model_name if available (lowercase)
        if "api_model_name" in c:
            idx[c["api_model_name"].lower()] = c
    
    out = []
    for name in model_names:
        key = name.lower()
        if key in idx and idx[key] not in out:
            out.append(idx[key])
        else:
            logging.warning(f"Model '{name}' not found. Available: {sorted(set(idx.keys()))}")
    
    return out


async def _save_model_results_to_odps(
    model_config: dict,
    df_eval: pd.DataFrame,
    temp_answers_store: dict,
    temp_metadata_store: dict,
    successful_query_ids: set,
    odps_writer,
    table_names: dict,
    set_id: int,
    dm_partition: str,
    game_id: int,
    game_name: str,
    overwrite_policy: bool
):
    """
    Save a single model's results immediately to ODPS.
    This allows parallel execution without conflicts since each model uses its own partition.
    """
    try:
        col_name_suffix = model_config["column_name_suffix"]
        model_name = model_config["model_name"]
        model_id = model_config["id"]
        
        # Only process successfully handled queries for this model
        model_results = []
        skipped_empty = 0
        
        # First log how many queries are being filtered by successful_query_ids
        total_queries = len(df_eval)
        filtered_queries = len(df_eval[df_eval[COL_QUERY_ID].isin(successful_query_ids)])
        if filtered_queries < total_queries:
            logging.info(f"{model_name}: Processing {filtered_queries}/{total_queries} queries (filtered by success status)")
        
        for row in df_eval[df_eval[COL_QUERY_ID].isin(successful_query_ids)].itertuples(index=False):
            query_id = getattr(row, COL_QUERY_ID)
            answer_content = temp_answers_store[query_id].get(col_name_suffix, "")
            metadata_content = temp_metadata_store[query_id].get(col_name_suffix, '{}')
            
            # No need to check for errors - successful_query_ids already filters them
            if answer_content:  # Basic non-empty check
                # Generate timestamp once to use for both answer_id and generated_at
                generation_timestamp = int(time.time())
                result_dict = {
                    'answer_id': make_answer_id(set_id, model_id, query_id, generation_timestamp),
                    'set_id': set_id,
                    'query_id': query_id,
                    'model_id': model_id,
                    'model_name': model_name,
                    'game_name': game_name,
                    'answer_content': answer_content,
                    'generation_metadata': metadata_content,
                    'generated_at': generation_timestamp
                }
                
                # Add query_time if available in the row
                if hasattr(row, COL_QUERY_TIME) and pd.notna(getattr(row, COL_QUERY_TIME, None)):
                    result_dict['query_time'] = getattr(row, COL_QUERY_TIME)
                
                model_results.append(result_dict)
            else:
                skipped_empty += 1
        
        if skipped_empty > 0:
            logging.warning(f"{model_name}: Skipped {skipped_empty} queries with empty answers despite being marked as successful")
        
        if model_results:
            # Remove model_id field from results before creating DataFrame
            for result in model_results:
                result.pop('model_id', None)
            
            df_model_results = pd.DataFrame(model_results)
            
            # Create partition string with model_id
            partition_str = f"dm='{dm_partition}',game_id={game_id},model_id={model_id}"
            
            logging.info(f"Storing {len(df_model_results)} answers for {model_name} to ODPS partition: {partition_str}")
            
            # Use global lock for ODPS write operations to ensure thread safety
            async with ODPS_WRITE_LOCK:
                insert_dataframe(
                    odps_writer, 
                    df_model_results, 
                    table_names["LLM_ANSWER_TABLE_NAME"], 
                    partition=partition_str, 
                    overwrite=overwrite_policy, 
                    set_id=set_id
                )
            
            logging.info(f"Successfully saved {model_name} results to ODPS")
        else:
            logging.warning(f"No valid results to save for {model_name}")
            
    except Exception as e:
        logging.error(f"Failed to save {model_config.get('model_name', 'unknown')} results to ODPS: {e}", exc_info=True)

async def _call_api_with_semaphore(semaphore: asyncio.Semaphore, 
                               provider_call_func, # e.g., collect_ans_call_openai_async
                               aclient: openai.AsyncOpenAI, 
                               query_text: str, model_name: str, 
                               query_id: Any, column_suffix: str, system_prompt: str,
                               params: dict = None, query_time = None,
                               provider_name: str = None) -> Tuple[Any, str, str, str, str]:
    """
    Call API with semaphore control, retry logic and error handling.
    """
    async with semaphore:
        # Enforce cooldown for GPT-5 to avoid rate limits
        if "gpt-5" in model_name.lower():
             # Get cooldown from env, default to 10s (faster than 30s)
             cooldown = float(os.getenv("GPT5_COOLDOWN_SECONDS", "10.0"))
             if cooldown > 0:
                 await asyncio.sleep(cooldown)

        start_time = time.monotonic()
        last_error = None
        timeout_s = _get_model_timeout(model_name)
        
        current_max_retries = 10 if "gpt-5" in model_name.lower() else MAX_RETRIES

        for attempt in range(current_max_retries):
            try:
                response = await asyncio.wait_for(
                    provider_call_func(aclient, query_text, model_name, system_prompt, params),
                    timeout=timeout_s
                )
                duration = time.monotonic() - start_time
                
                # Extract answer and reasoning
                message = response.choices[0].message if response.choices else None
                raw_answer = message.content.strip() if message and message.content else ""
                
                # Check for reasoning_content (DeepSeek/Doubao style)
                reasoning = getattr(message, "reasoning_content", "") if message else ""
                
                # If reasoning exists but not in content, prepend it with <think> tags
                if reasoning and "<think>" not in raw_answer:
                    raw_answer = f"<think>\n{reasoning}\n</think>\n\n{raw_answer}"
                
                sanitized_answer, cleaning_stats = sanitize_answer_text(raw_answer)

                # Validate answer
                min_len = int(os.getenv("ANS_MIN_LEN", "5"))
                if not _is_meaningful_answer(sanitized_answer, min_len):
                    raise ValueError(f"EMPTY_OR_MEANINGLESS_ANSWER(len={len(sanitized_answer)})")

                # Build metadata
                metadata = {
                    "duration_ms": int(duration * 1000),
                    "model": model_name,
                    "params_used": params,
                    "attempts": attempt + 1,
                    "cleaning": cleaning_stats
                }

                if not cleaning_stats.get("raw_equals_clean", True):
                    metadata["raw_answer_before_clean"] = raw_answer
                
                # Add model-specific metadata
                if response.choices and hasattr(response.choices[0].message, "metadata"):
                    metadata.update(response.choices[0].message.metadata or {})
                
                # Add usage info
                if hasattr(response, "usage") and response.usage:
                    metadata["prompt_tokens"] = response.usage.prompt_tokens
                    metadata["completion_tokens"] = response.usage.completion_tokens
                    metadata["total_tokens"] = response.usage.total_tokens
                    if hasattr(response.usage, "internal_metadata"):
                        metadata.update(response.usage.internal_metadata)
                
                return query_id, column_suffix, "ok", sanitized_answer, _dumps_safe(metadata)
                
            except asyncio.TimeoutError as e:
                last_error = e
                should_retry, delay = _get_retry_strategy(e, model_name, attempt)
                if attempt >= current_max_retries - 1:
                    should_retry = False

                if should_retry:
                    logging.warning(f"{model_name} timeout on attempt {attempt + 1}/{current_max_retries} (>{timeout_s}s); retry in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                    
            except Exception as e:
                last_error = e
                should_retry, delay = _get_retry_strategy(e, model_name, attempt)
                
                if attempt >= current_max_retries - 1:
                    should_retry = False

                if should_retry:
                    logging.warning(f"{model_name} attempt {attempt + 1}/{current_max_retries} failed: {e}; retry in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logging.error(f"{model_name} non-retryable error, giving up: {e}")
                    break
        
        # Failed after all retries
        duration = time.monotonic() - start_time
        
        # Error classification
        if isinstance(last_error, asyncio.TimeoutError):
            error_msg = f"ERROR_TIMEOUT_{model_name.upper().replace('-', '_')}"
        else:
            error_msg = f"ERROR_ASYNC_API_CALL_{model_name.upper().replace('-', '_')}_REASON_{type(last_error).__name__}"
        
        # Capture truncated response for debugging
        response_text = ""
        if hasattr(last_error, 'response') and hasattr(last_error.response, 'text'):
            full_text = last_error.response.text
            response_text = full_text[:2000] + ("..." if len(full_text) > 2000 else "")

        error_metadata = {
            "duration_ms": int(duration * 1000),
            "error": str(last_error),
            "response_body": response_text,
            "attempts": current_max_retries,
            "failed_after_retries": True
        }
        
        logging.error(f"API call to {model_name} failed after {current_max_retries} attempts. Error: {last_error}")
        return query_id, column_suffix, "error", error_msg, _dumps_safe(error_metadata)

# == SECTION 2: CONCURRENT MODEL PROCESSING == #

async def _process_single_model_concurrent(
    model_config: dict,
    df_eval: pd.DataFrame,
    clients: dict,
    call_func_map: dict,
    system_prompt_template: str,
    system_prompt_game_name: str,
    temp_answers_store: dict,
    temp_metadata_store: dict,
    # ODPS parameters for immediate saving
    odps_writer=None,
    table_names: dict = None,
    set_id: int = None,
    dm_partition: str = None,
    game_id: int = None,
    game_name: str = None,
    overwrite_policy: bool = True,
    # Retry parameters
    auto_retry_failed: bool = True,
    retry_only_mode: bool = False,
    failed_query_ids: set = None
):
    """
    Process a single model concurrently with all other models.
    Enhanced with all upgraded features: better timeout, metadata handling, error classification.
    
    Args:
        auto_retry_failed: If True, automatically retry failed queries once at the end
        retry_only_mode: If True, only process queries in failed_query_ids
        failed_query_ids: Set of query_ids that previously failed (used with retry_only_mode)
    """
    provider = model_config["provider"]
    model_name = model_config["model_name"]
    
    # Check if provider is available
    if provider not in clients and provider not in ["Google", "Internal"]:
        logging.info(f"Skipping {model_name} - provider not available")
        return
    
    # Filter queries if in retry-only mode
    if retry_only_mode and failed_query_ids:
        df_to_process = df_eval[df_eval[COL_QUERY_ID].isin(failed_query_ids)]
        logging.info(f"Retry mode: Processing {len(df_to_process)} failed queries for {model_name}")
    else:
        df_to_process = df_eval
        logging.info(f"Starting concurrent processing of {model_name} ({len(df_to_process)} queries)")
    
    # Determine effective concurrency limit (with environment overrides)
    model_concurrent_limit = _resolve_concurrency_limit(model_config, provider, model_name)
    model_semaphore = asyncio.Semaphore(model_concurrent_limit)

    if model_concurrent_limit != CONCURRENT_API_CALL_LIMIT:
        logging.info(f"Using custom concurrent limit for {model_name}: {model_concurrent_limit}")

    client_to_pass = clients.get(provider)  # Will be None for Gemini and Internal
    call_func = call_func_map[model_config["call_func_ref"]]
    params_json_str = model_config.get("params", "{}")

    try:
        limit_setting = int(os.getenv("ANS_QUERY_LIMIT", "0"))
    except ValueError:
        limit_setting = 0
    remaining_limit = None if limit_setting <= 0 else limit_setting

    rows_ordered = list(df_to_process.itertuples(index=False))
    if not rows_ordered:
        logging.info(f"No queries to process for {model_name}")
        return

    row_lookup = {getattr(row, COL_QUERY_ID): row for row in rows_ordered}
    effective_prompt_template = system_prompt_template

    async def run_rows(rows, attempt_index):
        nonlocal remaining_limit
        tasks = []
        for row in rows:
            if remaining_limit is not None and remaining_limit <= 0:
                break

            query_id = getattr(row, COL_QUERY_ID)
            query_text = str(getattr(row, COL_QUERY))
            current_query_time = getattr(row, COL_QUERY_TIME, None) if hasattr(row, COL_QUERY_TIME) else None

            params = _apply_env_overrides(json.loads(params_json_str), model_config.get("model_name"))
            system_prompt = _render_system_prompt_safe(
                effective_prompt_template,
                system_prompt_game_name,
                current_query_time
            )

            tasks.append(
                _call_api_with_semaphore(
                    model_semaphore,
                    call_func,
                    client_to_pass,
                    query_text,
                    model_config["api_model_name"],
                    query_id,
                    model_config["column_name_suffix"],
                    system_prompt,
                    params,
                    current_query_time,
                    provider_name=provider
                )
            )

            if remaining_limit is not None:
                remaining_limit -= 1

        if not tasks:
            return []

        desc_suffix = "" if attempt_index == 1 else " (retry)"
        return await tqdm_async.gather(
            *tasks,
            desc=f"Collecting answers from {model_name}{desc_suffix}"
        )

    def process_results(model_results, attempt_index):
        success_ids = set()
        retry_ids = set()
        error_examples = []
        failure_details = []
        success_count = 0
        error_count = 0

        for res_tuple in model_results:
            # Normalize tuple format: extract common fields
            if len(res_tuple) == 5:
                query_id, col_suffix, status, answer_content, metadata_content = res_tuple
                is_success = (status == "ok")
            elif len(res_tuple) == 4:
                query_id, col_suffix, answer_content, metadata_content = res_tuple
                is_success = bool(answer_content and not answer_content.startswith("ERROR"))
            else:
                logging.warning(f"Unexpected result tuple format: {res_tuple}")
                continue
            
            # Store results
            temp_answers_store[query_id][col_suffix] = answer_content
            temp_metadata_store[query_id][col_suffix] = metadata_content

            # Track success/failure
            if is_success and answer_content:
                success_ids.add(query_id)
                success_count += 1
            else:
                error_count += 1
                retry_ids.add(query_id)
                if len(error_examples) < 3:
                    error_examples.append(f"query_id {query_id}: {str(answer_content)[:100]}...")
                
                reason = "empty answer" if is_success and not answer_content else "error_status"
                failure_details.append((query_id, reason, answer_content))

        total_responses = len(model_results)
        failed_responses = total_responses - len(success_ids)

        if error_count > 0:
            logging.warning(f"{model_name}: {success_count} successes, {error_count} failures")
            for example in error_examples:
                logging.warning(f"  Error example: {example}")
        elif total_responses:
            logging.info(f"{model_name}: {success_count}/{total_responses} successful")

        return {
            "success_ids": success_ids,
            "retry_ids": retry_ids,
            "error_count": error_count,
            "success_count": success_count,
            "total_responses": total_responses,
            "failed_responses": failed_responses,
            "error_examples": error_examples,
            "failure_details": failure_details,
        }

    attempt_index = 1
    pending_rows = rows_ordered
    processed_any = False
    total_responses_collected = 0
    last_summary = None

    while pending_rows:
        model_results = await run_rows(pending_rows, attempt_index)
        if not model_results:
            if not processed_any:
                logging.info(f"No queries executed for {model_name} (ANS_QUERY_LIMIT may be reached)")
                return
            break

        processed_any = True
        total_responses_collected += len(model_results)
        logging.info(f"Completed {model_name} attempt {attempt_index} - {len(model_results)} responses collected")

        summary = process_results(model_results, attempt_index)
        last_summary = summary

        if attempt_index == 1 and summary["failed_responses"] > 0:
            logging.info(f"{model_name} Summary: Total={summary['total_responses']}, Successful={summary['success_count']}, Failed={summary['failed_responses']}")
            for qid, reason, answer_preview in summary["failure_details"]:
                logging.warning(f"  Failed query_id {qid}: {reason}, answer_preview: {str(answer_preview)[:100]}")
            if summary["failure_details"] and not retry_only_mode:
                failed_file = f"failed_queries_{model_name}_{int(time.time())}.json"
                failed_ids_for_file = sorted(_to_native_scalar(qid) for qid, _, _ in summary["failure_details"])
                failure_payload = {
                    "model_name": _to_native_scalar(model_name),
                    "failed_query_ids": failed_ids_for_file,
                    "timestamp": int(time.time()),
                    "set_id": _to_native_scalar(set_id),
                    "game_name": _to_native_scalar(game_name)
                }
                try:
                    with open(failed_file, 'w', encoding='utf-8') as f:
                        json.dump(failure_payload, f, cls=SafeJSONEncoder, ensure_ascii=False, indent=2)
                    logging.info(f"Saved {len(failed_ids_for_file)} failed query IDs to {failed_file}")
                except Exception as file_err:
                    logging.error(f"Failed to write failed query file {failed_file}: {file_err}", exc_info=True)

        if (
            attempt_index == 1
            and auto_retry_failed
            and not retry_only_mode
            and summary["retry_ids"]
            and (remaining_limit is None or remaining_limit > 0)
        ):
            pending_rows = [row_lookup[qid] for qid in summary["retry_ids"] if qid in row_lookup]
            logging.info(f"Auto-retrying {len(pending_rows)} failed queries for {model_name}")
            attempt_index += 1
            continue

        if attempt_index > 1:
            logging.info(f"{model_name} retry attempt results: {summary['success_count']} successes, {summary['error_count']} failures")

        break

    if not processed_any:
        logging.info(f"No results collected for {model_name}")
        return

    col_name_suffix = model_config["column_name_suffix"]
    successful_query_ids = set()
    for query_id, answers in temp_answers_store.items():
        answer_content = answers.get(col_name_suffix, "")
        if answer_content and not answer_content.startswith("ERROR"):
            successful_query_ids.add(query_id)

    if attempt_index > 1 and last_summary is not None:
        logging.info(f"After retry: {len(successful_query_ids)} successful responses for {model_name}")

    logging.info(f"Completed {model_name} - {total_responses_collected} responses collected across {attempt_index} attempt(s)")

    # Immediately save this model's results to ODPS (only if game_id is valid)
    if odps_writer and table_names and set_id is not None and dm_partition and game_id is not None and game_id > 0:
        await _save_model_results_to_odps(
            model_config=model_config,
            df_eval=df_eval,
            temp_answers_store=temp_answers_store,
            temp_metadata_store=temp_metadata_store,
            successful_query_ids=successful_query_ids,
            odps_writer=odps_writer,
            table_names=table_names,
            set_id=set_id,
            dm_partition=dm_partition,
            game_id=game_id,
            game_name=game_name,
            overwrite_policy=overwrite_policy
        )

    # Note: Memory cleanup is handled after DataFrame population
    logging.info(f"Results stored for {model_name}")

# == SECTION 3: MAIN CONCURRENT COLLECTION FUNCTION == #

async def collect_answers_stage_concurrent(
    df_eval: pd.DataFrame, 
    game_name: str,
    model_names: Optional[List[str]] = None,
    # ODPS parameters for immediate saving
    odps_writer = None,
    table_names: dict = None,
    set_id: int = None,
    dm_partition: str = None,
    game_id: int = None,
    overwrite_policy: bool = True,
    # Retry configuration
    auto_retry_failed: bool = True,
    retry_mode: str = None,  # None, 'auto', or 'manual'
    failed_query_ids: set = None  # For manual retry mode
) -> pd.DataFrame:
    """
    Stage 4: Collects answers from various LLMs for the evaluation queries (CONCURRENT V2).
    
    Based on answer_collector_upgraded.py with full concurrent processing:
    - All models run in parallel (not sequential)
    - Enhanced with GPT-5 Pro and Gemini 2.5 Pro support
    - Advanced metadata handling for all model types
    - Better error classification and timeout handling
    - Improved retry logic with exponential backoff + jitter
    - Auto-retry failed queries feature
    
    Args:
        df_eval: DataFrame with evaluation queries
        game_name: Name of the game being processed
        model_names: Optional list of specific models to run (if None, runs all)
        odps_writer: ODPS writer client for immediate saving
        table_names: Dictionary containing ODPS table names
        set_id: Evaluation set ID
        dm_partition: DM partition (e.g., '2025-06')
        game_id: Game ID for partitioning
        overwrite_policy: Whether to overwrite existing data (True) or use insert mode (False)
        auto_retry_failed: If True, automatically retry failed queries once at the end
        retry_mode: 'auto' for automatic retry, 'manual' to only process failed queries, None for normal mode
        failed_query_ids: Set of query IDs to retry (used with retry_mode='manual')
    """
    logging.info(f"Stage 4: Collect LLM Answers for game: {game_name}")
    
    # Filter models if specific ones are requested
    models_to_process = filter_models_by_name(model_names) if model_names else MODEL_CONFIGS
    
    if model_names:
        model_list = [config["model_name"] for config in models_to_process]
        logging.info(f"Processing only specified models: {model_list}")
    else:
        logging.info(f"Processing all {len(models_to_process)} configured models")
    
    logging.info(f"Running {len(models_to_process)} models concurrently")

    # Map call function references from strings to actual functions
    call_func_map = {
        "collect_ans_call_openai_async": collect_ans_call_openai_async,
        "collect_ans_call_gemini_async": collect_ans_call_gemini_async,
        "collect_ans_call_internal_async": collect_ans_call_internal_async,
        "collect_ans_call_gpt5_async": collect_ans_call_gpt5_async,
        "collect_ans_call_doubao_async": collect_ans_call_doubao_async,
    }

    # Simplified API Key Check and Client Initialization
    provider_configs = {
        "OpenAI": {
            "key": OPENAI_API_KEY,
            "key_name": "OPENAI_API_KEY",
            # 禁用 OpenAI SDK 的内部重试，由我们自己的重试逻辑控制
            # max_retries=0 可以避免单次请求因SDK内部重试而超时
            "client_init": lambda key: openai.AsyncOpenAI(
                api_key=key, 
                max_retries=0,  # 禁用 SDK 内部重试
                timeout=1800.0  # 单次请求最长等待30分钟
            ),
        },
        "Azure": {
            "key": os.getenv("AZURE_OPENAI_API_KEY_GPT5"),
            "key_name": "AZURE_OPENAI_API_KEY_GPT5",
            "client_init": lambda key: openai.AsyncAzureOpenAI(
                api_key=key,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_GPT5", "https://your-azure-endpoint.openai.azure.com/"), 
                api_version="2025-04-01-preview", 
                # azure_deployment is removed to allow dynamic model selection via 'model' param in create()
                max_retries=0,
                timeout=1800.0
            ),
        },
        "ByteDance": {
            "key": os.getenv("DOUBAO_API_KEY"),
            "key_name": "DOUBAO_API_KEY",
            "client_init": lambda key: (AsyncArk(
                api_key=key,
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            ) if AsyncArk else openai.AsyncOpenAI(
                api_key=key,
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )),
        },
        "Google": {
            "key": os.getenv("GEMINI_API_KEY"),
            "key_name": "GEMINI_API_KEY",
            "client_init": lambda key: None, # No client to init, handled in call func
        },
        "Internal": {
            "key": "not_needed",  # Internal API, no key required
            "key_name": "TARGET_MODEL_INTERNAL",
            "client_init": lambda key: None, # No client to init, handled in call func
        }
    }

    # Filter models by available providers instead of failing entirely
    active_providers = {m['provider'] for m in models_to_process}
    missing = {
        name: conf['key_name'] 
        for name, conf in provider_configs.items() 
        if name in active_providers and not conf['key']
    }
    
    if missing:
        logging.warning(f"Missing API key(s): {', '.join(missing.values())}. These providers will be filtered out from models_to_process.")
        models_to_process = [m for m in models_to_process if m["provider"] not in missing]
        
        if not models_to_process:
            logging.error("No models left to process after filtering out providers with missing keys. Please provide at least one API key.")
            return df_eval
        
        filtered_model_names = [config["model_name"] for config in models_to_process]
        logging.info(f"Continuing with {len(models_to_process)} models: {filtered_model_names}")
    
    clients = {}
    try:
        # Initialize clients for active providers
        for name in {m['provider'] for m in models_to_process}:
            if name in provider_configs and provider_configs[name]['key']:
                config = provider_configs[name]
                clients[name] = config["client_init"](config["key"])
            else:
                logging.warning(f"Skipping provider '{name}' due to missing API key.")

        # Use safe system prompt template (no double .format issues)
        # Note: We'll render the full prompt per query with actual query_time
        system_prompt_template = COLLECT_ANS_DEFAULT_SYSTEM_PROMPT_TEMPLATE
        system_prompt_game_name = game_name or df_eval[COL_GAME_NAME].iloc[0] if not df_eval.empty and COL_GAME_NAME in df_eval.columns else "通用"

        # Storage for all model results
        temp_answers_store = defaultdict(lambda: defaultdict(str))
        temp_metadata_store = defaultdict(lambda: defaultdict(str))
        
        # Create tasks for each model to run concurrently
        model_processing_tasks = []
        for model_config in models_to_process:
            task = _process_single_model_concurrent(
                model_config=model_config,
                df_eval=df_eval,
                clients=clients,
                call_func_map=call_func_map,
                system_prompt_template=system_prompt_template,
                system_prompt_game_name=system_prompt_game_name,
                temp_answers_store=temp_answers_store,
                temp_metadata_store=temp_metadata_store,
                # ODPS parameters for immediate saving
                odps_writer=odps_writer,
                table_names=table_names,
                set_id=set_id,
                dm_partition=dm_partition,
                game_id=game_id,
                game_name=game_name,
                overwrite_policy=overwrite_policy,
                # Retry parameters
                auto_retry_failed=auto_retry_failed,
                retry_only_mode=(retry_mode == 'manual'),
                failed_query_ids=failed_query_ids  # Will be set if retry_mode == 'manual'
            )
            model_processing_tasks.append(task)
        
        # Execute all models concurrently
        start_time = time.time()
        results = await asyncio.gather(*model_processing_tasks, return_exceptions=True)
        end_time = time.time()
        
        # Report any exceptions
        success_count = 0
        for i, result in enumerate(results):
            model_name = models_to_process[i]['model_name']
            if isinstance(result, Exception):
                logging.error(f"Model {model_name} failed: {result}")
            else:
                success_count += 1
        
        logging.info(f"CONCURRENT V2 PROCESSING COMPLETED in {end_time - start_time:.1f} seconds")
        logging.info(f"Results: {success_count}/{len(models_to_process)} models completed successfully")
        
        # Create and populate answer and metadata columns efficiently (vectorized)
        qid_series = df_eval[COL_QUERY_ID]
        default_placeholder = os.getenv("DEFAULT_NOT_RUN_PLACEHOLDER", "")
        for model_config in models_to_process:
            col_name_suffix = model_config["column_name_suffix"]
            answer_column_name = f"answer_{col_name_suffix}"
            metadata_column_name = f"metadata_{col_name_suffix}"

            answer_mapper = {q_id: temp_answers_store[q_id].get(col_name_suffix, "") for q_id in temp_answers_store}
            metadata_mapper = {q_id: temp_metadata_store[q_id].get(col_name_suffix, "") for q_id in temp_metadata_store}

            df_eval[answer_column_name] = qid_series.map(
                lambda q: answer_mapper.get(q, default_placeholder)
            )
            df_eval[metadata_column_name] = qid_series.map(
                lambda q: metadata_mapper.get(q, default_placeholder)
            )

    finally:
        # Close all clients with compatibility handling
        for client in clients.values():
            await _close_client_safe(client)

    try:
        logging.info(f"Stage 4 CONCURRENT V2: Answers collected. DataFrame has {len(df_eval)} rows with new answer columns.")
        return df_eval
    except Exception as e:
        logging.error(f"Stage 4: An error occurred: {e}")
        return df_eval 

# == SECTION 4: UTILITY FUNCTIONS == #

def show_available_models():
    """Show all available models with their configurations."""
    model_names = [config["model_name"] for config in MODEL_CONFIGS]
    logging.info(f"Available Models: {', '.join(model_names)}")

# == SECTION 5: CLI HELPER FUNCTIONS == #

async def get_latest_set_ids_per_game(odps_reader, table_names: dict, dm_partition: str) -> Optional[List[int]]:
    """
    Get the latest set_id for each game based on creation time, not ID value.
    
    Args:
        odps_reader: ODPS reader instance
        table_names: Dictionary containing table names
        dm_partition: DM partition to query
    
    Returns:
        List of latest set_ids (one per game) or None if not found
    """
    try:
        logging.info(f"Querying latest set_ids from partition dm='{dm_partition}'")
        
        # Query the query_set table which has created_at timestamp
        df_all = await read_rows_with_condition(
            odps_reader,
            table_name=table_names["QUERY_SET_TABLE_NAME"],
            partition_spec=f"dm='{dm_partition}'",
            where_clause=None,
            limit=None
        )
        
        if df_all.empty or 'set_id' not in df_all.columns or 'game_name' not in df_all.columns or 'created_at' not in df_all.columns:
            logging.error(f"No data found in partition dm='{dm_partition}' or missing required columns")
            return None
        
        # Find the latest set_id for each game based on created_at timestamp
        # Get the index of the row with max created_at for each game
        latest_indices = df_all.groupby('game_name')['created_at'].idxmax()
        latest_per_game = df_all.loc[latest_indices, ['game_name', 'set_id', 'created_at']].reset_index(drop=True)
        latest_set_ids = sorted(latest_per_game['set_id'].unique())
        
        logging.info(f"Found {len(latest_set_ids)} games with latest set_ids (by creation time):")
        for row in latest_per_game.itertuples(index=False):
            created_time = pd.to_datetime(row.created_at, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"    {row.game_name}: set_id={row.set_id} (created: {created_time})")
        
        return latest_set_ids
        
    except Exception as e:
        logging.error(f"Failed to query latest set_ids per game: {e}")
        return None

async def get_latest_set_id_for_game(odps_reader, table_names: dict, dm_partition: str, game_name: str) -> Optional[int]:
    """
    Get the latest set_id for a specific game based on creation time.
    """
    try:
        logging.info(f"Querying latest set_id for game '{game_name}' in dm='{dm_partition}'")
        df = await read_rows_with_condition(
            odps_reader,
            table_name=table_names["QUERY_SET_TABLE_NAME"],
            partition_spec=f"dm='{dm_partition}'",
            where_clause=f"game_name={_escape_sql_string(game_name)}",
            limit=None
        )
        if df.empty or 'set_id' not in df.columns or 'created_at' not in df.columns:
            logging.error(f"No rows for game '{game_name}' in dm='{dm_partition}' or missing columns")
            return None
        idx = df['created_at'].idxmax()
        latest = int(df.loc[idx, 'set_id'])
        created_time = pd.to_datetime(df.loc[idx, 'created_at'], unit='s').strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Latest set_id for {game_name}: {latest} (created: {created_time})")
        return latest
    except Exception as e:
        logging.error(f"Failed to query latest set_id for game '{game_name}': {e}")
        return None

# == SECTION 6: COMMAND LINE INTERFACE == #

"""
CONCURRENT V2 Usage Examples:

基本用法（所有模型并发）:
python answer_collector_concurrent_v2.py --set_id 123 --dm_partition 2025-08 --mode formal

指定模型并发运行:
python answer_collector_concurrent_v2.py --set_id 123 --dm_partition 2025-08 --mode formal --models gpt-5 gemini-2.5-pro

环境变量优化并发数:
export GPT5_CONCURRENT_LIMIT=3      # GPT-5并发数
export GEMINI_CONCURRENT_LIMIT=4    # Gemini并发数  
export LLM_CONCURRENT_LIMIT=8       # 全局并发数
python answer_collector_concurrent_v2.py --set_id 123 --dm_partition 2025-08 --mode formal

批处理多个set_id:
python answer_collector_concurrent_v2.py --set-ids 123 124 125 --dm_partition 2025-08 --mode formal

自动检测最新set_id:
python answer_collector_concurrent_v2.py --auto --dm_partition 2025-08 --mode formal

使用 insert 模式 (不覆盖):
python answer_collector_concurrent_v2.py --set_id 123 --dm_partition 2025-08 --mode formal --insert
"""

async def main():
    """Main entry point for CONCURRENT V2 answer collector."""
    parser = argparse.ArgumentParser(description="Collect LLM answers for evaluation sets (Stage 4) - CONCURRENT V2 with all upgraded features")
    # Set ID arguments - either single, multiple, auto-detect latest, or resolve via game_name
    set_id_group = parser.add_mutually_exclusive_group(required=False)
    set_id_group.add_argument("--set_id", type=int, help="Single Set ID of the evaluation set to process")
    set_id_group.add_argument("--set-ids", type=int, nargs="+", help="Multiple Set IDs to process (space-separated)")
    set_id_group.add_argument("--auto", action="store_true", help="Automatically use the latest set_id for each game from eval_item table")
    
    parser.add_argument("--game_name", type=str, help="Game to process. If set without set_id, resolves latest set_id for this game")
    parser.add_argument("--dm_partition", type=str, required=True, help="DM partition (e.g., '2025-06')")
    parser.add_argument("--mode", type=str, choices=["formal", "test"], default="test", help="Run mode: 'formal' for production tables, 'test' for test tables")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to run (optional, space-separated)")
    
    # ODPS overwrite behavior control
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true", help="Enable overwrite mode for ODPS tables")
    overwrite_group.add_argument("--insert", action="store_true", help="Use insert mode for ODPS tables (no overwrite)")
    
    # Target model configuration via environment variables
    parser.add_argument("--target-env", type=str, choices=["sh", "bj"], help="Target model environment: 'sh' for test, 'bj' for production (sets TARGET_MODEL_ENV)")
    parser.add_argument("--target-timeout", type=int, help="Target model request timeout in seconds (sets TARGET_MODEL_TIMEOUT)")
    
    # Retry configuration
    parser.add_argument("--no-auto-retry", action="store_true", help="Disable automatic retry of failed queries")
    parser.add_argument("--retry-mode", type=str, choices=["auto", "manual"], help="Retry mode: 'auto' for automatic retry, 'manual' to only process failed queries")
    parser.add_argument("--retry-file", type=str, help="JSON file containing failed query IDs to retry (used with --retry-mode manual)")
    
    args = parser.parse_args()
    
    # Set target model environment variables if provided
    if args.target_env:
        os.environ["TARGET_MODEL_ENV"] = args.target_env
    if args.target_timeout:
        os.environ["TARGET_MODEL_TIMEOUT"] = str(args.target_timeout)
    
    # Determine overwrite policy (aligned with eval_set_generator_refactored.py)
    if args.overwrite:
        overwrite_policy = True
        logging.info("Using overwrite mode (--overwrite specified)")
    elif args.insert:
        overwrite_policy = False
        logging.info("Using insert mode (--insert specified)")
    else:
        # Fall back to environment variable or default
        env_overwrite = os.getenv("ODPS_OVERWRITE_ANSWER", "").lower()
        if env_overwrite == "true":
            overwrite_policy = True
            logging.info("Using overwrite mode (ODPS_OVERWRITE_ANSWER=true)")
        else:
            # Default: insert mode (aligned with eval_set_generator_refactored.py)
            overwrite_policy = False
            if env_overwrite == "false":
                logging.info("Using insert mode (ODPS_OVERWRITE_ANSWER=false)")
            else:
                logging.info("Using insert mode (default behavior)")
    
    # Setup logging
    setup_logging()

    # Initialize clients  
    general_openai_client, odps_reader, odps_writer = initialize_clients()
    
    try:
        logging.info("Starting CONCURRENT V2 answer collection")
        
        # Get table names based on mode
        table_names = get_table_names(args.mode)
        
        # Update model configurations table at the beginning
        try:
            logging.info(f"Updating model configurations to {table_names['LLM_MODELS_TABLE_NAME']} for partition dm='{args.dm_partition}'")
            save_model_configs_to_odps(odps_writer, args.dm_partition, table_names)
            logging.info(f"Successfully updated model configurations")
        except Exception as e:
            logging.error(f"Failed to update model configurations: {e}")
            # Continue execution even if model config update fails
        
        # Get list of set_ids to process
        if args.game_name and not args.set_id and not args.set_ids and not args.auto:
            # Resolve latest set_id for a single game
            latest_for_game = await get_latest_set_id_for_game(odps_reader, table_names, args.dm_partition, args.game_name)
            if latest_for_game is None:
                logging.error(f"Could not find latest set_id for game '{args.game_name}' in dm='{args.dm_partition}'")
                return
            set_ids = [latest_for_game]
            logging.info(f"Resolved latest set_id for game '{args.game_name}': {set_ids[0]}")
        elif args.auto or (not args.set_id and not args.set_ids):
            # Auto-detect latest set_id for each game
            latest_set_ids = await get_latest_set_ids_per_game(odps_reader, table_names, args.dm_partition)
            if latest_set_ids is None or len(latest_set_ids) == 0:
                logging.error("Could not determine latest set_ids per game. Exiting.")
                return
            set_ids = latest_set_ids
            if args.auto:
                logging.info(f"Auto-detected latest set_ids for all games: {set_ids}")
            else:
                logging.info(f"No set_id provided, auto-detected latest set_ids for all games: {set_ids}")
        elif args.set_id:
            set_ids = [args.set_id]
        else:
            set_ids = args.set_ids
        
        logging.info(f"Will process {len(set_ids)} set_id(s) in CONCURRENT V2 MODE: {set_ids}")
        
        # Show available models if not specified
        if not args.models:
            show_available_models()
        
        # Process each set_id
        for i, set_id in enumerate(set_ids, 1):
            logging.info(f"CONCURRENT V2: Processing set_id {set_id} ({i}/{len(set_ids)})")
            
            try:
                # Load evaluation set from ODPS (exclude golden standard items)
                if args.game_name:
                    # Load specific game data
                    df_game_data = await read_rows_with_condition(
                        odps_reader,
                        table_name=table_names["QUERY_ITEM_TABLE_NAME"],
                        partition_spec=f"dm='{args.dm_partition}'",
                        where_clause=f"set_id={set_id} AND game_name={_escape_sql_string(args.game_name)} AND is_golden=0",
                        limit=None
                    )
                else:
                    # Load all games for this set_id
                    df_game_data = await read_rows_with_condition(
                        odps_reader,
                        table_name=table_names["QUERY_ITEM_TABLE_NAME"],
                        partition_spec=f"dm='{args.dm_partition}'",
                        where_clause=f"set_id={set_id} AND is_golden=0",
                        limit=None
                    )
                
                if df_game_data.empty:
                    logging.error(f"No evaluation set found with set_id={set_id} in partition dm='{args.dm_partition}'")
                    continue
                
                logging.info(f"Found {len(df_game_data)} queries in evaluation set {set_id}")
                
                # Check if query_time column is present
                if COL_QUERY_TIME in df_game_data.columns:
                    non_null_query_time = df_game_data[COL_QUERY_TIME].notna().sum()
                    if non_null_query_time == 0:
                        logging.warning("Query_time column found but no non-null values")
                else:
                    logging.warning("Query_time column not found in the loaded data")
                
                if args.models:
                    logging.info(f"Will process only specified models: {args.models}")
                    # Validate model names
                    available_models = get_available_models()
                    invalid_models = [m for m in args.models if m not in available_models]
                    if invalid_models:
                        logging.error(f"Invalid model names: {invalid_models}")
                        logging.info(f"Available models: {available_models}")
                        continue
                
                # Process each game separately
                games_to_process = df_game_data['game_name'].unique()
                
                for current_game_name in games_to_process:
                    df_game_eval = df_game_data[df_game_data['game_name'] == current_game_name]
                    
                    if df_game_eval.empty:
                        logging.warning(f"No data found for game: {current_game_name}, skipping...")
                        continue
                    
                    logging.info(f"CONCURRENT V2: Collecting answers for game: {current_game_name}")
                    
                    # Extract game_id for partitioning
                    if 'game_id' in df_game_eval.columns and pd.notna(df_game_eval['game_id'].iloc[0]):
                        game_id = df_game_eval['game_id'].iloc[0]
                        if game_id <= 0:  # Validate game_id is positive
                            logging.error(f"Invalid game_id ({game_id}) for game {current_game_name}. Skipping ODPS write to avoid data pollution.")
                            game_id = None  # Skip ODPS write
                    else:
                        logging.error(f"Missing or invalid game_id for game {current_game_name}. Skipping ODPS write to avoid creating game_id=-1 partition.")
                        game_id = None  # Skip ODPS write
                    
                    # Load failed query IDs if retry file is provided
                    failed_query_ids = None
                    if args.retry_mode == 'manual' and args.retry_file:
                        try:
                            with open(args.retry_file, 'r') as f:
                                retry_data = json.load(f)
                                failed_query_ids = set(retry_data.get("failed_query_ids", []))
                                logging.info(f"Loaded {len(failed_query_ids)} failed query IDs from {args.retry_file}")
                                logging.info(f"Original model: {retry_data.get('model_name')}, Set ID: {retry_data.get('set_id')}")
                        except Exception as e:
                            logging.error(f"Failed to load retry file {args.retry_file}: {e}")
                            continue
                    
                    # Collect answers using CONCURRENT processing
                    df_with_answers = await collect_answers_stage_concurrent(
                        df_eval=df_game_eval,
                        game_name=current_game_name,
                        model_names=args.models,
                        # ODPS parameters for immediate saving
                        odps_writer=odps_writer,
                        table_names=table_names,
                        set_id=set_id,
                        dm_partition=args.dm_partition,
                        game_id=game_id,
                        overwrite_policy=overwrite_policy,
                        # Retry configuration
                        auto_retry_failed=not args.no_auto_retry,
                        retry_mode=args.retry_mode,
                        failed_query_ids=failed_query_ids
                    )
                    
                    # Show statistics from the returned DataFrame
                    if df_with_answers is not None:
                        answer_columns = [col for col in df_with_answers.columns if col.startswith('answer_')]
                        total_answers = 0
                        total_errors = 0
                        
                        for col in answer_columns:
                            non_empty = df_with_answers[col].notna() & (df_with_answers[col] != "")
                            has_errors = df_with_answers[col].str.startswith("ERROR", na=False)
                            col_answers = non_empty.sum() - has_errors.sum() 
                            col_errors = has_errors.sum()
                            total_answers += col_answers
                            total_errors += col_errors
                            
                            model_name = col.replace('answer_', '')
                            logging.info(f"    {model_name}: {col_answers} answers, {col_errors} errors")
                        
                        logging.info(f"  Total valid answers: {total_answers}, Total errors: {total_errors}")
                    
                    logging.info(f"Game Complete: {current_game_name} - Data saved to ODPS")
                
                logging.info(f"Successfully completed set_id {set_id} in CONCURRENT V2 mode")
            except Exception as e:
                logging.error(f"Failed to process set_id {set_id}: {e}")
                # Continue with next set_id instead of stopping
                continue
        
        logging.info(f"Summary - Processed Set IDs: {set_ids}")
        
    finally:
        # Close clients with compatibility handling
        await _close_client_safe(general_openai_client)

if __name__ == "__main__":
    asyncio.run(main())
