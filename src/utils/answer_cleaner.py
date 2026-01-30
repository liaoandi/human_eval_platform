# -*- coding: utf-8 -*-
"""
Answer Cleaner: 独立的答案清理工具
用于清理已存储在 ODPS 中的 LLM 答案，移除 URL、emoji、引用标记等

使用方法:
python answer_cleaner.py --set_id 123 --dm_partition 2025-08 --mode test
python answer_cleaner.py --set_id 123 --dm_partition 2025-08 --mode formal --models gpt-4o gemini-2.0
python answer_cleaner.py --auto --dm_partition 2025-08 --mode test --dry-run
"""
import argparse
import asyncio
import re
import logging
import os
import sys
import time
import importlib
import json
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

# 支持通过环境变量切换 pipeline_common 模块
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

from pipeline_commons import (
    COL_QUERY_ID,
    initialize_clients,
    setup_logging,
    get_table_names,
    read_rows_with_condition,
    insert_dataframe,
    sanitize_answer_text,  # Import the cleaning function from pipeline_commons
)

# Additional tail trimming for link/title clutter
_TAIL_SITE_PATTERN = re.compile(
    r"(百度经验|知乎|小红书|3DM手游|九游|17173|TapTap|游民星空|NGA|论坛|综合讨论|手游网|门户站|WIKI|BWIKI|bilibili|哔哩哔哩|Fandom|游侠|18183|海泽网|米游社|观测枢|攻略)",
    re.IGNORECASE,
)
_TAIL_DOMAIN_PATTERN = re.compile(r"\.(?:com|cn|net|org|cc|co|top|fun|site)", re.IGNORECASE)
_SENT_END_PATTERN = re.compile(r"[。！？!?…]")
_FC_MARKER_PATTERN = re.compile(r"<\|FCResponse(?:Begin|End)\|>", re.IGNORECASE)
_INLINE_SITE_KEYWORDS = [
    "百度经验", "知乎", "小红书", "3DM", "3DM手游", "九游", "17173", "TapTap",
    "游民星空", "NGA", "WIKI", "BWIKI", "bilibili", "哔哩哔哩", "Fandom",
    "游侠", "18183", "海泽网", "米游社", "观测枢", "Game8", "9k9k", "wywyx"
]
_INLINE_SITE_PATTERN_PARENS = re.compile(
    r"[（(][^（）()]*?(?:" + "|".join(_INLINE_SITE_KEYWORDS) + r")[^（）()]*?[）)]"
)
_INLINE_SITE_PATTERN_BRACKETS = re.compile(
    r"[【\\[][^[\\]】]*?(?:" + "|".join(_INLINE_SITE_KEYWORDS) + r")[^[\\]】]*?[】\\]]"
)
_INLINE_SITE_PATTERN_SOURCE = re.compile(
    r"(?:来源[:：]\\s*)[^\\n。；;]*?(?:" + "|".join(_INLINE_SITE_KEYWORDS) + r")[^\\n。；;]*"
)
_INLINE_SITE_PATTERN_DASH = re.compile(
    r"[ \\t]*[-–—·•][ \\t]*(?:" + "|".join(_INLINE_SITE_KEYWORDS) + r")[^\\n。；;]*"
)


def trim_trailing_link_titles(text: str) -> Tuple[str, int]:
    """
    Trim site/link titles within a line, keeping the prefix before the site keyword.
    """
    if not text:
        return text, 0
    lines = text.splitlines()
    stripped = []
    removed = 0
    for line in lines:
        match_site = (_TAIL_SITE_PATTERN.search(line) or _TAIL_DOMAIN_PATTERN.search(line))
        if match_site:
            # Only trim when the site keyword is near the line end to avoid over-cutting正文
            if (len(line) - match_site.start()) <= 60:
                prefix = line[: match_site.start()].rstrip(" ：:、，,;-–—·•")
                if prefix:
                    stripped.append(prefix)
                removed += 1
                continue
        stripped.append(line)
    return "\n".join(stripped), removed


def strip_fc_markers(text: str) -> Tuple[str, int]:
    """Remove <|FCResponseBegin|> / <|FCResponseEnd|> markers."""
    cleaned, count = _FC_MARKER_PATTERN.subn("", text)
    return cleaned, count


def strip_inline_link_titles(text: str) -> Tuple[str, int]:
    """Remove in-line site/link titles (brackets, 来源:xxx, dash-separated)."""
    removed = 0
    for pat in (
        _INLINE_SITE_PATTERN_PARENS,
        _INLINE_SITE_PATTERN_BRACKETS,
        _INLINE_SITE_PATTERN_SOURCE,
    ):
        text, cnt = pat.subn("", text)
        removed += cnt
    return text, removed

# Note: All cleaning patterns and the sanitize_answer_text function are now
# imported from pipeline_commons to avoid code duplication.


def _escape_sql_string(value: str) -> str:
    """Basic SQL string escaping to prevent injection."""
    if value is None:
        return "NULL"
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


async def get_latest_set_ids_per_game(odps_reader, table_names: dict, dm_partition: str) -> Optional[List[int]]:
    """Get the latest set_id for each game based on creation time."""
    try:
        logging.info(f"Querying latest set_ids from partition dm='{dm_partition}'")
        
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


async def clean_answers_for_set(
    odps_reader,
    odps_writer,
    table_names: dict,
    set_id: int,
    dm_partition: str,
    model_ids: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    game_names: Optional[List[str]] = None,
    dry_run: bool = False
):
    """
    Clean answers for a specific set_id.
    
    Args:
        odps_reader: ODPS reader client
        odps_writer: ODPS writer client
        table_names: Dictionary of table names
        set_id: Evaluation set ID
        dm_partition: DM partition (e.g., '2025-08')
        model_ids: Optional list of model IDs to clean (if None, clean all)
        model_names: Optional list of model names to clean (if None, clean all)
        game_names: Optional list of game names to clean (if None, clean all)
        dry_run: If True, only show what would be cleaned without writing
    """
    logging.info(f"Cleaning answers for set_id={set_id}, dm='{dm_partition}'")
    
    # Build where clause
    where_parts = [f"set_id={set_id}"]
    
    if model_ids:
        model_ids_str = ",".join(str(mid) for mid in model_ids)
        where_parts.append(f"model_id IN ({model_ids_str})")
    
    if model_names:
        model_names_str = ",".join(_escape_sql_string(name) for name in model_names)
        where_parts.append(f"model_name IN ({model_names_str})")
    
    if game_names:
        game_names_str = ",".join(_escape_sql_string(name) for name in game_names)
        where_parts.append(f"game_name IN ({game_names_str})")
    
    where_clause = " AND ".join(where_parts)
    
    # Read answers from ODPS
    logging.info(f"Reading answers from ODPS with filter: {where_clause}")
    df_answers = await read_rows_with_condition(
        odps_reader,
        table_name=table_names["LLM_ANSWER_TABLE_NAME"],
        partition_spec=f"dm='{dm_partition}'",
        where_clause=where_clause,
        limit=None
    )
    
    if df_answers.empty:
        logging.warning(f"No answers found for set_id={set_id} with filter: {where_clause}")
        return
    
    logging.info(f"Found {len(df_answers)} answer records to clean")
    
    # Clean answers and collect statistics
    cleaned_count = 0
    unchanged_count = 0
    total_stats: Dict[str, int] = defaultdict(int)
    per_model_stats = defaultdict(lambda: {
        "total": 0,
        "changed": 0,
        "unchanged": 0,
        "counters": defaultdict(int),
    })
    
    cleaned_answers = []
    metadata_column_name = None
    for candidate in ["generation_metadata", "metadata", "answer_metadata"]:
        if candidate in df_answers.columns:
            metadata_column_name = candidate
            break
    
    for idx, row in df_answers.iterrows():
        original_answer = row['answer_content']
        model_key = row.get('model_name') or f"model_id:{row.get('model_id', 'unknown')}"
        per_model = per_model_stats[model_key]
        per_model["total"] += 1

        metadata_dict: Dict[str, Any] = {}
        if metadata_column_name:
            raw_metadata = row.get(metadata_column_name)
            if isinstance(raw_metadata, str) and raw_metadata.strip():
                try:
                    metadata_dict = json.loads(raw_metadata)
                except Exception as exc:
                    metadata_dict = {
                        "_answer_cleaner_metadata_parse_error": True,
                        "_answer_cleaner_metadata_parse_error_msg": str(exc),
                        "_answer_cleaner_metadata_raw": raw_metadata,
                    }

        def _get_raw_from_metadata(md: Dict[str, Any]) -> Optional[str]:
            if not isinstance(md, dict):
                return None
            candidates = []
            raw_field = md.get("raw_answer_before_clean")
            if isinstance(raw_field, list):
                candidates.extend(raw_field)
            elif isinstance(raw_field, str):
                candidates.append(raw_field)
            hist_field = md.get("raw_answer_before_clean_history")
            if isinstance(hist_field, list):
                candidates.extend(hist_field)
            elif isinstance(hist_field, str):
                candidates.append(hist_field)
            for item in candidates:
                if isinstance(item, str) and item.strip():
                    return item
            return None

        answer_to_clean = _get_raw_from_metadata(metadata_dict) or original_answer or ""
        cleaned_answer, stats = sanitize_answer_text(answer_to_clean)
        cleaned_answer, fc_removed = strip_fc_markers(cleaned_answer)
        if fc_removed:
            stats["removed_fc_markers"] = fc_removed
            stats["raw_equals_clean"] = False
        cleaned_answer, inline_removed = strip_inline_link_titles(cleaned_answer)
        if inline_removed:
            stats["removed_inline_link_titles"] = inline_removed
            stats["raw_equals_clean"] = False
        cleaned_answer, trimmed_tail = trim_trailing_link_titles(cleaned_answer)
        if trimmed_tail:
            stats["trimmed_trailing_link_titles"] = stats.get("trimmed_trailing_link_titles", 0) + trimmed_tail
            stats["raw_equals_clean"] = False
        if answer_to_clean != original_answer:
            stats["used_metadata_raw_answer"] = True

        if not stats.get("raw_equals_clean", True):
            cleaned_count += 1
            per_model["changed"] += 1
            # Aggregate statistics (capture all numeric counters)
            for key, value in stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total_stats[key] += int(value)
                    per_model["counters"][key] += int(value)

            previous_cleaning = metadata_dict.get("cleaning") if isinstance(metadata_dict.get("cleaning"), dict) else None
            if previous_cleaning:
                history = metadata_dict.setdefault("cleaning_history", [])
                history.append(previous_cleaning)

            cleaning_snapshot = dict(stats)
            cleaning_snapshot["source"] = "answer_cleaner"
            cleaning_snapshot["cleaned_at"] = int(time.time())
            metadata_dict["cleaning"] = cleaning_snapshot

            if original_answer is not None:
                prior_raw = metadata_dict.get("raw_answer_before_clean")
                if prior_raw is None:
                    metadata_dict["raw_answer_before_clean"] = original_answer
                elif isinstance(prior_raw, list):
                    if original_answer not in prior_raw:
                        prior_raw.append(original_answer)
                elif prior_raw != original_answer:
                    history_list = metadata_dict.setdefault("raw_answer_before_clean_history", [])
                    history_list.append(prior_raw)
                    metadata_dict["raw_answer_before_clean"] = original_answer
        else:
            unchanged_count += 1
            per_model["unchanged"] += 1
        
        # Create cleaned record
        cleaned_record = row.to_dict()
        cleaned_record['answer_content'] = cleaned_answer
        if metadata_column_name:
            try:
                cleaned_record[metadata_column_name] = json.dumps(metadata_dict, ensure_ascii=False)
            except Exception:
                # Fallback: store repr to avoid write failure
                cleaned_record[metadata_column_name] = json.dumps({
                    "_answer_cleaner_dump_error": True,
                    "_answer_cleaner_metadata_raw": str(metadata_dict),
                }, ensure_ascii=False)
        cleaned_answers.append(cleaned_record)
    
    # Report statistics
    logging.info(f"Cleaning statistics:")
    logging.info(f"  Total records: {len(df_answers)}")
    logging.info(f"  Changed: {cleaned_count}")
    logging.info(f"  Unchanged: {unchanged_count}")
    if total_stats:
        logging.info("  Cleaning counters:")
        for key, value in sorted(total_stats.items()):
            logging.info(f"    {key}: {value}")

    if per_model_stats:
        logging.info("Per-model statistics:")
        for model_name in sorted(per_model_stats.keys()):
            model_stat = per_model_stats[model_name]
            logging.info(
                f"  {model_name}: total={model_stat['total']}, changed={model_stat['changed']}, "
                f"unchanged={model_stat['unchanged']}"
            )
            counters = model_stat["counters"]
            if counters:
                for key, value in sorted(counters.items()):
                    logging.info(f"    {key}: {value}")
    
    if dry_run:
        logging.info("DRY RUN mode - no data written to ODPS")
        # Show a few examples
        examples_shown = 0
        for idx, row in df_answers.iterrows():
            original = row['answer_content']
            cleaned, stats = sanitize_answer_text(original)
            if not stats.get("raw_equals_clean", True) and examples_shown < 3:
                logging.info(f"\nExample {examples_shown + 1}:")
                logging.info(f"  Model: {row['model_name']}")
                logging.info(f"  Query ID: {row['query_id']}")
                logging.info(f"  Original length: {len(original)}")
                logging.info(f"  Cleaned length: {len(cleaned)}")
                logging.info(f"  Changes: {stats}")
                examples_shown += 1
        return
    
    # Write cleaned answers back to ODPS
    # Group by game_id and model_id for proper partitioning
    df_cleaned = pd.DataFrame(cleaned_answers)
    
    # Get unique partition combinations
    partition_groups = df_cleaned.groupby(['game_id', 'model_id'])
    
    logging.info(f"Writing cleaned answers to {len(partition_groups)} partitions...")
    
    for (game_id, model_id), group_df in partition_groups:
        # Remove partition columns from data (partition keys managed by ODPS)
        data_df = group_df.drop(columns=['dm', 'game_id', 'model_id'], errors='ignore')
        
        partition_str = f"dm='{dm_partition}',game_id={game_id},model_id={model_id}"
        
        logging.info(f"  Writing {len(data_df)} records to partition: {partition_str}")
        
        insert_dataframe(
            odps_writer,
            data_df,
            table_names["LLM_ANSWER_TABLE_NAME"],
            partition=partition_str,
            overwrite=True,  # Overwrite with cleaned data
            set_id=set_id
        )
    
    logging.info(f"Successfully cleaned and wrote {len(df_cleaned)} answer records")


async def main():
    """Main entry point for answer cleaner."""
    parser = argparse.ArgumentParser(
        description="Clean LLM answers stored in ODPS (remove URLs, emoji, citations, etc.)"
    )
    
    # Set ID arguments
    set_id_group = parser.add_mutually_exclusive_group(required=False)
    set_id_group.add_argument("--set_id", type=int, help="Single Set ID to clean")
    set_id_group.add_argument("--set-ids", type=int, nargs="+", help="Multiple Set IDs to clean (space-separated)")
    set_id_group.add_argument("--auto", action="store_true", help="Automatically clean latest set_id for each game")
    
    parser.add_argument("--dm_partition", type=str, required=True, help="DM partition (e.g., '2025-08')")
    parser.add_argument("--mode", type=str, choices=["formal", "test"], default="test", 
                       help="Run mode: 'formal' for production tables, 'test' for test tables")
    
    # Filtering options
    parser.add_argument("--models", type=str, nargs="+", help="Specific model names to clean (optional)")
    parser.add_argument("--model-ids", type=int, nargs="+", help="Specific model IDs to clean (optional)")
    parser.add_argument("--games", type=str, nargs="+", help="Specific game names to clean (optional)")
    
    # Execution options
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be cleaned without actually writing to ODPS")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize clients
    general_openai_client, odps_reader, odps_writer = initialize_clients()
    
    try:
        logging.info("Starting Answer Cleaner")
        
        # Get table names based on mode
        table_names = get_table_names(args.mode)
        
        # Determine set_ids to process
        if args.auto or (not args.set_id and not args.set_ids):
            latest_set_ids = await get_latest_set_ids_per_game(odps_reader, table_names, args.dm_partition)
            if latest_set_ids is None or len(latest_set_ids) == 0:
                logging.error("Could not determine latest set_ids per game. Exiting.")
                return
            set_ids = latest_set_ids
            if args.auto:
                logging.info(f"Auto-detected latest set_ids for all games: {set_ids}")
            else:
                logging.info(f"No set_id provided, auto-detected latest set_ids: {set_ids}")
        elif args.set_id:
            set_ids = [args.set_id]
        else:
            set_ids = args.set_ids
        
        logging.info(f"Will clean {len(set_ids)} set_id(s): {set_ids}")
        
        if args.dry_run:
            logging.info("DRY RUN mode enabled - no changes will be written")
        
        # Process each set_id
        for i, set_id in enumerate(set_ids, 1):
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing set_id {set_id} ({i}/{len(set_ids)})")
            logging.info(f"{'='*80}")
            
            try:
                await clean_answers_for_set(
                    odps_reader=odps_reader,
                    odps_writer=odps_writer,
                    table_names=table_names,
                    set_id=set_id,
                    dm_partition=args.dm_partition,
                    model_ids=args.model_ids,
                    model_names=args.models,
                    game_names=args.games,
                    dry_run=args.dry_run
                )
                
                logging.info(f"Successfully completed cleaning for set_id {set_id}")
                
            except Exception as e:
                logging.error(f"Failed to clean set_id {set_id}: {e}", exc_info=True)
                continue
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Answer Cleaner completed - Processed {len(set_ids)} set_id(s)")
        logging.info(f"{'='*80}")
        
    finally:
        # Close clients
        if hasattr(general_openai_client, 'close'):
            try:
                await general_openai_client.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())

