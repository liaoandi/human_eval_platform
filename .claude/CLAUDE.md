# Human Eval Platform - LLM 评测平台

## Project Overview
端到端 LLM 评测工程平台：生成评测数据集、并行收集多模型答案、计算胜率矩阵、生成 Bradley-Terry 排名（含偏差校正）。

## Key Directories
- `src/pipeline/` — 3 个核心模块：eval_set_generator, answer_collector_concurrent, pipeline_common（均为 v2）
- `src/analysis/Python/` — Bradley-Terry 分析、排名、可视化
- `src/utils/` — 共享工具

## Core Algorithms
- 多阶段去重：精确文本 → embedding KNN → LLM 语义判断
- 偏差校正框架：位置偏差、时间偏差、Golden Question 过滤
- 反作弊检测机制

## Critical Rules
- 不要简化去重流程——三阶段是经过迭代验证的最优方案
- 偏差框架是核心创新点，不要修改校正逻辑
- 设计迭代文档记录了算法演化过程，修改前先读
