-- =================== Bias Checks Summary ===================
-- 数据来源
--   - 明细来自 eval_tasks × comparisons 解码后的 votes
--   - 关键字段：left_model_id / right_model_id / winner_model_id / eval_dim_key
--   - 题目顺序：eval_tasks.no；按每个 eval_session_id 的 max(no) 将会话分为 EARLY / LATE
--   - 数据过滤：仅包含status=12（已完成）且eval_set_id>=10的任务
--   - **用户筛选**：只统计答完全部18题的用户（按content_id去重）
--
-- 统一区间估计：二项比例的 Wilson 95% 置信区间
--   常数：z=1.959963984540054（95%置信水平对应的z值），z2=3.841458820694125（z的平方）
--   输入：n（样本数）、k（成功数），p_hat=k/n（样本比例）
--   denom = 1 + z2/n
--   center = (p_hat + z2/(2n)) / denom
--   half = z * sqrt( p_hat*(1-p_hat)/n + z2/(4n^2) ) / denom
--   lo = max(0, center - half) ; hi = min(1, center + half)
--   优点：即使在小样本情况下也比正态近似更准确
--
-- 四项偏置检验：
--
--   偏置检验的2×2框架：
--   检验对象分为两类：【用户选择行为】和【模型出现分布】
--   检验维度分为两个：【位置维度 Left/Right】和【时间维度 Early/Late】
--
--   ┌─────────────────┬──────────────────────────┬──────────────────────────┐
--   │                 │   位置维度 (Left/Right)   │   时间维度 (Early/Late)   │
--   ├─────────────────┼──────────────────────────┼──────────────────────────┤
--   │ 用户选择行为     │ ① Position Bias          │ ③ Order Bias             │
--   │                 │   左侧是否更容易获胜？    │   早晚阶段选择是否不同？  │
--   ├─────────────────┼──────────────────────────┼──────────────────────────┤
--   │ 模型出现分布     │ ④ Model Position Bias    │ ② Time Exposure Bias     │
--   │                 │   模型是否更常出现在左侧？│   模型是否更常出现在早期？│
--   └─────────────────┴──────────────────────────┴──────────────────────────┘
--
-- 1) 位置偏置（Position Bias）- 用户选择的位置倾向
--    检验：左侧模型胜率是否显著偏离50%（分维度 + 全局）
--    决策：lo>0.5 → 'left'；hi<0.5 → 'right'；否则 'non-significant'
--
-- 2) 时间曝光偏置（Time Exposure Bias）- 模型出现的时间分布
--    检验：各模型在EARLY阶段的出现频率是否显著偏离全局基线（不分维度）
--    基线：baseline_q = SUM(early_cnt) / SUM(total_cnt)
--    决策：baseline_q < lo → 'early-heavy'；baseline_q > hi → 'late-heavy'；否则 'balanced'
--
-- 3) 顺序偏置（Order Bias）- 用户选择行为的时间差异
--    检验：EARLY vs LATE 阶段的左侧胜率是否有显著差异（全局，不分维度）
--    方法：比较两个阶段的左侧胜率，使用两样本比例检验
--    决策：差异CI不跨0 → 'early-higher'/'late-higher'；否则 'no-difference'
--
-- 4) 模型位置分布偏置（Model Position Bias）- 模型出现的位置分布
--    检验：各模型出现在左侧的频率是否显著偏离50%（不分维度）
--    决策：lo>0.5 → 'left-biased'；hi<0.5 → 'right-biased'；否则 'balanced'
--
-- 5) 模型整体曝光均匀性（Model Exposure Balance）** [新增] **
--    检验：各模型的总出现次数是否均匀分布
--    方法：计算每个模型的实际出现次数与期望均值的偏离程度（标准化残差）
--    决策：|标准化残差| > 1.96 → 'over-exposed'/'under-exposed'；否则 'balanced'
--
-- 6) 模型配对均匀性（Model Pairing Balance）** [新增] **
--    检验：模型两两配对的次数是否均匀分布
--    方法：计算每对模型的实际配对次数与期望均值的偏离程度（标准化残差）
--    决策：|标准化残差| > 1.96 → 'over-paired'/'under-paired'；否则 'balanced'
--
-- 输出字段
--   method ∈ {position_bias, time_exposure, order_bias, model_position_bias, 
--             model_exposure_balance, model_pairing_balance}
--   dimension_or_model：合并后的维度/模型标识（根据检验类型智能合并eval_dim_key和model_name）
--   n, k, p_hat, ci_lo, ci_hi, decision, significance
--   z_score（标准化得分）：偏离程度的标准误差倍数，|z|>1.96为显著
--   effect_size（效应量）：实际值与期望值的差异（百分点），值越大效应越强
--   ci_margin（显著性边距）：置信区间与临界值的最小距离，正值=显著且距离越大越稳健
-- =======================================================================

WITH 
-- ============================================================
-- 第一步：数据准备 - 提取用户选择和模型信息
-- ============================================================

-- 1) 炸平eval任务表，提取用户选择明细
--    解码result字段获取：winner_id、eval_dim_id、eval_dim_key
--    保留task_no用于后续区分EARLY/LATE阶段
user_choice_exploded AS (
	SELECT 
		t.eval_set_id,
		t.content_id,
		t.eval_session_id,
		t.no AS task_no,
		(t.end_at - t.start_at) / 60.0 AS duration_mins,
		r.winner_id,
		r.eval_dim_id,
		r.eval_dim_key
	FROM eval_tasks t
	LATERAL VIEW EXPLODE(
		FROM_JSON(
			t.result,
			'array<struct<winner_id:bigint, eval_dim_id:bigint, eval_dim_key:string>>'
		)
	) e AS r
	WHERE t.dt = MAX_PT('eval_tasks')
		AND t.status = 12
		AND eval_set_id >= 10
),

-- 2) 计算每个session的题目范围
--    获取每个session的最大题号（max_no），用于划分EARLY/LATE阶段
session_span AS (
	SELECT 
		eval_session_id,
		MAX(no) AS max_no
	FROM eval_tasks
	WHERE dt = MAX_PT('eval_tasks')
		AND status = 12
		AND eval_set_id >= 10
	GROUP BY eval_session_id
),

-- 3) 为用户选择添加时间标签
--    逻辑：如果题号 <= max_no/2，标记为EARLY；否则标记为LATE
user_choice_with_time_bin AS (
	SELECT 
		u.*,
		CASE 
			WHEN 2*u.task_no <= s.max_no THEN 'EARLY'
			ELSE 'LATE'
		END AS time_bin
	FROM user_choice_exploded u
	INNER JOIN session_span s ON u.eval_session_id = s.eval_session_id
),

-- 3.1) 关联用户ID
sessions_with_user AS (
	SELECT
		id AS eval_session_id,
		user_id
	FROM eval_sessions
	WHERE dt = MAX_PT('eval_sessions')
),

-- 3.2) comparison增加game_id信息
comparison_with_game AS (
	SELECT 
		c.id AS comparison_id,
		c.eval_set_id,
		c.eval_query_id,
		c.is_golden,
		a.game_id
	FROM eval_comparisons c
	LEFT JOIN (
		SELECT DISTINCT eval_set_id, eval_query_id, game_id
		FROM llm_answers
		WHERE dt = MAX_PT('llm_answers')
	) a
		ON a.eval_set_id = c.eval_set_id
		AND a.eval_query_id = c.eval_query_id
	WHERE c.dt = MAX_PT('eval_comparisons')
),

-- 3.3) 合并用户、game_id、golden等信息
user_choice_full AS (
	SELECT
		u.eval_set_id,
		u.content_id,
		u.eval_session_id,
		u.task_no,
		u.duration_mins,
		u.winner_id,
		u.eval_dim_id,
		u.eval_dim_key,
		u.time_bin,
		s.user_id,
		c.game_id,
		c.is_golden
	FROM user_choice_with_time_bin u
	LEFT JOIN sessions_with_user s 
		ON s.eval_session_id = u.eval_session_id
	LEFT JOIN comparison_with_game c
		ON c.comparison_id = u.content_id
		AND c.eval_set_id = u.eval_set_id
),

-- 3.4) 计算题目级别的时间阈值（15%分位）
question_time_threshold AS (
	SELECT 
		eval_set_id,
		content_id,
		PERCENTILE_APPROX(duration_mins, 0.15) AS p15_duration_mins
	FROM user_choice_full
	WHERE is_golden = 0
	GROUP BY eval_set_id, content_id
),

-- 3.5) session粒度的答题统计（去重题目）
session_question_base AS (
	SELECT 
		eval_session_id,
		eval_set_id,
		game_id,
		content_id,
		is_golden,
		MAX(duration_mins) AS duration_mins
	FROM user_choice_full
	GROUP BY eval_session_id, eval_set_id, game_id, content_id, is_golden
),

-- 3.6) 维度1：答完全部15道normal题
session_dim1 AS (
	SELECT 
		eval_session_id
	FROM session_question_base
	GROUP BY eval_session_id
	HAVING COUNT(DISTINCT CASE WHEN is_golden = 0 THEN content_id END) >= 15
),

-- 3.7) 维度3：所有正常题时间>=15%分位
fast_answer_check AS (
	SELECT 
		sqb.eval_session_id,
		CASE WHEN sqb.duration_mins < qt.p15_duration_mins THEN 1 ELSE 0 END AS is_too_fast,
		CASE WHEN sqb.duration_mins >= qt.p15_duration_mins THEN 1 ELSE 0 END AS is_good_timing
	FROM session_question_base sqb
	INNER JOIN question_time_threshold qt
		ON qt.eval_set_id = sqb.eval_set_id
		AND qt.content_id = sqb.content_id
	WHERE sqb.is_golden = 0
),

session_dim3 AS (
	SELECT 
		eval_session_id
	FROM (
		SELECT 
			eval_session_id,
			SUM(is_too_fast) AS too_fast_cnt
		FROM fast_answer_check
		GROUP BY eval_session_id
	) t
	WHERE too_fast_cnt = 0
),

session_dim3_half AS (
	SELECT 
		eval_session_id
	FROM (
		SELECT 
			eval_session_id,
			SUM(is_good_timing) AS good_cnt,
			COUNT(*) AS total_cnt
		FROM fast_answer_check
		GROUP BY eval_session_id
	) t
	WHERE good_cnt >= total_cnt / 2.0
),

session_quality_flags AS (
	SELECT 
		sqb.eval_session_id,
		MAX(sqb.game_id) AS game_id,
		MAX(CASE WHEN d1.eval_session_id IS NOT NULL THEN 1 ELSE 0 END) AS flag_dim1,
		MAX(CASE WHEN d3.eval_session_id IS NOT NULL THEN 1 ELSE 0 END) AS flag_dim3,
		MAX(CASE WHEN d3h.eval_session_id IS NOT NULL THEN 1 ELSE 0 END) AS flag_dim3_half
	FROM session_question_base sqb
	LEFT JOIN session_dim1 d1 ON d1.eval_session_id = sqb.eval_session_id
	LEFT JOIN session_dim3 d3 ON d3.eval_session_id = sqb.eval_session_id
	LEFT JOIN session_dim3_half d3h ON d3h.eval_session_id = sqb.eval_session_id
	GROUP BY sqb.eval_session_id
),

session_filter_map AS (
	SELECT 
		'no_filter' AS filter_key,
		'未过滤' AS filter_label,
		eval_session_id
	FROM session_quality_flags
	
	UNION ALL
	
	SELECT 
		'filter_dim1' AS filter_key,
		'维度1_答完全部normal题' AS filter_label,
		eval_session_id
	FROM session_quality_flags
	WHERE flag_dim1 = 1
	
	UNION ALL
	
	SELECT 
		'filter_dim3' AS filter_key,
		'维度3_全部题时间>=15pct' AS filter_label,
		eval_session_id
	FROM session_quality_flags
	WHERE flag_dim3 = 1
	
	UNION ALL
	
	SELECT 
		'filter_dim1_or_dim3' AS filter_key,
		'维度1或维度3' AS filter_label,
		eval_session_id
	FROM session_quality_flags
	WHERE flag_dim1 = 1 OR flag_dim3 = 1
	
	UNION ALL
	
	SELECT 
		'filter_dim1_and_dim3' AS filter_key,
		'维度1且维度3' AS filter_label,
		eval_session_id
	FROM session_quality_flags
	WHERE flag_dim1 = 1 AND flag_dim3 = 1
),

-- 4) 解码comparison表：建立answer_id到model_id的映射
--    目标：将answer_left/answer_right转换为left_model_id/right_model_id
normal_eval AS (
	SELECT 
		id AS comparison_id,
		eval_set_id,
		eval_query_id,
		answer_left,
		answer_right
	FROM eval_comparisons
	WHERE dt = MAX_PT('eval_comparisons')
		AND is_golden = 0
),

-- 5) 模型基本信息表
models AS (
	SELECT 
		id AS model_id,
		model_name
	FROM llm_models
	WHERE dt = MAX_PT('llm_models')
),

-- 6) 答案表（answer_id → model_id 映射）
answers AS (
	SELECT 
		id AS answer_id,
		eval_set_id,
		eval_query_id,
		llm_model_id,
		game_id,
		LENGTH(COALESCE(content, '')) AS answer_char_count
	FROM llm_answers
	WHERE dt = MAX_PT('llm_answers')
),

-- 7) 答案与模型信息关联
llm_info AS (
	SELECT 
		a.answer_id,
		a.eval_set_id,
		a.eval_query_id,
		a.llm_model_id,
		a.game_id,
		m.model_name
	FROM answers a
	LEFT JOIN models m ON m.model_id = a.llm_model_id
),

-- 7.1) 模型答案长度统计
model_answer_length_stats AS (
	SELECT 
		a.llm_model_id AS model_id,
		COUNT(*) AS total_answer_cnt,
		SUM(CASE WHEN a.answer_char_count > 0 THEN 1 ELSE 0 END) AS valid_answer_cnt,
		AVG(
			CASE 
				WHEN a.answer_char_count > 0 THEN CAST(a.answer_char_count AS DOUBLE)
			END
		) AS avg_answer_chars
	FROM answers a
	GROUP BY a.llm_model_id
),

-- 8) 将comparison的左右answer解码为具体的模型ID和名称
--    关联策略：通过(eval_set_id, eval_query_id, answer_id)三元组
comparison_with_models AS (
	SELECT 
		n.comparison_id,
		n.eval_set_id,
		l.game_id,
		n.eval_query_id,
		n.answer_left,
		n.answer_right,
		l.llm_model_id AS left_model_id,
		r.llm_model_id AS right_model_id,
		l.model_name AS left_model_name,
		r.model_name AS right_model_name
	FROM normal_eval n
	LEFT JOIN llm_info l 
		ON n.eval_set_id = l.eval_set_id 
		AND n.eval_query_id = l.eval_query_id 
		AND n.answer_left = l.answer_id
	LEFT JOIN llm_info r 
		ON n.eval_set_id = r.eval_set_id 
		AND n.eval_query_id = r.eval_query_id 
		AND n.answer_right = r.answer_id
),

votes_raw AS (
	SELECT 
		fm.filter_key,
		fm.filter_label,
		d.game_id,
		uc.eval_dim_key,
		uc.time_bin,
		d.left_model_id,
		d.right_model_id,
		d.left_model_name,
		d.right_model_name,
		d.answer_left AS left_answer_id,
		d.answer_right AS right_answer_id,
		CASE 
			WHEN uc.winner_id = d.answer_left THEN d.left_model_id
			WHEN uc.winner_id = d.answer_right THEN d.right_model_id
			ELSE NULL  -- 平局或无效投票
		END AS winner_model_id
	FROM user_choice_full uc
	INNER JOIN session_filter_map fm 
		ON fm.eval_session_id = uc.eval_session_id
	INNER JOIN comparison_with_models d 
		ON uc.eval_set_id = d.eval_set_id 
		AND uc.content_id = d.comparison_id
	WHERE uc.is_golden = 0
),

-- 10) 数据质量检查：统计NULL数量（按filter维度）
votes_quality_check AS (
	SELECT 
		filter_key,
		filter_label,
		COUNT(*) AS total_records,
		SUM(CASE WHEN winner_model_id IS NOT NULL THEN 1 ELSE 0 END) AS valid_records,
		SUM(CASE WHEN winner_model_id IS NULL THEN 1 ELSE 0 END) AS invalid_records,
		ROUND(100.0 * SUM(CASE WHEN winner_model_id IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS invalid_rate_pct
	FROM votes_raw
	GROUP BY filter_key, filter_label
),

-- 11) 过滤有效投票：只保留winner_model_id不为NULL的记录
votes AS (
	SELECT *
	FROM votes_raw
	WHERE winner_model_id IS NOT NULL
),

-- ============================================================
-- 11.1) 答案长度 vs 模型胜率相关性
--       目标：在“答复级别”验证单个答案长度与胜负结果之间的线性相关性
--       样本：每条有效对战的左右答案各记一条记录（length_value vs is_win）
--       长度:使用 answer_char_count
-- ============================================================
answer_length_win_samples AS (
	SELECT *
	FROM (
		SELECT 
			v.filter_key,
			v.filter_label,
			v.eval_dim_key,
			'left' AS position_side,
			v.left_model_id AS model_id,
			v.left_model_name AS model_name,
			a.answer_char_count,
			CASE WHEN a.answer_char_count IS NOT NULL AND a.answer_char_count > 0 
					THEN CAST(a.answer_char_count AS DOUBLE)
			END AS length_value,
			CASE WHEN v.winner_model_id = v.left_model_id THEN 1.0 ELSE 0.0 END AS is_win
		FROM votes v
		INNER JOIN answers a ON a.answer_id = v.left_answer_id
		
		UNION ALL
		
		SELECT 
			v.filter_key,
			v.filter_label,
			v.eval_dim_key,
			'right' AS position_side,
			v.right_model_id AS model_id,
			v.right_model_name AS model_name,
			a.answer_char_count,
			CASE WHEN a.answer_char_count IS NOT NULL AND a.answer_char_count > 0 
					THEN CAST(a.answer_char_count AS DOUBLE)
			END AS length_value,
			CASE WHEN v.winner_model_id = v.right_model_id THEN 1.0 ELSE 0.0 END AS is_win
		FROM votes v
		INNER JOIN answers a ON a.answer_id = v.right_answer_id
	) t
	WHERE t.length_value IS NOT NULL
),

answer_length_win_corr AS (
	SELECT 
		filter_key,
		filter_label,
		COUNT(*) AS sample_cnt,
		SUM(length_value) AS sum_len,
		SUM(is_win) AS sum_win,
		SUM(length_value * is_win) AS sum_len_win,
		SUM(POWER(length_value, 2)) AS sum_len_sq,
		SUM(POWER(is_win, 2)) AS sum_win_sq
	FROM answer_length_win_samples
	GROUP BY filter_key, filter_label
	HAVING COUNT(*) >= 100
),

answer_length_win_corr_enriched AS (
	SELECT 
		base.*,
		(base.sample_cnt * base.sum_len_win - base.sum_len * base.sum_win) AS numerator,
		(base.sample_cnt * base.sum_len_sq - POWER(base.sum_len, 2)) AS denom_len,
		(base.sample_cnt * base.sum_win_sq - POWER(base.sum_win, 2)) AS denom_win
	FROM answer_length_win_corr base
),

answer_length_win_corr_metrics AS (
	SELECT 
		e.filter_key,
		e.filter_label,
		e.sample_cnt,
		CASE 
			WHEN e.denom_len > 0 AND e.denom_win > 0 
			THEN GREATEST(
				LEAST(e.numerator / SQRT(e.denom_len * e.denom_win), 0.999999),
				-0.999999
			)
		END AS corr_coef
	FROM answer_length_win_corr_enriched e
),

answer_length_win_corr_stats AS (
	SELECT 
		m.*,
		CASE 
			WHEN m.corr_coef IS NOT NULL 
				AND ABS(m.corr_coef) < 1
				AND m.sample_cnt > 3
			THEN 0.5 * LN((1 + m.corr_coef) / (1 - m.corr_coef))
		END AS fisher_z,
		CASE 
			WHEN m.sample_cnt > 3 THEN 1 / SQRT(m.sample_cnt - 3) END AS fisher_se,
		CASE 
			WHEN m.corr_coef IS NOT NULL 
				AND ABS(m.corr_coef) < 1
				AND m.sample_cnt > 2
			THEN (m.corr_coef * SQRT(m.sample_cnt - 2)) / SQRT(1 - POWER(m.corr_coef, 2))
		END AS corr_z_score
	FROM answer_length_win_corr_metrics m
),

answer_length_win_corr_ci AS (
	SELECT 
		s.*,
		CASE 
			WHEN s.fisher_z IS NOT NULL AND s.fisher_se IS NOT NULL
			THEN (EXP(2 * (s.fisher_z - 1.959963984540054 * s.fisher_se)) - 1)
				/ (EXP(2 * (s.fisher_z - 1.959963984540054 * s.fisher_se)) + 1)
		END AS corr_ci_lo,
		CASE 
			WHEN s.fisher_z IS NOT NULL AND s.fisher_se IS NOT NULL
			THEN (EXP(2 * (s.fisher_z + 1.959963984540054 * s.fisher_se)) - 1)
				/ (EXP(2 * (s.fisher_z + 1.959963984540054 * s.fisher_se)) + 1)
		END AS corr_ci_hi
	FROM answer_length_win_corr_stats s
),

answer_length_winrate_output AS (
	SELECT 
		'answer_length_winrate_corr' AS method,
		ci.filter_key,
		ci.filter_label,
		'answer_length_vs_winrate' AS dimension_or_model,
		ci.sample_cnt AS n,
		CAST(NULL AS BIGINT) AS k,
		ROUND(ci.corr_coef, 4) AS p_hat,
		ROUND(ci.corr_ci_lo, 4) AS ci_lo,
		ROUND(ci.corr_ci_hi, 4) AS ci_hi,
		CASE 
			WHEN ci.corr_ci_lo IS NULL OR ci.corr_ci_hi IS NULL THEN 'insufficient-sample'
			WHEN ci.corr_ci_lo > 0 THEN 'length-positive'
			WHEN ci.corr_ci_hi < 0 THEN 'length-negative'
			ELSE 'no-correlation'
		END AS decision,
		CASE 
			WHEN ci.corr_ci_lo IS NULL OR ci.corr_ci_hi IS NULL THEN 'not-assessed'
			WHEN ci.corr_ci_lo > 0 OR ci.corr_ci_hi < 0 THEN 'significant'
			ELSE 'not-significant'
		END AS significance,
		ROUND(ABS(ci.corr_z_score), 2) AS z_score,
		ROUND(ci.corr_coef * 100, 2) AS effect_size,
		ROUND(
			CASE 
				WHEN ci.corr_ci_lo IS NULL OR ci.corr_ci_hi IS NULL THEN NULL
				WHEN ci.corr_ci_lo > 0 THEN ci.corr_ci_lo * 100
				WHEN ci.corr_ci_hi < 0 THEN ABS(ci.corr_ci_hi) * 100
				ELSE LEAST(ABS(ci.corr_ci_lo), ABS(ci.corr_ci_hi)) * -100
			END,
		2) AS ci_margin
	FROM answer_length_win_corr_ci ci
),

-- ============================================================
-- 第二步：检验1 - 位置偏置（Position Bias）
-- 检验左侧模型的胜率是否显著偏离50%
-- 分eval_dim_key和全局两个层面统计
-- ============================================================
position_vote_counts AS (
	SELECT 
		v.filter_key,
		v.filter_label,
		v.eval_dim_key,
		COUNT(*) AS n,
		SUM(CASE WHEN v.winner_model_id = v.left_model_id THEN 1 ELSE 0 END) AS k
	FROM votes v
	GROUP BY v.filter_key, v.filter_label, v.eval_dim_key
	
	UNION ALL
	
	SELECT 
		v.filter_key,
		v.filter_label,
		'_ALL_' AS eval_dim_key,
		COUNT(*) AS n,
		SUM(CASE WHEN v.winner_model_id = v.left_model_id THEN 1 ELSE 0 END) AS k
	FROM votes v
	GROUP BY v.filter_key, v.filter_label
),

position_vote_wilson AS (
	SELECT 
		base.filter_key,
		base.filter_label,
		base.eval_dim_key,
		base.n,
		base.k,
		base.p_hat,
		-- Wilson置信区间计算：z=1.959963984540054, z2=3.841458820694125
		((base.p_hat + 3.841458820694125 / (2 * base.n)) / (1 + 3.841458820694125 / base.n)) AS center,
		(1.959963984540054 * SQRT((base.p_hat * (1 - base.p_hat) + 3.841458820694125 / (4 * base.n)) / base.n) / (1 + 3.841458820694125 / base.n)) AS half
	FROM (
		SELECT 
			sc.filter_key,
			sc.filter_label,
			sc.eval_dim_key,
			sc.n,
			sc.k,
			CAST(sc.k AS DOUBLE) / sc.n AS p_hat
		FROM position_vote_counts sc
		WHERE sc.n > 0
	) base
),

position_bias_output AS (
	SELECT 
		'position_bias' AS method,
		sw.filter_key,
		sw.filter_label,
		sw.eval_dim_key AS dimension_or_model,
		sw.n,
		sw.k,
		ROUND(sw.p_hat, 4) AS p_hat,
		ROUND(CASE WHEN sw.center - sw.half < 0 THEN 0.0 ELSE sw.center - sw.half END, 4) AS ci_lo,
		ROUND(CASE WHEN sw.center + sw.half > 1 THEN 1.0 ELSE sw.center + sw.half END, 4) AS ci_hi,
		CASE 
			WHEN sw.center - sw.half > 0.5 THEN 'left'
			WHEN sw.center + sw.half < 0.5 THEN 'right'
			ELSE 'non-significant'
		END AS decision,
		-- 基于置信区间判定显著性（不跨0.5即显著）
		CASE 
			WHEN (sw.center - sw.half > 0.5) OR (sw.center + sw.half < 0.5)
			THEN 'significant'
			ELSE 'not-significant'
		END AS significance,
		-- z_score：标准化得分（偏离0.5的标准误差倍数）
		ROUND(
			ABS(sw.p_hat - 0.5) / NULLIF(SQRT((sw.p_hat * (1 - sw.p_hat)) / sw.n), 0),
			2
		) AS z_score,
		-- effect_size：效应量（实际胜率与50%的差异，单位：百分点）
		ROUND((sw.p_hat - 0.5) * 100, 2) AS effect_size,
		-- ci_margin：置信区间与临界值0.5的最小距离（正值表示显著）
		ROUND(
			CASE 
				WHEN sw.center - sw.half > 0.5 THEN (sw.center - sw.half - 0.5) * 100  -- 左侧偏置
				WHEN sw.center + sw.half < 0.5 THEN (0.5 - sw.center - sw.half) * 100  -- 右侧偏置
				ELSE LEAST(ABS(sw.center - sw.half - 0.5), ABS(sw.center + sw.half - 0.5)) * -100  -- 不显著，取负值
			END, 
		2) AS ci_margin
	FROM position_vote_wilson sw
),

-- ============================================================
-- 第三步：检验2 - 时间曝光偏置（Time Exposure Bias）
-- 检验各模型在EARLY阶段的出现频率是否显著偏离全局基线
-- 不分维度，全局层面统计
-- ============================================================
-- 统计每个模型在EARLY/LATE各出现多少次（左侧+右侧）
model_time_exposure_left_right AS (
	SELECT 
		v.filter_key,
		v.filter_label,
		v.time_bin,
		m.model_name,
		m.model_id,
		COUNT(1) AS cnt
	FROM votes_raw v
	INNER JOIN models m ON m.model_id = v.left_model_id
	GROUP BY v.filter_key, v.filter_label, v.time_bin, m.model_name, m.model_id
	
	UNION ALL
	
	SELECT 
		v.filter_key,
		v.filter_label,
		v.time_bin,
		m.model_name,
		m.model_id,
		COUNT(1) AS cnt
	FROM votes_raw v
	INNER JOIN models m ON m.model_id = v.right_model_id
	GROUP BY v.filter_key, v.filter_label, v.time_bin, m.model_name, m.model_id
),

model_time_exposure_by_model AS (
	SELECT 
		filter_key,
		filter_label,
		te.model_name,
		SUM(CASE WHEN te.time_bin='EARLY' THEN te.cnt ELSE 0 END) AS early_cnt,
		SUM(CASE WHEN te.time_bin='LATE' THEN te.cnt ELSE 0 END) AS late_cnt
	FROM model_time_exposure_left_right te
	GROUP BY filter_key, filter_label, te.model_name
),

model_time_exposure_rate AS (
	SELECT 
		filter_key,
		filter_label,
		'_ALL_' AS eval_dim_key,
		t.model_name,
		(t.early_cnt + t.late_cnt) AS n,
		t.early_cnt AS k,
		CASE 
			WHEN (t.early_cnt + t.late_cnt) > 0 
			THEN CAST(t.early_cnt AS DOUBLE) / (t.early_cnt + t.late_cnt)
		END AS p_hat
	FROM model_time_exposure_by_model t
),

model_time_exposure_wilson AS (
	SELECT 
		r.filter_key,
		r.filter_label,
		r.eval_dim_key,
		r.model_name,
		r.n,
		r.k,
		r.p_hat,
		((r.p_hat + 3.841458820694125 / (2 * r.n)) / (1 + 3.841458820694125 / r.n)) AS center,
		(1.959963984540054 * SQRT((r.p_hat * (1 - r.p_hat) + 3.841458820694125 / (4 * r.n)) / r.n) / (1 + 3.841458820694125 / r.n)) AS half
	FROM model_time_exposure_rate r
	WHERE r.n > 0
),

model_time_exposure_with_baseline AS (
	SELECT 
		tw.*,
		CAST(SUM(tw.k) OVER (PARTITION BY tw.filter_key, tw.filter_label) AS DOUBLE)
			/ NULLIF(SUM(tw.n) OVER (PARTITION BY tw.filter_key, tw.filter_label), 0) AS baseline_q
	FROM model_time_exposure_wilson tw
),

time_exposure_output AS (
	SELECT 
		'time_exposure' AS method,
		cb.filter_key,
		cb.filter_label,
		cb.model_name AS dimension_or_model,
		cb.n,
		cb.k,
		ROUND(cb.p_hat, 4) AS p_hat,
		ROUND(CASE WHEN cb.center - cb.half < 0 THEN 0.0 ELSE cb.center - cb.half END, 4) AS ci_lo,
		ROUND(CASE WHEN cb.center + cb.half > 1 THEN 1.0 ELSE cb.center + cb.half END, 4) AS ci_hi,
		CASE 
			WHEN cb.baseline_q < (cb.center - cb.half) THEN 'early-heavy'
			WHEN cb.baseline_q > (cb.center + cb.half) THEN 'late-heavy'
			ELSE 'balanced'
		END AS decision,
		CASE 
			WHEN (cb.baseline_q < cb.center - cb.half) OR (cb.baseline_q > cb.center + cb.half)
			THEN 'significant'
			ELSE 'not-significant'
		END AS significance,
		ROUND(
			ABS(cb.p_hat - cb.baseline_q) / NULLIF(SQRT((cb.p_hat * (1 - cb.p_hat)) / cb.n), 0),
			2
		) AS z_score,
		ROUND((cb.p_hat - cb.baseline_q) * 100, 2) AS effect_size,
		ROUND(
			CASE 
				WHEN cb.baseline_q < cb.center - cb.half THEN (cb.center - cb.half - cb.baseline_q) * 100
				WHEN cb.baseline_q > cb.center + cb.half THEN (cb.baseline_q - cb.center - cb.half) * 100
				ELSE LEAST(ABS(cb.center - cb.half - cb.baseline_q), ABS(cb.center + cb.half - cb.baseline_q)) * -100
			END, 
		2) AS ci_margin
	FROM model_time_exposure_with_baseline cb
),

-- ============================================================
-- 第四步：检验3 - 顺序偏置（Order Bias）
-- 检验EARLY vs LATE阶段左侧胜率的差异
-- 全局层面，不分维度
-- ============================================================
order_bias_vote_counts AS (
	SELECT 
		v.filter_key,
		v.filter_label,
		v.time_bin,
		COUNT(*) AS n,
		SUM(CASE WHEN v.winner_model_id = v.left_model_id THEN 1 ELSE 0 END) AS k
	FROM votes v
	GROUP BY v.filter_key, v.filter_label, v.time_bin
),

order_bias_rates AS (
	SELECT 
		filter_key,
		filter_label,
		time_bin,
		n,
		k,
		CAST(k AS DOUBLE) / n AS p_hat
	FROM order_bias_vote_counts
),

-- 使用条件聚合避免CROSS JOIN（ODPS不支持笛卡尔积）
order_bias_aggregated AS (
	SELECT 
		filter_key,
		filter_label,
		MAX(CASE WHEN time_bin = 'EARLY' THEN p_hat END) AS p_early,
		MAX(CASE WHEN time_bin = 'LATE' THEN p_hat END) AS p_late,
		MAX(CASE WHEN time_bin = 'EARLY' THEN n END) AS n_early,
		MAX(CASE WHEN time_bin = 'LATE' THEN n END) AS n_late
	FROM order_bias_rates
	GROUP BY filter_key, filter_label
),

order_bias_comparison AS (
	-- 计算EARLY和LATE两个比例的差异及其置信区间（两样本比例检验）
	SELECT 
		filter_key,
		filter_label,
		p_early,
		p_late,
		n_early,
		n_late,
		-- 差异：early - late（正值表示early更高）
		(p_early - p_late) AS diff,
		-- 合并标准误差（两样本比例差异）
		CASE 
			WHEN n_early > 0 AND n_late > 0 THEN SQRT(
				(p_early * (1 - p_early) / n_early) + (p_late * (1 - p_late) / n_late)
			)
			ELSE NULL
		END AS se_diff
	FROM order_bias_aggregated
),

order_bias_output AS (
	SELECT 
		'order_bias' AS method,
		oc.filter_key,
		oc.filter_label,
		'EARLY_vs_LATE' AS dimension_or_model,
		oc.n_early + oc.n_late AS n,
		CAST(NULL AS BIGINT) AS k,  -- 不适用于两样本检验
		ROUND(oc.diff, 4) AS p_hat,  -- 这里表示差异值
		-- 差异的95%置信区间
		ROUND(oc.diff - 1.959963984540054 * oc.se_diff, 4) AS ci_lo,
		ROUND(oc.diff + 1.959963984540054 * oc.se_diff, 4) AS ci_hi,
		CASE 
			WHEN (oc.diff - 1.959963984540054 * oc.se_diff) > 0 THEN 'early-higher'
			WHEN (oc.diff + 1.959963984540054 * oc.se_diff) < 0 THEN 'late-higher'
			ELSE 'no-difference'
		END AS decision,
		-- 基于置信区间判定显著性（不跨0即显著）
		CASE 
			WHEN (oc.diff - 1.959963984540054 * oc.se_diff > 0) OR (oc.diff + 1.959963984540054 * oc.se_diff < 0)
			THEN 'significant'
			ELSE 'not-significant'
		END AS significance,
		-- z_score：标准化得分（差异/标准误差）
		ROUND(ABS(oc.diff) / NULLIF(oc.se_diff, 0), 2) AS z_score,
		-- effect_size：效应量（两阶段左侧胜率的差异，单位：百分点）
		ROUND(oc.diff * 100, 2) AS effect_size,
		-- ci_margin：置信区间与0的最小距离（正值表示显著）
		ROUND(
			CASE 
				WHEN oc.diff - 1.959963984540054 * oc.se_diff > 0 THEN (oc.diff - 1.959963984540054 * oc.se_diff) * 100
				WHEN oc.diff + 1.959963984540054 * oc.se_diff < 0 THEN ABS(oc.diff + 1.959963984540054 * oc.se_diff) * 100
				ELSE LEAST(ABS(oc.diff - 1.959963984540054 * oc.se_diff), ABS(oc.diff + 1.959963984540054 * oc.se_diff)) * -100
			END, 
		2) AS ci_margin
	FROM order_bias_comparison oc
),

-- ============================================================
-- 第五步：检验4 - 模型位置分布偏置（Model Position Bias）
-- 检验各模型出现在左侧的频率是否显著偏离50%
-- 全局层面，不分维度
-- ============================================================
model_position_left_counts AS (
	-- 统计每个模型在左侧和右侧出现的次数
	SELECT 
		v.filter_key,
		v.filter_label,
		m.model_name,
		COUNT(1) AS cnt_left
	FROM votes_raw v
	INNER JOIN models m ON m.model_id = v.left_model_id
	GROUP BY v.filter_key, v.filter_label, m.model_name
),

model_position_right_counts AS (
	SELECT 
		v.filter_key,
		v.filter_label,
		m.model_name,
		COUNT(1) AS cnt_right
	FROM votes_raw v
	INNER JOIN models m ON m.model_id = v.right_model_id
	GROUP BY v.filter_key, v.filter_label, m.model_name
),

-- 汇总每个模型的左右出现次数
model_position_summary AS (
	SELECT 
		COALESCE(l.filter_key, r.filter_key) AS filter_key,
		COALESCE(l.filter_label, r.filter_label) AS filter_label,
		COALESCE(l.model_name, r.model_name) AS model_name,
		COALESCE(l.cnt_left, 0) AS cnt_left,
		COALESCE(r.cnt_right, 0) AS cnt_right,
		COALESCE(l.cnt_left, 0) + COALESCE(r.cnt_right, 0) AS n,
		COALESCE(l.cnt_left, 0) AS k
	FROM model_position_left_counts l
	FULL OUTER JOIN model_position_right_counts r 
		ON l.model_name = r.model_name
		AND l.filter_key = r.filter_key
		AND l.filter_label = r.filter_label
),

-- 计算每个模型出现在左侧的比例
model_position_left_rate AS (
	SELECT 
		filter_key,
		filter_label,
		'_ALL_' AS eval_dim_key,
		mp.model_name,
		mp.n,
		mp.k,
		CASE 
			WHEN mp.n > 0 THEN CAST(mp.k AS DOUBLE) / mp.n
			ELSE NULL
		END AS p_hat
	FROM model_position_summary mp
	WHERE mp.n > 0
),

-- 为每个模型的左侧出现比例计算Wilson置信区间
model_position_left_wilson AS (
	SELECT 
		r.filter_key,
		r.filter_label,
		r.eval_dim_key,
		r.model_name,
		r.n,
		r.k,
		r.p_hat,
		-- Wilson置信区间计算：z=1.959963984540054, z2=3.841458820694125
		((r.p_hat + 3.841458820694125 / (2 * r.n)) / (1 + 3.841458820694125 / r.n)) AS center,
		(1.959963984540054 * SQRT((r.p_hat * (1 - r.p_hat) + 3.841458820694125 / (4 * r.n)) / r.n) / (1 + 3.841458820694125 / r.n)) AS half
	FROM model_position_left_rate r
	WHERE r.n > 0
),

model_position_bias_output AS (
	SELECT 
		'model_position_bias' AS method,
		mw.filter_key,
		mw.filter_label,
		mw.model_name AS dimension_or_model,
		mw.n,
		mw.k,
		ROUND(mw.p_hat, 4) AS p_hat,
		ROUND(CASE WHEN mw.center - mw.half < 0 THEN 0.0 ELSE mw.center - mw.half END, 4) AS ci_lo,
		ROUND(CASE WHEN mw.center + mw.half > 1 THEN 1.0 ELSE mw.center + mw.half END, 4) AS ci_hi,
		CASE 
			WHEN mw.center - mw.half > 0.5 THEN 'left-biased'
			WHEN mw.center + mw.half < 0.5 THEN 'right-biased'
			ELSE 'balanced'
		END AS decision,
		-- 基于置信区间判定显著性（不跨0.5即显著）
		CASE 
			WHEN (mw.center - mw.half > 0.5) OR (mw.center + mw.half < 0.5)
			THEN 'significant'
			ELSE 'not-significant'
		END AS significance,
		-- z_score：标准化得分（偏离0.5的标准误差倍数）
		ROUND(
			ABS(mw.p_hat - 0.5) / NULLIF(SQRT((mw.p_hat * (1 - mw.p_hat)) / mw.n), 0),
			2
		) AS z_score,
		-- effect_size：效应量（左侧出现频率与50%的差异，单位：百分点）
		ROUND((mw.p_hat - 0.5) * 100, 2) AS effect_size,
		-- ci_margin：置信区间与临界值0.5的最小距离（正值表示显著）
		ROUND(
			CASE 
				WHEN mw.center - mw.half > 0.5 THEN (mw.center - mw.half - 0.5) * 100  -- 左侧偏置
				WHEN mw.center + mw.half < 0.5 THEN (0.5 - mw.center - mw.half) * 100  -- 右侧偏置
				ELSE LEAST(ABS(mw.center - mw.half - 0.5), ABS(mw.center + mw.half - 0.5)) * -100  -- 不显著，取负值
			END, 
		2) AS ci_margin
	FROM model_position_left_wilson mw
),

-- ============================================================
-- 第六步：检验5 - 模型整体曝光均匀性（Model Exposure Balance）
-- 检验各模型的总出现次数是否均匀分布
-- 方法：泊松分布的标准化残差
-- ============================================================
model_total_exposure_left_right AS (
	-- 统计每个模型的总出现次数（左侧+右侧）
	SELECT 
		v.filter_key,
		v.filter_label,
		m.model_name,
		COUNT(1) AS actual_count
	FROM votes_raw v
	INNER JOIN models m ON m.model_id = v.left_model_id
	GROUP BY v.filter_key, v.filter_label, m.model_name
	
	UNION ALL
	
	SELECT 
		v.filter_key,
		v.filter_label,
		m.model_name,
		COUNT(1) AS actual_count
	FROM votes_raw v
	INNER JOIN models m ON m.model_id = v.right_model_id
	GROUP BY v.filter_key, v.filter_label, m.model_name
),

-- 汇总每个模型的总出现次数
model_total_exposure_summary AS (
	SELECT 
		filter_key,
		filter_label,
		model_name,
		SUM(actual_count) AS actual_count
	FROM model_total_exposure_left_right
	GROUP BY filter_key, filter_label, model_name
),

-- 计算期望值（均值）和相关统计量
model_total_exposure_stats AS (
	SELECT 
		mes.*,
		AVG(mes.actual_count) OVER (PARTITION BY filter_key, filter_label) AS expected_count,
		COUNT(1) OVER (PARTITION BY filter_key, filter_label) AS num_models,
		SUM(mes.actual_count) OVER (PARTITION BY filter_key, filter_label) AS total_count
	FROM model_total_exposure_summary mes
),

model_exposure_balance_enriched AS (
	SELECT 
		ms.*,
		CASE 
			WHEN ms.expected_count > 0 
			THEN (ms.actual_count - ms.expected_count) / SQRT(ms.expected_count)
			ELSE NULL
		END AS z_residual
	FROM model_total_exposure_stats ms
),

model_exposure_balance_output AS (
	SELECT 
		'model_exposure_balance' AS method,
		me.filter_key,
		me.filter_label,
		me.model_name AS dimension_or_model,
		me.actual_count AS n,  -- 该模型样本量
		me.actual_count AS k,  -- 同上
		ROUND(
			me.actual_count / CAST(me.total_count AS DOUBLE),
			4
		) AS p_hat,  -- 该模型占比
		CAST(NULL AS DOUBLE) AS ci_lo,  -- 不适用
		CAST(NULL AS DOUBLE) AS ci_hi,  -- 不适用
		CASE 
			-- 使用泊松分布的标准化残差：(实际-期望) / sqrt(期望)
			WHEN me.z_residual > 1.96 THEN 'over-exposed'
			WHEN me.z_residual < -1.96 THEN 'under-exposed'
			ELSE 'balanced'
		END AS decision,
		CASE 
			WHEN ABS(me.z_residual) > 1.96
			THEN 'significant'
			ELSE 'not-significant'
		END AS significance,
		-- z_score：标准化残差
		ROUND(me.z_residual, 2) AS z_score,
		-- effect_size：实际次数与期望次数的差异（绝对值）
		ROUND(me.actual_count - me.expected_count, 2) AS effect_size,
		-- ci_margin：标准化残差超过临界值1.96的程度
		ROUND(CASE WHEN me.z_residual IS NOT NULL THEN ABS(me.z_residual) - 1.96 END, 2) AS ci_margin
	FROM model_exposure_balance_enriched me
),

-- ============================================================
-- 第七步：检验6 - 模型配对均匀性（Model Pairing Balance）
-- 检验模型两两配对的次数是否均匀分布
-- 方法：泊松分布的标准化残差
-- ============================================================
model_pairing_counts AS (
	-- 统计每对模型的配对次数（标准化为字母序，避免重复）
	SELECT 
		v.filter_key,
		v.filter_label,
		CASE 
			WHEN lm.model_name < rm.model_name THEN lm.model_name
			ELSE rm.model_name
		END AS model_a,
		CASE 
			WHEN lm.model_name < rm.model_name THEN rm.model_name
			ELSE lm.model_name
		END AS model_b,
		COUNT(1) AS actual_count
	FROM votes_raw v
	INNER JOIN models lm ON lm.model_id = v.left_model_id
	INNER JOIN models rm ON rm.model_id = v.right_model_id
	GROUP BY 
		v.filter_key,
		v.filter_label,
		CASE 
			WHEN lm.model_name < rm.model_name THEN lm.model_name
			ELSE rm.model_name
		END,
		CASE 
			WHEN lm.model_name < rm.model_name THEN rm.model_name
			ELSE lm.model_name
		END
),

-- 计算期望值（均值）和相关统计量
model_pairing_stats AS (
	SELECT 
		mpc.*,
		AVG(mpc.actual_count) OVER (PARTITION BY filter_key, filter_label) AS expected_count,
		COUNT(1) OVER (PARTITION BY filter_key, filter_label) AS num_pairs,
		SUM(mpc.actual_count) OVER (PARTITION BY filter_key, filter_label) AS total_count
	FROM model_pairing_counts mpc
),

model_pairing_balance_enriched AS (
	SELECT 
		mps.*,
		CASE 
			WHEN mps.expected_count > 0 
			THEN (mps.actual_count - mps.expected_count) / SQRT(mps.expected_count)
			ELSE NULL
		END AS z_residual
	FROM model_pairing_stats mps
),

model_pairing_balance_output AS (
	SELECT 
		'model_pairing_balance' AS method,
		mp.filter_key,
		mp.filter_label,
		CONCAT(mp.model_a, ' vs ', mp.model_b) AS dimension_or_model,
		mp.actual_count AS n,  -- 该配对样本量
		mp.actual_count AS k,  -- 同上
		ROUND(
			mp.actual_count / CAST(mp.total_count AS DOUBLE),
			4
		) AS p_hat,  -- 该配对占比
		CAST(NULL AS DOUBLE) AS ci_lo,  -- 不适用
		CAST(NULL AS DOUBLE) AS ci_hi,  -- 不适用
		CASE 
			-- 使用泊松分布的标准化残差：(实际-期望) / sqrt(期望)
			WHEN mp.z_residual > 1.96 THEN 'over-paired'
			WHEN mp.z_residual < -1.96 THEN 'under-paired'
			ELSE 'balanced'
		END AS decision,
		CASE 
			WHEN ABS(mp.z_residual) > 1.96
			THEN 'significant'
			ELSE 'not-significant'
		END AS significance,
		-- z_score：标准化残差
		ROUND(mp.z_residual, 2) AS z_score,
		-- effect_size：实际次数与期望次数的差异（绝对值）
		ROUND(mp.actual_count - mp.expected_count, 2) AS effect_size,
		-- ci_margin：标准化残差超过临界值的程度
		ROUND(CASE WHEN mp.z_residual IS NOT NULL THEN ABS(mp.z_residual) - 1.96 END, 2) AS ci_margin
	FROM model_pairing_balance_enriched mp
),

-- ============================================================
-- 数据质量报告（添加到输出中）
-- ============================================================
data_quality_output AS (
	SELECT 
		'data_quality' AS method,
		vqc.filter_key,
		vqc.filter_label,
		'NULL_filtering' AS dimension_or_model,
		vqc.total_records AS n,
		vqc.invalid_records AS k,
		ROUND(vqc.invalid_rate_pct / 100.0, 4) AS p_hat,
		CAST(NULL AS DOUBLE) AS ci_lo,
		CAST(NULL AS DOUBLE) AS ci_hi,
		CONCAT('filtered_out_', CAST(vqc.invalid_records AS STRING), '_records') AS decision,
		CASE 
			WHEN vqc.invalid_rate_pct > 5.0 THEN 'warning'
			ELSE 'normal'
		END AS significance,
		CAST(NULL AS DOUBLE) AS z_score,
		vqc.invalid_rate_pct AS effect_size,
		CAST(NULL AS DOUBLE) AS ci_margin
	FROM votes_quality_check vqc
)

-- ============================================================
-- 合并所有检验结果
-- ============================================================
SELECT * FROM data_quality_output
UNION ALL
SELECT * FROM position_bias_output
UNION ALL
SELECT * FROM time_exposure_output
UNION ALL
SELECT * FROM answer_length_winrate_output
UNION ALL
SELECT * FROM order_bias_output
UNION ALL
SELECT * FROM model_position_bias_output
UNION ALL
SELECT * FROM model_exposure_balance_output
UNION ALL
SELECT * FROM model_pairing_balance_output
;
