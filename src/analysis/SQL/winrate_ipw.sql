-- ========================================================================
-- 方案2: IPW (Inverse Propensity Weighting) 胜率校正
-- ========================================================================
-- 功能：使用逆倾向加权法校正多重偏差（位置、时间、曝光、配对）
-- 原理：为每个vote计算权重 = 1 / P(该vote出现的概率)
--      越"不正常"的vote（如某模型过度出现在某位置）权重越低
-- 输出：每个模型在各维度的原始胜率 vs IPW校正后胜率
-- ========================================================================

WITH 
-- ============================================================
-- 第一步：复用 bias_detection.sql 的数据准备逻辑
-- ============================================================

-- 1) 炸平用户选择
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

-- 2) 计算session范围，划分EARLY/LATE
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

-- 3) 筛选答完18题的用户
session_completion_check AS (
	SELECT 
		eval_session_id,
		COUNT(DISTINCT content_id) AS questions_answered
	FROM user_choice_with_time_bin
	GROUP BY eval_session_id
),

completed_sessions AS (
	SELECT eval_session_id
	FROM session_completion_check
	WHERE questions_answered = 18
),

user_choice_filtered AS (
	SELECT u.*
	FROM user_choice_with_time_bin u
	INNER JOIN completed_sessions cs 
		ON cs.eval_session_id = u.eval_session_id
),

-- 4) 模型信息
models AS (
	SELECT 
		id AS model_id,
		model_name
	FROM llm_models
	WHERE dt = MAX_PT('llm_models')
),

-- 5) 答案表
answers AS (
	SELECT 
		id AS answer_id,
		eval_set_id,
		eval_query_id,
		llm_model_id,
		game_id
	FROM llm_answers
	WHERE dt = MAX_PT('llm_answers')
),

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

-- 6) 解码comparison
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

-- 7) 生成投票明细（带位置和时间信息）
votes_detailed AS (
	SELECT 
		d.game_id,
		up.eval_dim_key,
		up.time_bin,
		d.left_model_id,
		d.right_model_id,
		d.left_model_name,
		d.right_model_name,
		CASE 
			WHEN up.winner_id = d.answer_left THEN d.left_model_id
			WHEN up.winner_id = d.answer_right THEN d.right_model_id
			ELSE NULL
		END AS winner_model_id,
		-- 标记位置（胜者在左还是右）
		CASE 
			WHEN up.winner_id = d.answer_left THEN 'left'
			WHEN up.winner_id = d.answer_right THEN 'right'
			ELSE NULL
		END AS winner_position
	FROM user_choice_filtered up
	INNER JOIN comparison_with_models d 
		ON up.eval_set_id = d.eval_set_id 
		AND up.content_id = d.comparison_id
	WHERE up.winner_id IS NOT NULL  -- 过滤平局
),

-- ============================================================
-- 第二步：计算倾向分数（Propensity Scores）
-- ============================================================

-- 2.1) 计算每个模型出现在左侧的概率
model_left_propensity AS (
	SELECT 
		m.model_name,
		m.model_id,
		COUNT(CASE WHEN v.left_model_id = m.model_id THEN 1 END) AS left_count,
		COUNT(CASE WHEN v.right_model_id = m.model_id THEN 1 END) AS right_count,
		COUNT(CASE WHEN v.left_model_id = m.model_id THEN 1 END) / 
			(COUNT(CASE WHEN v.left_model_id = m.model_id THEN 1 END) + 
			 COUNT(CASE WHEN v.right_model_id = m.model_id THEN 1 END)) AS p_left
	FROM votes_detailed v
	CROSS JOIN models m
	GROUP BY m.model_name, m.model_id
),

-- 2.2) 计算每个模型在EARLY阶段出现的概率
model_time_propensity AS (
	SELECT 
		m.model_name,
		m.model_id,
		SUM(CASE WHEN v.time_bin = 'EARLY' THEN 1 ELSE 0 END) / COUNT(*) AS p_early
	FROM (
		SELECT time_bin, left_model_id AS model_id FROM votes_detailed
		UNION ALL
		SELECT time_bin, right_model_id AS model_id FROM votes_detailed
	) v
	INNER JOIN models m ON m.model_id = v.model_id
	GROUP BY m.model_name, m.model_id
),

-- 2.3) 计算每对模型的配对概率
model_pairing_propensity AS (
	SELECT 
		CASE 
			WHEN v.left_model_name < v.right_model_name THEN v.left_model_name
			ELSE v.right_model_name
		END AS model_a,
		CASE 
			WHEN v.left_model_name < v.right_model_name THEN v.right_model_name
			ELSE v.left_model_name
		END AS model_b,
		COUNT(*) AS pair_count
	FROM votes_detailed v
	GROUP BY 
		CASE 
			WHEN v.left_model_name < v.right_model_name THEN v.left_model_name
			ELSE v.right_model_name
		END,
		CASE 
			WHEN v.left_model_name < v.right_model_name THEN v.right_model_name
			ELSE v.left_model_name
		END
),

total_pairs AS (
	SELECT SUM(pair_count) AS total FROM model_pairing_propensity
),

model_pairing_prob AS (
	SELECT 
		model_a,
		model_b,
		pair_count / tp.total AS p_pair
	FROM model_pairing_propensity
	CROSS JOIN total_pairs tp
),

-- ============================================================
-- 第三步：为每个vote计算IPW权重
-- ============================================================

-- 展开votes_detailed，为左右两个模型各生成一条记录
votes_expanded AS (
	-- 左侧模型的记录
	SELECT 
		v.game_id,
		v.eval_dim_key,
		v.time_bin,
		v.left_model_id AS model_id,
		v.left_model_name AS model_name,
		v.right_model_name AS opponent_name,
		'left' AS position,
		CASE WHEN v.winner_model_id = v.left_model_id THEN 1 ELSE 0 END AS is_win
	FROM votes_detailed v
	
	UNION ALL
	
	-- 右侧模型的记录
	SELECT 
		v.game_id,
		v.eval_dim_key,
		v.time_bin,
		v.right_model_id AS model_id,
		v.right_model_name AS model_name,
		v.left_model_name AS opponent_name,
		'right' AS position,
		CASE WHEN v.winner_model_id = v.right_model_id THEN 1 ELSE 0 END AS is_win
	FROM votes_detailed v
),

-- 关联倾向分数，计算IPW权重
votes_with_weights AS (
	SELECT 
		ve.*,
		-- 位置倾向分数
		CASE WHEN ve.position = 'left' THEN mlp.p_left 
		     ELSE (1 - mlp.p_left) 
		END AS p_position,
		-- 时间倾向分数
		CASE WHEN ve.time_bin = 'EARLY' THEN mtp.p_early 
		     ELSE (1 - mtp.p_early) 
		END AS p_time,
		-- 配对倾向分数
		COALESCE(mpp.p_pair, 1.0 / (SELECT COUNT(DISTINCT model_name) FROM models)) AS p_pairing,
		-- 联合倾向分数（假设独立）
		(CASE WHEN ve.position = 'left' THEN mlp.p_left ELSE (1 - mlp.p_left) END) *
		(CASE WHEN ve.time_bin = 'EARLY' THEN mtp.p_early ELSE (1 - mtp.p_early) END) *
		COALESCE(mpp.p_pair, 0.16667) AS propensity_score
	FROM votes_expanded ve
	LEFT JOIN model_left_propensity mlp 
		ON mlp.model_id = ve.model_id
	LEFT JOIN model_time_propensity mtp 
		ON mtp.model_id = ve.model_id
	LEFT JOIN model_pairing_prob mpp
		ON (mpp.model_a = LEAST(ve.model_name, ve.opponent_name) 
		    AND mpp.model_b = GREATEST(ve.model_name, ve.opponent_name))
),

-- 计算IPW权重（截断极端值，避免方差过大）
votes_with_ipw AS (
	SELECT 
		*,
		-- 权重 = 1 / propensity_score，但截断在 [0.2, 5] 之间
		CASE 
			WHEN 1.0 / propensity_score > 5.0 THEN 5.0
			WHEN 1.0 / propensity_score < 0.2 THEN 0.2
			ELSE 1.0 / propensity_score
		END AS ipw_weight
	FROM votes_with_weights
	WHERE propensity_score > 0  -- 过滤异常值
),

-- ============================================================
-- 第四步：计算原始胜率 vs IPW校正后胜率
-- ============================================================

-- 原始胜率（未加权）
raw_winrates AS (
	SELECT 
		game_id,
		eval_dim_key,
		model_name,
		COUNT(*) AS total_matches,
		SUM(is_win) AS wins,
		AVG(is_win) AS raw_winrate
	FROM votes_expanded
	GROUP BY game_id, eval_dim_key, model_name
),

-- IPW校正后胜率（加权）
ipw_winrates AS (
	SELECT 
		game_id,
		eval_dim_key,
		model_name,
		SUM(ipw_weight) AS weighted_total,
		SUM(is_win * ipw_weight) AS weighted_wins,
		SUM(is_win * ipw_weight) / SUM(ipw_weight) AS ipw_winrate,
		-- 有效样本量（Kish's effective sample size）
		POWER(SUM(ipw_weight), 2) / SUM(POWER(ipw_weight, 2)) AS effective_n
	FROM votes_with_ipw
	GROUP BY game_id, eval_dim_key, model_name
),

-- 诊断统计：权重分布
weight_diagnostics AS (
	SELECT 
		model_name,
		MIN(ipw_weight) AS min_weight,
		PERCENTILE_APPROX(ipw_weight, 0.25) AS p25_weight,
		PERCENTILE_APPROX(ipw_weight, 0.5) AS median_weight,
		PERCENTILE_APPROX(ipw_weight, 0.75) AS p75_weight,
		MAX(ipw_weight) AS max_weight,
		AVG(ipw_weight) AS avg_weight,
		STDDEV(ipw_weight) AS stddev_weight
	FROM votes_with_ipw
	GROUP BY model_name
),

-- ============================================================
-- 第五步：合并结果并计算校正效应
-- ============================================================

final_comparison AS (
	SELECT 
		rw.game_id,
		rw.eval_dim_key,
		rw.model_name,
		rw.total_matches,
		rw.wins,
		ROUND(rw.raw_winrate, 4) AS raw_winrate,
		ROUND(iw.ipw_winrate, 4) AS ipw_adjusted_winrate,
		ROUND((iw.ipw_winrate - rw.raw_winrate) * 100, 2) AS correction_pct,
		ROUND(iw.effective_n, 0) AS effective_sample_size,
		wd.min_weight,
		wd.median_weight,
		wd.max_weight,
		ROUND(wd.avg_weight, 2) AS avg_weight,
		ROUND(wd.stddev_weight, 2) AS stddev_weight
	FROM raw_winrates rw
	INNER JOIN ipw_winrates iw 
		ON rw.game_id = iw.game_id 
		AND rw.eval_dim_key = iw.eval_dim_key 
		AND rw.model_name = iw.model_name
	LEFT JOIN weight_diagnostics wd 
		ON wd.model_name = rw.model_name
)

-- ============================================================
-- 最终输出
-- ============================================================
SELECT 
	game_id,
	eval_dim_key,
	model_name,
	total_matches,
	wins,
	raw_winrate,
	ipw_adjusted_winrate,
	correction_pct,
	effective_sample_size,
	avg_weight,
	stddev_weight,
	min_weight,
	median_weight,
	max_weight,
	-- 校正方向和幅度
	CASE 
		WHEN correction_pct > 1.0 THEN 'upward_significant'
		WHEN correction_pct < -1.0 THEN 'downward_significant'
		ELSE 'minor_adjustment'
	END AS correction_direction
FROM final_comparison
WHERE game_id = 'all' OR game_id IN (
	-- 只输出最近的几个game_id，避免结果过多
	SELECT DISTINCT game_id 
	FROM votes_detailed 
	WHERE game_id != 'all' 
	LIMIT 10
)
ORDER BY 
	game_id,
	eval_dim_key,
	ABS(correction_pct) DESC
;


