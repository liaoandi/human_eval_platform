-- ========================================================================
-- 模型胜率矩阵（分层校正版 v2）
-- ========================================================================
-- 功能：输出 row_model vs col_model 的胜率矩阵
--  - raw_winrate              : 原始胜率 = (胜场 + 0.5×平局) / 总对局
--  - stratified_winrate       : 按 (position, time_bin) 四个层分别计算胜率，
--                               使用理想均匀分布 (0.25, 0.25, 0.25, 0.25) 加权平均
--                               校正位置偏差和时间偏差
--  - confidence_adjusted_winrate : 基于配对曝光度的简单置信度校正
--  - weight_variance          : 曝光分布的方差，衡量偏差程度（越大表示偏差越严重）
--  - 支持 game_id='all' 的全局结果 + 按 game_id 维度的明细
--  - 所有 game_id（包括 'all'）都会得到分层校正
-- ========================================================================

WITH 
-- ============================================================
-- 第一步：数据准备（简化版，移除filter）
-- ============================================================

user_choice_exploded AS (
	SELECT 
		t.eval_set_id,
		t.content_id,
		t.eval_session_id,
		t.no AS task_no,
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

user_choice_full AS (
	SELECT 
		u.eval_set_id,
		u.content_id,
		u.eval_session_id,
		u.task_no,
		u.winner_id,
		u.eval_dim_id,
		u.eval_dim_key,
		u.time_bin,
		c.game_id,
		c.is_golden
	FROM user_choice_with_time_bin u
	LEFT JOIN comparison_with_game c
		ON c.comparison_id = u.content_id
		AND c.eval_set_id = u.eval_set_id
),

models AS (
	SELECT 
		id AS model_id,
		model_name
	FROM llm_models
	WHERE dt = MAX_PT('llm_models')
),

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

votes_detailed AS (
	SELECT 
		d.game_id,
		uc.eval_dim_key,
		uc.time_bin,
		CASE 
			WHEN uc.winner_id = d.answer_left THEN 'left'
			WHEN uc.winner_id = d.answer_right THEN 'right'
			WHEN uc.winner_id = 0 THEN 'draw'
			ELSE 'unknown'
		END AS position_winner,
		d.left_model_id,
		d.right_model_id,
		d.left_model_name,
		d.right_model_name,
		CASE WHEN uc.winner_id = 0 THEN 1 ELSE 0 END AS is_draw,
		CASE 
			WHEN uc.winner_id = d.answer_left THEN d.left_model_id
			WHEN uc.winner_id = d.answer_right THEN d.right_model_id
			ELSE NULL
		END AS winner_model_id
	FROM user_choice_full uc
	INNER JOIN comparison_with_models d 
		ON uc.eval_set_id = d.eval_set_id 
		AND uc.content_id = d.comparison_id
	WHERE uc.is_golden = 0
		AND uc.winner_id IS NOT NULL
),

votes_with_all AS (
	SELECT * FROM votes_detailed
	
	UNION ALL
	
	SELECT 
		'all' AS game_id,
		eval_dim_key,
		time_bin,
		position_winner,
		left_model_id,
		right_model_id,
		left_model_name,
		right_model_name,
		is_draw,
		winner_model_id
	FROM votes_detailed
),

-- ============================================================
-- 第二步：真正的分层校正
-- ============================================================

-- 2.1 按 (row_model, col_model, position, time_bin) 分层统计
stratified_votes AS (
    SELECT 
        v.game_id,
        v.eval_dim_key,
		v.time_bin,
        v.left_model_id AS row_model_id,
        v.right_model_id AS col_model_id,
		'left' AS position,
        CASE 
            WHEN v.winner_model_id = v.left_model_id THEN 1.0
            WHEN v.is_draw = 1 THEN 0.5
            ELSE 0.0
        END AS result_points
    FROM votes_with_all v
    
    UNION ALL
    
    SELECT 
        v.game_id,
        v.eval_dim_key,
		v.time_bin,
        v.right_model_id AS row_model_id,
        v.left_model_id AS col_model_id,
		'right' AS position,
        CASE 
            WHEN v.winner_model_id = v.right_model_id THEN 1.0
            WHEN v.is_draw = 1 THEN 0.5
            ELSE 0.0
        END AS result_points
    FROM votes_with_all v
),

-- 2.2 按层统计对局数和胜率
strata_stats AS (
    SELECT 
        game_id,
        eval_dim_key,
        row_model_id,
        col_model_id,
		position,
		time_bin,
		COUNT(*) AS strata_matches,
		SUM(result_points) AS strata_points,
		CAST(SUM(result_points) AS DOUBLE) / NULLIF(COUNT(*), 0) AS strata_winrate
	FROM stratified_votes
    GROUP BY 
        game_id,
        eval_dim_key,
        row_model_id,
		col_model_id,
		position,
		time_bin
),

-- 2.3 计算每个配对在各层的实际曝光分布
pair_strata_distribution AS (
    SELECT 
		game_id,
		eval_dim_key,
		row_model_id,
		col_model_id,
		position,
		time_bin,
		strata_matches,
		SUM(strata_matches) OVER (
			PARTITION BY game_id, eval_dim_key, row_model_id, col_model_id
		) AS total_matches_for_pair,
		CAST(strata_matches AS DOUBLE) / NULLIF(
			SUM(strata_matches) OVER (
				PARTITION BY game_id, eval_dim_key, row_model_id, col_model_id
			), 0
		) AS actual_weight
	FROM strata_stats
),

-- 2.4 应用分层权重计算校正胜率（目标：均匀分布 = 1/4）
stratified_adjusted AS (
    SELECT 
		ss.game_id,
		ss.eval_dim_key,
		ss.row_model_id,
		ss.col_model_id,
		SUM(ss.strata_matches) AS total_matches,
		SUM(ss.strata_points) AS total_points,
		-- 原始胜率
		CAST(SUM(ss.strata_points) AS DOUBLE) / NULLIF(SUM(ss.strata_matches), 0) AS raw_winrate,
		-- 分层校正胜率：对出现的层按均匀权重取平均
		CAST(SUM(ss.strata_winrate) AS DOUBLE) / NULLIF(COUNT(*), 0) AS stratified_winrate,
		-- 各层的实际权重（用于诊断）
		MAX(CASE WHEN ss.position = 'left' AND ss.time_bin = 'EARLY' 
			THEN psd.actual_weight END) AS weight_left_early,
		MAX(CASE WHEN ss.position = 'left' AND ss.time_bin = 'LATE' 
			THEN psd.actual_weight END) AS weight_left_late,
		MAX(CASE WHEN ss.position = 'right' AND ss.time_bin = 'EARLY' 
			THEN psd.actual_weight END) AS weight_right_early,
		MAX(CASE WHEN ss.position = 'right' AND ss.time_bin = 'LATE' 
			THEN psd.actual_weight END) AS weight_right_late
	FROM strata_stats ss
	LEFT JOIN pair_strata_distribution psd
		ON psd.game_id = ss.game_id
		AND psd.eval_dim_key = ss.eval_dim_key
		AND psd.row_model_id = ss.row_model_id
		AND psd.col_model_id = ss.col_model_id
		AND psd.position = ss.position
		AND psd.time_bin = ss.time_bin
	GROUP BY 
		ss.game_id,
		ss.eval_dim_key,
		ss.row_model_id,
		ss.col_model_id
),

-- 2.5 简化的 confidence 校正（基于配对曝光度）
pair_exposure_stats AS (
    SELECT 
		game_id,
		eval_dim_key,
		row_model_id,
		col_model_id,
		total_matches,
		AVG(total_matches) OVER (
			PARTITION BY game_id, eval_dim_key
		) AS avg_pair_matches
	FROM stratified_adjusted
),

pair_confidence_computed AS (
	SELECT 
		pes.*,
		CASE 
			WHEN pes.avg_pair_matches IS NULL OR pes.avg_pair_matches = 0 THEN 1.0
			WHEN pes.total_matches >= pes.avg_pair_matches THEN 1.0
			ELSE pes.total_matches / pes.avg_pair_matches
		END AS pair_confidence
	FROM pair_exposure_stats pes
),

-- 2.6 合并所有指标
matrix_long AS (
    SELECT 
		sa.game_id,
		sa.eval_dim_key,
        rm.model_name AS row_model,
        cm.model_name AS col_model,
		sa.total_matches,
		ROUND(sa.raw_winrate, 4) AS raw_winrate,
		ROUND(sa.stratified_winrate, 4) AS stratified_winrate,
        ROUND(
			0.5 + (sa.raw_winrate - 0.5) * pcc.pair_confidence,
			4
		) AS confidence_adjusted_winrate,
		ROUND(pcc.pair_confidence, 4) AS pair_confidence,
		ROUND(pcc.avg_pair_matches, 2) AS avg_pair_matches,
		-- 各层的实际曝光权重（诊断用）
		ROUND(sa.weight_left_early, 4) AS weight_left_early,
		ROUND(sa.weight_left_late, 4) AS weight_left_late,
		ROUND(sa.weight_right_early, 4) AS weight_right_early,
		ROUND(sa.weight_right_late, 4) AS weight_right_late,
		-- 曝光不均衡程度（方差）
		ROUND(
			POWER(COALESCE(sa.weight_left_early, 0.25) - 0.25, 2) +
			POWER(COALESCE(sa.weight_left_late, 0.25) - 0.25, 2) +
			POWER(COALESCE(sa.weight_right_early, 0.25) - 0.25, 2) +
			POWER(COALESCE(sa.weight_right_late, 0.25) - 0.25, 2),
			6
		) AS weight_variance
	FROM stratified_adjusted sa
	LEFT JOIN pair_confidence_computed pcc
		ON pcc.game_id = sa.game_id
		AND pcc.eval_dim_key = sa.eval_dim_key
		AND pcc.row_model_id = sa.row_model_id
		AND pcc.col_model_id = sa.col_model_id
	LEFT JOIN models rm ON rm.model_id = sa.row_model_id
	LEFT JOIN models cm ON cm.model_id = sa.col_model_id
)

SELECT 
    game_id,
    eval_dim_key,
    row_model,
    col_model,
    total_matches,
	raw_winrate,
	stratified_winrate,
	confidence_adjusted_winrate,
    pair_confidence,
	avg_pair_matches,
	weight_left_early,
	weight_left_late,
	weight_right_early,
	weight_right_late,
	weight_variance
FROM matrix_long
WHERE row_model IS NOT NULL 
	AND col_model IS NOT NULL
ORDER BY 
    CASE WHEN game_id = 'all' THEN 0 ELSE 1 END,
    game_id,
    eval_dim_key,
    row_model,
    col_model
LIMIT 20000;
