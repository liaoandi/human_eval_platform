-- ========================================================================
-- 用户质量分析：多维度过滤条件的用户分布
-- ========================================================================
-- 功能：统计多个质量维度下的用户数量（按game_id分组统计）
-- 答完15道normal题
-- 答题时间满足要求：所有正常题的答题时间>=15%分位数
-- 
-- 输出：各维度单独满足的人数，以及OR/AND组合逻辑的人数（按game_id分组）
-- ========================================================================

-- ========== 1) 拍平用户评测结果 ==========
WITH user_choice_exploded AS (
	SELECT
		t.eval_set_id,
		t.eval_session_id,
		t.content_id,
		(t.end_at - t.start_at) / 60.0 AS duration_mins,
		r.winner_id,
		r.eval_dim_id,
		r.eval_dim_key
	FROM eval_tasks t
	LATERAL VIEW
		explode(
		from_json(
			t.result,
			'array<struct<winner_id:bigint, eval_dim_id:bigint, eval_dim_key:string>>'
		)
		) e AS r
	WHERE t.dt = MAX_PT('eval_tasks')   
		AND t.status = 12
		AND eval_set_id >= 10
),

-- ========== 2) 获取用户ID ==========
sessions_with_user AS (
	SELECT
		id AS session_id,
		user_id
	FROM eval_sessions
	WHERE dt = MAX_PT('eval_sessions')
),

-- ========== 3) 识别golden题的正确答案 ==========
answer_truth_id AS (
	SELECT 
		answers_map.answer_id,          
		answer_truth_uuid.is_correct
	FROM (
		SELECT 
			id   AS answer_id,
			uuid AS answer_uuid
		FROM llm_answers 
		WHERE dt = MAX_PT('llm_answers')
	) answers_map 
	INNER JOIN (
		SELECT 
			answer_id AS answer_uuid,
			CASE 
				WHEN NVL(GET_JSON_OBJECT(generation_metadata, '$.correct'), '') IN ('true','1','t','yes','y') THEN 1
				WHEN NVL(GET_JSON_OBJECT(generation_metadata, '$.correct'), '') IN ('false','0','f','no','n') THEN 0
				ELSE -1
			END AS is_correct
		FROM eval_answer_inc
		WHERE dm = MAX_PT('eval_answer_inc')
	) answer_truth_uuid 
	ON answers_map.answer_uuid = answer_truth_uuid.answer_uuid
),

-- ========== 4) 标记golden题的正确答案 ==========
golden_pairs AS (
	SELECT 
		comparisons.id          AS comparison_id, 
		comparisons.eval_set_id, 
		comparisons.answer_left, 
		comparisons.answer_right,
		CASE 
			WHEN al.is_correct = 1 AND NVL(ar.is_correct,0) = 0 THEN comparisons.answer_left
			WHEN ar.is_correct = 1 AND NVL(al.is_correct,0) = 0 THEN comparisons.answer_right
			ELSE NULL 
		END AS correct_answer_id
	FROM (
		SELECT id, eval_set_id, answer_left, answer_right
		FROM eval_comparisons 
		WHERE dt = MAX_PT('eval_comparisons')
			AND is_golden = 1
	) comparisons
	LEFT JOIN answer_truth_id al ON al.answer_id = comparisons.answer_left
	LEFT JOIN answer_truth_id ar ON ar.answer_id = comparisons.answer_right
),

-- ========== 5) 获取comparison的game_id信息 ==========
comparison_with_game AS (
	SELECT 
		c.id AS comparison_id,
		c.eval_set_id,
		c.is_golden,
		a.game_id
	FROM (
		SELECT id, eval_set_id, eval_query_id, is_golden
		FROM eval_comparisons
		WHERE dt = MAX_PT('eval_comparisons')
	) c
	LEFT JOIN (
		SELECT DISTINCT eval_set_id, eval_query_id, game_id
		FROM llm_answers
		WHERE dt = MAX_PT('llm_answers')
	) a
		ON a.eval_set_id = c.eval_set_id
		AND a.eval_query_id = c.eval_query_id
),

-- ========== 6) 用户答题明细（含golden题判定和game_id） ==========
user_choice_with_user AS (
	SELECT
		s.user_id,
		u.eval_set_id,
		u.eval_session_id,
		u.content_id,
		u.duration_mins,
		u.winner_id,
		u.eval_dim_id,
		u.eval_dim_key,
		c.is_golden,
		c.game_id,
		gp.correct_answer_id,
		CASE 
			WHEN c.is_golden = 1 AND u.winner_id = gp.correct_answer_id THEN 1
			WHEN c.is_golden = 1 AND u.winner_id != gp.correct_answer_id THEN 0
			ELSE NULL
		END AS is_golden_correct
	FROM user_choice_exploded u
	INNER JOIN sessions_with_user s 
		ON s.session_id = u.eval_session_id
	INNER JOIN comparison_with_game c
		ON c.comparison_id = u.content_id 
		AND c.eval_set_id = u.eval_set_id
	LEFT JOIN golden_pairs gp
		ON gp.comparison_id = u.content_id
		AND gp.eval_set_id = u.eval_set_id
),

-- ========== 7) 计算每道题的答题时长15%分位数 ==========
question_time_threshold AS (
	SELECT
		eval_set_id,
		content_id,
		PERCENTILE_APPROX(duration_mins, 0.15) AS p15_duration_mins
	FROM user_choice_with_user
	GROUP BY eval_set_id, content_id
),

-- ========== 8) 按用户统计答题情况（保留game_id维度） ==========
user_stats_base AS (
	SELECT
		user_id,
		eval_set_id,
		game_id,
		content_id,
		is_golden,
		is_golden_correct,
		duration_mins
	FROM (
		-- 按题去重（同一题多个维度只算一次，取任一维度的数据）
		SELECT 
			user_id, 
			eval_set_id,
			game_id,
			content_id, 
			is_golden, 
			is_golden_correct,
			MAX(duration_mins) AS duration_mins
		FROM user_choice_with_user
		GROUP BY user_id, eval_set_id, game_id, content_id, is_golden, is_golden_correct
	) t
),

-- ========== 9) 答完至少15道normal题的用户（按game_id统计） ==========
completed_all_users AS (
	SELECT
		user_id,
		eval_set_id,
		game_id,
		COUNT(DISTINCT CASE WHEN is_golden = 0 THEN content_id END) AS total_answered_normal
	FROM user_stats_base
	GROUP BY user_id, eval_set_id, game_id
	HAVING COUNT(DISTINCT CASE WHEN is_golden = 0 THEN content_id END) >= 15
),

-- ========== 10) 答题时间在15%分位数以上的用户 ==========
-- 只要有任意一题的答题时间低于该题的15%分位数，就不满足条件
fast_answer_check AS (
	SELECT
		u.user_id,
		u.eval_set_id,
		u.game_id,
		u.content_id,
		u.duration_mins,
		qt.p15_duration_mins,
		CASE 
			WHEN u.duration_mins < qt.p15_duration_mins THEN 1
			ELSE 0
		END AS is_too_fast,
		CASE 
			WHEN u.duration_mins >= qt.p15_duration_mins THEN 1
			ELSE 0
		END AS is_good_timing
	FROM user_stats_base u
	INNER JOIN question_time_threshold qt
		ON qt.eval_set_id = u.eval_set_id
		AND qt.content_id = u.content_id
	WHERE u.is_golden = 0  -- 只检查正常题
),

good_timing_users AS (
	SELECT
		user_id,
		eval_set_id,
		game_id
	FROM (
		SELECT
			user_id,
			eval_set_id,
			game_id,
			SUM(is_too_fast) AS too_fast_count
		FROM fast_answer_check
		GROUP BY user_id, eval_set_id, game_id
	) t
	WHERE too_fast_count = 0  -- 没有任何一题答题过快
),

-- ========== 11) 获取所有用户列表（包含game_id） ==========
all_users AS (
	SELECT DISTINCT
		user_id,
		eval_set_id,
		game_id
	FROM user_stats_base
),

-- ========== 12) 为每个用户打标签（保留game_id） ==========
user_quality_flags AS (
	SELECT
		a.user_id,
		a.eval_set_id,
		a.game_id,
		CASE WHEN c.user_id IS NOT NULL THEN 1 ELSE 0 END AS flag_completed_all,
		CASE WHEN t.user_id IS NOT NULL THEN 1 ELSE 0 END AS flag_good_timing
	FROM all_users a
	LEFT JOIN completed_all_users c
		ON c.user_id = a.user_id 
		AND c.eval_set_id = a.eval_set_id
		AND c.game_id = a.game_id
	LEFT JOIN good_timing_users t
		ON t.user_id = a.user_id 
		AND t.eval_set_id = a.eval_set_id
		AND t.game_id = a.game_id
)

-- ========== 13) 最终统计（按game_id分组） ==========
SELECT
	game_id,
	'答完至少15道normal题' AS filter_dimension,
	COUNT(DISTINCT CASE WHEN flag_completed_all = 1 THEN user_id END) AS user_count
FROM user_quality_flags
GROUP BY game_id

UNION ALL

SELECT
	game_id,
	'所有正常题答题时间>=15%分位数' AS filter_dimension,
	COUNT(DISTINCT CASE WHEN flag_good_timing = 1 THEN user_id END) AS user_count
FROM user_quality_flags
GROUP BY game_id

UNION ALL

SELECT
	game_id,
	'组合_满足任一条件' AS filter_dimension,
	COUNT(DISTINCT CASE 
		WHEN flag_completed_all = 1 
			OR flag_good_timing = 1 
		THEN user_id 
	END) AS user_count
FROM user_quality_flags
GROUP BY game_id

UNION ALL

SELECT
	game_id,
	'组合_同时满足两个条件' AS filter_dimension,
	COUNT(DISTINCT CASE 
		WHEN flag_completed_all = 1 
			AND flag_good_timing = 1 
		THEN user_id 
	END) AS user_count
FROM user_quality_flags
GROUP BY game_id

UNION ALL

SELECT
	game_id,
	'总用户数' AS filter_dimension,
	COUNT(DISTINCT user_id) AS user_count
FROM user_quality_flags
GROUP BY game_id

UNION ALL

-- ========== 14) 全局汇总统计（game_id='all'，不对用户去重） ==========
SELECT
	'all' AS game_id,
	'答完至少15道normal题' AS filter_dimension,
	SUM(CASE WHEN flag_completed_all = 1 THEN 1 ELSE 0 END) AS user_count
FROM user_quality_flags

UNION ALL

SELECT
	'all' AS game_id,
	'所有正常题答题时间>=15%分位数' AS filter_dimension,
	SUM(CASE WHEN flag_good_timing = 1 THEN 1 ELSE 0 END) AS user_count
FROM user_quality_flags

UNION ALL

SELECT
	'all' AS game_id,
	'组合_满足任一条件' AS filter_dimension,
	SUM(CASE 
		WHEN flag_completed_all = 1 
			OR flag_good_timing = 1 
		THEN 1 
		ELSE 0 
	END) AS user_count
FROM user_quality_flags

UNION ALL

SELECT
	'all' AS game_id,
	'组合_同时满足两个条件' AS filter_dimension,
	SUM(CASE 
		WHEN flag_completed_all = 1 
			AND flag_good_timing = 1 
		THEN 1 
		ELSE 0 
	END) AS user_count
FROM user_quality_flags

UNION ALL

SELECT
	'all' AS game_id,
	'总用户数' AS filter_dimension,
	COUNT(*) AS user_count
FROM user_quality_flags

ORDER BY game_id, filter_dimension
LIMIT 500;

