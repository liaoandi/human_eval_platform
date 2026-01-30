-- ========================================================================
-- 用户答题时间分位数分布分析
-- ========================================================================
-- 功能：统计正常题（非golden题）的答题时间分位数分布
-- 输出：每5%分位数一次（5%, 10%, 15%, ..., 95%, 100%）
-- 维度：按game_id分组，同时包含全局统计（game_id='all'）
-- 额外信息：计算以每个分位数为阈值时会排除的用户数
--          （即存在至少一道题答题时间低于该阈值的用户数）
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

-- ========== 3) 获取comparison的game_id和golden标记 ==========
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

-- ========== 4) 用户答题明细（含game_id） ==========
user_answer_with_game AS (
	SELECT
		s.user_id,
		u.eval_set_id,
		u.eval_session_id,
		u.content_id,
		u.duration_mins,
		c.is_golden,
		c.game_id
	FROM user_choice_exploded u
	INNER JOIN sessions_with_user s 
		ON s.session_id = u.eval_session_id
	INNER JOIN comparison_with_game c
		ON c.comparison_id = u.content_id 
		AND c.eval_set_id = u.eval_set_id
),

-- ========== 5) 按用户+题目去重，只保留正常题 ==========
normal_questions_only AS (
	SELECT
		user_id,
		eval_set_id,
		game_id,
		content_id,
		MAX(duration_mins) AS duration_mins
	FROM user_answer_with_game
	WHERE is_golden = 0  -- 只统计正常题
	GROUP BY user_id, eval_set_id, game_id, content_id
),

-- ========== 5.1) 计算每道题的各分位数阈值（用于判断用户是否被排除） ==========
question_thresholds AS (
	SELECT
		eval_set_id,
		content_id,
		PERCENTILE_APPROX(duration_mins, 0.05) AS p05_threshold,
		PERCENTILE_APPROX(duration_mins, 0.10) AS p10_threshold,
		PERCENTILE_APPROX(duration_mins, 0.15) AS p15_threshold,
		PERCENTILE_APPROX(duration_mins, 0.20) AS p20_threshold,
		PERCENTILE_APPROX(duration_mins, 0.25) AS p25_threshold,
		PERCENTILE_APPROX(duration_mins, 0.30) AS p30_threshold,
		PERCENTILE_APPROX(duration_mins, 0.35) AS p35_threshold,
		PERCENTILE_APPROX(duration_mins, 0.40) AS p40_threshold,
		PERCENTILE_APPROX(duration_mins, 0.45) AS p45_threshold,
		PERCENTILE_APPROX(duration_mins, 0.50) AS p50_threshold,
		PERCENTILE_APPROX(duration_mins, 0.55) AS p55_threshold,
		PERCENTILE_APPROX(duration_mins, 0.60) AS p60_threshold,
		PERCENTILE_APPROX(duration_mins, 0.65) AS p65_threshold,
		PERCENTILE_APPROX(duration_mins, 0.70) AS p70_threshold,
		PERCENTILE_APPROX(duration_mins, 0.75) AS p75_threshold,
		PERCENTILE_APPROX(duration_mins, 0.80) AS p80_threshold,
		PERCENTILE_APPROX(duration_mins, 0.85) AS p85_threshold,
		PERCENTILE_APPROX(duration_mins, 0.90) AS p90_threshold,
		PERCENTILE_APPROX(duration_mins, 0.95) AS p95_threshold,
		MAX(duration_mins) AS p100_threshold
	FROM normal_questions_only
	GROUP BY eval_set_id, content_id
),

-- ========== 5.2) 标记每个用户在每道题上是否低于各阈值 ==========
user_question_flags AS (
	SELECT
		n.user_id,
		n.eval_set_id,
		n.game_id,
		n.content_id,
		n.duration_mins,
		CASE WHEN n.duration_mins < t.p05_threshold THEN 1 ELSE 0 END AS below_p05,
		CASE WHEN n.duration_mins < t.p10_threshold THEN 1 ELSE 0 END AS below_p10,
		CASE WHEN n.duration_mins < t.p15_threshold THEN 1 ELSE 0 END AS below_p15,
		CASE WHEN n.duration_mins < t.p20_threshold THEN 1 ELSE 0 END AS below_p20,
		CASE WHEN n.duration_mins < t.p25_threshold THEN 1 ELSE 0 END AS below_p25,
		CASE WHEN n.duration_mins < t.p30_threshold THEN 1 ELSE 0 END AS below_p30,
		CASE WHEN n.duration_mins < t.p35_threshold THEN 1 ELSE 0 END AS below_p35,
		CASE WHEN n.duration_mins < t.p40_threshold THEN 1 ELSE 0 END AS below_p40,
		CASE WHEN n.duration_mins < t.p45_threshold THEN 1 ELSE 0 END AS below_p45,
		CASE WHEN n.duration_mins < t.p50_threshold THEN 1 ELSE 0 END AS below_p50,
		CASE WHEN n.duration_mins < t.p55_threshold THEN 1 ELSE 0 END AS below_p55,
		CASE WHEN n.duration_mins < t.p60_threshold THEN 1 ELSE 0 END AS below_p60,
		CASE WHEN n.duration_mins < t.p65_threshold THEN 1 ELSE 0 END AS below_p65,
		CASE WHEN n.duration_mins < t.p70_threshold THEN 1 ELSE 0 END AS below_p70,
		CASE WHEN n.duration_mins < t.p75_threshold THEN 1 ELSE 0 END AS below_p75,
		CASE WHEN n.duration_mins < t.p80_threshold THEN 1 ELSE 0 END AS below_p80,
		CASE WHEN n.duration_mins < t.p85_threshold THEN 1 ELSE 0 END AS below_p85,
		CASE WHEN n.duration_mins < t.p90_threshold THEN 1 ELSE 0 END AS below_p90,
		CASE WHEN n.duration_mins < t.p95_threshold THEN 1 ELSE 0 END AS below_p95,
		0 AS below_p100  -- 没有人会低于100%分位数
	FROM normal_questions_only n
	INNER JOIN question_thresholds t
		ON t.eval_set_id = n.eval_set_id
		AND t.content_id = n.content_id
),

-- ========== 5.3) 统计每个用户是否会被排除（按game_id） ==========
user_exclusion_by_game AS (
	SELECT
		user_id,
		eval_set_id,
		game_id,
		MAX(below_p05) AS excluded_at_p05,
		MAX(below_p10) AS excluded_at_p10,
		MAX(below_p15) AS excluded_at_p15,
		MAX(below_p20) AS excluded_at_p20,
		MAX(below_p25) AS excluded_at_p25,
		MAX(below_p30) AS excluded_at_p30,
		MAX(below_p35) AS excluded_at_p35,
		MAX(below_p40) AS excluded_at_p40,
		MAX(below_p45) AS excluded_at_p45,
		MAX(below_p50) AS excluded_at_p50,
		MAX(below_p55) AS excluded_at_p55,
		MAX(below_p60) AS excluded_at_p60,
		MAX(below_p65) AS excluded_at_p65,
		MAX(below_p70) AS excluded_at_p70,
		MAX(below_p75) AS excluded_at_p75,
		MAX(below_p80) AS excluded_at_p80,
		MAX(below_p85) AS excluded_at_p85,
		MAX(below_p90) AS excluded_at_p90,
		MAX(below_p95) AS excluded_at_p95,
		MAX(below_p100) AS excluded_at_p100
	FROM user_question_flags
	GROUP BY user_id, eval_set_id, game_id
),

-- ========== 5.4) 统计每个game的总用户数 ==========
total_users_by_game AS (
	SELECT
		game_id,
		COUNT(DISTINCT user_id) AS total_users
	FROM normal_questions_only
	GROUP BY game_id
),

-- ========== 5.5) 统计全局总用户数 ==========
total_users_all AS (
	SELECT
		COUNT(DISTINCT user_id) AS total_users
	FROM normal_questions_only
),

-- ========== 6) 计算各game_id的分位数 ==========
percentiles_by_game AS (
	SELECT
		game_id,
		'P05' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.05) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P10' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.10) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P15' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.15) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P20' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.20) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P25' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.25) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P30' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.30) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P35' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.35) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P40' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.40) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P45' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.45) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P50' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.50) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P55' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.55) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P60' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.60) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P65' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.65) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P70' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.70) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P75' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.75) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P80' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.80) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P85' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.85) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P90' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.90) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P95' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.95) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
	
	UNION ALL
	
	SELECT
		game_id,
		'P100' AS percentile_label,
		MAX(duration_mins) AS percentile_value
	FROM normal_questions_only
	GROUP BY game_id
),

-- ========== 7) 计算全局分位数（game_id='all'） ==========
percentiles_all AS (
	SELECT
		'all' AS game_id,
		'P05' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.05) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P10' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.10) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P15' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.15) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P20' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.20) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P25' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.25) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P30' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.30) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P35' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.35) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P40' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.40) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P45' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.45) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P50' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.50) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P55' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.55) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P60' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.60) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P65' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.65) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P70' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.70) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P75' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.75) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P80' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.80) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P85' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.85) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P90' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.90) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P95' AS percentile_label,
		PERCENTILE_APPROX(duration_mins, 0.95) AS percentile_value
	FROM normal_questions_only
	
	UNION ALL
	
	SELECT
		'all' AS game_id,
		'P100' AS percentile_label,
		MAX(duration_mins) AS percentile_value
	FROM normal_questions_only
),

-- ========== 6.1) 统计各game的排除用户数 ==========
excluded_users_by_game AS (
	SELECT game_id, 'P05' AS percentile_label, SUM(excluded_at_p05) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P10' AS percentile_label, SUM(excluded_at_p10) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P15' AS percentile_label, SUM(excluded_at_p15) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P20' AS percentile_label, SUM(excluded_at_p20) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P25' AS percentile_label, SUM(excluded_at_p25) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P30' AS percentile_label, SUM(excluded_at_p30) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P35' AS percentile_label, SUM(excluded_at_p35) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P40' AS percentile_label, SUM(excluded_at_p40) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P45' AS percentile_label, SUM(excluded_at_p45) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P50' AS percentile_label, SUM(excluded_at_p50) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P55' AS percentile_label, SUM(excluded_at_p55) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P60' AS percentile_label, SUM(excluded_at_p60) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P65' AS percentile_label, SUM(excluded_at_p65) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P70' AS percentile_label, SUM(excluded_at_p70) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P75' AS percentile_label, SUM(excluded_at_p75) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P80' AS percentile_label, SUM(excluded_at_p80) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P85' AS percentile_label, SUM(excluded_at_p85) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P90' AS percentile_label, SUM(excluded_at_p90) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P95' AS percentile_label, SUM(excluded_at_p95) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
	UNION ALL
	SELECT game_id, 'P100' AS percentile_label, SUM(excluded_at_p100) AS excluded_users FROM user_exclusion_by_game GROUP BY game_id
),

-- ========== 6.2) 统计全局排除用户数 ==========
excluded_users_all AS (
	SELECT 'all' AS game_id, 'P05' AS percentile_label, SUM(excluded_at_p05) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P10' AS percentile_label, SUM(excluded_at_p10) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P15' AS percentile_label, SUM(excluded_at_p15) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P20' AS percentile_label, SUM(excluded_at_p20) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P25' AS percentile_label, SUM(excluded_at_p25) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P30' AS percentile_label, SUM(excluded_at_p30) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P35' AS percentile_label, SUM(excluded_at_p35) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P40' AS percentile_label, SUM(excluded_at_p40) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P45' AS percentile_label, SUM(excluded_at_p45) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P50' AS percentile_label, SUM(excluded_at_p50) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P55' AS percentile_label, SUM(excluded_at_p55) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P60' AS percentile_label, SUM(excluded_at_p60) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P65' AS percentile_label, SUM(excluded_at_p65) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P70' AS percentile_label, SUM(excluded_at_p70) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P75' AS percentile_label, SUM(excluded_at_p75) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P80' AS percentile_label, SUM(excluded_at_p80) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P85' AS percentile_label, SUM(excluded_at_p85) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P90' AS percentile_label, SUM(excluded_at_p90) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P95' AS percentile_label, SUM(excluded_at_p95) AS excluded_users FROM user_exclusion_by_game
	UNION ALL
	SELECT 'all' AS game_id, 'P100' AS percentile_label, SUM(excluded_at_p100) AS excluded_users FROM user_exclusion_by_game
)

-- ========== 8) 合并结果并排序 ==========
SELECT
	p.game_id,
	p.percentile_label,
	ROUND(p.percentile_value, 2) AS duration_mins,
	NVL(e.excluded_users, 0) AS excluded_users,
	t.total_users
FROM percentiles_by_game p
LEFT JOIN excluded_users_by_game e
	ON e.game_id = p.game_id
	AND e.percentile_label = p.percentile_label
LEFT JOIN total_users_by_game t
	ON t.game_id = p.game_id

UNION ALL

SELECT
	p.game_id,
	p.percentile_label,
	ROUND(p.percentile_value, 2) AS duration_mins,
	NVL(e.excluded_users, 0) AS excluded_users,
	(SELECT total_users FROM total_users_all) AS total_users
FROM percentiles_all p
LEFT JOIN excluded_users_all e
	ON e.game_id = p.game_id
	AND e.percentile_label = p.percentile_label

ORDER BY 
	CASE WHEN game_id = 'all' THEN 'ZZZZZ' ELSE game_id END,
	CASE percentile_label
		WHEN 'P05' THEN 1
		WHEN 'P10' THEN 2
		WHEN 'P15' THEN 3
		WHEN 'P20' THEN 4
		WHEN 'P25' THEN 5
		WHEN 'P30' THEN 6
		WHEN 'P35' THEN 7
		WHEN 'P40' THEN 8
		WHEN 'P45' THEN 9
		WHEN 'P50' THEN 10
		WHEN 'P55' THEN 11
		WHEN 'P60' THEN 12
		WHEN 'P65' THEN 13
		WHEN 'P70' THEN 14
		WHEN 'P75' THEN 15
		WHEN 'P80' THEN 16
		WHEN 'P85' THEN 17
		WHEN 'P90' THEN 18
		WHEN 'P95' THEN 19
		WHEN 'P100' THEN 20
	END
LIMIT 1000;

