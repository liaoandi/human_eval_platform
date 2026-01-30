-- ========================================================================
-- 用户答题行为统计
-- ========================================================================
-- 功能：统计每个用户的答题行为指标
-- 指标：答题数、跳过题数、反作弊答对题数、答题时间统计（平均/中位数）
-- 维度：按user_id和game_id分组，同时支持game_id='all'的全局统计
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

-- ========== 7) 扩展用户数据：添加game_id='all'的聚合视图 ==========
user_choice_with_all_games AS (
	-- 原始数据：按实际game_id
	SELECT
		user_id,
		eval_set_id,
		content_id,
		is_golden,
		is_golden_correct,
		duration_mins,
		game_id
	FROM user_choice_with_user
	
	UNION ALL
	
	-- 聚合数据：所有game_id合并，标记为'all'
	SELECT
		user_id,
		eval_set_id,
		content_id,
		is_golden,
		is_golden_correct,
		duration_mins,
		'all' AS game_id
	FROM user_choice_with_user
),

-- ========== 8) 按题聚合：判断每个题目是否所有维度都答对 ==========
user_question_aggregated AS (
	SELECT 
		user_id, 
		eval_set_id,
		game_id,
		content_id, 
		is_golden,
		-- 计算该题目的总维度数和答对维度数
		COUNT(*) AS total_dims,
		SUM(CASE WHEN is_golden_correct = 1 THEN 1 ELSE 0 END) AS correct_dims,
		SUM(CASE WHEN is_golden_correct = 0 THEN 1 ELSE 0 END) AS wrong_dims,
		-- 只有所有维度都答对，才算真正答对
		CASE 
			WHEN is_golden = 1 AND MIN(is_golden_correct) = 1 AND MAX(is_golden_correct) = 1 THEN 1
			WHEN is_golden = 1 AND (MIN(is_golden_correct) = 0 OR MAX(is_golden_correct) = 0) THEN 0
			ELSE NULL
		END AS is_golden_correct_all_dims,
		MAX(duration_mins) AS duration_mins
	FROM user_choice_with_all_games
	GROUP BY user_id, eval_set_id, game_id, content_id, is_golden
),

-- ========== 9) 最终去重结果：每个用户×题目一条记录 ==========
user_question_dedupe AS (
	SELECT
		user_id,
		eval_set_id,
		game_id,
		content_id,
		is_golden,
		is_golden_correct_all_dims AS is_golden_correct,
		duration_mins
	FROM user_question_aggregated
),

-- ========== 10) 统计每个用户的答题行为 ==========
user_behavior_stats AS (
	SELECT
		user_id,
		eval_set_id,
		game_id,
		-- 答题数（已作答的题目数）
		COUNT(DISTINCT content_id) AS answered_count,
		-- 反作弊题答对数
		SUM(CASE WHEN is_golden = 1 AND is_golden_correct = 1 THEN 1 ELSE 0 END) AS golden_correct_count,
		-- 反作弊题总数
		SUM(CASE WHEN is_golden = 1 THEN 1 ELSE 0 END) AS golden_total_count,
		-- 正常题答题数
		SUM(CASE WHEN is_golden = 0 THEN 1 ELSE 0 END) AS normal_answered_count,
		-- 平均答题时间（分钟）
		ROUND(AVG(duration_mins), 2) AS avg_duration_mins,
		-- 中位数答题时间（分钟）
		ROUND(PERCENTILE_APPROX(duration_mins, 0.5), 2) AS median_duration_mins,
		-- 最小答题时间
		ROUND(MIN(duration_mins), 2) AS min_duration_mins,
		-- 最大答题时间
		ROUND(MAX(duration_mins), 2) AS max_duration_mins
	FROM user_question_dedupe
	GROUP BY user_id, eval_set_id, game_id
),

-- ========== 11) 用户级别的统计 ==========
user_level_stats AS (
	SELECT
		user_id,
		eval_set_id,
		game_id,
		answered_count,                                        -- 答题数
		18 - answered_count AS skipped_count,                  -- 跳过题数（假设总题数为18）
		golden_correct_count,                                  -- 反作弊题答对数
		golden_total_count,                                    -- 反作弊题总数
		normal_answered_count,                                 -- 正常题答题数
		avg_duration_mins,                                     -- 该用户的平均答题时间（分钟）
		median_duration_mins                                   -- 该用户的中位数答题时间（分钟）
	FROM user_behavior_stats
)

-- ========== 12) 总体分布统计：按game_id聚合 ==========
SELECT
	game_id,
	-- 用户数量
	COUNT(DISTINCT user_id) AS total_users,
	-- 人均答题数
	ROUND(AVG(answered_count), 2) AS avg_answered_per_user,
	-- 人均跳过题数
	ROUND(AVG(skipped_count), 2) AS avg_skipped_per_user,	
	-- 人均反作弊答对题数
	ROUND(AVG(golden_correct_count), 2) AS avg_golden_correct_per_user,
	-- 人均正常题答题数
	ROUND(AVG(normal_answered_count), 2) AS avg_normal_answered_per_user,
	-- 人均答题平均时间（所有用户的平均答题时间的平均值）
	ROUND(AVG(avg_duration_mins), 2) AS avg_of_avg_duration_mins,
	-- 人均答题中位数时间（所有用户的中位数答题时间的平均值）
	ROUND(AVG(median_duration_mins), 2) AS avg_of_median_duration_mins,
	-- 完成率（答完全部题的用户比例）
	ROUND(SUM(CASE WHEN answered_count = 18 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS completion_rate_pct,
	-- 反作弊题全对的用户比例
	ROUND(SUM(CASE WHEN golden_correct_count = golden_total_count AND golden_total_count > 0 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 2) AS golden_perfect_rate_pct
FROM user_level_stats
GROUP BY game_id
ORDER BY game_id
LIMIT 500;

-- ========================================================================
-- 反作弊题正确率统计
-- ========================================================================
-- 功能：统计每个反作弊题（golden题）的正确率
-- 维度：按content_id（题目ID）和game_id分组
-- ========================================================================

-- ========== 13) 每个反作弊题的正确率统计 ==========
SELECT
	game_id,
	content_id AS golden_question_id,
	-- 作答该题的用户数
	COUNT(DISTINCT user_id) AS total_attempts,
	-- 答对的用户数
	SUM(CASE WHEN is_golden_correct = 1 THEN 1 ELSE 0 END) AS correct_count,
	-- 答错的用户数
	SUM(CASE WHEN is_golden_correct = 0 THEN 1 ELSE 0 END) AS wrong_count,
	-- 正确率
	ROUND(SUM(CASE WHEN is_golden_correct = 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 4) AS correct_rate,
	-- 平均答题时间
	ROUND(AVG(duration_mins), 2) AS avg_duration_mins,
	-- 中位数答题时间
	ROUND(PERCENTILE_APPROX(duration_mins, 0.5), 2) AS median_duration_mins
FROM user_question_dedupe
WHERE is_golden = 1
GROUP BY game_id, content_id
ORDER BY game_id, correct_rate ASC
LIMIT 500;

-- ========================================================================
-- 反作弊题详细信息（含原始问题文本）
-- ========================================================================
-- 功能：展示每个反作弊题的详细信息，包括问题文本、正确答案等
-- ========================================================================

-- ========== 14) 反作弊题详细信息（含问题文本和正确率） ==========
WITH golden_stats AS (
	SELECT
		game_id,
		content_id AS comparison_id,
		COUNT(DISTINCT user_id) AS total_attempts,
		SUM(CASE WHEN is_golden_correct = 1 THEN 1 ELSE 0 END) AS correct_count,
		ROUND(SUM(CASE WHEN is_golden_correct = 1 THEN 1 ELSE 0 END) / COUNT(DISTINCT user_id), 4) AS correct_rate,
		ROUND(AVG(duration_mins), 2) AS avg_duration_mins
	FROM user_question_dedupe
	WHERE is_golden = 1
	GROUP BY game_id, content_id
),
golden_comparisons AS (
	SELECT
		c.id AS comparison_id,
		c.eval_set_id,
		c.eval_query_id,
		c.answer_left,
		c.answer_right,
		gp.correct_answer_id,
		q.query_text,
		cg.game_id
	FROM (
		SELECT id, eval_set_id, eval_query_id, answer_left, answer_right
		FROM eval_comparisons
		WHERE dt = MAX_PT('eval_comparisons')
			AND is_golden = 1
	) c
	LEFT JOIN golden_pairs gp
		ON gp.comparison_id = c.id
		AND gp.eval_set_id = c.eval_set_id
	LEFT JOIN (
		SELECT id, content AS query_text
		FROM eval_queries
		WHERE dt = MAX_PT('eval_queries')
	) q
		ON q.id = c.eval_query_id
	LEFT JOIN comparison_with_game cg
		ON cg.comparison_id = c.id
		AND cg.eval_set_id = c.eval_set_id
),
answer_texts AS (
	SELECT
		id AS answer_id,
		answer_text
	FROM llm_answers
	WHERE dt = MAX_PT('llm_answers')
)
SELECT
	gs.game_id,
	gc.comparison_id AS golden_question_id,
	gc.eval_set_id,
	gc.query_text AS question_text,
	gc.answer_left AS answer_left_id,
	al.answer_text AS answer_left_text,
	gc.answer_right AS answer_right_id,
	ar.answer_text AS answer_right_text,
	gc.correct_answer_id,
	CASE 
		WHEN gc.correct_answer_id = gc.answer_left THEN 'LEFT'
		WHEN gc.correct_answer_id = gc.answer_right THEN 'RIGHT'
		ELSE 'UNKNOWN'
	END AS correct_position,
	gs.total_attempts,
	gs.correct_count,
	gs.correct_rate,
	gs.avg_duration_mins
FROM golden_stats gs
INNER JOIN golden_comparisons gc
	ON gc.comparison_id = gs.comparison_id
LEFT JOIN answer_texts al
	ON al.answer_id = gc.answer_left
LEFT JOIN answer_texts ar
	ON ar.answer_id = gc.answer_right
WHERE gs.game_id != 'all'  -- 排除聚合数据，只看具体game_id
ORDER BY gs.game_id, gs.correct_rate ASC
LIMIT 500;

