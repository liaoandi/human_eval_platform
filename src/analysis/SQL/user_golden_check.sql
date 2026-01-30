--odps sql 
--********************************************************************--
--author:廖安迪
--create time:2025-08-21 17:15:44
--********************************************************************--
-- odps sql
-- ********************************************************************
-- 目的：按用户统计 golden 反作弊完成情况（修正 dwd.answer_id 为 UUID 的映射）
-- ********************************************************************
WITH
-- ① dwd：每个"UUID答案"的正确性
answer_truth_uuid AS (
	SELECT 
		answer_id, -- 注意：这是 UUID（字符串）
		CASE 
			WHEN LOWER(NVL(get_json_object(generation_metadata, '$.correct'), '')) IN ('true','1','t','yes','y') THEN 1
			WHEN LOWER(NVL(get_json_object(generation_metadata, '$.correct'), '')) IN ('false','0','f','no','n') THEN 0
			ELSE NULL 
		END AS is_correct
	FROM eval_answer_inc
	WHERE dm = '2025-06'
),

-- ② 从 ODS 取 UUID↔ID 映射（去重取最新）
answers_map AS (
	SELECT 
		id AS answer_id, -- 数值型 id（用于 comparisons.answer_left/right 关联）
		uuid AS answer_uuid -- 字符串 UUID（用于跟 dwd 关联）
	FROM (
		SELECT 
			a.id, 
			a.uuid, 
			ROW_NUMBER() OVER (PARTITION BY a.uuid ORDER BY a.updated_at DESC, a.created_at DESC) AS rn
		FROM llm_answers a
		WHERE a.dt = MAX_PT('llm_answers')
	) t
	WHERE rn = 1
),

-- ③ 把 dwd 的 UUID 正确性转成以 ID 为主键
answer_truth_id AS (
	SELECT 
		m.answer_id, -- 数值型 id
		u.is_correct
	FROM answers_map m
	LEFT JOIN answer_truth_uuid u ON u.answer_id = m.answer_uuid
),

-- ④ golden 对比题及"正确答案"（以 ID 连接）
golden_pairs AS (
	SELECT 
		c.id AS comparison_id, 
		c.eval_set_id, 
		c.eval_query_id, 
		c.answer_left, 
		c.answer_right,
		al.is_correct AS left_correct, 
		ar.is_correct AS right_correct,
		CASE 
			WHEN al.is_correct = 1 AND NVL(ar.is_correct,0) = 0 THEN c.answer_left
			WHEN ar.is_correct = 1 AND NVL(al.is_correct,0) = 0 THEN c.answer_right
			ELSE NULL 
		END AS correct_answer_id
	FROM eval_comparisons c
	LEFT JOIN answer_truth_id al ON al.answer_id = c.answer_left
	LEFT JOIN answer_truth_id ar ON ar.answer_id = c.answer_right
	WHERE c.dt = MAX_PT('eval_comparisons')
		AND c.is_golden = 1
),

-- ⑤ 用户选择（按维度展开 → 去重到"每个 comparison 只计一次"）
user_choice_raw AS (
	SELECT 
		t.id AS task_id, 
		t.eval_session_id AS session_id, 
		t.eval_set_id, 
		t.content_id AS comparison_id, -- = comparisons.id
		(t.end_at - t.start_at) / 60.0 AS duration_mins,
		r.winner_id AS winner_answer_id, -- 这里是 answers.id（数值型）
		r.eval_dim_id AS eval_dim_id
	FROM eval_tasks t
	LATERAL VIEW explode(
		from_json(
			t.result,
			'array<struct<winner_id:bigint, eval_dim_id:bigint, eval_dim_key:string>>'
		)
	) e AS r
	WHERE t.dt = MAX_PT('eval_tasks')
		AND t.status = 12
),

user_choice_dedup AS (
	SELECT 
		task_id, 
		session_id, 
		eval_set_id, 
		comparison_id, 
		duration_mins, 
		winner_answer_id
	FROM (
		SELECT 
			u.*, 
			ROW_NUMBER() OVER (PARTITION BY u.task_id, u.comparison_id ORDER BY u.eval_dim_id) AS rn
		FROM user_choice_raw u
	) x
	WHERE rn = 1
),

-- ⑥ 贴上 user_id，并只保留 golden 对比题
golden_votes AS (
	SELECT 
		s.user_id, 
		d.task_id, 
		d.session_id, 
		d.eval_set_id, 
		d.comparison_id, 
		d.duration_mins, 
		d.winner_answer_id, 
		gp.correct_answer_id
	FROM user_choice_dedup d
	INNER JOIN eval_sessions s 
		ON s.dt = MAX_PT('eval_sessions') 
		AND s.id = d.session_id
	INNER JOIN golden_pairs gp 
		ON gp.eval_set_id = d.eval_set_id 
		AND gp.comparison_id = d.comparison_id
),

-- ⑦ 打标（阈值直接常量 0.25 分钟；如不要时长过滤见下方替换）
scored AS (
	SELECT 
		v.*,
		CASE 
			WHEN v.duration_mins < 0.25 THEN 1 
			ELSE 0 
		END AS is_too_fast,
		CASE 
			WHEN v.duration_mins >= 0.25 AND v.correct_answer_id IS NOT NULL THEN 1 
			ELSE 0 
		END AS is_scored,
		CASE 
			WHEN v.correct_answer_id IS NOT NULL AND v.winner_answer_id = v.correct_answer_id THEN 1
			WHEN v.correct_answer_id IS NOT NULL THEN 0
			ELSE NULL 
		END AS is_correct
	FROM golden_votes v
),

-- ⑧ 用户级聚合
user_golden AS (
	SELECT 
		user_id,
		COUNT(*) AS golden_total,
		SUM(CASE WHEN is_scored = 1 THEN 1 ELSE 0 END) AS golden_scored,
		SUM(CASE WHEN is_scored = 1 AND is_correct = 1 THEN 1 ELSE 0 END) AS golden_correct,
		SUM(CASE WHEN is_too_fast = 1 THEN 1 ELSE 0 END) AS too_fast_cnt,
		MEDIAN(duration_mins) AS median_duration_mins,
		AVG(duration_mins) AS avg_duration_mins
	FROM scored
	GROUP BY user_id
)

SELECT 
	u.user_id,
	u.golden_total,
	u.golden_scored,
	u.golden_correct,
	ROUND(CASE WHEN u.golden_scored > 0 THEN u.golden_correct / CAST(u.golden_scored AS DOUBLE) END, 4) AS golden_accuracy,
	u.too_fast_cnt,
	ROUND(CASE WHEN u.golden_total > 0 THEN u.too_fast_cnt / CAST(u.golden_total AS DOUBLE) END, 4) AS fast_rate,
	ROUND(u.median_duration_mins, 3) AS median_dur_mins,
	ROUND(u.avg_duration_mins, 3) AS avg_dur_mins
FROM user_golden u;