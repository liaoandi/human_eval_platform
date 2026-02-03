WITH
	user_choice AS (
		SELECT
			t.eval_set_id,
			t.content_id AS comparison_id,
			r.winner_id,
			r.eval_dim_key
		FROM eval_tasks t
		LATERAL VIEW explode(
			from_json(
				t.result,
				'array<struct<winner_id:bigint, eval_dim_id:bigint, eval_dim_key:string>>'
			)
		) e AS r
		WHERE t.dt = MAX_PT('eval_tasks')
			AND t.status = 12
			AND t.eval_set_id >= 10
			AND r.winner_id IS NOT NULL
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
		LEFT JOIN models m
			ON m.model_id = a.llm_model_id
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
			n.eval_query_id,
			COALESCE(l.game_id, r.game_id) AS game_id,
			n.answer_left,
			n.answer_right,
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
	votes_with_models AS (
		SELECT
			u.eval_set_id,
			cwm.comparison_id,
			cwm.eval_query_id,
			cwm.game_id,
			cwm.answer_left,
			cwm.answer_right,
			cwm.left_model_name,
			cwm.right_model_name,
			u.winner_id,
			u.eval_dim_key
		FROM user_choice u
		INNER JOIN comparison_with_models cwm
			ON cwm.comparison_id = u.comparison_id
			AND cwm.eval_set_id = u.eval_set_id
	),
	eval_queries AS (
		SELECT
			q.id AS eval_query_id,
			q.uuid AS eval_query_uuid,
			q.content AS raw_query
		FROM eval_queries q
		WHERE q.dt = MAX_PT('eval_queries')
	),
	query_items AS (
		SELECT
			qi.query_id,
			qi.category
		FROM eval_query_item_inc qi
		WHERE qi.dm = MAX_PT('eval_query_item_inc')
	),
	target_model_vote_records AS (
		SELECT
			COALESCE(v.game_id, 'unknown') AS game_id,
			COALESCE(qi.category, 'UNCLASSIFIED') AS category,
			v.eval_dim_key,
			v.eval_set_id,
			v.eval_query_id,
			eq.raw_query,
			v.comparison_id,
			v.answer_left,
			v.answer_right,
			v.left_model_name,
			v.right_model_name,
			CASE
				WHEN v.left_model_name = 'target-model' THEN 'left'
				ELSE 'right'
			END AS target_position,
			CASE
				WHEN v.left_model_name = 'target-model' THEN v.answer_left
				ELSE v.answer_right
			END AS target_answer_id,
			CASE
				WHEN v.left_model_name = 'target-model' THEN v.right_model_name
				ELSE v.left_model_name
			END AS opponent_model,
			CASE
				WHEN v.winner_id = 0 THEN 'draw'
				WHEN v.winner_id = (
					CASE
						WHEN v.left_model_name = 'target-model' THEN v.answer_left
						ELSE v.answer_right
					END
				) THEN 'win'
				ELSE 'loss'
			END AS target_result
		FROM votes_with_models v
		LEFT JOIN eval_queries eq
			ON eq.eval_query_id = v.eval_query_id
		LEFT JOIN query_items qi
			ON qi.query_id = eq.eval_query_uuid
		WHERE v.left_model_name = 'target-model'
			OR v.right_model_name = 'target-model'
	),
	category_summary AS (
		SELECT
			game_id,
			category,
			COUNT(DISTINCT eval_query_id) AS query_count,
			COUNT(DISTINCT comparison_id) AS comparison_count,
			COUNT(*) AS total_votes,
			SUM(CASE WHEN target_result = 'win' THEN 1 ELSE 0 END) AS win_cnt,
			SUM(CASE WHEN target_result = 'loss' THEN 1 ELSE 0 END) AS loss_cnt,
			SUM(CASE WHEN target_result = 'draw' THEN 1 ELSE 0 END) AS draw_cnt,
			CASE
				WHEN COUNT(*) - SUM(CASE WHEN target_result = 'draw' THEN 1 ELSE 0 END) > 0
				THEN ROUND(
					CAST(SUM(CASE WHEN target_result = 'win' THEN 1 ELSE 0 END) AS DOUBLE)
					/ CAST(
						COUNT(*) - SUM(CASE WHEN target_result = 'draw' THEN 1 ELSE 0 END)
						AS DOUBLE
					),
					4
				)
				ELSE NULL
			END AS target_winrate
		FROM target_model_vote_records
		GROUP BY game_id, category
	),
	category_dimension_summary AS (
		SELECT
			game_id,
			category,
			eval_dim_key,
			COUNT(DISTINCT eval_query_id) AS query_count,
			COUNT(DISTINCT comparison_id) AS comparison_count,
			COUNT(*) AS total_votes,
			SUM(CASE WHEN target_result = 'win' THEN 1 ELSE 0 END) AS win_cnt,
			SUM(CASE WHEN target_result = 'loss' THEN 1 ELSE 0 END) AS loss_cnt,
			SUM(CASE WHEN target_result = 'draw' THEN 1 ELSE 0 END) AS draw_cnt,
			CASE
				WHEN COUNT(*) - SUM(CASE WHEN target_result = 'draw' THEN 1 ELSE 0 END) > 0
				THEN ROUND(
					CAST(SUM(CASE WHEN target_result = 'win' THEN 1 ELSE 0 END) AS DOUBLE)
					/ CAST(
						COUNT(*) - SUM(CASE WHEN target_result = 'draw' THEN 1 ELSE 0 END)
						AS DOUBLE
					),
					4
				)
				ELSE NULL
			END AS target_winrate
		FROM target_model_vote_records
		GROUP BY game_id, category, eval_dim_key
	)
SELECT
	'by_category' AS aggregation_level,
	game_id,
	category,
	CAST(NULL AS STRING) AS eval_dim_key,
	query_count,
	comparison_count,
	total_votes,
	win_cnt,
	loss_cnt,
	draw_cnt,
	target_winrate
FROM category_summary

UNION ALL

SELECT
	'by_category_eval_dim' AS aggregation_level,
	game_id,
	category,
	eval_dim_key,
	query_count,
	comparison_count,
	total_votes,
	win_cnt,
	loss_cnt,
	draw_cnt,
	target_winrate
FROM category_dimension_summary

ORDER BY
	aggregation_level,
	game_id,
	category,
	eval_dim_key
LIMIT 5000;

