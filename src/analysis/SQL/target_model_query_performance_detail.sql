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
			game_id,
			content AS answer_text
		FROM llm_answers
		WHERE dt = MAX_PT('llm_answers')
			AND eval_set_id >= 10
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
	model_answers_raw AS (
		SELECT
			a.eval_query_id,
			m.model_name,
			a.content AS answer_text
		FROM llm_answers a
		LEFT JOIN (
			SELECT *
			FROM llm_models
			WHERE dt = MAX_PT('llm_models')
		) m ON m.id = a.llm_model_id
		WHERE a.dt = MAX_PT('llm_answers')
			AND a.eval_set_id >= 10
	),
	model_answers_pivot AS (
		SELECT
			eval_query_id,
			MAX(CASE WHEN LOWER(model_name) LIKE 'target-model%' THEN answer_text END) AS target_model_answer,
			MAX(CASE WHEN LOWER(model_name) LIKE 'gemini%' THEN answer_text END) AS gemini_answer,
			MAX(CASE WHEN LOWER(model_name) LIKE 'gpt%' THEN answer_text END) AS gpt_answer,
			MAX(CASE WHEN LOWER(model_name) LIKE 'perplexity%' THEN answer_text END) AS pplx_answer
		FROM model_answers_raw
		GROUP BY eval_query_id
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
				WHEN LOWER(CASE WHEN v.left_model_name = 'target-model' THEN v.right_model_name ELSE v.left_model_name END) LIKE 'gemini%' THEN 'gemini'
				WHEN LOWER(CASE WHEN v.left_model_name = 'target-model' THEN v.right_model_name ELSE v.left_model_name END) LIKE 'gpt%' THEN 'gpt'
				WHEN LOWER(CASE WHEN v.left_model_name = 'target-model' THEN v.right_model_name ELSE v.left_model_name END) LIKE 'perplexity%' THEN 'pplx'
				ELSE NULL
			END AS opponent_family,
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
	target_model_query_stats AS (
		SELECT
			game_id,
			category,
			eval_query_id,
			MAX(raw_query) AS raw_query,
			COUNT(DISTINCT comparison_id) AS comparison_count,
			COUNT(*) AS total_votes,
			SUM(CASE WHEN target_position = 'left' THEN 1 ELSE 0 END) AS target_left_votes,
			SUM(CASE WHEN target_position = 'right' THEN 1 ELSE 0 END) AS target_right_votes
		FROM target_model_vote_records
		GROUP BY
			game_id,
			category,
			eval_query_id
	),
	dimension_winrates AS (
		SELECT
			game_id,
			category,
			eval_query_id,
			eval_dim_key,
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
			END AS dim_winrate
		FROM target_model_vote_records
		GROUP BY
			game_id,
			category,
			eval_query_id,
			eval_dim_key
	),
	dimension_winrates_pivot AS (
		SELECT
			game_id,
			category,
			eval_query_id,
			MAX(CASE WHEN eval_dim_key = 'model_style' THEN dim_winrate END) AS wr_model_style,
			MAX(CASE WHEN eval_dim_key = 'result_relevance' THEN dim_winrate END) AS wr_result_relevance,
			MAX(CASE WHEN eval_dim_key = 'result_usefulness' THEN dim_winrate END) AS wr_result_usefulness
		FROM dimension_winrates
		GROUP BY
			game_id,
			category,
			eval_query_id
	),
	dimension_winrates_family AS (
		SELECT
			game_id,
			category,
			eval_query_id,
			opponent_family,
			eval_dim_key,
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
			END AS dim_winrate
		FROM target_model_vote_records
		WHERE opponent_family IS NOT NULL
		GROUP BY
			game_id,
			category,
			eval_query_id,
			opponent_family,
			eval_dim_key
	),
	dimension_winrates_family_pivot AS (
		SELECT
			game_id,
			category,
			eval_query_id,
			MAX(CASE WHEN eval_dim_key = 'model_style' AND opponent_family = 'gemini' THEN dim_winrate END) AS wr_model_style_gemini,
			MAX(CASE WHEN eval_dim_key = 'model_style' AND opponent_family = 'gpt' THEN dim_winrate END) AS wr_model_style_gpt,
			MAX(CASE WHEN eval_dim_key = 'model_style' AND opponent_family = 'pplx' THEN dim_winrate END) AS wr_model_style_pplx,
			MAX(CASE WHEN eval_dim_key = 'result_relevance' AND opponent_family = 'gemini' THEN dim_winrate END) AS wr_result_relevance_gemini,
			MAX(CASE WHEN eval_dim_key = 'result_relevance' AND opponent_family = 'gpt' THEN dim_winrate END) AS wr_result_relevance_gpt,
			MAX(CASE WHEN eval_dim_key = 'result_relevance' AND opponent_family = 'pplx' THEN dim_winrate END) AS wr_result_relevance_pplx,
			MAX(CASE WHEN eval_dim_key = 'result_usefulness' AND opponent_family = 'gemini' THEN dim_winrate END) AS wr_result_usefulness_gemini,
			MAX(CASE WHEN eval_dim_key = 'result_usefulness' AND opponent_family = 'gpt' THEN dim_winrate END) AS wr_result_usefulness_gpt,
			MAX(CASE WHEN eval_dim_key = 'result_usefulness' AND opponent_family = 'pplx' THEN dim_winrate END) AS wr_result_usefulness_pplx
		FROM dimension_winrates_family
		GROUP BY
			game_id,
			category,
			eval_query_id
	),
	target_model_query_detail AS (
		SELECT
			stats.game_id,
			stats.category,
			stats.eval_query_id,
			stats.raw_query,
			stats.comparison_count,
			stats.total_votes,
			stats.target_left_votes,
			stats.target_right_votes,
			ma.target_model_answer,
			ma.gemini_answer,
			ma.gpt_answer,
			ma.pplx_answer,
			pivot_wr.wr_model_style,
			pivot_wr.wr_result_relevance,
			pivot_wr.wr_result_usefulness,
			pivot_wr_family.wr_model_style_gemini,
			pivot_wr_family.wr_model_style_gpt,
			pivot_wr_family.wr_model_style_pplx,
			pivot_wr_family.wr_result_relevance_gemini,
			pivot_wr_family.wr_result_relevance_gpt,
			pivot_wr_family.wr_result_relevance_pplx,
			pivot_wr_family.wr_result_usefulness_gemini,
			pivot_wr_family.wr_result_usefulness_gpt,
			pivot_wr_family.wr_result_usefulness_pplx
		FROM target_model_query_stats stats
		LEFT JOIN model_answers_pivot ma
			ON ma.eval_query_id = stats.eval_query_id
		LEFT JOIN dimension_winrates_pivot pivot_wr
			ON pivot_wr.game_id = stats.game_id
			AND pivot_wr.category = stats.category
			AND pivot_wr.eval_query_id = stats.eval_query_id
		LEFT JOIN dimension_winrates_family_pivot pivot_wr_family
			ON pivot_wr_family.game_id = stats.game_id
			AND pivot_wr_family.category = stats.category
			AND pivot_wr_family.eval_query_id = stats.eval_query_id
	)
SELECT
	game_id,
	category,
	eval_query_id,
	raw_query,
	total_votes,
	target_left_votes,
	target_right_votes,
	wr_model_style,
	wr_result_relevance,
	wr_result_usefulness,
	wr_model_style_gemini,
	wr_model_style_gpt,
	wr_model_style_pplx,
	wr_result_relevance_gemini,
	wr_result_relevance_gpt,
	wr_result_relevance_pplx,
	wr_result_usefulness_gemini,
	wr_result_usefulness_gpt,
	wr_result_usefulness_pplx,
	target_model_answer,
	gemini_answer,
	gpt_answer,
	pplx_answer
FROM target_model_query_detail
ORDER BY
	game_id,
	category,
	total_votes DESC,
	eval_query_id
LIMIT 10000;

