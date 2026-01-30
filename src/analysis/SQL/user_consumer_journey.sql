--odps sql 
--********************************************************************--
--author:廖安迪
--create time:2025-10-28 11:08:38
--updated: 2025-11-03 (added game_id = 'all' aggregation)
--********************************************************************--

-- 1）看整体答题的状态,作答用户数及时长中位数
-- 包含各个game_id的统计 + game_id = 'all'的整体统计

WITH base AS (
	SELECT
		eval_set_id,
		user_id, 
		game_id, 
		status   AS eval_status,
		submitted, 
		FROM_UNIXTIME(start_at) AS start_time,
		(end_at - start_at) / 60 AS duration_mins
	FROM eval_sessions
	WHERE dt = MAX_PT('eval_sessions')
	AND eval_set_id >= 10
),

-- 按game_id分组的统计
answer_by_game AS (
	SELECT  
		-- 1为进行中，4为已放弃, 5为已经提交
		game_id,
		eval_set_id,
		eval_status, 
		SUM(IF(submitted == 0 , 1, 0))                        AS quick_quit_user, 
		COUNT(DISTINCT user_id)                               AS user_cnt,
		MEDIAN(duration_mins)                                 AS median_duration
	FROM base 
	GROUP BY 
		game_id,
		eval_set_id,
		eval_status
),

alipay_by_game AS (
	SELECT 
		eval_set_id,
		game_id,
		eval_status,
		SUM(IF(alipay_account IS NOT NULL, 1, 0)) AS alipay_user_cnt
	FROM base
	LEFT JOIN (
		SELECT  id, alipay_account, phone
		FROM    creators
		WHERE   ds = MAX_PT('creators')
	) creator
	ON base.user_id = creator.id
	GROUP BY 
		eval_set_id,
		game_id,
		eval_status
),

-- 所有game_id的整体统计
answer_all AS (
	SELECT  
		'all' AS game_id,
		eval_set_id,
		eval_status, 
		SUM(IF(submitted == 0 , 1, 0))                        AS quick_quit_user, 
		COUNT(DISTINCT user_id)                               AS user_cnt,
		MEDIAN(duration_mins)                                 AS median_duration
	FROM base 
	GROUP BY 
		eval_set_id,
		eval_status
),

alipay_all AS (
	SELECT 
		eval_set_id,
		'all' AS game_id,
		eval_status,
		SUM(IF(alipay_account IS NOT NULL, 1, 0)) AS alipay_user_cnt
	FROM base
	LEFT JOIN (
		SELECT  id, alipay_account, phone
		FROM    creators
		WHERE   ds = MAX_PT('creators')
	) creator
	ON base.user_id = creator.id
	GROUP BY 
		eval_set_id,
		eval_status
)

-- 合并按game_id的统计结果
SELECT 
	answer_by_game.game_id,
	answer_by_game.eval_set_id,
	CASE WHEN answer_by_game.eval_status = 1 THEN 'quit'
		WHEN answer_by_game.eval_status = 4  THEN 'ongoing'
		WHEN answer_by_game.eval_status = 5  THEN 'submit'
	END AS eval_status,
	user_cnt,
	median_duration,
	quick_quit_user,
	alipay_user_cnt,
	quick_quit_user / user_cnt AS quick_quit_rate,
	alipay_user_cnt / user_cnt AS alipay_bind_rate
FROM answer_by_game 
LEFT JOIN alipay_by_game
ON answer_by_game.eval_set_id = alipay_by_game.eval_set_id
AND answer_by_game.game_id = alipay_by_game.game_id
AND answer_by_game.eval_status = alipay_by_game.eval_status

UNION ALL

-- 合并game_id = 'all'的统计结果
SELECT 
	answer_all.game_id,
	answer_all.eval_set_id,
	CASE WHEN answer_all.eval_status = 1 THEN 'quit'
		WHEN answer_all.eval_status = 4  THEN 'ongoing'
		WHEN answer_all.eval_status = 5  THEN 'submit'
	END AS eval_status,
	user_cnt,
	median_duration,
	quick_quit_user,
	alipay_user_cnt,
	quick_quit_user / user_cnt AS quick_quit_rate,
	alipay_user_cnt / user_cnt AS alipay_bind_rate
FROM answer_all 
LEFT JOIN alipay_all
ON answer_all.eval_set_id = alipay_all.eval_set_id
AND answer_all.game_id = alipay_all.game_id
AND answer_all.eval_status = alipay_all.eval_status

ORDER BY eval_set_id, game_id, eval_status
;

