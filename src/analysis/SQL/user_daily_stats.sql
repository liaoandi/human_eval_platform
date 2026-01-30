--odps sql 
--********************************************************************--
--author:廖安迪
--create time:2025-11-03
--********************************************************************--

-- 计算每天的总eval人数和总sub人数

WITH base AS (
	SELECT
		TO_CHAR(FROM_UNIXTIME(start_at), 'yyyy-MM-dd') AS eval_date,
		user_id,
		status AS eval_status,
		submitted
	FROM eval_sessions
	WHERE dt = MAX_PT('eval_sessions')
	AND eval_set_id >= 10
)

SELECT 
	eval_date,
	COUNT(DISTINCT user_id) AS total_eval_users,
	COUNT(DISTINCT CASE WHEN eval_status = 5 THEN user_id END) AS total_submit_users,
	COUNT(DISTINCT CASE WHEN eval_status = 5 THEN user_id END) / COUNT(DISTINCT user_id) AS submit_rate
FROM base
GROUP BY eval_date
ORDER BY eval_date
;








