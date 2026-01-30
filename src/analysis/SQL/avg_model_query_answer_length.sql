-- ========================================================================
-- 统计各模型的 Query 和 Answer 平均长度
-- ========================================================================
-- 
-- 关联表：
-- 1. eval_answer_inc_test (Answer 表)
-- 2. eval_query_item_inc_test (Query 表)
--
-- 统计指标：
-- - avg_query_chars: 平均 Query 字符数 (CHAR_LENGTH)
-- - avg_answer_chars: 平均 Answer 字符数 (CHAR_LENGTH)
-- - avg_query_bytes: 平均 Query 字节数 (LENGTH)
-- - avg_answer_bytes: 平均 Answer 字节数 (LENGTH)
-- - total_samples: 样本数量

SELECT
    a.model_id,
    a.model_name,
    -- Query 长度统计 (字符数 & 字节数)
    AVG(CHAR_LENGTH(q.raw_query)) AS avg_query_chars,
    AVG(LENGTH(q.raw_query)) AS avg_query_bytes,
    -- Answer 长度统计 (字符数 & 字节数)
    AVG(CHAR_LENGTH(a.answer_content)) AS avg_answer_chars,
    AVG(LENGTH(a.answer_content)) AS avg_answer_bytes,
    -- 样本量
    COUNT(*) AS total_samples
FROM
    eval_answer_inc_test a
JOIN
    eval_query_item_inc_test q
ON
    a.query_id = q.query_id
    AND a.game_id = q.game_id -- 关联分区字段以提高效率
WHERE
    -- 通常建议限制在最新分区，或根据需要去掉 WHERE 子句以统计全量
    a.dm = MAX_PT('eval_answer_inc_test')
    AND q.dm = MAX_PT('eval_query_item_inc_test')
GROUP BY
    a.model_id,
    a.model_name
ORDER BY
    avg_answer_chars DESC;

