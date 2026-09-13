# TRAIN 转录语义候选首批标注

模型：OpenLux claude-sonnet-4-6。34 条完整转录，来源于现有17 session试点，每session两条；仅发送转录，不发送媒体、listener、session标识或原始文件名。

- 完成34/34请求；290个候选。
- 33条通过格式、原文引用、引用顺序、unknown说话者与空时间检查；1条quote_order失败，保留原始响应待复核。
- 输入14901 token，输出27647 token，合计42548 token；平均1251.4，中位数1161.0，范围507–2071 token/条。
- 本轮提示词新增说话者unknown，最多16候选，与此前单条8候选成本测试不同。用量不含之前healthcheck和单条测试；平台缓存计价需单独核算。
- 人工核验0；可训练标签0。没有时间对齐、listener观察或paired关联标注。
- review_candidates.jsonl包含全部候选及检查状态；检查失败记录不可自动接收。原始response.txt保留。
- 这些是34条整段文本的候选，不能对应为已完成200个带时标事件窗口。
