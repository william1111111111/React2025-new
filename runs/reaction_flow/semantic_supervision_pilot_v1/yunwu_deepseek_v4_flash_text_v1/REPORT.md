# DeepSeek V4 Flash TRAIN文本标注

Yunwu，thinking disabled。首批34条完整转录，16路并发；对5条失败记录各做一次修正请求，所有响应保留。

{
  "records": 34,
  "requests": 39,
  "initial_concurrency": 16,
  "repair_concurrency": 5,
  "checks_passed": 32,
  "pending_failure_review": [
    "rec_4aca068a3f7d6d60",
    "rec_e70c4c7a1f8e5d96"
  ],
  "checked_candidates": 234,
  "usage_including_repairs": {
    "prompt_tokens": 15414,
    "completion_tokens": 26035,
    "total_tokens": 41449
  },
  "human_reviewed": 0,
  "training_eligible": 0
}

通过检查只代表格式、引用及显式权限字段通过；不是语义正确性结论。复核表仅导出通过检查的候选。剩余2条需复核原文引用/数量。时间与说话者归属仍unknown，未进入训练。用量含本批次修正请求，不含之前平台测试。
