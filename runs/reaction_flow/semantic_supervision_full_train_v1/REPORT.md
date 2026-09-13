# 全量 TRAIN 文本标注

DeepSeek V4 Flash，Yunwu，关闭思考。只发送现有speaker转录，不上传音视频或listener数据。复用已通过检查的试点记录，每条最多3次有效输出尝试，保留全部版本。

{
  "stage": "finished_with_pending_review",
  "total_records": 1660,
  "reused_records": 32,
  "checks_passed": 1645,
  "pending_records": 15,
  "attempts_recorded": 1868,
  "requests_with_usage": 1868,
  "usage_new_attempts": {
    "prompt_tokens": 728060,
    "completion_tokens": 1233079,
    "total_tokens": 1961139
  },
  "concurrency": 16,
  "elapsed_s": 1421.9,
  "training_eligible": 0,
  "human_reviewed": 0,
  "updated_unix": 1789303366.9679163,
  "candidate_count": 13580
}

自动检查不代表语义正确。时间与说话者归属未知，人工核验0，未用于训练。pending_review.json列出仍待处理记录；新请求用量不含复用试点的历史费用。
