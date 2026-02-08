### 帮我分析code agent生成代码是否正确，如果不正确，是执行过程中的什么原因导致的

输入（for each task）：生成的代码脚本、ground truth脚本、生成代码的log、task描述

输出：
1. 一个表格 包括ground truth的代码实现的要素（比如model, toolkit, method...，但是不包括prompt和task并未指定的参数）
2. 该脚本是否成功运行，如果成功运行，则输出成功运行日志的输出（比如"[Message 24]
Role: tool
Content: Today in New York City, the temperature is approximately 25.48°Fahrenheit, feeling like 12.88°Fahrenheit. The maximum temperature forecasted for today is 26.55°Fahrenheit, while the minimum temperature is expected to be around 23.0°Fahrenheit. The wind is blowing at a speed of 28.7670484 miles per hour from the northwest direction. The visibility is reported to be 6.21 miles. The sun will rise at 11:59 AM and set at 10:20 PM local time.
"），如果遇到了报错，则输出报错的日志作为证据（比如[Message 10]
Role: tool
Content: 2026-02-07 14:49:21,868 - camel.models.model_manager - ERROR - Error processing with model: <camel.models.gemini_model.GeminiModel object at 0x7f90262dc150>
2026-02-07 14:49:21,868 - camel.camel.agents.chat_agent - WARNING - Rate limit hit (attempt 1/3). Retrying in 0.4s
2026-02-07 14:49:25,854 - camel.models.model_manager - ERROR - Error processing with model: <camel.models.gemini_model.GeminiModel object at 0x7f90262dc150>
2026-02-07 14:49:25,854 - camel.camel.agents.chat_agent - WARNING - Rate limit hit (attempt 2/3). Retrying in 1.7s
2026-02-07 14:49:31,458 - camel.models.model_manager - ERROR - Error processing with model: <camel.models.gemini_model.GeminiModel object at 0x7f90262dc150>
2026-02-07 14:49:31,459 - camel.camel.agents.chat_agent - ERROR - Rate limit exhausted after 3 attempts
Traceback (most recent call last):
  File "/home/yangz0h/Programming/meta-agent/camel/task-script/single_agent/2_duckduckgo_agent.py", line 25, in <module>
    response = agent.step(question)
               ^^^^^^^^^^^^^^^^^^^^
  File "/home/yangz0h/Programming/meta-agent/camel/camel/agents/chat_agent.py", line 2799, in step
    return future.result(timeout=self.step_timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yangz0h/anaconda3/envs/meta-agent/lib/python3.11/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/home/yangz0h/anaconda3/envs/meta-agent/lib/python3.11/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/home/yangz0h/anaconda3/envs/meta-agent/lib/python3.11/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yangz0h/Programming/meta-agent/camel/camel/agents/chat_agent.py", line 2880, in _step_impl
    response = self._get_model_response(
               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yangz0h/Programming/meta-agent/camel/camel/agents/chat_agent.py", line 3401, in _get_model_response
    raise ModelProcessingError(
camel.models.model_manager.ModelProcessingError: Unable to process messages: Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3-pro\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3-pro\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3-pro\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3-pro\nPlease retry in 28.699406771s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3-pro'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3-pro'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3-pro'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerDay-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3-pro'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '28s'}]}}]
）

表格的格式：
| Item | Ground Truth | Generated | Match? | Acceptable? |
|------|-------------------|--------------|-----------|--------|
| Model | ... | ... | ✅/❌ | ✅/❌ |
| Toolkit | ... | ... | ✅/❌ | ✅/❌ |
| Method | ... | ... | ✅/❌ | ✅/❌ |

注意在对比task描述的过程中，要严格对比task描述中的每个要素，不能有遗漏，比如task描述中说要使用duckduckgo search，那么你就要严格对比ground truth和generated中是否都使用了duckduckgo search，如果只用了search_toolkit，是错误的，因为task描述中说要使用duckduckgo search，而不是search_toolkit

（在分析acceptable的时候参考task描述，是不是完成任务所必须的参数，如果参数不是任务必须的，而且/Users/yangz0h/Documents/Programming/meta-agent/camel/logsno-context和camel/logsw-context和camel/logs_singlew-context-single里面对应的脚本成功跑起来了，就算正确）

另外：log的开头是execution status，这个只代表了生成任务过程的状态，不代表你要分析的代码是否正确，你要分析的是代码是否正确，所以你要从log的结尾开始分析，看最后输出的结果、以及和ground truth对比是否正确

