# 022 用到的網路模板

這一輪不在 `test-reason-flow.md` 第 1–62 條。清單裡和模板有關的只有兩處：`enable_thinking=false` 會預填空的 `<think></think>`（003 已跑），以及訓練時多輪對話要不要保留歷史思考。Ornith-1.5-9B 自帶的 `chat_template.jinja` 沒有 low / medium / xhigh 的句子，所以只傳 `reasoning_effort` 不會改思考長度。

這一輪沒有改 `/content/models/ornith-1.5-9B/chat_template.jinja`。`chat_template.jinja` 是 Ornith 原模板的複本，只在無工具路徑加上下面那一句。工具呼叫和 `<think>` 預填沒動。

## 採用

Qwen3.8 官方模板在 `reasoning_effort=low` 時寫進系統訊息的句子，原文照抄：

> Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.

來源：<https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8/blob/main/chat_template.jinja>

同一份模板裡：

- `medium` 不加句子。
- `xhigh` 是「Please think carefully through the task, validate key assumptions, consider plausible alternatives...」，那是要求想更久，這輪不用。
- `enable_thinking=false` 會直接報錯，和 Ornith 預填空思考區不同。

同一段 low / xhigh 句子也出現在 <https://huggingface.co/FINAL-Bench/Darwin-180B-RSI/blob/main/chat_template.jinja>。022 抄的是 Qwen3.8 那一份。

## 看過但沒用

- Qwen3 把思考做成可關的預填：<https://huggingface.co/blog/qwen-3-chat-template-deep-dive>。空的 `<think></think>` 就是 003，不重跑。
- DeepSeek-R1-0528 討論串裡加的也是同一種空思考區：<https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/discussions/69>。
- Qwen Sharp 另加一整段 terseness，預設打開，可用 `chat_template_kwargs.terse=false` 關掉。開頭是「Answer directly, after thinking. Lead with the answer, then only what it needs to be correct and usable.」這是另一個旋鈕，022 沒有疊上去。
  - 專案：<https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates>
  - 句子出現在這個提交：<https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/commit/628f2e4b202cad6d597703ac268e91f8b414530b>
  - 較長的一版寫在討論：<https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/discussions/1>
  - 關掉 terseness 的參數：<https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/discussions/5>
- Sharp 上游的修正模板：<https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates>。修的是工具呼叫和思考開關，不是這句 low。
- 另一份把 `terse` 預設關掉的 Qwen 3.x 模板：<https://huggingface.co/OliviaRossi/Improved-Chat-Template-for-Qwen-3.x>。
- SmolLM3 用 `/think`、`/no_think`，GLM 用另一套 `Reasoning Effort` 標記。格式和 Ornith 的 Qwen 模板不同，沒有套用。
