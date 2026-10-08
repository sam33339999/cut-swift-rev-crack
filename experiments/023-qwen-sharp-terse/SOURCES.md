# 023 用到的 Qwen Sharp 段落

這一輪不在 `test-reason-flow.md` 第 1–62 條。沒有改 `/content/models/ornith-1.5-9B/chat_template.jinja`。`chat_template.jinja` 是 Ornith 原模板的複本，只在無工具路徑加上下面這一段。工具呼叫和 `<think>` 預填沒動，也沒有 022 的 Qwen3.8 low 句子。

來源：<https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates>

Sharp 本身是 froggeric 的修正模板再加一段強制系統提示，不改權重。這 14 題是單輪數學、沒有工具，所以只用思考開啟時的 terseness 段落。froggeric 的工具修正和多輪保留思考沒有套用。

思考開啟時的原文：

```text
Answer directly, after thinking. Lead with the answer, then only what it needs to be correct and usable.
Never: open with preamble or pleasantries; restate the question; add filler transitions; hedge with niceties; or repeat a point you've already made.
Always: keep essential steps, caveats, uncertainties, and specifics — never drop correctness or a needed warning for brevity. Keep the final answer lean. Use the least structure that conveys it (plain prose when short; lists or code only when they earn their place). If genuinely uncertain, say so and explain why — never omit uncertainty for the sake of brevity.
If a user request is genuinely ambiguous, ask a sharp question, don't guess.
```

Sharp 可以用 `chat_template_kwargs` 的 `{"terse": false}` 拿掉這一段。這一輪沒有做那個開關，模板裡這一段一律附上。
