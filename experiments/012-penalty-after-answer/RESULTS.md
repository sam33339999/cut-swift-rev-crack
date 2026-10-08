# 012-penalty-after-answer

方法：14。

這一輪在抽樣前先由本程式關掉 port 8000 上的 vLLM，再用 `start_server.sh` 重新載入 Ornith-1.5-9B。
重新載入完成時間：2026-10-08T00:23:57.132805+00:00。models id：Ornith-1.5-9B。max_model_len：20480。

## 旋鈕

思考區出現成形答案之前不罰。成形答案指「the answer is」、「answer:」、「\boxed{」或「答案是」後面緊接整數，而且只看 </think> 之前的文字。出現之後才擋 002 的四個片語（含句首大寫）的下一個 token：let me reconsider、Let me reconsider、to be safe、To be safe、make sure、Make sure、let me recheck、Let me recheck。不是全程 bad_words。

```json
{
  "method": "14",
  "rule": "思考區出現成形答案之前不罰。成形答案指「the answer is」、「answer:」、「\\boxed{」或「答案是」後面緊接整數，而且只看 </think> 之前的文字。出現之後才擋 002 的四個片語（含句首大寫）的下一個 token：let me reconsider、Let me reconsider、to be safe、To be safe、make sure、Make sure、let me recheck、Let me recheck。不是全程 bad_words。",
  "system": null,
  "extra_body": {
    "vllm_xargs": {
      "penalty_after_answer": 1
    }
  }
}
```

## 分數

正確的定義和基線相同：沒有截斷，而且可見回答最後一個 `ANSWER:` 整數等於標準答案。截斷留在分母裡。

| 籃 | 答對 | 截斷 | 思考平均 | 思考中位數 | 相對基線答對 | 相對基線平均 | 相對基線中位數 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easy | 24/24 | 0 | 44.3750 | 37.0000 | +0 | +0.0000 | +0.0000 |
| hard | 29/32 | 1 | 1685.4688 | 888.5000 | -1 | +64.2812 | +0.0000 |

難題思考中位數沒有變短。不把這輪記成有效縮短。

## 成形答案有沒有出現

用和線上相同的規則回看思考區：21/56 筆出現成形答案，罰分有機會打開。
沒有出現的那幾筆，這輪和基線的差別只會是抽樣波動。
