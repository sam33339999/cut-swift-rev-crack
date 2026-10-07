# 003-no-thinking

方法：55，以及第 9 條在這個模板上唯一不同的一檔。

這一輪在抽樣前先由本程式關掉 port 8000 上的 vLLM，再用 `start_server.sh` 重新載入 Ornith-1.5-9B。
重新載入完成時間：2026-10-07T16:41:25.772243+00:00。models id：Ornith-1.5-9B。max_model_len：20480。

## 旋鈕

chat_template_kwargs enable_thinking=false。模板會預填空的思考區。low/medium/high 不會改這個模板，所以不另開一輪。

```json
{
  "method": "55，以及第 9 條在這個模板上唯一不同的一檔",
  "rule": "chat_template_kwargs enable_thinking=false。模板會預填空的思考區。low/medium/high 不會改這個模板，所以不另開一輪。",
  "system": null,
  "extra_body": {
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }
}
```

## 分數

正確的定義和基線相同：沒有截斷，而且可見回答最後一個 `ANSWER:` 整數等於標準答案。截斷留在分母裡。

| 籃 | 答對 | 截斷 | 思考平均 | 思考中位數 | 相對基線答對 | 相對基線平均 | 相對基線中位數 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| easy | 24/24 | 0 | 0.0000 | 0.0000 | +0 | -44.3750 | -37.0000 |
| hard | 25/32 | 0 | 0.0000 | 0.0000 | -5 | -1621.1875 | -888.5000 |

難題中位數變短，但答對少了 2 筆以上。不把這輪記成可用的縮短。
