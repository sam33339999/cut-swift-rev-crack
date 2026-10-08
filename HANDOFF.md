# 交接

日期：2026-10-07。待辦以 `TODO.md` 為準。使用者說「繼續」時做那個檔的第一個未完成項，做完就勾選。這份是環境和評分規則。不要重跑已經完成的 001–008。

## 現在狀態

- 模型：`/content/models/ornith-1.5-9B`（Ornith-1.5-9B，Qwen3.5 架構）。
- 服務：vLLM 0.31.0，執行檔 `/content/.venv/bin/vllm`，聽 `0.0.0.0:8000`，`max-model-len 20480`，最多 4 條並行，思考預設打開，沒有 tool parser。啟動腳本是 `experiments/001-baseline-and-mine/start_server.sh`。
- vLLM 裝在 `/content/.venv`（uv workspace `/content`，依賴寫在這個 repo 的 `pyproject.toml`）。不要改用系統 Python 的 `vllm`。
- GPU：A100 40GB，服務約占 35GB。新的一輪要先用 `experiments/server_ctl.py` 的 `reload_server` 關掉再載入。不要另開第二個引擎。
- 不要把服務改回原本的 262144 上下文加 tool parser，除非使用者明確要求。原本那條命令寫在 `experiments/001-baseline-and-mine/PROTOCOL.md`。

## 已經做完的

開發集 14 題（`experiments/001-baseline-and-mine/problems.json`），seed 101–104，共 56 筆。解碼鎖死：temperature 1.0、top_p 0.95、top_k 20、min_p 0、presence_penalty 0、repetition_penalty 1、max_tokens 8192、思考打開（003 除外）。

正確的定義在 `experiments/grader.py`：沒有截斷，而且可見回答最後一個 `ANSWER:` 整數等於標準答案。截斷留在分母。傳輸錯誤不算進分母。

| 輪 | 旋鈕 | 難題 | 思考中位數 | 結論 |
| --- | --- | --- | --- | --- |
| 001 | 基線 | 30/32 | 888.5 | 對照 |
| 002 | 四個空轉片語 `bad_words` 硬封鎖 | 28/32 | 829.5 | 沒有變短；平均還變長 |
| 003 | `enable_thinking=false` | 25/32 | 0 | 思考是 0，答對少 5 |
| 004 | 系統提示「推理寫短，想到答案就停。」 | 31/32 | 593.5 | 目前唯一達標的一輪 |
| 005 | token 37781（` reconsider`）logit −2 | 29/32 | 821.5 | 看不出縮短 |
| 006 | NoWait 詞 logit −2 | 31/32 | 886.5 | 幾乎沒變 |
| 007 | `thinking_token_budget=2048` | 28/32 | 829.5 | 平均下降是長尾被砍，答對少 2 |
| 008 | `thinking_token_budget=512` | 21/32 | 511 | 答對少 9 |
| 009 | 系統提示「先給答案，再用最多三句驗證。」 | 29/32 | 424.0 | 開發集上達標。簡單題 24/24，中位數 68.5 |
| 010 | 系統提示「逐步推理，每一步最多五個詞。」 | 27/32 | 380.5 | 答對少 3。簡單題 18/24。答案 token 難題中位數 169 |
| 011 | 留出題，空提示對 004 | 31/32 對 31/32 | 641.0 對 598.5 | 004 的縮短不在。簡單題 24/24 對 22/24 |
| 012 | 答完才罰四個片語 | 29/32 | 888.5 | 中位數沒有變短。21/56 筆出現成形答案 |
| 013 | 系統提示禁 wait / alternatively / hmm / 再檢查一次 / 再想一次 | 30/32 | 697.5 | 開發集上達標。簡單題 22/24 |
| 014 | 題目後「思考不得超過 256 個 token。」 | 28/32 | 344.0 | 答對少 2。沒有達標 |
| 015 | 題目後「思考不得超過 512 個 token。」 | 29/32 | 420.5 | 開發集上達標 |
| 016 | 題目後「思考不得超過 1024 個 token。」 | 29/32 | 470.5 | 開發集上達標 |
| 017 | 題目後「思考不得超過 2048 個 token。」 | 30/32 | 631.0 | 開發集上達標。答對和基線相同 |
| 018 | 簡單題「用短推理」，難題「可以想完，但不要重複檢查。」 | 31/32 | 703.0 | 開發集上達標。簡單題中位數 56.5 |
| 019 | 004 的英文版 | 29/32 | 522.0 | 開發集上達標。簡單題 21/24 |
| 020 | 013 的英文版 | 31/32 | 682.0 | 開發集上達標。簡單題 23/24 |
| 021 | 009 的英文版 | 27/32 | 597.5 | 答對少 3。沒有達標 |
| 022 | Qwen3.8 low 模板句 | 32/32 | 553.0 | 開發集上達標。簡單題 24/24，中位數 36 |
| 023 | Qwen Sharp terseness 模板段 | 26/32 | 552.0 | 答對少 4。沒有達標 |
| 047 | 第 45 條 LoRA，優勢只在思考區 | 31/32 | 560.5 | 開發集上達標。第一次 adapter 因 `ANS` 迴圈作廢 |

達標規則在 `experiments/summary.py` 的 `judgement`：難題中位數至少少 15%，而且答對筆數沒有少超過 1。沒達標不要寫成有效縮短。004 和 009 達標，但是開發集，不是測試集。簡單題在 004 從 24/24 掉到 23/24。009 的簡單題答對沒掉，思考中位數變長。

這個聊天模板沒有 low / medium / high。`reasoning_effort` 只有 `none` 會關掉思考，其餘和基線相同。不要再為第 9 條的檔位各跑一輪。

每輪目錄裡有 `RESULTS.md`、`summary.json`、`results.jsonl`、`logs/reload.json`。總表是 `experiments/COMPARE.md`。

## 下一輪要做的

`TODO.md` 的 001–023 都做完了。開發集上達標的有 004、009、013、015、016、017、018、019、020、022。022 是 Qwen3.8 的 low 模板句，難題 32/32、中位數 553.0。023 是 Qwen Sharp 的 terseness 段落，難題 26/32、中位數 552.0，沒有達標，所以不開第 45 條。011 的留出題只測過 004，那次難題中位數只少約 6.6%，縮短不在。

留出題沒有站住之前，不要做 LoRA、合併或強化學習。

不要做的：

- 重跑 001–023。
- 把 `experiments/045-dual-reward/adapter/` 掛上去當可用模型。那次更新讓簡單題重複 `ANS`，說明在 `experiments/045-dual-reward/FAILURE.md`。
- 把 `zhgeneral`（`experiments/049-zh-general/adapter/`）當通用繁體模型。它沒有把小題和第二輪答完，說明在 `experiments/049-zh-general/FAILURE.md`。通用聊天用基座 `Ornith-1.5-9B`。
- 把 005、006 的懲罰加大後再掃一次。輕罰已經沒有差距。
- 在同一份 14 題上做 LoRA、合併或 GSPO。文件第 6 節寫明，解碼沒有在留出的題上站住之前不要訓練。
- 把中位數少幾個 token 的結果寫成成功。

## 怎麼開一輪

```bash
cd /content/swift-rev-crack
/usr/bin/python3 -m unittest experiments.test_grader experiments.test_next_rounds
```

新的一輪加進 `experiments/next_rounds.py` 的 `round_specs`，或照 `run_round.py` 自己重載再抽樣。抽樣前必須看到 `/v1/models` 回 `Ornith-1.5-9B`。每筆結果立刻追加到 jsonl。傳輸失敗的列可以重跑，已經成功的列不要覆蓋。
