# 交接

日期：2026-10-07。這份給下一個 session 的 agent。先讀這份，再讀 `experiments/COMPARE.md`。不要重跑已經完成的 001–008。

## 現在狀態

- 模型：`/content/models/ornith-1.5-9B`（Ornith-1.5-9B，Qwen3.5 架構）。
- 服務：vLLM 0.31.0，`/usr/bin/python3 /usr/local/bin/vllm`，聽 `0.0.0.0:8000`，`max-model-len 20480`，最多 4 條並行，思考預設打開，沒有 tool parser。啟動腳本是 `experiments/001-baseline-and-mine/start_server.sh`。
- `/content/.venv` 沒有 vLLM，不要用它啟動服務。
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

達標規則在 `experiments/summary.py` 的 `judgement`：難題中位數至少少 15%，而且答對筆數沒有少超過 1。沒達標不要寫成有效縮短。004 達標，但是開發集，不是測試集。簡單題在 004 從 24/24 掉到 23/24。

這個聊天模板沒有 low / medium / high。`reasoning_effort` 只有 `none` 會關掉思考，其餘和基線相同。不要再為第 9 條的檔位各跑一輪。

每輪目錄裡有 `RESULTS.md`、`summary.json`、`results.jsonl`、`logs/reload.json`。總表是 `experiments/COMPARE.md`。

## 下一輪要做的

一次只改一個旋鈕。每一輪抽樣前先重載。沿用 `experiments/next_rounds.py` 的寫法：重載、抽樣、用 `grader.py` 評分、寫 `RESULTS.md` 和 `summary.json`，並把新的一列補進 `COMPARE.md`。

建議順序：

1. **009 先答案、後短驗證（文件第 3 條）。** 系統提示只加「先給答案，再用最多三句驗證。」004 說明提示有用，這一條是下一句該試的提示，不是再罰 token。
2. **010 Chain of Draft（文件第 4 條）。** 系統提示只加「逐步推理，每一步最多五個詞。」思考 token 和答案 token 分開數。
3. **另做一份測試集，再只跑基線和 004。** 題目要用程式對答案，不能和現有 14 題重複，也不能拿來調提示。004 在開發集上的縮短要在這份題上再看一次。沒有這一步，不要做 LoRA 或強化學習。
4. **答完才罰（文件第 14 條）先不要用全域 `bad_words`。** 002 已經證明整段硬封鎖會讓軌跡改道變長。這一條要在思考裡出現成形答案之後才罰。vLLM 這次啟動沒有掛自訂 logits processor；若做不到「出現答案才罰」，就寫明做不到，不要改成另一種全域懲罰充數。

不要做的：

- 重跑 001–008。
- 把 005、006 的懲罰加大後再掃一次。輕罰已經沒有差距。
- 在同一份 14 題上做 LoRA、合併或 GSPO。文件第 6 節寫明，解碼沒有在留出的題上站住之前不要訓練。
- 把中位數少幾個 token 的結果寫成成功。

## 怎麼開一輪

```bash
cd /content/swift-rev-crack
/usr/bin/python3 -m unittest experiments.test_grader experiments.test_next_rounds
```

新的一輪加進 `experiments/next_rounds.py` 的 `round_specs`，或照 `run_round.py` 自己重載再抽樣。抽樣前必須看到 `/v1/models` 回 `Ornith-1.5-9B`。每筆結果立刻追加到 jsonl。傳輸失敗的列可以重跑，已經成功的列不要覆蓋。
