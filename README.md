# 縮短推理長度

在 Ornith-1.5-9B 上試「思考變短，正確率只差一點」。對照的是 UkisAI Swift 公開過的方向：罰空轉，不直接罰思考總長度。方法清單在 `test-reason-flow.md`。

這是後訓練和解碼干預的實驗，不是重跑預訓練。權重配方沒有公開，所以每一輪只改一個旋鈕，並跟同一份開發集比。

## 現在

模型在 `/content/models/ornith-1.5-9B`。vLLM 0.31.0 聽 `0.0.0.0:8000`，上下文 20480。啟動腳本是 `experiments/001-baseline-and-mine/start_server.sh`。`/content/.venv` 裡沒有 vLLM。

開發集 14 題、每題 4 個 seed，已經跑完 001 到 008。分數在 `experiments/COMPARE.md`。

目前只有 **004** 達標：系統提示「推理寫短，想到答案就停。」難題思考中位數從 888.5 降到 593.5，答對 31/32（基線 30/32）。這仍是開發集，不是測試集。關掉思考、片語硬封鎖、輕罰 token、思考預算都沒有同時保住正確率。

## 繼續

待辦在 [`TODO.md`](TODO.md)。進到這個專案後跟 agent 說「繼續」，它會從第一個未完成項做到驗收條件成立，把 `- [ ]` 改成 `- [x]`，寫上結果，然後做下一項。已經勾選的項不要重跑。

操作規則在 `AGENTS.md` 和 `HANDOFF.md`。

## 目錄

| 路徑 | 內容 |
| --- | --- |
| `TODO.md` | 待辦。勾選以這個檔為準 |
| `HANDOFF.md` | 環境、評分、已經做完的輪、不要做的事 |
| `experiments/COMPARE.md` | 各輪分數 |
| `experiments/00N-*/` | 該輪的 `RESULTS.md`、`summary.json`、`results.jsonl` |
| `experiments/grader.py` | 正確與否的定義 |
| `test-reason-flow.md` | 全部可試方法 |

## 怎樣算答對

沒有被截斷，而且可見回答最後一個 `ANSWER:` 的整數等於標準答案。截斷留在分母裡，算錯。思考區裡出現過答案不算答對。

難題中位數至少比基線少 15%，而且答對筆數沒有少超過 1，才算這輪達標。差幾個 token 不要寫成成功。
