# 001 實驗紀錄

日期：2026-10-07。

這次只做兩件事。

1. 開發集基線。空系統提示，思考打開，解碼參數鎖死。簡單題、難題都跑。頂到輸出上限算答錯。
2. 第 24 條。同一題裡，把答對的軌跡分成較短和較長，挖長組中後段才常見的 token 和片語。

沒有做 logit bias，沒有微調。

## 環境

- 模型：`/content/models/ornith-1.5-9B`（Ornith-1.5-9B，Qwen3.5 架構，思考標記 `<think>` / `</think>`）。
- GPU：NVIDIA A100-SXM4-40GB。
- vLLM 0.31.0，實際程式是 `/usr/bin/python3 /usr/local/bin/vllm`。
- `/content/.venv` 裡沒有裝 vLLM，不能拿來啟動服務。

## 原來的服務

2026-10-07 15:03 左右由既有工作階段啟動，工作目錄 `/content`：

```text
uv run vllm serve /content/models/ornith-1.5-9B \
  --served-model-name Ornith-1.5-9B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

行程：`uv` pid 6407，python pid 6410，引擎 pid 6831。約占 35 GB。

這個上下文長度對本實驗太大。題目不到 200 token，單次輸出上限 8192，用 262144 會把 KV cache 拉滿，同時能跑的序列變少。

## 這次的服務

腳本：`start_server.sh`。

改了三處：

- `--max-model-len 20480`。夠放下題目加 8192 輸出。
- `--max-num-seqs 4`。和客戶端 4 條並行對齊。
- 拿掉 tool-call parser。這次沒有工具呼叫，避免工具標記干擾思考區切分。`--reasoning-parser qwen3` 保留。
- 啟動時預設 `enable_thinking=true`。請求裡也再傳一次。

直接呼叫 `/usr/local/bin/vllm`，不經過 `uv run`，避免 uv 重新解析套件。版本仍是 0.31.0。

要回到原來的服務，把上面那條 `uv run` 命令再跑一次即可。本實驗結束後先留著這份較短上下文的服務。

## 開發集

`problems.json` 由 `make_problems.py` 產生。每一題的標準答案都是程式算出來的，腳本裡有斷言。6 題簡單、8 題難題。題幹是英文，因為先前探測時這個模型的思考區是英文。中文題幹留到文件第 7 條，這次不混在一起。

題目後面固定加一句答案格式：可見回答的最後一行必須是 `ANSWER: <整數>`。

| id | 籃 | 答案 | 核對方式 |
| --- | --- | --- | --- |
| e01 | easy | 391 | 17×23 |
| e02 | easy | 1024 | 2^10 |
| e03 | easy | 6 | gcd(84, 30) |
| e04 | easy | 9 | 36 的正因數個數 |
| e05 | easy | 5050 | 1 加到 100 |
| e06 | easy | 3 | 17^3 mod 5 |
| h01 | hard | 405 | n! 至少 100 個尾端 0 的最小 n |
| h02 | hard | 401 | ≤1000，被 3 或 5 整除，但不被 7 整除 |
| h03 | hard | 31875000 | 唯一的 a<b<c、a+b+c=1000 的畢氏三元組，取乘積 |
| h04 | hard | 5521 | 3^12+2^12 的最大質因數 |
| h05 | hard | 615 | n<100，n 和 n+1 都能被大於 1 的平方整除，取和 |
| h06 | hard | 376 | 2^100 mod 1000 |
| h07 | hard | 2 | 100 寫成至少兩個連續正整數之和的方法數 |
| h08 | hard | 1060 | 小於 100 的質數之和 |

h05 的 n 是 8、24、27、44、48、49、63、75、80、98、99。h07 的兩種寫法是 18+19+20+21+22 和 9+10+…+16。h03 只有 (200, 375, 425)。

## 抽樣

鎖死，每一題 4 個 seed：101、102、103、104。

- temperature 1.0
- top_p 0.95
- top_k 20
- min_p 0
- presence_penalty 0
- repetition_penalty 1
- max_tokens 8192
- 無系統提示

先前探測過，會做的題大約 40 到 800 個思考 token。8192 是上限，不是目標長度。頂到上限記截斷，正確率裡算答錯，不從分母拿掉。

正確：可見回答最後一個 `ANSWER:` 行的整數等於標準答案，而且 `finish_reason` 不是 `length`。思考區裡出現標準答案不算答對。

傳輸錯誤另外記，不進正確率分母。`run_baseline.py` 每完成一筆就追加到 jsonl，中斷後重跑會跳過已經成功的列。

## 挖詞規則

實作在 `analyze.py`。

- 長度用 tokenizer 對思考字串的切分。指標表的思考 token 用 API 的 `reasoning_tokens`。兩套數字都留下。
- 只拿沒截斷且答對的軌跡。依長度排序，前半短、後半長。奇數條時中間那條不進任何一組。
- 長組中位數不到短組的 1.25 倍，這題不進嚴格彙總。
- 單 token 看長組後段（後 50%）對上短組整段。片語是長組後段裡的 3 詞和 4 詞。
- 分數是各題（長組後段出現率 − 短組出現率）的平均。
- 嚴格留下：至少 2 題、分數 ≥ 0.35、出現位置偏後段（後段次數占該詞在長組次數的 60% 以上）、短組出現率 ≤ 0.25。
- 純數字、單一字元、極常見功能詞不收。跨題才彙總，避免把某一題的運算符號當成空轉詞。

信賴區間：以題為單位重抽 1000 次，seed 固定為 0。這份開發集題數少，區間只描述這一次，不用來分辨 2 到 3 個百分點。

## 操作紀錄

- 15:42:18Z：對 pgid 6407 送 SIGTERM。port 8000 隨即關掉，GPU 記憶體回到 0。
- 15:42:41Z：用 `start_server.sh` 啟動。vLLM 0.31.0 接受了 `--max-model-len 20480`、`--max-num-seqs 4`、`--default-chat-template-kwargs {"enable_thinking": true}`。權重 17.66 GiB，編譯和 warmup 之後才開始聽 port。
- 15:46:21Z：`/v1/models` 回 Ornith-1.5-9B，`max_model_len` 20480，GPU 約 34890 MiB。
- 格式試跑：`pilot.jsonl`，e01 與 h01，seed 101 和 102。四筆都答對，`ANSWER:` 格式可用。試跑沒有併進 `results.jsonl`。
- 正式基線：`run_baseline.py`，56 筆，無傳輸錯誤，牆鐘約 254 秒。結果見 `RESULTS.md`。
- 挖詞：`analyze.py` 寫出 `metrics.json`、`mined_tokens.json`、`RESULTS.md`。人工從表裡挑出的懲罰候選在 `recommended.json`，挑的理由寫在 `RESULTS.md` 的解讀。

服務現在仍是這份實驗設定，聽在 `0.0.0.0:8000`。要回到原本 262144 上下文和 tool parser 的服務，用本檔「原來的服務」那一條命令。

## 檔案

| 檔 | 內容 |
| --- | --- |
| `PROTOCOL.md` | 這份紀錄 |
| `make_problems.py` | 產生並斷言標準答案 |
| `problems.json` | 開發集 |
| `start_server.sh` | 這次的 vLLM 命令 |
| `run_baseline.py` | 抽樣、評分、可續跑 |
| `analyze.py` | 指標、挖詞、自我測試 |
| `pilot.jsonl` | 格式試跑，4 筆 |
| `results.jsonl` | 正式 56 筆，含完整思考 |
| `metrics.json` | 正確率、長度、信賴區間 |
| `mined_tokens.json` | 門檻篩過的 token 和片語 |
| `recommended.json` | 解讀後建議拿去罰的少數片語 |
| `RESULTS.md` | 給人看的結果 |
| `logs/vllm.log` | 服務日誌 |
| `logs/server_history.jsonl` | 開關服務和跑完基線的時間 |
