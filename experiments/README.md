# 實驗

- `001-baseline-and-mine/`：Ornith-1.5-9B 的開發集基線，以及同一題內長短軌跡的空轉詞。紀錄在該目錄的 `PROTOCOL.md` 和 `RESULTS.md`。
- `002-phrase-penalty/`：同一份開發集上，用 vLLM `bad_words` 封鎖第一檔空轉片語。進入點是 `run_round.py`，它會先自己關掉並重載 vLLM。
