# 實驗

根目錄的 `README.md` 是說明，`TODO.md` 是待辦。下一個 session 說「繼續」時做 `TODO.md` 的第一個未完成項。

- `001-baseline-and-mine/`：Ornith-1.5-9B 的開發集基線，以及同一題內長短軌跡的空轉詞。紀錄在該目錄的 `PROTOCOL.md` 和 `RESULTS.md`。
- `002-phrase-penalty/`：同一份開發集上，用 vLLM `bad_words` 封鎖第一檔空轉片語。進入點是 `run_round.py`，它會先自己關掉並重載 vLLM。
- `003-no-thinking/` 到 `008-think-budget-512/`：關掉思考、短提示、輕罰、NoWait、兩檔思考預算。由 `next_rounds.py` 逐輪重載後評分。
- `COMPARE.md`：八輪的分數對照。
