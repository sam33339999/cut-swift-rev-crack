# 實驗

根目錄的 `README.md` 是說明，`TODO.md` 是待辦。下一個 session 說「繼續」時做 `TODO.md` 的第一個未完成項。

- `001-baseline-and-mine/`：Ornith-1.5-9B 的開發集基線，以及同一題內長短軌跡的空轉詞。紀錄在該目錄的 `PROTOCOL.md` 和 `RESULTS.md`。
- `002-phrase-penalty/`：同一份開發集上，用 vLLM `bad_words` 封鎖第一檔空轉片語。進入點是 `run_round.py`，它會先自己關掉並重載 vLLM。
- `003-no-thinking/` 到 `010-chain-of-draft/`：關掉思考、短提示、輕罰、NoWait、兩檔思考預算、先答案、Chain of Draft。由 `next_rounds.py` 逐輪重載後評分。009 和 010 用 `--only 009-answer-first,010-chain-of-draft`。
- `011-heldout/`：另一份 14 題，只跑空提示和 004。題目由 `make_problems.py` 產生，`experiments/test_heldout.py` 再核對一次。
- `012-penalty-after-answer/`：思考裡出現成形答案之後才罰 002 的四個片語。規則在 `penalty_after_answer.py`，`experiments/test_penalty_after_answer.py` 檢查沒答完不罰。
- `COMPARE.md`：開發集各輪的分數對照。
