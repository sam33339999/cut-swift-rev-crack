# 050 Thinking-Cap，接到本機 Ornith-1.5-9B

2026-10-08 已跑完。開發集結論和繁體三題在 `RESULTS.md`。下面是這次實際用的設定。

來源是 [khudgins/ornith-thinking-cap](https://github.com/khudgins/ornith-thinking-cap)。那份是 Ornith-1.0-9B 的 GRPO，不是這個機器上的 Ornith-1.5-9B。獎勵是：

```
reward = 答對(1 或 0)
       - (λ × 思考長度 / 2048，只有答對才扣)
       + 格式分(0.1，有 </think> 而且可見回答不是空的)
```

λ 在原專案從 `1e-3` 拉到 `5e-2`。長度扣分最多 0.05，錯的答案加格式分也只有 0.1，所以錯的不能贏過對的。這次一輪用他們的上限 `λ = 0.05`。`rewards.py` 的單元測試鎖住這條。

## 和原專案不同的地方

原專案是線上 GRPO：每步抽 8 條，約 1000 步，GSM8K 加 MBPP 加 HumanEval，LoRA rank 32，還含 MLP。這個 A100 40GB 放不下那個設定，也放不下 vLLM 和訓練同時開。

這次：

- 基座是 `/content/models/ornith-1.5-9B`。不寫進那個目錄。
- 樣本用 `experiments/045-dual-reward/rollouts.jsonl`。32 題、seed 201–204，英文可核對數學，沒有 Sharp，不和開發集、留出題重複。不再抽一次。
- 只做一輪離線更新。不是 1000 步線上 GRPO，不能拿來對他們 README 的 GSM8K −74%。
- 學習率 `2e-6`，KL `0.04`，梯度裁到 1.0。這兩個數是他們 9B 設定。
- LoRA rank 8、alpha 16。目標是 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`in_proj_qkv`、`out_proj`。不加 MLP，避免 40GB 爆掉。
- 優勢只乘在 `<think>` 裡和寫出 `</think>` 的 token。`ANSWER` 那幾個 token 優勢是 0。本機第一次第 45 條就是因為優勢蓋到答案儀式，才重複 `ANS`。
- 新 adapter 在這個目錄的 `adapter/`。服務名稱 `m50cap`。不覆蓋 `m45`、`m45retry`、`zhgeneral`。

## 什麼算有調整

開發集仍是旁證：14 題、seed 101–104。達標句子只用 `experiments/summary.py` 的 `judgement`。沒達標就寫沒達標。

通用繁體不是這條獎勵能直接練的。對話沒有唯一整數。訓練後用三題對一下基座和 `m50cap`（每題最多 400 token）：國慶日、把一句話改短、Redis 選擇後再問快取失效。這三題不過，就不把 `m50cap` 當成通用模型。通用聊天仍用基座 `Ornith-1.5-9B`。
