#!/usr/bin/env python3
"""Score the baseline and mine stall tokens within each problem.

Length groups use the tokenizer length of the reasoning string. The metrics
table uses the server's reasoning_tokens count. A problem enters the strict
mine only when it has at least two non-truncated correct traces and the long
half is at least 1.25 times the short half.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "/content/models/ornith-1.5-9B"

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "is", "it",
    "that", "this", "we", "for", "with", "by", "be", "as", "at", "from",
    "are", "was", "so", "if", "then", "i", "my", "our", "can", "will",
}
WORD_RE = re.compile(r"[a-z]+(?:[-'][a-z]+)*|[\u4e00-\u9fff]+")
REFERENCE = [
    ("wait", r"\bwait\b"),
    ("hmm", r"\bhmm\b"),
    ("alternatively", r"\balternatively\b"),
    ("actually", r"\bactually\b"),
    ("however", r"\bhowever\b"),
    ("let me", r"\blet me\b"),
    ("double-check", r"\bdouble-?check\b"),
    ("reconsider", r"\breconsider\b"),
    ("re-examine", r"\bre-?examine\b"),
    ("hold on", r"\bhold on\b"),
    ("make sure", r"\bmake sure\b"),
    ("check again", r"\bcheck again\b"),
    ("another way", r"\banother way\b"),
    ("等等", r"等等"),
    ("不對", r"不對"),
    ("再想", r"再想"),
    ("再檢查", r"再檢查"),
    ("換一種", r"換一種"),
    ("或者", r"或者"),
    ("但是", r"但是"),
]


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ok_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("error")]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[index]


def bootstrap_stat(groups: dict[str, list[float]], stat, rng: random.Random, n: int = 1000):
    ids = list(groups)
    if not ids:
        return None
    point_pool = [value for pid in ids for value in groups[pid]]
    point = stat(point_pool)
    samples = []
    for _ in range(n):
        pool = []
        for _ in ids:
            pool.extend(groups[rng.choice(ids)])
        samples.append(stat(pool))
    samples.sort()
    return {
        "point": point,
        "low": percentile(samples, 0.025),
        "high": percentile(samples, 0.975),
        "n_problems": len(ids),
        "n_samples": len(point_pool),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def basket_metrics(rows: list[dict], basket: str) -> dict:
    chosen = [row for row in rows if row["basket"] == basket]
    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in chosen:
        by_problem[row["problem_id"]].append(row)
    correct_groups = {
        pid: [1.0 if row["strict_correct"] else 0.0 for row in items]
        for pid, items in by_problem.items()
    }
    length_groups = {}
    for pid, items in by_problem.items():
        lengths = [row["reasoning_tokens_api"] for row in items if row.get("reasoning_tokens_api") is not None]
        if lengths:
            length_groups[pid] = lengths
    rng = random.Random(0)
    n = len(chosen)
    n_correct = sum(1 for row in chosen if row["strict_correct"])
    n_truncated = sum(1 for row in chosen if row["truncated"])
    return {
        "basket": basket,
        "n_samples": n,
        "n_problems": len(by_problem),
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else None,
        "n_truncated": n_truncated,
        "n_transport_errors_excluded": 0,
        "accuracy_bootstrap_95": bootstrap_stat(correct_groups, mean, rng),
        "reasoning_tokens_mean_bootstrap_95": bootstrap_stat(length_groups, mean, rng),
        "reasoning_tokens_median_bootstrap_95": bootstrap_stat(length_groups, median, rng),
        "problems": problem_rows(by_problem),
    }


def problem_rows(by_problem: dict[str, list[dict]]) -> list[dict]:
    out = []
    for pid in sorted(by_problem):
        items = sorted(by_problem[pid], key=lambda row: row["seed"])
        lengths = [row.get("reasoning_tokens_api") for row in items]
        correct_lengths = [
            row["reasoning_tokens_api"]
            for row in items
            if row["strict_correct"] and row.get("reasoning_tokens_api") is not None
        ]
        out.append(
            {
                "problem_id": pid,
                "n": len(items),
                "n_correct": sum(1 for row in items if row["strict_correct"]),
                "n_truncated": sum(1 for row in items if row["truncated"]),
                "extracted": [row.get("extracted_answer") for row in items],
                "gold": items[0]["gold"],
                "reasoning_tokens": lengths,
                "correct_reasoning_tokens_median": median(correct_lengths) if correct_lengths else None,
                "gold_in_reasoning_when_wrong": sum(
                    1
                    for row in items
                    if (not row["strict_correct"]) and row.get("gold_in_reasoning")
                ),
            }
        )
    return out


def keep_unigram(surface: str) -> bool:
    text = surface.strip().lower()
    if not text:
        return False
    if len(text) <= 1 and not ("\u4e00" <= text <= "\u9fff"):
        return False
    if all(ch.isdigit() or ch in ".,%/+-^_" for ch in text):
        return False
    if text in STOPWORDS:
        return False
    return True


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def split_problem(traces: list[dict]) -> dict:
    good = [trace for trace in traces if trace["correct"] and not trace["truncated"] and trace["token_ids"]]
    good.sort(key=lambda trace: len(trace["token_ids"]))
    if len(good) < 2:
        return {"used": False, "reason": "fewer than 2 correct non-truncated traces", "n_correct": len(good)}
    mid = len(good) // 2
    if len(good) % 2 == 1:
        short = good[:mid]
        long = good[mid + 1 :]
        dropped = 1
    else:
        short = good[:mid]
        long = good[mid:]
        dropped = 0
    short_med = median([len(trace["token_ids"]) for trace in short])
    long_med = median([len(trace["token_ids"]) for trace in long])
    ratio = long_med / short_med if short_med else float("inf")
    strict = ratio >= 1.25
    return {
        "used": True,
        "strict": strict,
        "reason": None if strict else "length ratio under 1.25",
        "n_correct": len(good),
        "dropped_middle": dropped,
        "short_n": len(short),
        "long_n": len(long),
        "short_median_tokens": short_med,
        "long_median_tokens": long_med,
        "ratio": ratio,
        "short": short,
        "long": long,
    }


def late_ids(trace: dict) -> list[int]:
    ids = trace["token_ids"]
    return ids[len(ids) // 2 :]


def early_ids(trace: dict) -> list[int]:
    ids = trace["token_ids"]
    return ids[: len(ids) // 2]


def late_text(trace: dict, tokenizer) -> str:
    ids = late_ids(trace)
    return tokenizer.decode(ids) if ids else ""


def presence_token(traces: list[dict], token_id: int, late_only: bool) -> float:
    if not traces:
        return 0.0
    hits = 0
    for trace in traces:
        span = late_ids(trace) if late_only else trace["token_ids"]
        if token_id in span:
            hits += 1
    return hits / len(traces)


def presence_phrase(traces: list[dict], phrase: str, tokenizer, late_only: bool) -> float:
    if not traces:
        return 0.0
    hits = 0
    for trace in traces:
        text = late_text(trace, tokenizer) if late_only else trace["text"]
        if contains_phrase(" ".join(words_of(text)), phrase):
            hits += 1
    return hits / len(traces)


def mine(traces_by_problem: dict[str, list[dict]], tokenizer) -> dict:
    splits = {pid: split_problem(traces) for pid, traces in traces_by_problem.items()}
    strict_ids = [pid for pid, split in splits.items() if split.get("strict")]
    token_problems = defaultdict(set)
    phrase_problems = defaultdict(set)
    for pid in strict_ids:
        split = splits[pid]
        for trace in split["long"]:
            for token_id in set(late_ids(trace)):
                token_problems[token_id].add(pid)
            grams = words_of(late_text(trace, tokenizer))
            for n in (3, 4):
                for i in range(len(grams) - n + 1):
                    gram = grams[i : i + n]
                    if all(word in STOPWORDS for word in gram):
                        continue
                    phrase_problems[" ".join(gram)].add(pid)

    def token_record(token_id: int) -> dict:
        support = sorted(token_problems[token_id])
        diffs = []
        long_ps = []
        short_ps = []
        early = late = 0
        for pid in support:
            split = splits[pid]
            long_p = presence_token(split["long"], token_id, True)
            short_p = presence_token(split["short"], token_id, False)
            diffs.append(long_p - short_p)
            long_ps.append(long_p)
            short_ps.append(short_p)
            for trace in split["long"]:
                early += sum(1 for item in early_ids(trace) if item == token_id)
                late += sum(1 for item in late_ids(trace) if item == token_id)
        late_bias = late / (early + late) if early + late else 0.0
        surface = tokenizer.decode([token_id])
        return {
            "kind": "token",
            "surface": surface,
            "token_ids": [token_id],
            "support_problems": support,
            "support": len(support),
            "mean_presence_long_late": mean(long_ps),
            "mean_presence_short": mean(short_ps),
            "score": mean(diffs),
            "late_bias": late_bias,
            "strict_pass": (
                len(support) >= 2
                and mean(diffs) >= 0.35
                and late_bias >= 0.60
                and mean(short_ps) <= 0.25
                and keep_unigram(surface)
            ),
        }

    def phrase_record(phrase: str) -> dict:
        support = sorted(phrase_problems[phrase])
        diffs = []
        long_ps = []
        short_ps = []
        early_hits = late_hits = 0
        for pid in support:
            split = splits[pid]
            long_p = presence_phrase(split["long"], phrase, tokenizer, True)
            short_p = presence_phrase(split["short"], phrase, tokenizer, False)
            diffs.append(long_p - short_p)
            long_ps.append(long_p)
            short_ps.append(short_p)
            for trace in split["long"]:
                early_words = " ".join(words_of(tokenizer.decode(early_ids(trace))))
                late_words = " ".join(words_of(late_text(trace, tokenizer)))
                early_hits += int(contains_phrase(early_words, phrase))
                late_hits += int(contains_phrase(late_words, phrase))
        late_bias = late_hits / (early_hits + late_hits) if early_hits + late_hits else 0.0
        pieces = phrase.split()
        return {
            "kind": "phrase",
            "surface": phrase,
            "support_problems": support,
            "support": len(support),
            "mean_presence_long_late": mean(long_ps),
            "mean_presence_short": mean(short_ps),
            "score": mean(diffs),
            "late_bias": late_bias,
            "strict_pass": (
                len(support) >= 2
                and mean(diffs) >= 0.35
                and late_bias >= 0.60
                and mean(short_ps) <= 0.25
                and any(word not in STOPWORDS for word in pieces)
            ),
        }

    tokens = [token_record(token_id) for token_id in token_problems if keep_unigram(tokenizer.decode([token_id]))]
    phrases = [phrase_record(phrase) for phrase in phrase_problems if len(phrase_problems[phrase]) >= 2]
    tokens.sort(key=lambda item: (item["strict_pass"], item["score"], item["support"]), reverse=True)
    phrases.sort(key=lambda item: (item["strict_pass"], item["score"], item["support"]), reverse=True)

    def examples_for_token(token_id: int, limit: int = 2) -> list[str]:
        found = []
        for pid in strict_ids:
            for trace in splits[pid]["long"]:
                ids = trace["token_ids"]
                cut = len(ids) // 2
                if token_id not in ids[cut:]:
                    continue
                index = cut + ids[cut:].index(token_id)
                window = ids[max(0, index - 6) : index + 7]
                found.append(tokenizer.decode(window).replace("\n", " "))
                if len(found) >= limit:
                    return found
        return found

    def examples_for_phrase(phrase: str, limit: int = 2) -> list[str]:
        found = []
        for pid in strict_ids:
            for trace in splits[pid]["long"]:
                text = late_text(trace, tokenizer)
                match = re.search(re.escape(phrase), " ".join(words_of(text)))
                if not match:
                    continue
                raw = re.search(phrase.replace(" ", r"\s+"), text, re.I)
                if raw:
                    start = max(0, raw.start() - 40)
                    found.append(text[start : raw.end() + 40].replace("\n", " "))
                else:
                    found.append(phrase)
                if len(found) >= limit:
                    return found
        return found

    for item in tokens[:40]:
        item["examples"] = examples_for_token(item["token_ids"][0])
        item["encodings"] = encodings(tokenizer, item["surface"].strip())
    for item in phrases[:40]:
        item["examples"] = examples_for_phrase(item["surface"])
        item["encodings"] = encodings(tokenizer, item["surface"])

    reference = reference_table(strict_ids, splits, tokenizer)
    public_splits = {}
    for pid, split in splits.items():
        public_splits[pid] = {key: value for key, value in split.items() if key not in {"short", "long"}}
    return {
        "strict_problems": strict_ids,
        "splits": public_splits,
        "tokens": tokens[:40],
        "phrases": phrases[:40],
        "reference": reference,
    }


def encodings(tokenizer, phrase: str) -> list[dict]:
    variants = []
    seen = set()
    texts = [phrase, " " + phrase]
    if phrase[:1].isalpha():
        texts.append(phrase[:1].upper() + phrase[1:])
        texts.append(" " + phrase[:1].upper() + phrase[1:])
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            {
                "text": text,
                "ids": ids,
                "pieces": [tokenizer.decode([token_id]) for token_id in ids],
                "single_token": len(ids) == 1,
            }
        )
    return variants


def reference_table(strict_ids: list[str], splits: dict, tokenizer) -> list[dict]:
    rows = []
    for name, pattern in REFERENCE:
        regex = re.compile(pattern, re.I)
        diffs = []
        long_ps = []
        short_ps = []
        used = []
        for pid in strict_ids:
            split = splits[pid]
            def hits(traces, late_only: bool) -> float:
                if not traces:
                    return 0.0
                count = 0
                for trace in traces:
                    text = late_text(trace, tokenizer) if late_only else trace["text"]
                    if regex.search(text):
                        count += 1
                return count / len(traces)
            long_p = hits(split["long"], True)
            short_p = hits(split["short"], False)
            if long_p == 0 and short_p == 0:
                continue
            used.append(pid)
            long_ps.append(long_p)
            short_ps.append(short_p)
            diffs.append(long_p - short_p)
        rows.append(
            {
                "phrase": name,
                "support_problems": used,
                "support": len(used),
                "mean_presence_long_late": mean(long_ps) if long_ps else 0.0,
                "mean_presence_short": mean(short_ps) if short_ps else 0.0,
                "score": mean(diffs) if diffs else 0.0,
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def attach_tokens(rows: list[dict], tokenizer) -> dict[str, list[dict]]:
    by_problem = defaultdict(list)
    for row in ok_rows(rows):
        text = row.get("reasoning") or ""
        by_problem[row["problem_id"]].append(
            {
                "problem_id": row["problem_id"],
                "basket": row["basket"],
                "seed": row["seed"],
                "correct": bool(row["strict_correct"]),
                "truncated": bool(row["truncated"]),
                "text": text,
                "token_ids": tokenizer.encode(text, add_special_tokens=False),
                "api_reasoning_tokens": row.get("reasoning_tokens_api"),
            }
        )
    return by_problem


def length_agreement(rows: list[dict], tokenizer) -> dict:
    ratios = []
    for row in ok_rows(rows):
        api = row.get("reasoning_tokens_api")
        if not api:
            continue
        local = len(tokenizer.encode(row.get("reasoning") or "", add_special_tokens=False))
        ratios.append(local / api)
    if not ratios:
        return {"n": 0}
    return {
        "n": len(ratios),
        "median_tokenizer_over_api": median(ratios),
        "min": min(ratios),
        "max": max(ratios),
    }


def fmt(value, digits=1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def ci(block: dict | None, digits=1) -> str:
    if not block:
        return "—"
    return f"{block['point']:.{digits}f} [{block['low']:.{digits}f}, {block['high']:.{digits}f}]"


def write_results(path: Path, metrics: dict, mined: dict, meta: dict) -> None:
    lines = []
    lines.append("# 001 基線與空轉詞")
    lines.append("")
    lines.append("模型：Ornith-1.5-9B。這次只做開發集基線（文件第 1、2 步裡的第 8 條對照，加上第 24 條挖詞）。沒有改權重，沒有加 logit bias。")
    lines.append("")
    lines.append("## 抽樣")
    lines.append("")
    lines.append("鎖死的解碼：`temperature=1.0`、`top_p=0.95`、`top_k=20`、`min_p=0`、`presence_penalty=0`、`repetition_penalty=1`。思考打開。`max_tokens=8192`，頂到上限的回答算答錯，仍留在分母裡。空系統提示，題目後面只加答案格式。seed 是 101、102、103、104。")
    lines.append("")
    lines.append("正確的定義：可見回答裡最後一行 `ANSWER:` 的整數等於標準答案，而且沒有被截斷。思考區裡出現過標準答案、但可見回答沒寫對，不算對。")
    lines.append("")
    lines.append("## 總表")
    lines.append("")
    lines.append("| 籃 | 題數 | 作答 | 答對 | 正確率（題為單位 95% 區間） | 截斷 | 思考 token 平均 | 思考 token 中位數 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for basket in ("easy", "hard"):
        block = metrics[basket]
        lines.append(
            f"| {basket} | {block['n_problems']} | {block['n_samples']} | {block['n_correct']} | "
            f"{ci(block['accuracy_bootstrap_95'], 3)} | {block['n_truncated']} | "
            f"{ci(block['reasoning_tokens_mean_bootstrap_95'], 0)} | "
            f"{ci(block['reasoning_tokens_median_bootstrap_95'], 0)} |"
        )
    lines.append("")
    lines.append("區間是以題為單位重抽 1000 次。題少，區間寬，只拿來描述這份開發集，不拿來宣稱小差距。")
    lines.append("")
    lines.append("API 的 `reasoning_tokens` 和 tokenizer 對思考字串的長度，中位數比值（tokenizer / API）是 "
                 f"{meta['length_agreement'].get('median_tokenizer_over_api')}。挖詞的位置用 tokenizer。")
    lines.append("")
    lines.append("## 每題")
    lines.append("")
    lines.append("| 題 | 籃 | 標準答案 | 抽出的答案 | 答對 | 截斷 | 思考 token | 答對軌跡的中位數 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for basket in ("easy", "hard"):
        for row in metrics[basket]["problems"]:
            extracted = ", ".join(str(item) for item in row["extracted"])
            lengths = ", ".join(str(item) for item in row["reasoning_tokens"])
            lines.append(
                f"| {row['problem_id']} | {basket} | {row['gold']} | {extracted} | "
                f"{row['n_correct']}/{row['n']} | {row['n_truncated']} | {lengths} | "
                f"{fmt(row['correct_reasoning_tokens_median'], 0)} |"
            )
    lines.append("")
    lines.append("## 挖詞")
    lines.append("")
    lines.append("同一題裡，只拿沒被截斷而且答對的軌跡。依思考字串的 tokenizer 長度排序，前半是短組、後半是長組。奇數條時中間那條兩邊都不進。長組中位數至少是短組的 1.25 倍，這題才進入嚴格彙總。")
    lines.append("")
    lines.append("單 token：長組後段出現、短組整段很少出現。片語：長組後段的 3 或 4 個詞。分數是各題（長組後段出現率 − 短組出現率）的平均。嚴格留下要同時滿足：至少 2 題、分數 ≥ 0.35、後段占比 ≥ 0.60、短組出現率 ≤ 0.25。")
    lines.append("")
    entered = ", ".join(mined["strict_problems"]) or "（沒有）"
    lines.append(f"進入嚴格挖詞的題：{entered}。")
    lines.append("")
    lines.append("| 題 | 用了 | 答對條數 | 短組中位數 | 長組中位數 | 比值 | 原因 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for pid, split in sorted(mined["splits"].items()):
        lines.append(
            f"| {pid} | {split.get('strict', False)} | {split.get('n_correct')} | "
            f"{fmt(split.get('short_median_tokens'), 0)} | {fmt(split.get('long_median_tokens'), 0)} | "
            f"{fmt(split.get('ratio'), 2)} | {split.get('reason') or ''} |"
        )
    lines.append("")
    lines.append("### 嚴格留下的片語")
    lines.append("")
    strict_phrases = [item for item in mined["phrases"] if item["strict_pass"]]
    lines.extend(candidate_table(strict_phrases))
    lines.append("")
    lines.append("### 嚴格留下的單 token")
    lines.append("")
    strict_tokens = [item for item in mined["tokens"] if item["strict_pass"]]
    lines.extend(candidate_table(strict_tokens))
    lines.append("")
    lines.append("### 參考詞在嚴格題上的長短差")
    lines.append("")
    lines.append("這份是 NoWait 一類英文詞和常見中文複查詞，用來對照，不是挖出來的清單。")
    lines.append("")
    lines.append("| 片語 | 有出現的題數 | 長組後段 | 短組 | 差 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in mined["reference"]:
        if item["support"] == 0:
            continue
        lines.append(
            f"| {item['phrase']} | {item['support']} | {item['mean_presence_long_late']:.2f} | "
            f"{item['mean_presence_short']:.2f} | {item['score']:.2f} |"
        )
    lines.append("")
    lines.append("## 傳輸錯誤")
    lines.append("")
    lines.append(f"排除在正確率分母之外的傳輸錯誤：{meta['n_errors']}。")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def candidate_table(items: list[dict]) -> list[str]:
    if not items:
        return ["（沒有通過門檻的項目。）", ""]
    lines = [
        "| 表面 | 題 | 長組後段 | 短組 | 差 | 後段占比 | 單 token id |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in items:
        single = []
        for encoding in item.get("encodings") or []:
            if encoding.get("single_token"):
                single.append(f"{encoding['ids'][0]}:{encoding['text']!r}")
        lines.append(
            f"| `{item['surface']}` | {', '.join(item['support_problems'])} | "
            f"{item['mean_presence_long_late']:.2f} | {item['mean_presence_short']:.2f} | "
            f"{item['score']:.2f} | {item['late_bias']:.2f} | {', '.join(single) or '多 token'} |"
        )
    return lines


def run(results_path: Path, out_dir: Path, tokenizer) -> None:
    rows = load_rows(results_path)
    good = ok_rows(rows)
    meta = {
        "n_rows": len(rows),
        "n_ok": len(good),
        "n_errors": len(rows) - len(good),
        "length_agreement": length_agreement(good, tokenizer),
    }
    metrics = {
        "easy": basket_metrics(good, "easy"),
        "hard": basket_metrics(good, "hard"),
        "meta": meta,
    }
    mined = mine(attach_tokens(good, tokenizer), tokenizer)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "mined_tokens.json").write_text(json.dumps(mined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_results(out_dir / "RESULTS.md", metrics, mined, meta)
    print(f"wrote metrics, mined_tokens, RESULTS n_ok={meta['n_ok']} errors={meta['n_errors']}")
    for basket in ("easy", "hard"):
        block = metrics[basket]
        print(
            basket,
            "acc",
            block["n_correct"],
            "/",
            block["n_samples"],
            "trunc",
            block["n_truncated"],
        )


def self_test(tokenizer) -> None:
    def trace(pid: str, text: str, correct: bool = True) -> dict:
        return {
            "problem_id": pid,
            "basket": "hard",
            "seed": abs(hash(text)) % 1000,
            "correct": correct,
            "truncated": False,
            "text": text,
            "token_ids": tokenizer.encode(text, add_special_tokens=False),
            "api_reasoning_tokens": None,
        }

    short = "We add two and two and get four. The arithmetic is finished."
    long = (
        "We add two and two and get four. "
        "We add two and two and get four. "
        "We add two and two and get four. "
        "We add two and two and get four. "
        "Wait, let me double-check the arithmetic before finishing."
    )
    long_b = (
        "We add two and two and get four. "
        "We add two and two and get four. "
        "We add two and two and get four. "
        "The total is four after the addition. "
        "Wait, let me double-check the arithmetic once more."
    )
    traces = {
        "p1": [trace("p1", short, True), trace("p1", short + " Done.", True), trace("p1", long, True)],
        "p2": [trace("p2", short, True), trace("p2", long_b, True)],
        "p3": [trace("p3", short, True)],
    }
    # A word that only the long trace of p1 uses.
    traces["p1"][2]["text"] += " xylophone"
    traces["p1"][2]["token_ids"] = tokenizer.encode(traces["p1"][2]["text"], add_special_tokens=False)
    mined = mine(traces, tokenizer)
    assert "p1" in mined["strict_problems"] and "p2" in mined["strict_problems"]
    assert "p3" not in mined["strict_problems"]
    strict_text = " ".join(item["surface"].lower() for item in mined["phrases"] + mined["tokens"] if item["strict_pass"])
    assert "wait" in strict_text or "double-check" in strict_text or "double" in strict_text, strict_text
    assert "xylophone" not in strict_text
    print("self-test ok", "strict", [item["surface"] for item in mined["phrases"] if item["strict_pass"]][:8])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results.jsonl")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if args.self_test:
        self_test(tokenizer)
        return
    run(args.results, ROOT, tokenizer)


if __name__ == "__main__":
    main()
