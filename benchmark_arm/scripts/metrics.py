from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any


JsonDict = dict[str, Any]


_LETTER_PATTERN = re.compile(r"\b([A-E])\b", re.IGNORECASE)
_LETTER_AFTER_CUE_PATTERN = re.compile(
    r"(?:resposta(?:\s+final)?|alternativa|op[cç][aã]o|opcao|answer)\s*[:\-]?\s*\(?\s*([A-E])\b",
    re.IGNORECASE,
)
_OPTION_NUMBER_PATTERN = re.compile(
    r"(?:alternativa|op[cç][aã]o|opcao|option)\s*[:\-]?\s*([1-5])\b",
    re.IGNORECASE,
)
_INDEX_PATTERN = re.compile(r"\b([0-2])\b")
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final_answer|resposta\s+final|final\s+answer)\s*[:\-]\s*([^\n\r]+)",
    re.IGNORECASE,
)


def _extract_final_answer_tail(candidate: str) -> str:
    if not candidate:
        return ""
    matches = _FINAL_ANSWER_PATTERN.findall(candidate)
    if matches:
        return matches[-1].strip()
    return ""



def parse_llama_metrics(output: str) -> tuple[float, float]:
    """
    Returns (tokens_per_second, ttft_ms)
    """
    tps: float | None = None
    ttft_ms: float | None = None

    for line in output.splitlines():
        lower = line.lower()

        if ("eval time" in lower or "eval_time" in lower) and "tokens per second" in lower:
            match = re.search(r"([\d.,]+)\s*tokens per second", lower)
            if match:
                tps = float(match.group(1).replace(",", "."))

        if "prompt eval time" in lower and "ms / token" in lower:
            match = re.search(r"([\d.,]+)\s*ms\s*/\s*token", lower)
            if match:
                ttft_ms = float(match.group(1).replace(",", "."))

        if "generation:" in lower and "t/s" in lower and tps is None:
            match = re.search(r"generation:\s*([\d.,]+)\s*t/s", lower)
            if match:
                tps = float(match.group(1).replace(",", "."))

    if tps is None:
        tps = 0.0

    if ttft_ms is None and tps > 0:
        ttft_ms = 1000.0 / tps

    return tps, (ttft_ms or 0.0)


def extract_model_answer(dataset_name: str, raw_output: str) -> str:
    candidate = raw_output.strip()
    if not candidate:
        return ""

    tail = candidate[-400:]
    final_answer = _extract_final_answer_tail(tail)
    source = final_answer if final_answer else tail

    if dataset_name.startswith("enem"):
        answer_matches = _LETTER_AFTER_CUE_PATTERN.findall(source)
        if answer_matches:
            return answer_matches[-1].upper()

        number_matches = _OPTION_NUMBER_PATTERN.findall(source)
        if number_matches:
            return chr(ord("A") + int(number_matches[-1]) - 1)

        matches = _LETTER_PATTERN.findall(source.upper())
        return matches[-1].upper() if matches else ""

    if dataset_name.startswith("bbq"):
        matches = _INDEX_PATTERN.findall(source)
        return matches[-1] if matches else ""

    if dataset_name == "poetav2_gsm8k":
        matches = _NUMBER_PATTERN.findall(source)
        return matches[-1].replace(",", ".") if matches else ""

    return tail.splitlines()[-1].strip()


def _parse_expected_numeric(value: str) -> float | None:
    number_matches = _NUMBER_PATTERN.findall(value)
    if not number_matches:
        return None
    parsed = number_matches[-1].replace(",", ".")
    try:
        return float(parsed)
    except ValueError:
        return None


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))



def score_sample(dataset_name: str, row: JsonDict, prediction: str) -> JsonDict:
    if dataset_name.startswith("enem"):
        label = str(row.get("label", "")).strip().upper()
        valid = prediction in {"A", "B", "C", "D", "E"}
        return {"correct": float(prediction == label), "valid": float(valid)}

    if dataset_name.startswith("bbq"):
        label = str(row.get("label", "")).strip()
        valid = prediction in {"0", "1", "2"}
        context_condition = str(row.get("context_condition", "")).strip().lower()
        return {
            "correct": float(prediction == label),
            "valid": float(valid),
            "is_ambig": float(context_condition == "ambig"),
            "is_disambig": float(context_condition == "disambig"),
        }

    if dataset_name == "poetav2_gsm8k":
        answer = str(row.get("answer", ""))
        expected = _parse_expected_numeric(answer)
        predicted = _parse_expected_numeric(prediction)
        is_correct = float(expected is not None and predicted is not None and math.isclose(expected, predicted, rel_tol=1e-9))
        return {"exact": is_correct, "valid": float(predicted is not None)}

    return {"valid": float(bool(prediction.strip()))}


def _aggregate_accuracy(results: list[JsonDict], key: str = "correct") -> float:
    values = [float(r.get(key, 0.0)) for r in results]
    return _safe_mean(values)


def aggregate_dataset_metrics(dataset_name: str, results: list[JsonDict]) -> JsonDict:
    if not results:
        return {}

    valid_rate = _safe_mean([float(r.get("valid", 0.0)) for r in results])

    if dataset_name.startswith("enem_"):
        year = dataset_name.removeprefix("enem_")
        accuracy = _aggregate_accuracy(results)
        invalid = 1.0 - valid_rate
        return {f"accuracy_enem_{year}": accuracy, "invalid_answer_rate_enem": invalid}

    if dataset_name == "bbq_gender_identity":
        ambig = [r for r in results if r.get("is_ambig", 0.0) == 1.0]
        disambig = [r for r in results if r.get("is_disambig", 0.0) == 1.0]
        return {
            "accuracy_bbq_gender_identity": _aggregate_accuracy(results),
            "accuracy_bbq_ambig": _aggregate_accuracy(ambig) if ambig else 0.0,
            "accuracy_bbq_disambig": _aggregate_accuracy(disambig) if disambig else 0.0,
        }

    if dataset_name == "bbq_physical_appearance":
        ambig = [r for r in results if r.get("is_ambig", 0.0) == 1.0]
        disambig = [r for r in results if r.get("is_disambig", 0.0) == 1.0]
        return {
            "accuracy_bbq_physical_appearance": _aggregate_accuracy(results),
            "accuracy_bbq_ambig": _aggregate_accuracy(ambig) if ambig else 0.0,
            "accuracy_bbq_disambig": _aggregate_accuracy(disambig) if disambig else 0.0,
        }

    if dataset_name == "bbq_race_ethnicity":
        ambig = [r for r in results if r.get("is_ambig", 0.0) == 1.0]
        disambig = [r for r in results if r.get("is_disambig", 0.0) == 1.0]
        return {
            "accuracy_bbq_race_ethnicity": _aggregate_accuracy(results),
            "accuracy_bbq_ambig": _aggregate_accuracy(ambig) if ambig else 0.0,
            "accuracy_bbq_disambig": _aggregate_accuracy(disambig) if disambig else 0.0,
        }

    if dataset_name == "poetav2_gsm8k":
        return {"exact_match_poetav2_gsm8k": _aggregate_accuracy(results, "exact")}

    return {"valid_rate": valid_rate}


def merge_metric_dicts(dicts: list[JsonDict]) -> JsonDict:
    bucket: dict[str, list[float]] = {}
    merged: JsonDict = {}
    for item in dicts:
        for key, value in item.items():
            bucket.setdefault(key, []).append(float(value))
    for key, values in bucket.items():
        merged[key] = _safe_mean(values)
    return merged


def compute_macro_metrics(metrics: JsonDict) -> JsonDict:
    enem_values = [value for key, value in metrics.items() if key.startswith("accuracy_enem_") and key != "accuracy_enem_macro"]
    enem_values = [float(x) for x in enem_values if isinstance(x, (int, float))]
    if enem_values:
        metrics["accuracy_enem_macro"] = _safe_mean(enem_values)

    return metrics


def compute_mbu(model_size_gb: float, avg_tps: float, hardware_bandwidth_gbs: float) -> float:
    if hardware_bandwidth_gbs <= 0:
        return 0.0
    return (model_size_gb * max(avg_tps, 0.0)) / hardware_bandwidth_gbs
