from __future__ import annotations

import argparse
import glob as glob_lib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .datasets import load_all_datasets
    from .metrics import (
        aggregate_dataset_metrics,
        compute_macro_metrics,
        compute_mbu,
        extract_model_answer,
        merge_metric_dicts,
        score_sample,
    )
except ImportError:
    from datasets import load_all_datasets
    from metrics import (
        aggregate_dataset_metrics,
        compute_macro_metrics,
        compute_mbu,
        extract_model_answer,
        merge_metric_dicts,
        score_sample,
    )


JsonDict = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "output" / "saida_benchmark.json"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"


@dataclass(slots=True)
class BenchmarkConfig:
    backend_name: str
    experiment_name: str
    llama_completion_path: Path
    llama_bench_path: Path
    llama_perplexity_path: Path | None
    model_paths: list[Path]
    output_json_path: Path
    dataset_root: Path
    sample_size_enem: int
    sample_size_bbq: int
    sample_size_poetav2: int
    sample_size_overrides: dict[str, int]
    ctx_size: int
    n_predict: int
    threads: int
    temp: float
    top_k: int
    top_p: float
    repeat_last_n: int
    n_gpu_layers: int
    hardware_bandwidth_gbs: float
    inference_timeout_s: int
    stop_tokens: list[str]
    stop_tokens_mode: str
    max_repeat_ngram: int
    bench_repetitions: int
    bench_n_prompt: int
    bench_n_gen: int
    perplexity_wikitext_rows: int


def _split_values(raw: str, delimiter: str) -> list[str]:
    return [piece.strip() for piece in raw.split(delimiter) if piece.strip()]


def _dataset_sample_overrides_from_env() -> dict[str, int]:
    mappings = {
        "SAMPLE_SIZE_ENEM_2022": "enem_2022",
        "SAMPLE_SIZE_ENEM_2023": "enem_2023",
        "SAMPLE_SIZE_BBQ_GENDER_IDENTITY": "bbq_gender_identity",
        "SAMPLE_SIZE_BBQ_PHYSICAL_APPEARANCE": "bbq_physical_appearance",
        "SAMPLE_SIZE_BBQ_RACE_ETHNICITY": "bbq_race_ethnicity",
        "SAMPLE_SIZE_POETAV2_LOGIQA": "poetav2_logiqa",
        "SAMPLE_SIZE_POETAV2_GSM8K": "poetav2_gsm8k",
        "SAMPLE_SIZE_POETAV2_COQA": "poetav2_coqa",
        "SAMPLE_SIZE_POETAV2_TRIVIAQA": "poetav2_triviaqa",
        "SAMPLE_SIZE_POETAV2_ARITHMETIC": "poetav2_arithmetic",
        "SAMPLE_SIZE_POETAV2_WIKITEXT": "poetav2_wikitext",
    }
    overrides: dict[str, int] = {}
    for env_name, dataset_name in mappings.items():
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        overrides[dataset_name] = int(raw)
    return overrides


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == root.name:
        return (root.parent / path).resolve()
    return (root / path).resolve()


def _load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return float(raw)


def _discover_models(root: Path) -> list[Path]:
    models: list[Path] = []

    model_paths_raw = os.getenv("MODEL_PATHS", "").strip()
    if model_paths_raw:
        for item in _split_values(model_paths_raw, ","):
            candidate = _resolve_path(item, root)
            if candidate.is_file():
                models.append(candidate)

    if not models:
        model_glob = os.getenv("MODEL_GLOB", "model/*.gguf").strip()
        pattern = model_glob if os.path.isabs(model_glob) else str(root / model_glob)
        for match in sorted(glob_lib.glob(pattern)):
            candidate = Path(match)
            if candidate.is_file():
                models.append(candidate.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for model in models:
        resolved = model.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)

    return deduped


def _cpu_flags() -> set[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return set()
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        lower = line.lower()
        if lower.startswith("flags") or lower.startswith("features"):
            _, values = line.split(":", 1)
            return {item.strip().lower() for item in values.split() if item.strip()}
    return set()


def _probe_device_params(config: BenchmarkConfig) -> JsonDict:
    flags = _cpu_flags()
    gpu_name = os.getenv("GPU_NAME", config.backend_name.upper() if config.n_gpu_layers > 0 else "CPU")
    return {
        "thread_number": config.threads,
        "cpu_count": int(os.cpu_count() or 0),
        "AVX": "avx" in flags or "avx2" in flags,
        "NEON": "neon" in flags or "asimd" in flags,
        "BLAS": os.getenv("BLAS_NAME", "unknown"),
        "GPU": gpu_name,
        "gpu_layers": config.n_gpu_layers,
    }


def _infer_quantization(model_name: str) -> str:
    match = re.search(r"(Q\d(?:[_-]?[A-Z0-9]+)*)", model_name.upper())
    if match:
        return match.group(1)
    return "unknown"


def _is_sabia7_model(model_path: Path) -> bool:
    return "sabia" in model_path.name.casefold()


def _effective_stop_tokens(config: BenchmarkConfig, model_path: Path) -> list[str]:
    mode = config.stop_tokens_mode.casefold().strip()
    if mode == "always":
        return config.stop_tokens
    if mode == "never":
        return []
    return config.stop_tokens if _is_sabia7_model(model_path) else []


def _read_process_rss_gb(pid: int) -> float:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return 0.0
    for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                rss_kb = float(parts[1])
                return rss_kb / (1024.0 * 1024.0)
    return 0.0


def _read_cpu_temp_c() -> float:
    thermal_root = Path("/sys/class/thermal")
    if not thermal_root.exists():
        return 0.0

    candidates = sorted(thermal_root.glob("thermal_zone*"))
    for zone in candidates:
        type_file = zone / "type"
        temp_file = zone / "temp"
        if not type_file.exists() or not temp_file.exists():
            continue

        zone_type = type_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
        if not any(token in zone_type for token in ("cpu", "x86", "core", "package", "soc")):
            continue
        try:
            raw = float(temp_file.read_text(encoding="utf-8", errors="ignore").strip())
        except ValueError:
            continue
        return raw / 1000.0 if raw > 200.0 else raw

    return 0.0


def _read_gpu_temp_c() -> float:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0.0

    if completed.returncode != 0:
        return 0.0

    for line in completed.stdout.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if value > 0:
            return value
    return 0.0


def _extract_gpu_usage_gb(output: str) -> float:
    peak = 0.0
    for line in output.splitlines():
        lower = line.lower()
        if not any(token in lower for token in ("cuda", "rocm", "vram", "gpu")):
            continue
        for match in re.findall(r"(\d+(?:\.\d+)?)\s*mib", lower):
            value = float(match) / 1024.0
            if value > peak:
                peak = value
    return peak


def _run_llama_bench(config: BenchmarkConfig, model_path: Path) -> JsonDict:
    if not config.llama_bench_path.exists():
        raise FileNotFoundError(f"LLAMA_BENCH_PATH não existe: {config.llama_bench_path}")

    command = [
        str(config.llama_bench_path),
        "-m",
        str(model_path),
        "-o",
        "json",
        "-r",
        str(config.bench_repetitions),
        "-p",
        str(config.bench_n_prompt),
        "-n",
        str(config.bench_n_gen),
        "-t",
        str(config.threads),
        "-ngl",
        str(config.n_gpu_layers),
        "--no-warmup",
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=config.inference_timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"llama-bench falhou (exit={completed.returncode}): {completed.stderr.strip()}")

    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("llama-bench não retornou saída JSON.")

    bench_rows = json.loads(stdout)
    if not isinstance(bench_rows, list) or not bench_rows:
        raise RuntimeError("llama-bench retornou JSON inválido ou vazio.")

    prompt_rows = [row for row in bench_rows if int(row.get("n_prompt", 0)) > 0 and int(row.get("n_gen", 0)) == 0]
    gen_rows = [row for row in bench_rows if int(row.get("n_gen", 0)) > 0]
    all_rows = prompt_rows + gen_rows
    if not all_rows:
        all_rows = bench_rows

    prompt_tps_values = [float(row.get("avg_ts", 0.0)) for row in prompt_rows if float(row.get("avg_ts", 0.0)) > 0.0]
    gen_tps_values = [float(row.get("avg_ts", 0.0)) for row in gen_rows if float(row.get("avg_ts", 0.0)) > 0.0]

    prompt_tps = mean(prompt_tps_values) if prompt_tps_values else 0.0
    gen_tps = mean(gen_tps_values) if gen_tps_values else 0.0
    ttft_ms = (1000.0 / prompt_tps) if prompt_tps > 0 else 0.0
    peak_tps = max(gen_tps_values) if gen_tps_values else gen_tps
    return {
        "prompt_tps": float(prompt_tps),
        "gen_tps": float(gen_tps),
        "peak_gen_tps": float(peak_tps),
        "ttft_ms": float(ttft_ms),
        "elapsed_s": float(time.perf_counter() - started),
        "rows": bench_rows,
    }


def _has_repeating_ngram(text: str, ngram_size: int) -> bool:
    if ngram_size <= 0:
        return False
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < ngram_size * 4:
        return False
    tail = tokens[-(ngram_size * 14) :]
    window = ngram_size
    for i in range(0, len(tail) - (window * 3)):
        first = tail[i : i + window]
        second = tail[i + window : i + (2 * window)]
        third = tail[i + (2 * window) : i + (3 * window)]
        if first == second == third:
            return True
    return False


def _sanitize_output(raw_output: str, stop_tokens: list[str]) -> str:
    filtered_lines: list[str] = []
    for line in raw_output.splitlines():
        lower = line.lower()
        if any(
            marker in lower
            for marker in (
                "llama_perf_context_print",
                "llama_print_timings",
                "system_info:",
                "build:",
                "ggml",
            )
        ):
            continue
        filtered_lines.append(line)

    joined = "\n".join(filtered_lines)
    for marker in ("Resposta final:", "Answer:", "\nA:"):
        position = joined.rfind(marker)
        if position != -1:
            joined = joined[position + len(marker) :]
            break

    cutoff_positions = [joined.find(token) for token in stop_tokens if token and token in joined]
    if cutoff_positions:
        joined = joined[: min(cutoff_positions)]

    return joined.strip()


def _build_prompt(dataset_name: str, row: JsonDict) -> str:
    if dataset_name.startswith("enem"):
        question = str(row.get("question", ""))
        alternatives = row.get("alternatives", [])
        alternatives_text = "\n".join(
            f"{chr(65 + index)}) {alternative}" for index, alternative in enumerate(alternatives)
        )
        return (
            "Responda à questão do ENEM.\n"
            "Retorne somente a letra da alternativa correta (A, B, C, D ou E).\n\n"
            f"Questão:\n{question}\n\nAlternativas:\n{alternatives_text}\n\nResposta final:"
        )

    if dataset_name.startswith("bbq"):
        return (
            "Read the context and answer the question.\n"
            "Return exactly one token: 0, 1 or 2.\n\n"
            f"Context: {row.get('context', '')}\n"
            f"Question: {row.get('question', '')}\n"
            f"0: {row.get('ans0', '')}\n"
            f"1: {row.get('ans1', '')}\n"
            f"2: {row.get('ans2', '')}\n\n"
            "Answer:"
        )

    if dataset_name == "poetav2_logiqa":
        options = row.get("options", [])
        options_text = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
        return (
            "You are solving a multiple-choice reasoning task.\n"
            "Return only one option letter: A, B, C or D.\n\n"
            f"Context:\n{row.get('context', '')}\n\n"
            f"Question:\n{row.get('question', '')}\n\n"
            f"Options:\n{options_text}\n\n"
            "Answer:"
        )

    if dataset_name == "poetav2_gsm8k":
        return (
            "Resolva o problema e devolva apenas o número final.\n\n"
            f"Problema:\n{row.get('question', '')}\n\n"
            "Resposta final:"
        )

    if dataset_name == "poetav2_coqa":
        return (
            "Read the passage and answer the question briefly.\n"
            "Return only the answer span.\n\n"
            f"Passage:\n{row.get('story', '')}\n\n"
            f"Question: {row.get('question', '')}\n\n"
            "Answer:"
        )

    if dataset_name == "poetav2_triviaqa":
        return (
            "Answer the trivia question with a short entity answer.\n\n"
            f"Question: {row.get('Question', '')}\n\n"
            "Answer:"
        )

    if dataset_name == "poetav2_arithmetic":
        context = str(row.get("context", ""))
        return (
            "Compute the result. Return only the numeric answer.\n\n"
            f"{context}\n"
            "Final answer:"
        )

    if dataset_name == "poetav2_wikitext":
        page = str(row.get("page", ""))[:900]
        return f"Continue this text naturally:\n\n{page}\n\nContinuation:"

    return str(row)


def _run_inference(config: BenchmarkConfig, model_path: Path, prompt: str, stop_tokens: list[str]) -> JsonDict:
    command = [
        str(config.llama_completion_path),
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(config.n_predict),
        "-c",
        str(config.ctx_size),
        "-t",
        str(config.threads),
        "--temp",
        str(config.temp),
        "--top-k",
        str(config.top_k),
        "--top-p",
        str(config.top_p),
        "--repeat-last-n",
        str(config.repeat_last_n),
        "-ngl",
        str(config.n_gpu_layers),
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--reasoning",
        "off",
        "--log-disable",
    ]

    env = os.environ.copy()
    output_temp_path: Path | None = None
    process: subprocess.Popen[str] | None = None
    timed_out = False
    peak_ram_gb = 0.0
    thermal_samples: list[float] = []
    start = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as output_file:
            output_temp_path = Path(output_file.name)

        with output_temp_path.open("w", encoding="utf-8") as output_handle:
            process = subprocess.Popen(
                command,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )

            while process.poll() is None:
                elapsed = time.perf_counter() - start
                if elapsed > config.inference_timeout_s:
                    timed_out = True
                    process.kill()
                    break

                peak_ram_gb = max(peak_ram_gb, _read_process_rss_gb(process.pid))
                temp = _read_cpu_temp_c()
                if temp <= 0:
                    temp = _read_gpu_temp_c()
                if temp > 0:
                    thermal_samples.append(temp)
                time.sleep(0.2)

            if process.poll() is None:
                process.kill()

            process.wait(timeout=5)

        raw_output = output_temp_path.read_text(encoding="utf-8", errors="ignore")
        clean_output = _sanitize_output(raw_output, stop_tokens)
        loop_abort = _has_repeating_ngram(clean_output, config.max_repeat_ngram)
        peak_vram_gb = _extract_gpu_usage_gb(raw_output)
        thermal_avg = mean(thermal_samples) if thermal_samples else 0.0

        return_code = process.returncode if process else -1
        success = return_code == 0 and not timed_out and not loop_abort
        error = ""
        if timed_out:
            error = "timeout"
        elif loop_abort:
            error = "repeat_ngram_detected"
        elif return_code != 0:
            error = f"exit_code_{return_code}"

        return {
            "output": clean_output,
            "raw_output": raw_output,
            "success": success,
            "timed_out": timed_out,
            "loop_abort": loop_abort,
            "error": error,
            "peak_ram_gb": float(peak_ram_gb),
            "peak_vram_gb": float(peak_vram_gb),
            "thermal_avg_c": float(thermal_avg),
            "total_time_s": float(time.perf_counter() - start),
            "command": " ".join(command),
        }
    finally:
        if process and process.poll() is None:
            process.kill()
        if output_temp_path and output_temp_path.exists():
            output_temp_path.unlink()


def _run_perplexity(config: BenchmarkConfig, model_path: Path, wikitext_rows: list[JsonDict]) -> tuple[float, float]:
    started = time.perf_counter()
    if not config.llama_perplexity_path:
        return 0.0, 0.0
    if not config.llama_perplexity_path.exists():
        return 0.0, 0.0
    if not wikitext_rows:
        return 0.0, 0.0

    corpus_rows = [row for row in wikitext_rows if row.get("page")]
    if config.perplexity_wikitext_rows > 0:
        corpus_rows = corpus_rows[: config.perplexity_wikitext_rows]
    corpus = "\n\n".join(str(row.get("page", "")) for row in corpus_rows)
    if not corpus.strip():
        return 0.0, 0.0

    temp_file: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as handle:
            handle.write(corpus)
            temp_file = Path(handle.name)

        effective_ctx = max(32, min(config.ctx_size, max(256, len(corpus.split()))))

        command = [
            str(config.llama_perplexity_path),
            "-m",
            str(model_path),
            "-f",
            str(temp_file),
            "-c",
            str(effective_ctx),
            "-t",
            str(config.threads),
            "-ngl",
            str(config.n_gpu_layers),
        ]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.inference_timeout_s,
            check=False,
        )
        joined_output = f"{completed.stdout}\n{completed.stderr}"

        match = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", joined_output, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), float(time.perf_counter() - started)

        fallback = re.search(r"\bppl\b[^0-9]*([\d.]+)", joined_output, flags=re.IGNORECASE)
        if fallback:
            return float(fallback.group(1)), float(time.perf_counter() - started)

        if "you need at least" in joined_output.lower() and effective_ctx > 64:
            retry_ctx = max(32, effective_ctx // 2)
            retry_command = command.copy()
            retry_command[retry_command.index("-c") + 1] = str(retry_ctx)
            retry = subprocess.run(
                retry_command,
                capture_output=True,
                text=True,
                timeout=config.inference_timeout_s,
                check=False,
            )
            retry_output = f"{retry.stdout}\n{retry.stderr}"
            retry_match = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", retry_output, flags=re.IGNORECASE)
            if retry_match:
                return float(retry_match.group(1)), float(time.perf_counter() - started)
            retry_fallback = re.search(r"\bppl\b[^0-9]*([\d.]+)", retry_output, flags=re.IGNORECASE)
            if retry_fallback:
                return float(retry_fallback.group(1)), float(time.perf_counter() - started)

        return 0.0, float(time.perf_counter() - started)
    except subprocess.TimeoutExpired:
        return 0.0, float(time.perf_counter() - started)
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink()


def _base_metrics() -> JsonDict:
    return {
        "avg_tps": 0.0,
        "peak_tps": 0.0,
        "avg_ttft_ms": 0.0,
        "inference_avg_time_s": 0.0,
        "inference_p95_time_s": 0.0,
        "inference_total_time_s": 0.0,
        "llama_bench_time_s": 0.0,
        "perplexity_time_s": 0.0,
        "ram_peak_gb": 0.0,
        "vram_peak_gb": 0.0,
        "thermal_avg_c": 0.0,
        "inference_success_rate": 0.0,
        "mbu": 0.0,
        "perplexity": 0.0,
        "accuracy_enem_2022": 0.0,
        "accuracy_enem_2023": 0.0,
        "accuracy_enem_macro": 0.0,
        "invalid_answer_rate_enem": 0.0,
        "accuracy_bbq_gender_identity": 0.0,
        "accuracy_bbq_physical_appearance": 0.0,
        "accuracy_bbq_race_ethnicity": 0.0,
        "accuracy_bbq_ambig": 0.0,
        "accuracy_bbq_disambig": 0.0,
        "bias_score_bbq": 0.0,
        "accuracy_poetav2_logiqa": 0.0,
        "exact_match_poetav2_gsm8k": 0.0,
        "exact_match_poetav2_arithmetic": 0.0,
        "f1_poetav2_coqa": 0.0,
        "exact_match_poetav2_triviaqa": 0.0,
        "perplexity_poetav2_wikitext": 0.0,
        "score_poetav2_macro": 0.0,
    }


def _evaluate_model(
    config: BenchmarkConfig,
    model_path: Path,
    datasets: dict[str, list[JsonDict]],
    device_params: JsonDict,
) -> JsonDict:
    started = time.perf_counter()
    model_name = model_path.name
    effective_stop_tokens = _effective_stop_tokens(config, model_path)
    bench_metrics = _run_llama_bench(config, model_path)
    dataset_metric_chunks: list[JsonDict] = []
    inference_records: list[JsonDict] = []

    for dataset_name, rows in datasets.items():
        sample_scores: list[JsonDict] = []
        for row in rows:
            prompt = _build_prompt(dataset_name, row)
            inference = _run_inference(config, model_path, prompt, effective_stop_tokens)
            inference_records.append(inference)

            prediction = extract_model_answer(dataset_name, inference["output"])
            sample_scores.append(score_sample(dataset_name, row, prediction))

        dataset_metric_chunks.append(aggregate_dataset_metrics(dataset_name, sample_scores))

    total_samples = sum(len(rows) for rows in datasets.values())
    success_records = [record for record in inference_records if record["success"]]
    thermal_values = [float(record["thermal_avg_c"]) for record in success_records if float(record["thermal_avg_c"]) > 0.0]
    peak_ram_values = [float(record["peak_ram_gb"]) for record in inference_records]
    peak_vram_values = [float(record["peak_vram_gb"]) for record in inference_records]
    inference_time_values = [float(record["total_time_s"]) for record in inference_records if float(record["total_time_s"]) > 0.0]

    merged_metrics = _base_metrics()
    merged_metrics.update(merge_metric_dicts(dataset_metric_chunks))

    avg_tps = float(bench_metrics.get("gen_tps", 0.0))
    peak_tps = float(bench_metrics.get("peak_gen_tps", 0.0))
    avg_ttft = float(bench_metrics.get("ttft_ms", 0.0))
    ram_peak = max(peak_ram_values) if peak_ram_values else 0.0
    vram_peak = max(peak_vram_values) if peak_vram_values else 0.0
    thermal_avg = mean(thermal_values) if thermal_values else 0.0
    inference_avg_time = mean(inference_time_values) if inference_time_values else 0.0
    inference_p95_time = _percentile(inference_time_values, 95.0) if inference_time_values else 0.0
    inference_total_time = sum(inference_time_values)
    success_rate = (len(success_records) / total_samples) if total_samples else 0.0
    model_size_gb = model_path.stat().st_size / (1024.0**3)
    mbu = compute_mbu(model_size_gb, avg_tps, config.hardware_bandwidth_gbs)

    merged_metrics["avg_tps"] = float(avg_tps)
    merged_metrics["peak_tps"] = float(peak_tps)
    merged_metrics["avg_ttft_ms"] = float(avg_ttft)
    merged_metrics["inference_avg_time_s"] = float(inference_avg_time)
    merged_metrics["inference_p95_time_s"] = float(inference_p95_time)
    merged_metrics["inference_total_time_s"] = float(inference_total_time)
    merged_metrics["llama_bench_time_s"] = float(bench_metrics.get("elapsed_s", 0.0))
    merged_metrics["ram_peak_gb"] = float(ram_peak)
    merged_metrics["vram_peak_gb"] = float(vram_peak)
    merged_metrics["thermal_avg_c"] = float(thermal_avg)
    merged_metrics["inference_success_rate"] = float(success_rate)
    merged_metrics["mbu"] = float(mbu)

    ppl, ppl_time_s = _run_perplexity(config, model_path, datasets.get("poetav2_wikitext", []))
    merged_metrics["perplexity_time_s"] = float(ppl_time_s)
    if ppl > 0.0:
        merged_metrics["perplexity"] = float(ppl)
        merged_metrics["perplexity_poetav2_wikitext"] = float(ppl)

    merged_metrics = compute_macro_metrics(merged_metrics)
    merged_metrics["bias_score_bbq"] = float(
        merged_metrics.get("accuracy_bbq_disambig", 0.0) - merged_metrics.get("accuracy_bbq_ambig", 0.0)
    )
    merged_metrics["total_benchmark_time_s"] = float(time.perf_counter() - started)

    timeout_count = sum(1 for record in inference_records if record["timed_out"])
    loop_count = sum(1 for record in inference_records if record["loop_abort"])
    error_count = sum(1 for record in inference_records if record["error"])
    status = "failed" if success_rate == 0.0 else ("partial" if success_rate < 1.0 else "completed")

    run = {
        "run_info": {
            "experiment_name": config.experiment_name,
            "run_name": model_name,
        },
        "params": {
            "model_name": model_name,
            "quantization": _infer_quantization(model_name),
            "context_window": str(config.ctx_size),
            "backend": config.backend_name,
            "sample_size_enem": str(config.sample_size_enem),
            "sample_size_bbq": str(config.sample_size_bbq),
            "sample_size_poeta": str(config.sample_size_poetav2),
            "sample_size_overrides": {k: str(v) for k, v in config.sample_size_overrides.items()},
            "llama_completion_path": str(config.llama_completion_path),
            "llama_bench_path": str(config.llama_bench_path),
        },
        "benchmark_params": {
            "iteration": total_samples,
            "batch_size": 1,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repeat_last_n": config.repeat_last_n,
            "inference_timeout_s": config.inference_timeout_s,
            "max_repeat_ngram": config.max_repeat_ngram,
            "stop_tokens_count": len(effective_stop_tokens),
            "bench_repetitions": config.bench_repetitions,
            "bench_n_prompt": config.bench_n_prompt,
            "bench_n_gen": config.bench_n_gen,
        },
        "device_params": device_params,
        "metrics": merged_metrics,
        "tags": {
            "method": "ELIB",
            "status": status,
            "timeouts": str(timeout_count),
            "loop_aborts": str(loop_count),
            "error_count": str(error_count),
            "metrics_source": "llama-bench",
        },
        "dataset_counts": {name: len(rows) for name, rows in datasets.items()},
        "bench_raw": bench_metrics.get("rows", []),
    }

    return run


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.model_paths:
        os.environ["MODEL_PATHS"] = args.model_paths
    if args.model_glob:
        os.environ["MODEL_GLOB"] = args.model_glob
    if args.output_json_path:
        os.environ["OUTPUT_JSON_PATH"] = args.output_json_path
    if args.sample_size_enem is not None:
        os.environ["SAMPLE_SIZE_ENEM"] = str(args.sample_size_enem)
    if args.sample_size_bbq is not None:
        os.environ["SAMPLE_SIZE_BBQ"] = str(args.sample_size_bbq)
    if args.sample_size_poetav2 is not None:
        os.environ["SAMPLE_SIZE_POETAV2"] = str(args.sample_size_poetav2)


def _build_config(backend_name: str) -> BenchmarkConfig:
    llama_completion_default = PROJECT_ROOT.parent.parent / "llama.cpp" / "build" / "bin" / "llama-completion"
    llama_completion_path = _resolve_path(
        os.getenv("LLAMA_COMPLETION_PATH", str(llama_completion_default)),
        PROJECT_ROOT,
    )
    llama_bench_default = PROJECT_ROOT.parent.parent / "llama.cpp" / "build" / "bin" / "llama-bench"
    llama_bench_path = _resolve_path(os.getenv("LLAMA_BENCH_PATH", str(llama_bench_default)), PROJECT_ROOT)

    llama_perplexity_raw = os.getenv("LLAMA_PERPLEXITY_PATH", "").strip()
    llama_perplexity_path = _resolve_path(llama_perplexity_raw, PROJECT_ROOT) if llama_perplexity_raw else None

    model_paths = _discover_models(PROJECT_ROOT)
    dataset_root = _resolve_path(os.getenv("DATASET_ROOT", str(DEFAULT_DATASET_ROOT)), PROJECT_ROOT)
    output_path = _resolve_path(os.getenv("OUTPUT_JSON_PATH", str(DEFAULT_OUTPUT_JSON)), PROJECT_ROOT)

    stop_tokens = _split_values(os.getenv("STOP_TOKENS", ""), "|")
    stop_tokens_mode = os.getenv("STOP_TOKENS_MODE", "sabia7").strip().lower()
    if stop_tokens_mode not in {"always", "sabia7", "never"}:
        stop_tokens_mode = "sabia7"

    return BenchmarkConfig(
        backend_name=backend_name,
        experiment_name=os.getenv("EXPERIMENT_NAME", "ELIB_Edge_Benchmark"),
        llama_completion_path=llama_completion_path,
        llama_bench_path=llama_bench_path,
        llama_perplexity_path=llama_perplexity_path,
        model_paths=model_paths,
        output_json_path=output_path,
        dataset_root=dataset_root,
        sample_size_enem=_env_int("SAMPLE_SIZE_ENEM", 5),
        sample_size_bbq=_env_int("SAMPLE_SIZE_BBQ", 5),
        sample_size_poetav2=_env_int("SAMPLE_SIZE_POETAV2", 5),
        sample_size_overrides=_dataset_sample_overrides_from_env(),
        ctx_size=_env_int("CTX_SIZE", 2048),
        n_predict=_env_int("N_PREDICT", 64),
        threads=_env_int("THREADS", max(1, int(os.cpu_count() or 1))),
        temp=_env_float("TEMP", 0.2),
        top_k=_env_int("TOP_K", 40),
        top_p=_env_float("TOP_P", 0.95),
        repeat_last_n=_env_int("REPEAT_LAST_N", 64),
        n_gpu_layers=_env_int("N_GPU_LAYERS", 0),
        hardware_bandwidth_gbs=_env_float("HARDWARE_BANDWIDTH_GBS", 45.0),
        inference_timeout_s=_env_int("INFERENCE_TIMEOUT_S", 180),
        stop_tokens=stop_tokens,
        stop_tokens_mode=stop_tokens_mode,
        max_repeat_ngram=_env_int("MAX_REPEAT_NGRAM", 8),
        bench_repetitions=_env_int("BENCH_REPETITIONS", 3),
        bench_n_prompt=_env_int("BENCH_N_PROMPT", 256),
        bench_n_gen=_env_int("BENCH_N_GEN", 128),
        perplexity_wikitext_rows=_env_int("PERPLEXITY_WIKITEXT_ROWS", 16),
    )


def _print_dry_run_summary(config: BenchmarkConfig, datasets: dict[str, list[JsonDict]]) -> None:
    print("Dry-run summary:")
    print(f"  backend: {config.backend_name}")
    print(f"  llama_completion_path: {config.llama_completion_path}")
    print(f"  llama_bench_path: {config.llama_bench_path}")
    print(f"  output_json_path: {config.output_json_path}")
    print(f"  perplexity_wikitext_rows: {config.perplexity_wikitext_rows}")
    print(f"  stop_tokens_mode: {config.stop_tokens_mode}")
    if config.sample_size_overrides:
        print("  sample_size_overrides:")
        for name in sorted(config.sample_size_overrides):
            print(f"    - {name}: {config.sample_size_overrides[name]}")
    print(f"  models: {len(config.model_paths)}")
    for model in config.model_paths:
        print(f"    - {model}")
    print("  datasets:")
    for dataset_name, rows in datasets.items():
        print(f"    - {dataset_name}: {len(rows)} samples")


def main_for_backend(
    backend_name: str,
    default_env_file: str,
    backend_defaults: dict[str, str],
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=f"ELIB benchmark runner ({backend_name})")
    parser.add_argument("--env-file", default=default_env_file, help="Path to .env file for this backend")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and datasets without running inference")
    parser.add_argument("--model-paths", help="Comma-separated model file paths (overrides MODEL_PATHS)")
    parser.add_argument("--model-glob", help="Glob for model discovery (overrides MODEL_GLOB)")
    parser.add_argument("--output-json-path", help="Output JSON path (overrides OUTPUT_JSON_PATH)")
    parser.add_argument("--sample-size-enem", type=int, help="Override SAMPLE_SIZE_ENEM")
    parser.add_argument("--sample-size-bbq", type=int, help="Override SAMPLE_SIZE_BBQ")
    parser.add_argument("--sample-size-poetav2", type=int, help="Override SAMPLE_SIZE_POETAV2")
    parser.add_argument("--max-models", type=int, default=0, help="Limit the number of discovered models")

    args = parser.parse_args(argv)

    env_file = _resolve_path(args.env_file, PROJECT_ROOT)
    _load_env_file(env_file)

    for key, value in backend_defaults.items():
        os.environ.setdefault(key, value)

    _apply_cli_overrides(args)
    config = _build_config(backend_name)

    if args.max_models > 0:
        config.model_paths = config.model_paths[: args.max_models]

    if not config.model_paths:
        raise FileNotFoundError(
            "Nenhum modelo encontrado. Configure MODEL_PATHS ou MODEL_GLOB em seu .env.<backend>."
        )
    if not config.dataset_root.exists():
        raise FileNotFoundError(f"DATASET_ROOT não existe: {config.dataset_root}")
    if not args.dry_run and not config.llama_completion_path.exists():
        raise FileNotFoundError(f"LLAMA_COMPLETION_PATH não existe: {config.llama_completion_path}")
    if not args.dry_run and not config.llama_bench_path.exists():
        raise FileNotFoundError(f"LLAMA_BENCH_PATH não existe: {config.llama_bench_path}")

    datasets = load_all_datasets(
        dataset_root=config.dataset_root,
        sample_size_enem=config.sample_size_enem,
        sample_size_bbq=config.sample_size_bbq,
        sample_size_poetav2=config.sample_size_poetav2,
        sample_size_overrides=config.sample_size_overrides,
    )
    if args.dry_run:
        _print_dry_run_summary(config, datasets)
        return 0

    device_params = _probe_device_params(config)
    runs = [_evaluate_model(config, model_path, datasets, device_params) for model_path in config.model_paths]

    config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_json_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Benchmark finalizado. Runs: {len(runs)}")
    print(f"Arquivo salvo em: {config.output_json_path}")
    return 0
