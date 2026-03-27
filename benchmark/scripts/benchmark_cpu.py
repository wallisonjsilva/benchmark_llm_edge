from __future__ import annotations

import sys

try:
    from .benchmark_runner import main_for_backend
except ImportError:
    from benchmark_runner import main_for_backend


# Defaults para CPU puro — sem GPU layers, sem flash attention, largura de banda de RAM
CPU_DEFAULTS: dict[str, str] = {
    "N_GPU_LAYERS": "0",
    "FLASH_ATTN": "false",
    "THREADS": "8",
    "HARDWARE_BANDWIDTH_GBS": "45.0",
    "CTX_SIZE": "4096",
    "N_PREDICT": "1024",
    "TEMP": "0.0",
    "TOP_K": "40",
    "TOP_P": "0.95",
    "REPEAT_LAST_N": "64",
    "INFERENCE_TIMEOUT_S": "180",
    "MAX_REPEAT_NGRAM": "8",
    "BENCH_REPETITIONS": "3",
    "BENCH_N_PROMPT": "256",
    "BENCH_N_GEN": "128",
    "PERPLEXITY_WIKITEXT_ROWS": "16",
    "STOP_TOKENS_MODE": "always",
}


def main() -> int:
    return main_for_backend(
        backend_name="cpu",
        default_env_file=".env.cpu",
        backend_defaults=CPU_DEFAULTS,
        argv=sys.argv[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
