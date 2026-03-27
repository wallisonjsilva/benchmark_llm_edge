from __future__ import annotations

import sys

try:
    from .benchmark_runner import main_for_backend
except ImportError:
    from benchmark_runner import main_for_backend


# Defaults para CUDA — GPU layers máximos, flash attention habilitado, largura de banda de GPU
CUDA_DEFAULTS: dict[str, str] = {
    "N_GPU_LAYERS": "99",
    "FLASH_ATTN": "true",
    "THREADS": "8",
    "HARDWARE_BANDWIDTH_GBS": "300.0",
    "CTX_SIZE": "4096",
    "N_PREDICT": "512",
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
    # GPU_NAME é detectado automaticamente pelo _probe_device_params via backend_name="cuda"
    # mas pode ser sobrescrito no .env.cuda com: GPU_NAME=NVIDIA L4
}


def main() -> int:
    return main_for_backend(
        backend_name="cuda",
        default_env_file=".env.cuda",
        backend_defaults=CUDA_DEFAULTS,
        argv=sys.argv[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
