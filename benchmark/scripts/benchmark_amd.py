from __future__ import annotations

import sys

try:
    from .benchmark_runner import main_for_backend
except ImportError:
    from benchmark_runner import main_for_backend


def main() -> int:
    backend_defaults = {
        "N_GPU_LAYERS": "24",
        "THREADS": "8",
        "HARDWARE_BANDWIDTH_GBS": "256.0",
        "CTX_SIZE": "4096",
        "N_PREDICT": "512",
        "TEMP": "0.0",
        "TOP_K": "40",
        "TOP_P": "0.95",
        "REPEAT_LAST_N": "64",
        "INFERENCE_TIMEOUT_S": "180",
        "MAX_REPEAT_NGRAM": "8",
        "GPU_NAME": "ROCm",
    }
    return main_for_backend(
        backend_name="amd",
        default_env_file=".env.amd",
        backend_defaults=backend_defaults,
        argv=sys.argv[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
