from __future__ import annotations

import sys

try:
    from .benchmark_runner import main_for_backend
except ImportError:
    from benchmark_runner import main_for_backend


def main() -> int:
    backend_defaults = {
        "N_GPU_LAYERS": "0",
        "THREADS": "8",
        "HARDWARE_BANDWIDTH_GBS": "45.0",
        "CTX_SIZE": "2048",
        "N_PREDICT": "64",
        "TEMP": "0.2",
        "TOP_K": "40",
        "TOP_P": "0.95",
        "REPEAT_LAST_N": "64",
        "INFERENCE_TIMEOUT_S": "180",
        "MAX_REPEAT_NGRAM": "8",
    }
    return main_for_backend(
        backend_name="cpu",
        default_env_file=".env.cpu",
        backend_defaults=backend_defaults,
        argv=sys.argv[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
