from __future__ import annotations

import sys

try:
    from .benchmark_runner import main_for_backend
except ImportError:
    from benchmark_runner import main_for_backend


# Defaults para ARM (Snapdragon 8 Gen 3 — Samsung S24) via Termux
#
# HARDWARE_BANDWIDTH_GBS: LPDDR5X do Snapdragon 8 Gen 3 = ~77 GB/s
#   (vs 45 GB/s DDR4 desktop no benchmark_cpu.py)
#
# THREADS: 4 cores de performance (P-cores) é o ponto ótimo para inferência
#   no Snapdragon 8 Gen 3. Usar todos os 8 acelera o throttling térmico.
#   Ajuste via THREADS= no .env.arm conforme seu modelo e temperatura ambiente.
#
# INFERENCE_TIMEOUT_S: 300s (vs 180s no CPU) — ARM é mais lento para modelos
#   maiores; evita abortos prematuros.
#
# N_PREDICT: 256 é conservador para não estourar memória RAM do aparelho.
#
# CTX_SIZE: 2048 reduz pressão de memória. O S24 tem 12 GB RAM mas o Android
#   reserva boa parte para o sistema. Ajuste conforme o modelo.
ARM_DEFAULTS: dict[str, str] = {
    "N_GPU_LAYERS": "0",
    "FLASH_ATTN": "false",
    "THREADS": "4",
    "HARDWARE_BANDWIDTH_GBS": "77.0",
    "CTX_SIZE": "2048",
    "N_PREDICT": "256",
    "TEMP": "0.0",
    "TOP_K": "40",
    "TOP_P": "0.95",
    "REPEAT_LAST_N": "64",
    "INFERENCE_TIMEOUT_S": "300",
    "MAX_REPEAT_NGRAM": "8",
    "BENCH_REPETITIONS": "3",
    "BENCH_N_PROMPT": "256",
    "BENCH_N_GEN": "128",
    "PERPLEXITY_WIKITEXT_ROWS": "16",
    "STOP_TOKENS_MODE": "always",
    # GPU_NAME aparece no device_params do JSON como identificador do hardware
    "GPU_NAME": "Snapdragon_8_Gen3",
}


def main() -> int:
    return main_for_backend(
        backend_name="arm",
        default_env_file=".env.arm",
        backend_defaults=ARM_DEFAULTS,
        argv=sys.argv[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
