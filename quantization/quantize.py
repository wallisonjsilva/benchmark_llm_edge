#!/usr/bin/env python3
"""
quantize.py — Converte modelos Hugging Face (SafeTensors) para GGUF FP16
e gera variantes quantizadas (Q8_0, Q4_K_M, Q5_0, Q4_1, Q5_1).

Uso:
    python quantization/quantize.py                    # todos os modelos em models/
    python quantization/quantize.py --model llama-8b   # modelo específico
    python quantization/quantize.py --model llama-8b --quant Q4_K_M
    python quantization/quantize.py --convert-only
    python quantization/quantize.py --list

Logs por operação são gravados em logs/convert_{modelo}_{tipo}.log
e também impressos no console.

Variáveis de ambiente (opcionais):
    MODELS_DIR      Diretório com os modelos HF   (padrão: <projeto>/models)
    GGUF_FP16_DIR   Diretório com os GGUF FP16     (padrão: <projeto>/models/gguf_fp16)
    GGUF_DIR        Diretório de saída quantizados  (padrão: <projeto>/models/gguf)
    LOGS_DIR        Diretório de logs              (padrão: <projeto>/logs)
    LLAMA_CPP_DIR   Diretório raiz do llama.cpp   (padrão: <projeto>/llama.cpp)
    CONVERT_SCRIPT  Caminho do convert_hf_to_gguf.py
    LLAMA_QUANTIZE  Caminho do binário llama-quantize (padrão: llama.cpp/build-cpu/bin/llama-quantize)
    USE_DOCKER      Usar Docker em vez de binários locais (1=sim, padrão: 0)
    DOCKER_IMAGE    Nome da imagem Docker (padrão: tcc-engine)
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — configuráveis via variáveis de ambiente
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR    = Path(os.environ.get("MODELS_DIR",    PROJECT_ROOT / "models"))
GGUF_FP16_DIR = Path(os.environ.get("GGUF_FP16_DIR", PROJECT_ROOT / "models" / "gguf_fp16"))
GGUF_DIR      = Path(os.environ.get("GGUF_DIR",      PROJECT_ROOT / "models" / "gguf"))
LOGS_DIR      = Path(os.environ.get("LOGS_DIR",      PROJECT_ROOT / "logs"))

# --- Modo Docker ---
USE_DOCKER   = os.environ.get("USE_DOCKER", "0") != "0"
DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "tcc-engine")

# Paths dentro do container (conforme Dockerfile.engine)
CONTAINER_MODELS_DIR     = Path("/models")
CONTAINER_GGUF_FP16_DIR  = Path("/gguf_fp16")
CONTAINER_GGUF_DIR       = Path("/gguf")
CONTAINER_CONVERT_SCRIPT = Path("/opt/llama.cpp/convert_hf_to_gguf.py")
CONTAINER_LLAMA_QUANTIZE = Path("/usr/local/bin/llama-quantize")

# --- Modo local (USE_DOCKER=0) ---
LLAMA_CPP_DIR  = Path(os.environ.get("LLAMA_CPP_DIR",  PROJECT_ROOT / "llama.cpp"))
CONVERT_SCRIPT = Path(os.environ.get("CONVERT_SCRIPT", LLAMA_CPP_DIR / "convert_hf_to_gguf.py"))
LLAMA_QUANTIZE = Path(os.environ.get("LLAMA_QUANTIZE", LLAMA_CPP_DIR / "build-cpu" / "bin" / "llama-quantize"))

QUANT_TYPES: list[str] = ["Q8_0", "Q4_K_M", "Q5_0", "Q4_1", "Q5_1"]

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

log = get_logger("quantize")


@contextlib.contextmanager
def log_to_file(tag: str):
    """Adiciona um FileHandler ao logger para a duração do bloco."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"convert_{tag}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(handler)
    try:
        yield log_path
    finally:
        log.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    safetensors_dir: Path
    quant_types: list[str] = field(default_factory=lambda: list(QUANT_TYPES))

    # Paths no host
    @property
    def fp16_gguf(self) -> Path:
        return GGUF_FP16_DIR / f"{self.name}-fp16.gguf"

    def quantized_gguf(self, qtype: str) -> Path:
        return GGUF_DIR / f"{self.name}-{qtype}.gguf"

    # Paths dentro do container Docker
    @property
    def container_safetensors_dir(self) -> Path:
        return CONTAINER_MODELS_DIR / self.name

    @property
    def container_fp16_gguf(self) -> Path:
        return CONTAINER_GGUF_FP16_DIR / f"{self.name}-fp16.gguf"

    def container_quantized_gguf(self, qtype: str) -> Path:
        return CONTAINER_GGUF_DIR / f"{self.name}-{qtype}.gguf"

# ---------------------------------------------------------------------------
# Descoberta automática de modelos em MODELS_DIR
# ---------------------------------------------------------------------------
def discover_models() -> list[ModelSpec]:
    """Descobre modelos a partir dos arquivos *-fp16.gguf em GGUF_FP16_DIR."""
    if not GGUF_FP16_DIR.exists():
        log.error("Diretório FP16 não encontrado: %s", GGUF_FP16_DIR)
        return []

    specs = [
        ModelSpec(
            name=fp16.stem.removesuffix("-fp16"),
            safetensors_dir=MODELS_DIR / fp16.stem.removesuffix("-fp16"),
        )
        for fp16 in sorted(GGUF_FP16_DIR.glob("*-fp16.gguf"))
    ]

    if not specs:
        log.warning("Nenhum arquivo *-fp16.gguf encontrado em %s", GGUF_FP16_DIR)
    return specs

# ---------------------------------------------------------------------------
# Flush de caches (Linux) — libera memória antes de quantizar
# ---------------------------------------------------------------------------
def flush_caches() -> None:
    try:
        subprocess.run(["sync"], check=True, timeout=30)
        cache_file = Path("/proc/sys/vm/drop_caches")
        if cache_file.exists():
            try:
                cache_file.write_text("3\n")
            except PermissionError:
                pass  # sem root, ignora silenciosamente
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Docker runner
# ---------------------------------------------------------------------------
def docker_run(inner_cmd: list[str]) -> list[str]:
    """Envolve um comando com `docker run`, montando os diretórios necessários."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{MODELS_DIR}:{CONTAINER_MODELS_DIR}:ro",
        "-v", f"{GGUF_FP16_DIR}:{CONTAINER_GGUF_FP16_DIR}:ro",
        "-v", f"{GGUF_DIR}:{CONTAINER_GGUF_DIR}",
    ]

    # Passthrough de dispositivos ROCm/GPU se disponíveis
    for dev in ("/dev/kfd", "/dev/dri"):
        if Path(dev).exists():
            cmd += ["--device", dev]

    cmd.append(DOCKER_IMAGE)
    cmd.extend(inner_cmd)
    return cmd

# ---------------------------------------------------------------------------
# Conversão HF → GGUF FP16
# ---------------------------------------------------------------------------
def convert_hf_to_gguf_fp16(spec: ModelSpec) -> Path:
    """Converte um modelo HuggingFace SafeTensors para GGUF FP16."""
    output = spec.fp16_gguf
    if output.exists():
        log.info("FP16 GGUF já existe: %s", output)
        return output

    if not spec.safetensors_dir.exists():
        raise FileNotFoundError(
            f"Diretório de pesos não encontrado: {spec.safetensors_dir}"
        )

    GGUF_FP16_DIR.mkdir(parents=True, exist_ok=True)

    with log_to_file(f"{spec.name}_fp16") as log_path:
        log.info("Log: %s", log_path)
        if USE_DOCKER:
            inner = [
                "python3", str(CONTAINER_CONVERT_SCRIPT),
                str(spec.container_safetensors_dir),
                "--outtype", "f16",
                "--outfile", str(spec.container_fp16_gguf),
            ]
            cmd = docker_run(inner)
        else:
            if not CONVERT_SCRIPT.exists():
                raise FileNotFoundError(
                    f"Script de conversão não encontrado: {CONVERT_SCRIPT}\n"
                    f"Compile o llama.cpp ou defina LLAMA_CPP_DIR."
                )
            cmd = [
                sys.executable, str(CONVERT_SCRIPT),
                str(spec.safetensors_dir),
                "--outtype", "f16",
                "--outfile", str(output),
            ]

        log.info("Convertendo %s → FP16 GGUF ... [%s]", spec.name,
                 "docker" if USE_DOCKER else "local")
        log.debug("CMD: %s", " ".join(cmd))

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if proc.returncode != 0:
            log.error("Conversão falhou:\nSTDOUT: %s\nSTDERR: %s",
                      proc.stdout[-1000:], proc.stderr[-1000:])
            raise RuntimeError(f"convert_hf_to_gguf falhou para {spec.name}")

        log.info("✓ FP16 GGUF criado: %s (%.2f GB)",
                 output, output.stat().st_size / (1024**3))
    return output


# ---------------------------------------------------------------------------
# Quantização GGUF FP16 → Q8_0 / Q4_K_M / Q5_0 / Q4_1 / Q5_1
# ---------------------------------------------------------------------------
def quantize_gguf(spec: ModelSpec, qtype: str) -> Path:
    """Gera variante quantizada a partir do FP16 GGUF."""
    fp16_path = spec.fp16_gguf
    output = spec.quantized_gguf(qtype)

    if output.exists():
        log.info("Quantizado %s já existe: %s", qtype, output)
        return output

    if not fp16_path.exists():
        raise FileNotFoundError(
            f"FP16 GGUF não encontrado: {fp16_path}. Execute a conversão primeiro."
        )

    flush_caches()
    output.parent.mkdir(parents=True, exist_ok=True)

    with log_to_file(f"{spec.name}_{qtype}") as log_path:
        log.info("Log: %s", log_path)
        if USE_DOCKER:
            inner = [
                str(CONTAINER_LLAMA_QUANTIZE),
                str(spec.container_fp16_gguf),
                str(spec.container_quantized_gguf(qtype)),
                qtype,
            ]
            cmd = docker_run(inner)
        else:
            if not LLAMA_QUANTIZE.exists():
                raise FileNotFoundError(
                    f"llama-quantize não encontrado: {LLAMA_QUANTIZE}\n"
                    f"Compile o llama.cpp ou defina LLAMA_CPP_DIR."
                )
            cmd = [
                str(LLAMA_QUANTIZE),
                str(fp16_path),
                str(output),
                qtype,
            ]

        log.info("Quantizando %s → %s ... [%s]", spec.name, qtype,
                 "docker" if USE_DOCKER else "local")
        log.debug("CMD: %s", " ".join(cmd))

        run_env = None
        if not USE_DOCKER:
            run_env = os.environ.copy()
            lib_dir = str(LLAMA_QUANTIZE.parent)
            existing = run_env.get("LD_LIBRARY_PATH", "")
            run_env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                              env=run_env)

        if proc.returncode != 0:
            log.error("Quantização falhou:\nSTDOUT: %s\nSTDERR: %s",
                      proc.stdout[-1000:], proc.stderr[-1000:])
            raise RuntimeError(f"llama-quantize falhou para {spec.name} / {qtype}")

        log.info("✓ Quantizado: %s (%.2f GB)",
                 output, output.stat().st_size / (1024**3))
    return output


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def process_model(spec: ModelSpec, quant_types: list[str] | None = None) -> None:
    """Executa pipeline completo: HF → FP16 → quantizações."""
    qtypes = quant_types or spec.quant_types
    log.info("=" * 60)
    log.info("Processando modelo: %s", spec.name)
    log.info("Quantizações: %s", qtypes)

    # Etapa 1: conversão FP16
    try:
        convert_hf_to_gguf_fp16(spec)
    except Exception as exc:
        log.error("Falha na conversão FP16 de %s: %s", spec.name, exc)
        return

    # Etapa 2: quantizações
    for qtype in qtypes:
        try:
            quantize_gguf(spec, qtype)
        except Exception as exc:
            log.error("Falha na quantização %s de %s: %s", qtype, spec.name, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte e quantiza modelos HuggingFace para GGUF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  %(prog)s                        # todos os modelos em models/\n"
            "  %(prog)s --model llama-8b       # modelo específico\n"
            "  %(prog)s --quant Q8_0 Q4_K_M   # escolher quantizações\n"
            "  %(prog)s --convert-only         # apenas FP16, sem quantizar\n"
        ),
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Nome da pasta do modelo em models/.")
    parser.add_argument("--quant", type=str, nargs="+", default=None,
                        help=f"Tipos de quantização (padrão: {QUANT_TYPES}).")
    parser.add_argument("--convert-only", action="store_true",
                        help="Apenas converte para FP16 GGUF, sem quantizar.")
    parser.add_argument("--list", action="store_true",
                        help="Lista os modelos disponíveis em models/.")
    args = parser.parse_args()

    models = discover_models()

    if args.list:
        if models:
            print("Modelos disponíveis (FP16 em models/gguf_fp16):")
            for m in models:
                print(f"  {m.name}  ({m.fp16_gguf})")
        else:
            print("Nenhum modelo encontrado.")
        return

    if args.model:
        models = [m for m in models if m.name == args.model]
        if not models:
            log.error("Modelo '%s' não encontrado em %s.", args.model, MODELS_DIR)
            sys.exit(1)

    if not models:
        log.error("Nenhum modelo para processar. Coloque arquivos *-fp16.gguf em %s.", GGUF_FP16_DIR)
        sys.exit(1)

    for spec in models:
        if args.convert_only:
            log.info("=" * 60)
            log.info("Convertendo (FP16 only): %s", spec.name)
            try:
                convert_hf_to_gguf_fp16(spec)
            except Exception as exc:
                log.error("Falha na conversão de %s: %s", spec.name, exc)
        else:
            process_model(spec, args.quant)

    log.info("Pipeline concluído.")


if __name__ == "__main__":
    main()