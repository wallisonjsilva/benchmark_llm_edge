# Benchmark ELIB + llama.cpp

Implementação baseada em `docs/plano_benchmark_elib_llamacpp.md`.

## Scripts

- `scripts/benchmark_cpu.py`
- `scripts/benchmark_cuda.py`
- `scripts/benchmark_amd.py`

Todos usam o runner comum `scripts/benchmark_runner.py` e carregam `.env` específico.

## Execução

```bash
uv run python -m scripts.benchmark_cpu --dry-run
uv run python -m scripts.benchmark_cuda --dry-run
uv run python -m scripts.benchmark_amd --dry-run
```

Para executar benchmark real (sem `--dry-run`), ajuste antes os caminhos em:

- `.env.cpu`
- `.env.cuda`
- `.env.amd`

Saída padrão: `refs/saida_benchmark.json`.
