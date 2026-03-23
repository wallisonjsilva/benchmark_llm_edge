# Benchmark ELIB + llama.cpp

Implementação baseada em `docs/plano_benchmark_elib_llamacpp.md`.

## Scripts

- `scripts/benchmark_cpu.py`
- `scripts/benchmark_cuda.py`
- `scripts/benchmark_amd.py`

Todos usam o runner comum `scripts/benchmark_runner.py` e carregam `.env` específico.

## Métricas x Qualidade

- **Métricas de desempenho (TPS/TTFT)**: coletadas via `llama-bench` (saída JSON).
- **Qualidade por dataset (acurácia/F1/EM)**: calculada no runner a partir das respostas inferidas.
- A saída interativa do `llama-cli` não é usada para métricas.

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

## Variáveis importantes no `.env`

- `LLAMA_COMPLETION_PATH`: binário de inferência para gerar respostas do dataset.
- `LLAMA_BENCH_PATH`: binário usado para benchmark de desempenho (TPS/TTFT).
- `LLAMA_PERPLEXITY_PATH`: opcional para perplexidade em wikitext.
- `BENCH_REPETITIONS`, `BENCH_N_PROMPT`, `BENCH_N_GEN`: parâmetros do `llama-bench`.
