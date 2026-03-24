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
- **Tempo de execução**: disponível em `total_benchmark_time_s`, `inference_avg_time_s`,
  `inference_p95_time_s`, `inference_total_time_s`, `llama_bench_time_s` e `perplexity_time_s`.

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

Saída padrão: `output/saida_benchmark_poetav2.json`.

## Variáveis importantes no `.env`

- `LLAMA_COMPLETION_PATH`: binário de inferência para gerar respostas do dataset.
- `LLAMA_BENCH_PATH`: binário usado para benchmark de desempenho (TPS/TTFT).
- `LLAMA_PERPLEXITY_PATH`: opcional para perplexidade em wikitext.
- `BENCH_REPETITIONS`, `BENCH_N_PROMPT`, `BENCH_N_GEN`: parâmetros do `llama-bench`.
- `PERPLEXITY_WIKITEXT_ROWS`: quantidade de páginas usadas para calcular perplexidade (aumente para estabilizar PPL).
- `STOP_TOKENS_MODE`: controla quando aplicar `STOP_TOKENS`:
  - `sabia7` (padrão): aplica só para modelos Sabiá.
  - `always`: aplica para todos os modelos.
  - `never`: não aplica.
- `SAMPLE_SIZE_ENEM`, `SAMPLE_SIZE_BBQ`, `SAMPLE_SIZE_POETAV2`: cortes globais por família.
- Overrides por dataset (sobrescrevem globais quando definidos):
  - `SAMPLE_SIZE_ENEM_2022`, `SAMPLE_SIZE_ENEM_2023`, `SAMPLE_SIZE_ENEM_2024`
  - `SAMPLE_SIZE_BBQ_GENDER_IDENTITY`, `SAMPLE_SIZE_BBQ_PHYSICAL_APPEARANCE`, `SAMPLE_SIZE_BBQ_RACE_ETHNICITY`
  - `SAMPLE_SIZE_POETAV2_LOGIQA`, `SAMPLE_SIZE_POETAV2_GSM8K`, `SAMPLE_SIZE_POETAV2_COQA`
  - `SAMPLE_SIZE_POETAV2_TRIVIAQA`, `SAMPLE_SIZE_POETAV2_ARITHMETIC`, `SAMPLE_SIZE_POETAV2_WIKITEXT`

Guia recomendado para comparação FP16 vs quantizados (Q8_0, Q5_1, Q5_0, Q4_K_M, Q4_1):

- `docs/recomendacoes_quantizacao_edge.md`
