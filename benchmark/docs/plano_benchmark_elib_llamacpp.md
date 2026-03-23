# Plano de implementação — Benchmark ELIB com `llama.cpp` (GGUF)

## 1) Objetivo

Construir um pipeline de benchmark para modelos `.gguf` executados via `llama.cpp`, com:

- métricas ELIB (desempenho + eficiência de hardware);
- métricas de avaliação dos datasets `ENEM`, `BBQ` e datasets do `PoETaV2` presentes em `benchmark/datasets/poetav2`;
- saída em JSON padronizada para ingestão no MLflow;
- 3 pontos de entrada separados para execução em `CUDA`, `AMD/ROCm` e `CPU`;
- configuração 100% por variáveis de ambiente (`.env` por backend);
- execução agnóstica ao tipo de modelo (reasoning/instruct/base).

## 2) Referências analisadas e decisões

- `benchmark/sample/benchmark_elib.py`: base de cálculo de TPS, latência, RAM/VRAM, térmica, sucesso de inferência e MBU.
- `benchmark/refs/saida_benchmark.json` e `benchmark/refs/saida.json`: estrutura alvo de saída (`run_info`, `params`, `benchmark_params`, `device_params`, `metrics`, `tags`).
- `benchmark/refs/elib.md`: orientação para mapeamento de métricas ELIB.
- `benchmark/refs/poetav2_benchmark_funcionamento_metricas.md`: famílias de métricas PoETaV2 (acc/f1/exact/mse/pearson etc.).
- Schemas reais dos datasets em `benchmark/datasets/*`:
  - ENEM: `question`, `alternatives`, `label`.
  - BBQ: `context`, `question`, `ans0..ans2`, `label`, `context_condition`, `question_polarity`.
  - PoETaV2 local: `gsm8k`, `coqa`, `triviaqa`, `logiqa`, `arithmetic`, `wikitext`.

## 3) Arquitetura proposta

### 3.1 Scripts de execução (3 backends)

- `benchmark/scripts/benchmark_cuda.py`
- `benchmark/scripts/benchmark_amd.py`
- `benchmark/scripts/benchmark_cpu.py`

Cada script:

1. carrega seu arquivo `.env` específico;
2. define parâmetros default do backend (threads, gpu-layers, binários, flags);
3. chama um runner comum para evitar duplicação.

### 3.2 Núcleo comum (reuso)

- `benchmark/scripts/benchmark_runner.py` (núcleo agnóstico de backend):
  - descoberta de modelos (`MODEL_PATHS` e/ou `MODEL_GLOB`);
  - loop modelo × dataset × amostras;
  - geração de prompt;
  - inferência `llama.cpp`;
  - extração/cálculo de métricas;
  - escrita do JSON final.

- `benchmark/scripts/metrics.py`:
  - métricas ELIB e métricas de qualidade por dataset.

- `benchmark/scripts/datasets.py`:
  - leitura JSONL e normalização de exemplos por dataset.

## 4) Configuração por `.env` (um por backend)

Arquivos:

- `benchmark/.env.cuda`
- `benchmark/.env.amd`
- `benchmark/.env.cpu`

Variáveis mínimas:

- `LLAMA_CLI_PATH`
- `LLAMA_PERPLEXITY_PATH` (opcional)
- `MODEL_PATHS` (lista separada por vírgula) **ou** `MODEL_GLOB`
- `OUTPUT_JSON_PATH=benchmark/refs/saida_benchmark.json`
- `DATASET_ROOT=benchmark/datasets`
- `SAMPLE_SIZE_ENEM`, `SAMPLE_SIZE_BBQ`, `SAMPLE_SIZE_POETAV2`
- `CTX_SIZE`, `N_PREDICT`, `THREADS`, `TEMP`, `TOP_K`, `TOP_P`, `REPEAT_LAST_N`
- `N_GPU_LAYERS` (CUDA/AMD > 0, CPU = 0)
- `HARDWARE_BANDWIDTH_GBS` (para MBU)
- `INFERENCE_TIMEOUT_S`
- `STOP_TOKENS` (lista separada por `|`)
- `MAX_REPEAT_NGRAM` (detecção de loop)

## 5) Estratégia de inferência agnóstica a modelo (reasoning/instruct/base)

### 5.1 Prompting

- Prompt por dataset com instruções de saída curta/estruturada (ex.: letra no ENEM, `0/1/2` no BBQ).
- Extração da resposta final via parser robusto (regex + normalização), sem depender de token especial do modelo.

### 5.2 Contenção de loop/saída infinita

- limite rígido de tokens (`N_PREDICT`);
- timeout por inferência (`INFERENCE_TIMEOUT_S`);
- detecção de repetição de n-grama (`MAX_REPEAT_NGRAM`) para abortar geração;
- `STOP_TOKENS` configuráveis por `.env` (quando aplicável ao binário/versão);
- marcação explícita de amostra inválida em vez de “silenciar” erro.

## 6) Métricas a calcular

### 6.1 ELIB (core)

- `avg_tps`, `peak_tps`
- `avg_ttft_ms` (ou `avg_latency_ms_per_token`, conforme log disponível)
- `ram_peak_gb`, `vram_peak_gb`
- `thermal_avg_c`
- `inference_success_rate`
- `mbu`
- `perplexity` (quando `llama-perplexity` e dataset de PPL estiverem configurados)

### 6.2 ENEM

- `accuracy_enem_2022`
- `accuracy_enem_2023`
- `accuracy_enem_macro`
- `invalid_answer_rate_enem`

### 6.3 BBQ

- `accuracy_bbq_gender_identity`
- `accuracy_bbq_physical_appearance`
- `accuracy_bbq_race_ethnicity`
- `accuracy_bbq_ambig`
- `accuracy_bbq_disambig`
- `bias_score_bbq` (quando viável pelo par estereotipado/antiestereotipado)

### 6.4 PoETaV2 (datasets locais)

- `accuracy_poetav2_logiqa`
- `exact_match_poetav2_gsm8k`
- `exact_match_poetav2_arithmetic`
- `f1_poetav2_coqa`
- `exact_match_poetav2_triviaqa` (com aliases)
- `perplexity_poetav2_wikitext`
- `score_poetav2_macro`

## 7) Padrão de saída JSON para MLflow

A saída seguirá o padrão de `benchmark/refs/saida_benchmark.json`: **lista de runs**, um objeto por modelo/backend.

Estrutura alvo por run:

```json
{
  "run_info": {
    "experiment_name": "ELIB_Edge_Benchmark",
    "run_name": "nome_modelo.gguf"
  },
  "params": {},
  "benchmark_params": {},
  "device_params": {},
  "metrics": {},
  "tags": {}
}
```

Padronização para MLflow:

- `metrics`: somente valores numéricos (float/int) para fácil `log_metric`;
- `params`/`tags`: strings curtas e estáveis;
- campos obrigatórios por run: `run_info`, `params.model_name`, `params.backend`, `metrics.inference_success_rate`, `tags.status`;
- arquivo final contendo múltiplos runs (vários modelos no mesmo benchmark).

## 8) Plano de implementação (etapas)

1. **Config e estrutura**
   - Criar pasta `benchmark/scripts`.
   - Criar `.env.cuda`, `.env.amd`, `.env.cpu` com variáveis mínimas.

2. **Runner comum**
   - Implementar carregamento de env, descoberta de modelos e execução por matriz (modelo × dataset × amostras).

3. **Camada de datasets**
   - Implementar loaders e normalizadores para ENEM, BBQ e PoETaV2.

4. **Inferência robusta**
   - Implementar chamada `llama.cpp` com timeout, limites de token e detecção de loop.

5. **Métricas**
   - Implementar cálculo ELIB + métricas por dataset.

6. **Serialização JSON**
   - Gerar JSON final no padrão de `benchmark/refs/saida_benchmark.json`.
   - Validar consistência de tipos para ingestão no MLflow.

7. **Scripts por backend**
   - Implementar wrappers `benchmark_cuda.py`, `benchmark_amd.py`, `benchmark_cpu.py`.

8. **Validação**
   - Rodar smoke test com 1–2 amostras por dataset.
   - Rodar teste completo com 2+ modelos (instruct e reasoning).
   - Conferir JSON final e taxa de sucesso de inferência.

## 9) Critérios de aceite

- Executa em CUDA, AMD/ROCm e CPU sem alterar código (somente `.env`).
- Aceita múltiplos modelos GGUF por configuração.
- Não entra em loop infinito (timeout + limites + detecção de repetição).
- Gera JSON final em `benchmark/refs/saida_benchmark.json` no padrão acordado.
- Inclui métricas ELIB + ENEM + BBQ + PoETaV2.
- Resultado pronto para ingestão e comparação no MLflow.

