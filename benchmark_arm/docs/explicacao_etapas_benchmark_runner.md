# Explicação completa das etapas do `benchmark_runner.py`

Este documento descreve, de ponta a ponta, como o script `benchmark/scripts/benchmark_runner.py` executa o benchmark, quais decisões ele toma e como os resultados são gerados.

---

## 1) Objetivo do runner

O `benchmark_runner.py` é o orquestrador central do benchmark.  
Ele coordena:

- descoberta de modelos;
- leitura de configurações (CLI + `.env` + defaults);
- carregamento dos datasets;
- execução de performance (`llama-bench`);
- execução de inferência por amostra (`llama-completion`);
- scoring por dataset;
- perplexity (`llama-perplexity`);
- agregação e persistência de métricas em JSON.

Entradas principais:

- parâmetros de CLI (ex.: `--sample-size-enem`);
- variáveis de ambiente (ex.: `MODEL_GLOB`, `INFERENCE_TIMEOUT_S`);
- arquivos de dataset;
- executáveis `llama-completion`, `llama-bench`, `llama-perplexity`.

Saída principal:

- arquivo JSON com uma lista de `runs`, um por modelo.

---

## 2) Estruturas-base e constantes

### 2.1 `BenchmarkConfig` (dataclass)

Agrupa toda a configuração efetiva da execução:

- paths (binários, datasets, saída, modelos);
- tamanhos de amostra;
- parâmetros de inferência (`ctx_size`, `n_predict`, sampling);
- parâmetros de benchmark (`bench_repetitions`, `bench_n_prompt`, `bench_n_gen`);
- timeouts;
- modo de stop tokens;
- parâmetros de perplexity.

### 2.2 Constantes

- `PROJECT_ROOT`: raiz de `benchmark/`;
- `DEFAULT_OUTPUT_JSON`: `benchmark/output/saida_benchmark_poetav2.json`;
- `DEFAULT_DATASET_ROOT`: `benchmark/datasets`.

---

## 3) Resolução de configuração

Esta parte transforma entradas dispersas em uma configuração única e reproduzível.

### 3.1 `_load_env_file(env_file)`

Lê arquivo `.env` linha a linha e injeta pares `KEY=VALUE` no ambiente.

Comportamento atual (corrigido):

- usa `os.environ.setdefault(key, value)`;
- ou seja, se a variável já veio exportada pelo shell/CLI/CI, **ela prevalece**;
- o `.env` funciona como fallback.

Isso evita que overrides de execução sejam involuntariamente ignorados.

### 3.2 `_env_int` e `_env_float`

Helpers para ler variáveis numéricas com default quando ausentes.

### 3.3 `_dataset_sample_overrides_from_env`

Monta mapa de override por dataset específico, por exemplo:

- `SAMPLE_SIZE_POETAV2_GSM8K` -> `poetav2_gsm8k`;
- `SAMPLE_SIZE_ENEM_2024` -> `enem_2024`.

### 3.4 `_resolve_path`

Converte caminhos relativos/absolutos em `Path` resolvido de forma consistente, ancorando no `PROJECT_ROOT`.

### 3.5 `_discover_models`

Descobre modelos na ordem:

1. `MODEL_PATHS` (lista explícita separada por vírgula);
2. fallback para `MODEL_GLOB` (padrão `model/*.gguf`).

Depois remove duplicados.

### 3.6 `_build_config(backend_name)`

Consolida toda a configuração final:

- paths de binários;
- modelo(s);
- datasets;
- parâmetros de inferência/bench/perplexity;
- sample sizes gerais e por-dataset;
- stop tokens e modo de aplicação.

---

## 4) Fluxo de entrada da CLI

### 4.1 `main_for_backend(...)`

É o ponto de entrada usado por `benchmark_cuda.py`, `benchmark_cpu.py`, `benchmark_amd.py`.

Etapas:

1. define argumentos de CLI;
2. carrega `.env` informado (`--env-file`);
3. aplica defaults do backend (`backend_defaults`);
4. aplica overrides explícitos via CLI (`_apply_cli_overrides`);
5. constrói `BenchmarkConfig`.

### 4.2 Validações de pré-execução

Antes de rodar:

- exige ao menos um modelo;
- exige `DATASET_ROOT` existente;
- exige binários `llama-completion` e `llama-bench` (quando não é `--dry-run`).

### 4.3 `--dry-run`

Se ativo:

- carrega datasets;
- imprime resumo de configuração e contagens;
- não executa inferência.

---

## 5) Carregamento dos datasets

`load_all_datasets(...)` (em `datasets.py`) entrega um dicionário:

- chaves: nomes dos datasets (`enem_2022`, `bbq_gender_identity`, etc.);
- valores: lista de registros (rows).

O runner só consome esse dicionário; parsing específico e flattening ficam no módulo de datasets.

---

## 6) Avaliação por modelo: `_evaluate_model(...)`

Para cada modelo:

1. executa `llama-bench` para métricas de performance;
2. percorre cada dataset e cada amostra;
3. faz inferência com `llama-completion`;
4. extrai resposta (`extract_model_answer`);
5. computa score da amostra (`score_sample`);
6. agrega por dataset (`aggregate_dataset_metrics`);
7. agrega métricas globais;
8. roda perplexity no wikitext;
9. computa macro-métricas e tags;
10. retorna objeto `run`.

---

## 7) Etapa de performance: `_run_llama_bench(...)`

Monta comando com:

- modelo;
- `-o json`;
- reps (`-r`);
- tokens de prompt (`-p`);
- tokens de geração (`-n`);
- threads;
- gpu layers.

Parseia JSON retornado e calcula:

- `prompt_tps`;
- `gen_tps`;
- `peak_gen_tps`;
- `ttft_ms` aproximado (`1000 / prompt_tps`);
- `elapsed_s`;
- `rows` crus de bench.

### Tratamento de erro (corrigido)

Agora a função **não derruba o benchmark inteiro** em falhas de bench.

Em caso de:

- timeout;
- exit code != 0;
- saída vazia;
- JSON inválido;

ela retorna estrutura válida com métricas zeradas e campo:

- `error` (`timeout`, `exit_code_X`, `invalid_json`, etc.).

Esse erro é propagado para `tags.llama_bench_error`.

---

## 8) Etapa de inferência por amostra: `_run_inference(...)`

Para cada prompt:

1. monta comando `llama-completion` com parâmetros de sampling e contexto;
2. executa via `subprocess.Popen`;
3. monitora timeout manual;
4. mede RAM de processo e temperatura (CPU/GPU quando disponível);
5. lê saída bruta do arquivo temporário;
6. sanitiza saída (`_sanitize_output`);
7. detecta loops por n-grama (`_has_repeating_ngram`);
8. calcula sucesso/falha e metadados.

Retorna, entre outros:

- `output` (limpo);
- `raw_output`;
- `success`, `timed_out`, `loop_abort`, `error`;
- `peak_ram_gb`, `peak_vram_gb`, `thermal_avg_c`;
- `total_time_s`.

---

## 9) Prompting por dataset: `_build_prompt(...)`

Define template específico por dataset, com forte instrução de formato final.

Exemplos:

- ENEM: `FINAL_ANSWER: <A|B|C|D|E>`;
- BBQ: `FINAL_ANSWER: <0|1|2>`;
- numéricos: `FINAL_ANSWER: <number>`;
- span curto (CoQA/TriviaQA): `FINAL_ANSWER: <texto>`.

Isso reduz ambiguidade de parser e diminui taxa de respostas inválidas.

---

## 10) Scoring e agregação (módulo `metrics.py`)

Fluxo de qualidade:

1. `extract_model_answer(dataset_name, output)` extrai predição canônica;
2. `score_sample(...)` gera score por linha (accuracy/exact/f1/valid);
3. `aggregate_dataset_metrics(...)` resume por dataset;
4. `merge_metric_dicts(...)` combina todos os blocos;
5. `compute_macro_metrics(...)` calcula métricas macro.

Inclui:

- acurácia ENEM por ano + macro;
- acurácia BBQ + ambig/disambig + `bias_score_bbq`;
- LogiQA, GSM8K, CoQA, TriviaQA, Arithmetic;
- score macro PoETaV2.

---

## 11) Etapa de perplexity: `_run_perplexity(...)`

Usa amostras de `poetav2_wikitext`:

1. monta corpus textual;
2. salva em arquivo temporário;
3. executa `llama-perplexity`;
4. parseia `Final estimate: PPL = ...` (com fallback regex);
5. retorna `(ppl, elapsed_s)`.

Se não houver binário/corpus/parse, retorna `0.0`.

---

## 12) Métricas finais e status do run

Além das métricas de dataset, o runner calcula:

- `avg_tps`, `peak_tps`, `avg_ttft_ms`;
- tempos de inferência (`avg`, `p95`, `total`);
- `ram_peak_gb`, `vram_peak_gb`, `thermal_avg_c`;
- `inference_success_rate`;
- `mbu` (com `compute_mbu`).

Define `status`:

- `failed`: success rate 0;
- `partial`: entre 0 e 1;
- `completed`: 1.0.

Também registra:

- contagem de timeouts/loops/errors;
- fonte de métricas (`metrics_source=llama-bench`);
- erro do bench (`llama_bench_error`).

---

## 13) Persistência do resultado

No final do `main_for_backend`:

1. executa `_evaluate_model` para cada modelo descoberto;
2. cria diretório de saída;
3. grava JSON (`ensure_ascii=False`, `indent=2`);
4. imprime caminho e número de runs.

Estrutura do JSON por run:

- `run_info`;
- `params`;
- `benchmark_params`;
- `device_params`;
- `metrics`;
- `tags`;
- `dataset_counts`;
- `bench_raw`.

---

## 14) Resumo do fluxo em uma linha

`CLI/.env -> config -> datasets -> llama-bench -> inferência+scoring -> perplexity -> agregação -> JSON final`.

---

## 15) Observações práticas

- Se quiser smoke rápido: reduza `N_PREDICT`, `CTX_SIZE`, `THREADS`, `BENCH_N_PROMPT`, `BENCH_N_GEN` e `SAMPLE_SIZE_*`.
- Para comparações reais: mantenha sample size fixo entre modelos.
- Se `llama-bench` estourar timeout, agora o run continua e o erro fica rastreável em `tags.llama_bench_error`.
