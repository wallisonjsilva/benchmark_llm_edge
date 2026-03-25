# Benchmark ELIB + llama.cpp

Framework de avaliação de LLMs para dispositivos edge, combinando métricas de desempenho
**ELIB** (Edge Library Benchmarking) com métricas de qualidade em português via **PoETaV2**.
Executa inferência local usando [llama.cpp](https://github.com/ggerganov/llama.cpp) em
múltiplos backends de hardware.

## Arquitetura

```
benchmark/
├── scripts/                    # Código principal (~1.700 LOC Python)
│   ├── benchmark_runner.py     # Orquestrador central
│   ├── benchmark_cuda.py       # Wrapper NVIDIA (L4/L40S/T4)
│   ├── benchmark_cpu.py        # Wrapper CPU (ARM/x86)
│   ├── benchmark_amd.py        # Wrapper AMD ROCm
│   ├── metrics.py              # Extração de respostas e scoring
│   └── datasets.py             # Carregamento e amostragem dos datasets
├── datasets/                   # 29 arquivos JSONL (ver seção Datasets)
│   ├── poetav2/                # ENEM, Arithmetic, CoQA, GSM8K, LogiQA, TriviaQA, Wikitext
│   └── bbq/                    # Bias Benchmark for QA
├── docs/                       # Documentação técnica detalhada
├── refs/                       # Referências sobre PoETaV2 e ELIB
├── output/                     # Resultados JSON gerados
├── model/                      # Modelos GGUF (não versionados)
├── .env.cuda / .env.cpu / .env.amd   # Configuração por backend
└── pyproject.toml              # Python ≥ 3.13, gerenciado com uv
```

## Fluxo de Execução

```
Configuração (.env + CLI)
    ↓
Descoberta de modelos (glob em model/*.gguf)
    ↓
Carregamento dos datasets (JSONL → amostragem determinística, seed=75)
    ↓
Para cada modelo:
    ├─ llama-bench → TPS, TTFT (desempenho bruto)
    ├─ Para cada dataset:
    │   ├─ Construção do prompt (formato específico por dataset)
    │   ├─ Detecção de chat template (Qwen/ChatML, Llama3, DeepSeek, Sabiá)
    │   ├─ llama-completion (com timeout, monitoramento de RAM/VRAM/temperatura)
    │   ├─ Extração da resposta (regex com fallbacks)
    │   └─ Scoring (accuracy / F1 / exact_match)
    ├─ llama-perplexity (Wikitext PPL)
    └─ Cálculo de métricas macro e MBU
    ↓
Saída: output/saida_benchmark_poetav2.json (compatível com MLflow)
```

## Métricas Coletadas

### Desempenho (ELIB)

| Métrica | Fonte | Descrição |
|---------|-------|-----------|
| `avg_tps` / `peak_tps` | llama-bench | Tokens por segundo (geração) |
| `avg_ttft_ms` | llama-bench | Latência até o primeiro token |
| `mbu` | calculado | Utilização de banda de memória (Memory Bandwidth Utilization) |
| `ram_peak_gb` / `vram_peak_gb` | monitoramento | Pico de uso de memória |
| `thermal_avg_c` | monitoramento | Temperatura média durante inferência |
| `inference_success_rate` | runner | Taxa de inferências concluídas com sucesso |
| `perplexity_poetav2_wikitext` | llama-perplexity | Qualidade de modelagem de linguagem |

### Qualidade por Dataset

| Família | Métricas | Tipo |
|---------|----------|------|
| ENEM (2022/2023/2024) | `accuracy_enem_*`, `invalid_answer_rate_enem` | Múltipla escolha (A–E) |
| BBQ (3 categorias) | `accuracy_bbq_*`, `bias_score_bbq` | Viés/fairness (0–2) |
| GSM8K | `exact_match_poetav2_gsm8k` | Raciocínio matemático |
| LogiQA | `accuracy_poetav2_logiqa` | Raciocínio lógico | (Não estar em uso atualmente, exaustivo e estoura a janela de contexto)
| CoQA | `f1_poetav2_coqa` | QA conversacional | (Não estar em uso atualmente)
| TriviaQA | `exact_match_poetav2_triviaqa` | QA factual | (Não estar em uso atualmente)
| Arithmetic | `exact_match_poetav2_arithmetic` | Aritmética (1–5 dígitos) | (Não estar em uso atualmente)

### Tempo de Execução

`total_benchmark_time_s`, `inference_avg_time_s`, `inference_p95_time_s`,
`inference_total_time_s`, `llama_bench_time_s`, `perplexity_time_s`

## Datasets

Todos os datasets estão no diretório `datasets/` em formato JSONL. A amostragem é
determinística (seed fixa = 75) para garantir reprodutibilidade entre execuções e hardwares.

### ENEM — Exame Nacional do Ensino Médio

Questões de múltipla escolha (5 alternativas, rótulo A–E) das provas oficiais brasileiras.

| Arquivo | Registros | Uso |
|---------|-----------|-----|
| `poetav2/enem/2022.jsonl` | 179 | Avaliação |
| `poetav2/enem/2023.jsonl` | 179 | Avaliação |
| `poetav2/enem/2024.jsonl` | 179 | Avaliação |
| `poetav2/enem/train.jsonl` | 537 | Referência |

**Campos**: `id`, `exam`, `context`, `question`, `options` (lista de 5), `label`

### BBQ — Bias Benchmark for QA

Avalia viés do modelo em relação a grupos demográficos. Cada registro traz um contexto
ambíguo ou desambiguado e 3 alternativas (índices 0–2).

| Arquivo | Registros | Categoria |
|---------|-----------|-----------|
| `bbq/Gender_identity.jsonl` | 5.672 | Identidade de gênero |
| `bbq/Race_ethnicity.jsonl` | 6.880 | Raça e etnia |
| `bbq/Physical_appearance.jsonl` | 1.576 | Aparência física |

**Campos**: `example_id`, `category`, `context`, `question`, `ans0`/`ans1`/`ans2`,
`label`, `context_condition` (ambig/disambig), `question_polarity` (neg/nonneg)

### PoETaV2 — Avaliação em Português

Subconjunto de tarefas do framework PoETaV2, cobrindo raciocínio, matemática, QA e
modelagem de linguagem.

| Dataset | Arquivos | Registros (avaliação) | Métrica |
|---------|----------|-----------------------|---------|
| **Arithmetic** | 10 variantes (1–5 dígitos, adição/subtração) | 2.000 cada | exact_match |
| **GSM8K** | train (7.473) + test (1.319) | 1.319 | exact_match |
| **LogiQA** | train (7.376) + validation (651) + test (651) | 651 | accuracy |
| **CoQA** | train (7.199) + validation (500) | 500 | F1 |
| **TriviaQA** | train (87.622) + validation (11.313) | 11.313 | exact_match |
| **Wikitext** | train (629) + validation (60) + test (62) | 60 | perplexity |

### Tamanhos de Amostra

Cada backend define tamanhos de amostra adequados ao seu throughput. Exemplos:

- **CUDA** (GPU rápida): 60 ENEM / 30 BBQ / 30 PoETaV2
- **CPU** (mais lento): 15 ENEM / 10 BBQ / 10 PoETaV2

Overrides por dataset via variáveis de ambiente (ex: `SAMPLE_SIZE_ENEM_2022=100`).

## Execução

### Pré-requisitos

- Python ≥ 3.13 com [uv](https://github.com/astral-sh/uv)
- Binários compilados do llama.cpp (`llama-bench`, `llama-completion`, `llama-perplexity`)
- Modelos GGUF em `model/`

### Dry-run (validação sem inferência)

```bash
uv run python -m scripts.benchmark_cuda --dry-run
uv run python -m scripts.benchmark_cpu --dry-run
uv run python -m scripts.benchmark_amd --dry-run
```

### Execução real

Configure os caminhos dos binários e parâmetros no `.env` correspondente:

```bash
# NVIDIA GPU
uv run python -m scripts.benchmark_cuda

# CPU (ARM/x86)
uv run python -m scripts.benchmark_cpu

# AMD ROCm
uv run python -m scripts.benchmark_amd
```

### Customização de amostras

```bash
SAMPLE_SIZE_ENEM_2022=100 SAMPLE_SIZE_POETAV2_GSM8K=50 \
  uv run python -m scripts.benchmark_cuda
```

Saída padrão: `output/saida_benchmark_poetav2.json`

## Hardware Suportado

| Plataforma | Exemplos | Bandwidth ref. | N_GPU_LAYERS | Notas |
|------------|----------|----------------|--------------|-------|
| **NVIDIA CUDA** | L4, L40S, T4 | 300 GB/s | 99 (full offload) | Flash Attention recomendado para Ada Lovelace |
| **AMD ROCm** | RX 7600, RDNA/CDNA | 256 GB/s | 99 (full offload) | Suporte a Flash Attention varia |
| **CPU** | ARM Graviton, Apple M, x86 | 45 GB/s | 0 | CTX_SIZE=2048, foco em quantizações |

## Variáveis de Ambiente (.env)

### Binários e caminhos

| Variável | Descrição |
|----------|-----------|
| `LLAMA_COMPLETION_PATH` | Binário de inferência (gera respostas) |
| `LLAMA_BENCH_PATH` | Binário de benchmark de desempenho |
| `LLAMA_PERPLEXITY_PATH` | Binário para cálculo de perplexidade (Wikitext) |
| `MODEL_GLOB` | Padrão glob para descoberta de modelos (default: `model/*.gguf`) |
| `OUTPUT_JSON_PATH` | Caminho da saída JSON |
| `DATASET_ROOT` | Diretório raiz dos datasets |

### Parâmetros de hardware

| Variável | Descrição |
|----------|-----------|
| `N_GPU_LAYERS` | Camadas offloaded para GPU (0 = CPU only, 99 = full) |
| `THREADS` | Threads de CPU |
| `HARDWARE_BANDWIDTH_GBS` | Bandwidth de memória do hardware (para cálculo MBU) |
| `CTX_SIZE` | Tamanho do contexto (tokens) |
| `FLASH_ATTN` | Habilitar Flash Attention (`true`/`false`) |

### Parâmetros de inferência

| Variável | Descrição |
|----------|-----------|
| `TEMP` | Temperatura de amostragem (0.0 = greedy/determinístico) |
| `N_PREDICT` | Máximo de tokens gerados por inferência |
| `INFERENCE_TIMEOUT_S` | Timeout por inferência (segundos) |
| `MAX_REPEAT_NGRAM` | Limite de n-gram repetido (detecção de loops) |
| `STOP_TOKENS` | Lista de tokens de parada (separados por `\|`) |
| `STOP_TOKENS_MODE` | Quando aplicar stop tokens: `always`, `sabia7` (default), `never` |

### Parâmetros de benchmark

| Variável | Descrição |
|----------|-----------|
| `BENCH_REPETITIONS` | Repetições no llama-bench |
| `BENCH_N_PROMPT` | Tokens de prompt no benchmark |
| `BENCH_N_GEN` | Tokens gerados no benchmark |
| `PERPLEXITY_WIKITEXT_ROWS` | Páginas do Wikitext para PPL (mais = mais estável) |

### Tamanhos de amostra

Globais: `SAMPLE_SIZE_ENEM`, `SAMPLE_SIZE_BBQ`, `SAMPLE_SIZE_POETAV2`

Overrides por dataset (sobrescrevem os globais):

- `SAMPLE_SIZE_ENEM_2022`, `SAMPLE_SIZE_ENEM_2023`, `SAMPLE_SIZE_ENEM_2024`
- `SAMPLE_SIZE_BBQ_GENDER_IDENTITY`, `SAMPLE_SIZE_BBQ_PHYSICAL_APPEARANCE`, `SAMPLE_SIZE_BBQ_RACE_ETHNICITY`
- `SAMPLE_SIZE_POETAV2_LOGIQA`, `SAMPLE_SIZE_POETAV2_GSM8K`, `SAMPLE_SIZE_POETAV2_COQA`
- `SAMPLE_SIZE_POETAV2_TRIVIAQA`, `SAMPLE_SIZE_POETAV2_ARITHMETIC`, `SAMPLE_SIZE_POETAV2_WIKITEXT`

## Estratégia de Quantização

O benchmark suporta avaliação sistemática do trade-off entre qualidade e performance
em diferentes níveis de quantização:

1. **Baseline**: FP16/BF16 em GPU com Flash Attention
2. **Quantizações**: Q8_0 → Q4_K_M → Q3_K_M → Q2_K_S
3. **Critérios de aceitação**:
   - Queda de qualidade < 3–5% (accuracy_enem_macro, perplexity)
   - Ganho de throughput > 1.5× (avg_tps)
   - Taxa de sucesso ≥ 95% (inference_success_rate)

Detalhes: [`docs/recomendacoes_quantizacao_edge.md`](docs/recomendacoes_quantizacao_edge.md)

## Documentação Complementar

| Documento | Descrição |
|-----------|-----------|
| [`docs/plano_benchmark_elib_llamacpp.md`](docs/plano_benchmark_elib_llamacpp.md) | Plano de implementação e arquitetura |
| [`docs/guia_benchmark_multi_hardware.md`](docs/guia_benchmark_multi_hardware.md) | Guia de setup por hardware (NVIDIA, AMD, CPU) |
| [`docs/explicacao_etapas_benchmark_runner.md`](docs/explicacao_etapas_benchmark_runner.md) | Explicação detalhada de cada etapa do runner |
| [`docs/extracao_metricas_bench_custom.md`](docs/extracao_metricas_bench_custom.md) | Pipeline de extração de métricas |
| [`docs/recomendacoes_quantizacao_edge.md`](docs/recomendacoes_quantizacao_edge.md) | Recomendações para avaliação de quantizações |
| [`docs/llama-bench/README.md`](docs/llama-bench/README.md) | Referência do llama-bench |
| [`docs/llama-cli/README.md`](docs/llama-cli/README.md) | Referência do llama-cli |
| [`docs/llama-perplexity/README.md`](docs/llama-perplexity/README.md) | Referência do llama-perplexity |
