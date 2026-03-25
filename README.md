# benchmark_llm_edge

Framework de benchmarking de Large Language Models (LLMs) para cenários de **edge computing**,
com foco em modelos de linguagem em português.

O projeto avalia simultaneamente **desempenho de hardware** (throughput, latência, uso de memória)
e **qualidade de respostas** (acurácia, F1, exact match, perplexidade) em diferentes backends
(NVIDIA CUDA, AMD ROCm, CPU ARM/x86), permitindo análise sistemática do trade-off entre
quantização e qualidade para implantação em dispositivos com recursos limitados.

## Motivação

Implantar LLMs em produção no edge — GPUs compactas, dispositivos ARM, servidores com VRAM
limitada — exige encontrar o equilíbrio certo entre velocidade de inferência e qualidade das
respostas. Este framework automatiza essa avaliação, respondendo perguntas como:

- Qual o impacto real de quantizar um modelo de FP16 para Q4_K_M em tarefas em português?
- Qual backend de hardware entrega o melhor custo-benefício para modelos 7–8B?
- O modelo mantém ≥95% da qualidade original com 2× o throughput após quantização?

## Componentes

```
benchmark_llm_edge/
├── benchmark/          # Suite principal de benchmark (ELIB + llama.cpp)
├── PoETaV2/            # Framework de avaliação de LLMs em português (40+ tarefas)
└── README.md
```

### [benchmark/](benchmark/)

Suite de benchmark que combina métricas **ELIB** (Edge Library Benchmarking) com inferência
local via [llama.cpp](https://github.com/ggerganov/llama.cpp). Orquestra a execução de
`llama-bench`, `llama-completion` e `llama-perplexity` sobre modelos GGUF, coletando
30+ métricas por modelo em formato JSON compatível com MLflow.

**Principais capacidades:**
- Benchmark de desempenho: TPS (tokens/s), TTFT (latência), MBU (utilização de banda)
- Avaliação de qualidade em 3 famílias de datasets (ENEM, BBQ, PoETaV2)
- Suporte a 3 backends: NVIDIA CUDA, AMD ROCm, CPU
- Avaliação de 6 níveis de quantização (FP16 → Q4_1)
- Monitoramento de recursos (RAM, VRAM, temperatura)
- Detecção de loops e timeout automático
- Saída JSON estruturada para MLflow

### [PoETaV2/](PoETaV2/)

**Portuguese LLM Evaluation Toolkit v2** — framework de avaliação abrangente para modelos
de linguagem em português, baseado no [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Inclui 40+ tarefas cobrindo compreensão de texto, raciocínio lógico, matemática, QA,
análise de sentimento e detecção de viés, com cálculo de NPM (Normalized Preferred Metric)
para comparação padronizada entre modelos.

## Datasets

O benchmark utiliza 29 arquivos JSONL organizados em 3 famílias:

| Família | Datasets | Total de registros | O que avalia |
|---------|----------|--------------------|--------------|
| **ENEM** | Provas 2022, 2023, 2024 + treino | 1.074 | Conhecimento geral e interpretação em português (múltipla escolha) |
| **BBQ** | Gender Identity, Race/Ethnicity, Physical Appearance | 14.128 | Viés e fairness do modelo em grupos demográficos |
| **PoETaV2** | Arithmetic (10 variantes), GSM8K, LogiQA, CoQA, TriviaQA, Wikitext | ~130.000 | Raciocínio matemático, lógico, QA conversacional/factual, perplexidade |

A amostragem é determinística (seed fixa = 75) para reprodutibilidade entre hardwares.
Tamanhos de amostra configuráveis por backend e por dataset via variáveis de ambiente.

## Hardware Alvo

| Plataforma | Exemplos | Cenário |
|------------|----------|---------|
| **NVIDIA CUDA** | L4, L40S, T4 | Baseline FP16/BF16 + quantizações |
| **AMD ROCm** | RX 7600, RDNA/CDNA | Quantizações Q4–Q8-Q3-Q2 |
| **CPU** | ARM Graviton, Apple M, x86 | Edge agressivo (Q4, Q3, Q2 principalmente e demonstrar porque Q8 não é bom) |

## Modelos Avaliados

O framework é agnóstico a modelos — qualquer GGUF compatível com llama.cpp pode ser
avaliado. Os modelos documentados incluem:

- **Qwen3-8B** — ChatML template, raciocínio avançado
- **Llama-3.1-8B** — Meta, template Llama 3
- **Gemma3-12B** - Modelo google
- **Sabiá-7B** — Otimizado para português (Maritaca AI)

## Quick Start

```bash
cd benchmark/

# Instalar dependências
uv sync

# Colocar modelos GGUF em model/
# Configurar .env.cuda (ou .env.cpu / .env.amd)

# Validar setup (sem inferência real)
uv run python -m scripts.benchmark_cuda --dry-run

# Executar benchmark completo
uv run python -m scripts.benchmark_cuda
```

Resultados em `benchmark/output/saida_benchmark_poetav2.json`.

## Requisitos

- Python ≥ 3.13
- [uv](https://github.com/astral-sh/uv) (gerenciador de pacotes)
- Binários compilados do [llama.cpp](https://github.com/ggerganov/llama.cpp)
  (`llama-bench`, `llama-completion`, `llama-perplexity`)
- Modelos no formato GGUF

## Documentação

A documentação detalhada está em [`benchmark/docs/`](benchmark/docs/):

- [Plano de implementação](benchmark/docs/plano_benchmark_elib_llamacpp.md)
- [Guia multi-hardware](benchmark/docs/guia_benchmark_multi_hardware.md)
- [Etapas do benchmark runner](benchmark/docs/explicacao_etapas_benchmark_runner.md)
- [Extração de métricas](benchmark/docs/extracao_metricas_bench_custom.md)
- [Recomendações de quantização](benchmark/docs/recomendacoes_quantizacao_edge.md)

## Licença

MIT
