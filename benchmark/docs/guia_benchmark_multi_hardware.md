# Guia de Benchmark Multi-Hardware — FP16/BF16 Baseline + Quantizações

## 1. Resumo do Problema

O benchmark anterior (`saida_benchmark_poetav2.json`) rodou o modelo **Qwen3-8B FP16** em uma **Tesla T4** com os seguintes problemas:

| Problema | Impacto | Solução |
|---|---|---|
| **Prompts sem chat template** | 100% de `invalid_answer_rate_enem` — modelo tratou input como texto puro e ignorou instruções de formato | Adicionado sistema de chat templates por família de modelo (ChatML, Llama3, Llama2) |
| **N_GPU_LAYERS=20** (de 33 camadas) | ~60% do modelo rodou na CPU → TPS de apenas 2.99 | Alterado para 99 (offload total na GPU) |
| **Flash Attention desativado** | Perda de 2-4x de throughput em GPUs Ada Lovelace (L4/L40S) | Adicionada flag `-fa` condicional via `FLASH_ATTN=true` |
| **Temperatura 0.2** | Introduzia aleatoriedade → respostas inconsistentes entre runs | Alterada para 0.0 (greedy decoding, 100% determinístico) |
| **Stop tokens incompletos** | Tokens de fim de turno do Qwen (`<\|im_end\|>`) e Llama3 (`<\|eot_id\|>`) não estavam na lista | Lista expandida + detecção automática por família de modelo |
| **Instrução "Não use \<think\>"** | Confundia modelos que não conhecem esse token | Removida do prompt; tratada por `--reasoning off` e `_sanitize_output` |

---

## 2. Mudanças Realizadas

### 2.1 Chat Templates por Família de Modelo

O script agora detecta automaticamente a família do modelo a partir do nome do arquivo GGUF e aplica o template de chat correto:

| Família | Detecção (filename) | Template | Exemplo |
|---|---|---|---|
| **Qwen3 / DeepSeek-R1-Distill** | `qwen` ou `deepseek` no nome | **ChatML** | `<\|im_start\|>system\n...<\|im_end\|>\n<\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n` |
| **Llama 3.1** | `llama` no nome | **Llama 3** | `<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>\n\n...<\|eot_id\|>...` |
| **Sabiá-7B** | `sabia` no nome | **Llama 2** | `[INST] <<SYS>>\n...\n<</SYS>>\n\n{prompt} [/INST]` |
| **Genérico** | Nenhuma detecção | Sem template | Prompt enviado como texto puro |

**Função**: `_detect_model_family()` em `benchmark_runner.py`

**Importante**: Nomeie seus arquivos GGUF de forma que o nome contenha a família do modelo. Exemplos:
- `Qwen3-8b-fp16.gguf` → detecta `qwen`
- `llama-3.1-8b-instruct-fp16.gguf` → detecta `llama3`
- `deepseek-r1-distill-qwen-7b-fp16.gguf` → detecta `deepseek`
- `sabia-7b-instruct-fp16.gguf` → detecta `sabia`

### 2.2 Flash Attention

Adicionada variável de ambiente `FLASH_ATTN=true` que injeta a flag `-fa` nos comandos `llama-bench` e `llama-completion`.

- **L4/L40S (Ada Lovelace, CC 8.9)**: Suporta Flash Attention 2. **Obrigatório** para desempenho estado da arte.
- **T4 (Turing, CC 7.5)**: **Não suporta**. Deixe `FLASH_ATTN=false` ou omita.
- **AMD ROCm**: Suporte depende do GPU e versão do ROCm. Teste antes de ativar.
- **CPU**: **Não aplicável**. Omita a variável.

### 2.3 Stop Tokens Automáticos

O sistema agora injeta stop tokens específicos do modelo **automaticamente**, independente do que estiver no `.env`. A variável `STOP_TOKENS` no `.env` agora serve apenas para tokens genéricos adicionais (`###`, `User:`).

### 2.4 Temperatura Greedy (0.0)

Para benchmarks de acurácia, `TEMP=0.0` garante:
- Determinismo total (mesma entrada → mesma saída)
- Modelo sempre escolhe o token mais provável
- Resultados reprodutíveis entre runs

---

## 3. Configuração por Hardware

### 3.1 NVIDIA L4 / L40S (Baseline FP16/BF16)

**Objetivo**: Estabelecer o "teto" de qualidade e velocidade para comparação com quantizações.

#### Compilação do llama.cpp
```bash
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

> ⚠️ **CRÍTICO**: `-DCMAKE_CUDA_ARCHITECTURES=89` é específico para Ada Lovelace (L4/L40S). Sem isso, o binário não usará os kernels otimizados para o hardware.

#### Arquivo `.env.cuda`
```env
N_GPU_LAYERS=99
TEMP=0.0
FLASH_ATTN=true
STOP_TOKENS_MODE=always
HARDWARE_BANDWIDTH_GBS=300.0   # L4: 300 GB/s | L40S: 864 GB/s (ajuste conforme a GPU)
```

#### Conversão de modelos para GGUF FP16/BF16
```bash
# FP16
python3 llama.cpp/convert_hf_to_gguf.py models/Qwen3-8B-Instruct/ \
  --outtype f16 --outfile benchmark/model/Qwen3-8b-fp16.gguf

# BF16 (preferível para modelos treinados em BF16: Qwen3, Llama3.1)
python3 llama.cpp/convert_hf_to_gguf.py models/Qwen3-8B-Instruct/ \
  --outtype bf16 --outfile benchmark/model/Qwen3-8b-bf16.gguf
```

#### Modelos FP16 a converter
| Modelo | HuggingFace | Tamanho FP16 | VRAM necessária |
|---|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | ~16 GB | L4 (24GB) ✅ / L40S (48GB) ✅ |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB | L4 ✅ / L40S ✅ |
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14 GB | L4 ✅ / L40S ✅ |
| Sabiá-7B | `maritaca-ai/sabia-7b` | ~14 GB | L4 ✅ / L40S ✅ |

#### Executar benchmark
```bash
cd benchmark
python -m scripts.benchmark_cuda
```

#### O que esperar (L4)
| Modelo | TPS esperado (FP16+FA) | Perplexidade (referência) |
|---|---|---|
| Qwen3-8B | ~25-40 t/s | ~6.5-7.5 |
| Llama-3.1-8B | ~25-40 t/s | ~6.0-7.0 |
| DeepSeek-R1-Distill-7B | ~30-45 t/s | ~7.0-8.5 |
| Sabiá-7B | ~30-45 t/s | ~8.0-10.0 |

---

### 3.2 AMD ROCm — GPU 8 GB VRAM (Quantizações)

**Objetivo**: Benchmark de quantizações Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.

#### Compilação do llama.cpp
```bash
cd llama.cpp
cmake -B build \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx1030"  # Ajuste para sua GPU: gfx1030 (RX 6800), gfx1100 (RX 7900)
cmake --build build --config Release -j$(nproc)
```

#### Arquivo `.env.amd`
```env
N_GPU_LAYERS=99
TEMP=0.0
FLASH_ATTN=false              # Testar com true se ROCm >= 6.0 + GPU suportada
STOP_TOKENS_MODE=always
HARDWARE_BANDWIDTH_GBS=256.0   # Ajuste para sua GPU (RX 6800: 256, RX 7900 XTX: 960)
```

#### Quantizações e VRAM estimada (modelo 8B)
| Quantização | Tamanho | VRAM (aprox.) | Cabe em 8 GB? |
|---|---|---|---|
| Q8_0 | ~8.5 GB | ~9.0 GB | ⚠️ Apertado — pode precisar `N_GPU_LAYERS` < 99 |
| Q5_1 | ~5.7 GB | ~6.2 GB | ✅ Sim |
| Q5_0 | ~5.3 GB | ~5.8 GB | ✅ Sim |
| Q4_1 | ~4.8 GB | ~5.3 GB | ✅ Sim |
| Q4_0 | ~4.3 GB | ~4.8 GB | ✅ Sim |

> ⚠️ **Q8_0 em 8 GB VRAM**: Teste com `N_GPU_LAYERS=99` primeiro. Se der OOM, reduza para ~28-30 camadas para modelos 8B (33 camadas total). Use `N_GPU_LAYERS=28` no `.env.amd`.

#### Quantizar modelos
```bash
# A partir do GGUF FP16
./build/bin/llama-quantize model/Qwen3-8b-fp16.gguf model/Qwen3-8b-Q4_0.gguf Q4_0
./build/bin/llama-quantize model/Qwen3-8b-fp16.gguf model/Qwen3-8b-Q4_1.gguf Q4_1
./build/bin/llama-quantize model/Qwen3-8b-fp16.gguf model/Qwen3-8b-Q5_0.gguf Q5_0
./build/bin/llama-quantize model/Qwen3-8b-fp16.gguf model/Qwen3-8b-Q5_1.gguf Q5_1
./build/bin/llama-quantize model/Qwen3-8b-fp16.gguf model/Qwen3-8b-Q8_0.gguf Q8_0
```

#### Executar benchmark
```bash
cd benchmark
python -m scripts.benchmark_amd
```

---

### 3.3 NVIDIA CUDA 16 GB VRAM (Quantizações — ex: T4)

**Objetivo**: Benchmark de quantizações no ecossistema CUDA com GPU de 16 GB.

#### Compilação do llama.cpp
```bash
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_BUILD_TYPE=Release
# T4: CC 7.5 → ARCHITECTURES=75
# A10: CC 8.6 → ARCHITECTURES=86
cmake --build build --config Release -j$(nproc)
```

#### Arquivo `.env.cuda` (ajustes para T4 + quantizações)
```env
N_GPU_LAYERS=99
TEMP=0.0
FLASH_ATTN=false               # T4 não suporta Flash Attention
HARDWARE_BANDWIDTH_GBS=320.0   # T4: 320 GB/s
```

> **Nota**: Para rodar quantizações na T4, copie `.env.cuda` → `.env.cuda.quant` e ajuste `FLASH_ATTN=false` e `HARDWARE_BANDWIDTH_GBS=320.0`. Execute com:
> ```bash
> python -m scripts.benchmark_cuda --env-file .env.cuda.quant
> ```

---

### 3.4 CPU ARM 32 GB RAM (Quantizações)

**Objetivo**: Benchmark de quantizações em CPU ARM (ex: Raspberry Pi 5, Apple M-series, AWS Graviton).

#### Compilação do llama.cpp
```bash
cd llama.cpp
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release
# Para Apple Silicon, adicione: -DGGML_METAL=ON
# Para ARM com NEON (ativado por padrão na maioria dos ARM64)
cmake --build build --config Release -j$(nproc)
```

#### Arquivo `.env.cpu`
```env
N_GPU_LAYERS=0
TEMP=0.0
CTX_SIZE=2048                  # Contexto menor para economizar RAM
THREADS=8                     # Ajuste para o número de cores eficientes do seu ARM
HARDWARE_BANDWIDTH_GBS=45.0   # DDR4: ~35-50, DDR5: ~50-80, Apple M2: ~100
INFERENCE_TIMEOUT_S=600       # CPU é mais lento — aumente o timeout
```

#### RAM estimada (modelo 8B)
| Quantização | Tamanho modelo | RAM total (modelo + KV cache) |
|---|---|---|
| Q8_0 | ~8.5 GB | ~10-12 GB |
| Q5_1 | ~5.7 GB | ~7-9 GB |
| Q5_0 | ~5.3 GB | ~7-8 GB |
| Q4_1 | ~4.8 GB | ~6-8 GB |
| Q4_0 | ~4.3 GB | ~6-7 GB |

> Todos cabem em 32 GB com folga. Use `CTX_SIZE=2048` para limitar o KV cache.

#### Executar benchmark
```bash
cd benchmark
python -m scripts.benchmark_cpu
```

---

## 4. Configuração por Modelo

### 4.1 Qwen3-8B

- **Família detectada**: `qwen` (ChatML template)
- **Formato de treino**: BF16 (preferir `--outtype bf16` na conversão)
- **Stop tokens automáticos**: `<|im_end|>`, `<|endoftext|>`, `<|im_start|>`
- **Reasoning mode**: Qwen3 suporta modo thinking. O flag `--reasoning off` desativa. O `_sanitize_output` remove blocos `<think>` como fallback.
- **Nomeação GGUF**: `Qwen3-8b-fp16.gguf`, `Qwen3-8b-Q4_0.gguf`, etc.

### 4.2 Llama-3.1-8B-Instruct

- **Família detectada**: `llama3` (Llama 3 template)
- **Formato de treino**: BF16
- **Stop tokens automáticos**: `<|eot_id|>`, `<|end_of_text|>`, `<|start_header_id|>`
- **Nota**: Use a versão **Instruct** (não a base). A versão base não segue instruções.
- **Nomeação GGUF**: `llama-3.1-8b-instruct-fp16.gguf`

### 4.3 DeepSeek-R1-Distill-Qwen-7B

- **Família detectada**: `deepseek` (ChatML template, mesmo do Qwen)
- **Formato de treino**: BF16
- **Stop tokens automáticos**: `<|im_end|>`, `<|endoftext|>`, `<|im_start|>`
- **⚠️ Atenção**: Este é um modelo de **raciocínio** (Chain of Thought). Mesmo com `--reasoning off`, ele pode gerar blocos `<think>` longos. O `_sanitize_output` limpa isso, mas o N_PREDICT pode precisar ser maior (128-256) se as respostas estiverem sendo cortadas.
- **Quantização**: Modelos de raciocínio sofrem mais com quantizações agressivas (< Q5). O benchmark FP16 é essencial como baseline.
- **Nomeação GGUF**: `deepseek-r1-distill-qwen-7b-fp16.gguf`

### 4.4 Sabiá-7B

- **Família detectada**: `sabia` (Llama 2 template)
- **Formato de treino**: FP16 (baseado em Llama-2)
- **Stop tokens automáticos**: `</s>`, `[INST]`
- **Nota**: Sabiá-7B é extremamente estável em quantizações. A perda de Q4_K_M→FP16 costuma ser < 1% de PPL.
- **Nomeação GGUF**: `sabia-7b-instruct-fp16.gguf`

---

## 5. Estratégia de Trade-off: FP16 vs Quantizações

### 5.1 Métricas de Comparação

Para cada par (modelo, quantização), colete:

| Métrica | Fonte | Significado |
|---|---|---|
| **avg_tps** | `llama-bench` | Velocidade de geração (tokens/segundo) |
| **perplexity** | `llama-perplexity` | Qualidade do modelo (menor = melhor) |
| **accuracy_enem_macro** | Inferência + parsing | Acurácia em múltipla escolha |
| **vram_peak_gb** | Monitoramento runtime | Consumo de VRAM |
| **mbu** | Calculado | Memory Bandwidth Utilization |

### 5.2 Tabela de Comparação (Template)

```
| Modelo        | Quant | HW       | VRAM  | TPS   | PPL   | ENEM% | MBU  |
|---------------|-------|----------|-------|-------|-------|-------|------|
| Qwen3-8B      | FP16  | L4       | 16 GB | 35    | 6.8   | 65%   | 0.52 |
| Qwen3-8B      | Q8_0  | T4       | 9 GB  | 55    | 6.9   | 64%   | 0.49 |
| Qwen3-8B      | Q5_1  | AMD 8GB  | 6 GB  | 40    | 7.2   | 60%   | 0.35 |
| Qwen3-8B      | Q4_0  | CPU ARM  | 5 GB  | 8     | 7.8   | 55%   | 0.12 |
```

### 5.3 Ordem de Execução Recomendada

1. **Converter todos os modelos para FP16/BF16 GGUF**
2. **Rodar baseline FP16 na L4/L40S** (`python -m scripts.benchmark_cuda`)
3. **Quantizar todos os modelos** (Q8_0, Q5_1, Q5_0, Q4_1, Q4_0)
4. **Rodar quantizações na L4/L40S** (para comparação direta de qualidade)
5. **Copiar modelos quantizados para cada hardware alvo**
6. **Rodar benchmark em cada hardware** (AMD, T4, CPU)
7. **Comparar resultados** usando a tabela acima

---

## 6. Checklist Pré-Execução

- [ ] llama.cpp compilado com arquitetura CUDA correta (`-DCMAKE_CUDA_ARCHITECTURES=89` para L4/L40S)
- [ ] Modelo GGUF presente em `benchmark/model/`
- [ ] Nome do arquivo GGUF contém a família do modelo (qwen, llama, deepseek, sabia)
- [ ] `.env.<backend>` configurado corretamente
- [ ] `FLASH_ATTN=true` **apenas** para GPUs que suportam (L4, L40S, A100, H100)
- [ ] `N_GPU_LAYERS=99` para GPUs com VRAM suficiente
- [ ] `HARDWARE_BANDWIDTH_GBS` ajustado para o hardware real
- [ ] Dry-run funciona: `python -m scripts.benchmark_cuda --dry-run`
- [ ] Datasets presentes em `benchmark/datasets/`

---

## 7. Troubleshooting

### 100% invalid_answer_rate
**Causa**: O modelo não está seguindo o formato `FINAL_ANSWER: <resposta>`.
**Solução**: Verifique se o nome do arquivo GGUF contém a família correta (qwen, llama, deepseek, sabia). O chat template é aplicado automaticamente com base no nome.

### TPS muito baixo (< 5 t/s em GPU)
**Causa provável**: `N_GPU_LAYERS` muito baixo ou Flash Attention desativado.
**Solução**: `N_GPU_LAYERS=99` e `FLASH_ATTN=true` (se GPU suportar).

### OOM (Out of Memory)
**Causa**: Modelo FP16 muito grande para a VRAM disponível.
**Solução**: Reduza `N_GPU_LAYERS` ou use quantização. FP16 8B requer ~16GB + KV cache.

### Modelo gera lixo ou repete tokens
**Causa**: Temperatura muito alta ou modelo não instrução-tuned.
**Solução**: Use `TEMP=0.0` e modelos **Instruct** (não base).

### DeepSeek gera `<think>` muito longo
**Causa**: Modelo de raciocínio com `N_PREDICT=64` insuficiente.
**Solução**: Aumente `N_PREDICT=256` no `.env`. O `_sanitize_output` remove blocos `<think>` automaticamente.
