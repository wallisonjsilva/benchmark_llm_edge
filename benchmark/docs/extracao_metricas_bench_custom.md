# Extração de métricas para benchmark customizado (llama.cpp)

Este guia define como separar corretamente **desempenho** e **qualidade** no seu benchmark, usando:

- `llama-bench` para throughput/latência de inferência;
- `llama-perplexity` para PPL em `wikitext`;
- `llama-cli` (ou `llama-completion`) para respostas dos datasets e cálculo de qualidade.

Também explica por que em `output/old/saida_benchmark_poetav2.json` as métricas de dataset ficaram em `0.0`.

---

## 1) O que cada binário mede (e o que não mede)

## `llama-bench`

Mede desempenho da engine (`t/s`, `avg_ns`, `stddev`, etc.) em testes `pp`, `tg`, `pg`.

Ponto crítico do README:

- as medições **não incluem tokenização nem sampling**;
- não mede acurácia/F1/EM de dataset;
- saída ideal para parser: `-o json` (ou `jsonl`, `sql`).

Conclusão: use para **performance baseline**, não para corretude semântica.

## `llama-perplexity`

Mede PPL de LM em corpus (convencionalmente Wikitext-2).

Pontos críticos do README:

- PPL menor é melhor (no mesmo tokenizer/setup);
- bom para comparar perda de qualidade de quantização vs FP16;
- opcionalmente pode medir KL com `--kl-divergence-base` e `--kl-divergence`.

Conclusão: use para **qualidade intrínseca de linguagem** (especialmente `poetav2_wikitext`).

## `llama-cli`

Gera respostas para prompts reais de tarefa (ENEM/BBQ/PoETaV2).

Pontos úteis para benchmark:

- `--show-timings` e `--perf` para timing interno;
- fixar sampling (`--seed`, `--temp`, `--top-k`, `--top-p`, `--repeat-last-n`);
- controlar contexto e geração (`--ctx-size`, `--n-predict`).

Conclusão: use para **coletar predições** e calcular métricas de qualidade por dataset.

---

## 2) Análise do JSON zerado (`saida_benchmark_poetav2.json`)

No arquivo analisado:

- `inference_success_rate = 1.0` (inferência executou);
- `invalid_answer_rate_enem = 1.0`;
- acurácias/F1/EM dos datasets estão em `0.0`;
- `perplexity_poetav2_wikitext = 7.7458` (PPL foi calculada).

Leitura técnica: o pipeline rodou, mas o parser de resposta não extraiu/validou saídas corretas para os datasets de QA/classificação.  
Então as métricas não estão “faltando”; elas foram computadas como erro total (ou resposta inválida) na amostra usada.

Possíveis causas práticas:

- formato de prompt não forçou resposta canônica (ex.: só letra A-E, só `0/1/2`, número final);
- saída do modelo veio com raciocínio longo e sem marcador final consistente;
- regex de extração não casou com o estilo de resposta do modelo atual;
- `sample_size` pequeno amplifica efeito (10 exemplos).

---

## 3) Estratégia proposta: faz sentido?

Sua ideia:

- `llama-bench` com dataset `poetav2/wikitext`;
- `llama-perplexity` para PPL no wikitext;
- `llama-cli` com todos datasets para trade-off usando `sample size`.

Avaliação: **sim, faz sentido**, com um ajuste de framing:

- `llama-bench` não usa dataset semântico; ele só roda cenários de tokens (`-p/-n/-pg`).
- então o “wikitext no llama-bench” deve ser entendido como **configuração de carga equivalente** ao perfil desejado (ex.: `n_prompt` próximo do tamanho típico de contexto), não como avaliação de conteúdo do wikitext.

Pipeline recomendado:

1. **Performance**: `llama-bench` (`-o json`, reps >= 3)  
   métricas: `gen_tps`, `prompt_tps`, `ttft` aproximado, variância.

2. **Qualidade LM**: `llama-perplexity` em Wikitext  
   métricas: `perplexity`, e opcionalmente `kld`/`Δppl` para quantização.

3. **Qualidade de tarefa**: `llama-cli` + parser + scorer por dataset  
   métricas: ENEM accuracy, BBQ accuracy/bias, PoETaV2 (LogiQA/GSM8K/CoQA/TriviaQA/Arithmetic).

4. **Trade-off**: cruzar `t/s` vs `score_poetav2_macro` e `perplexity`.

---

## 4) Mapeamento de métricas por ferramenta

## De `llama-bench` (JSON)

- `avg_ts` (t/s médio por teste)
- `stddev_ts`
- `avg_ns`, `stddev_ns`
- metadados de execução (`n_prompt`, `n_gen`, `n_batch`, `n_gpu_layers`, backend, etc.)

Transformações sugeridas no runner:

- `avg_tps` := `avg_ts` do teste `tg`;
- `peak_tps` := maior `samples_ts` observado;
- `llama_bench_time_s` := duração total da chamada do bench.

## De `llama-perplexity`

- `perplexity` global (parse de `Final estimate: PPL = ...` ou fallback robusto).
- opcional: `kld`, `delta_ppl`, correlação de probabilidades (quando usar base logits).

## De `llama-cli`/`llama-completion`

Saída bruta por amostra:

- texto gerado;
- tempo total inferência;
- timeout/abort/erro;
- uso de RAM/VRAM (se monitorado externamente).

Após parser/scorer:

- ENEM: `accuracy_enem_20xx`, `invalid_answer_rate_enem`;
- BBQ: `accuracy_bbq_*`, `accuracy_bbq_ambig`, `accuracy_bbq_disambig`, `bias_score_bbq`;
- PoETaV2: `accuracy_poetav2_logiqa`, `exact_match_poetav2_gsm8k`, `f1_poetav2_coqa`, `exact_match_poetav2_triviaqa`, `exact_match_poetav2_arithmetic`.

---

## 5) Comandos-base (exemplos)

## 5.1 `llama-bench` (desempenho)

```bash
./llama-bench \
  -m /caminho/model.gguf \
  -o json \
  -r 3 \
  -p 256 \
  -n 128 \
  -t 8 \
  -ngl 20
```

## 5.2 `llama-perplexity` (wikitext)

```bash
./llama-perplexity \
  -m /caminho/model.gguf \
  -f /caminho/wikitext.txt
```

Com comparação FP16 vs quantizado (opcional):

```bash
# 1) gerar logits base
./llama-perplexity -m /caminho/model-fp16.gguf -f /caminho/wikitext.txt --kl-divergence-base /tmp/base.kld

# 2) comparar quantizado contra base
./llama-perplexity -m /caminho/model-q4.gguf -f /caminho/wikitext.txt --kl-divergence-base /tmp/base.kld --kl-divergence
```

## 5.3 `llama-cli` (qualidade por dataset)

```bash
./llama-cli \
  -m /caminho/model.gguf \
  -p "PERGUNTA...\nResponda apenas com: A, B, C, D ou E." \
  --seed 42 \
  --temp 0.2 \
  --top-k 40 \
  --top-p 0.95 \
  -c 4096 \
  -n 64 \
  --show-timings \
  --perf \
  -st
```

---

## 6) Regras para evitar métricas zeradas

1. Padronize prompt de saída final por dataset.
   - Use um marcador único: `FINAL_ANSWER: ...`.
   - ENEM: `FINAL_ANSWER: A|B|C|D|E`.
   - BBQ: `FINAL_ANSWER: 0|1|2`.
   - Numéricos: `FINAL_ANSWER: <numero>`.
   - Span curto: `FINAL_ANSWER: <texto>`.

2. Ajuste parser para “answer cue”.
   - priorize `FINAL_ANSWER: X`;
   - mantenha fallback para `Resposta final: X` e `Final answer: X`.

3. Logue e audite amostras inválidas.
   - guardar `prompt`, `raw_output`, `prediction`, `expected`, `valid`.

4. Use `sample_size` progressivo.
   - smoke: 3–5;
   - calibração parser: 20–50;
   - comparação final: tamanho fixo para todos modelos.

5. Não misture métricas de performance com qualidade.
   - `metrics_source=llama-bench` para desempenho;
   - qualidade vem do scorer de datasets.

---

## 7) Curva de trade-off recomendada

Para cada modelo/quant:

- eixo X: `avg_tps` (`llama-bench`);
- eixo Y1: `score_poetav2_macro` (`llama-cli` + scorer);
- eixo Y2: `perplexity` (`llama-perplexity`, menor melhor).

Regra prática:

- modelos dominados (menor t/s e pior qualidade) podem ser descartados;
- mantenha candidatos de fronteira de Pareto para decisão final.

---

## 8) Checklist rápido de execução

- `llama-bench` com `-o json` e `-r >= 3`.
- `llama-perplexity` em wikitext com parse validado.
- `llama-cli` com sampling fixo + prompt canônico por dataset.
- `sample_size` idêntico entre modelos ao comparar.
- auditoria de inválidos antes de confiar em acurácia.
