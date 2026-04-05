# Recomendações práticas de benchmark de quantização para edge (FP16 vs GGUF quantizado)

## Objetivo da validação

Responder de forma objetiva:

**“Quais são os limites práticos da quantização em LLM que trazem ganhos relevantes de eficiência computacional mantendo qualidade aceitável?”**

Cenário alvo:

- comparação entre baseline **FP16** e quantizações **Q8_0, Q5_1, Q5_0, Q4_K_M, Q4_1**;
- foco em decisão de uso em **computação edge**;
- benchmark **não exaustivo**, mas estatisticamente útil para decisão de trade-off.

---

## Quantizações consideradas (ordem de referência)

1. `FP16` (baseline de qualidade e custo)
2. `Q8_0` (queda mínima esperada de qualidade, ganho moderado de eficiência)
3. `Q5_1` / `Q5_0` (trade-off intermediário)
4. `Q4_K_M` (agressivo com boa relação custo/qualidade em muitos casos)
5. `Q4_1` (mais agressivo; validar impacto na qualidade com atenção)

---

## Tamanho de amostra recomendado (não exaustivo)

Para este cenário, o corte recomendado por modelo é **100–150 amostras totais**.

Configuração ideal sugerida:

- `enem_2022`: **15**
- `enem_2023`: **15**
- `bbq_gender_identity`: **10**
- `bbq_physical_appearance`: **10**
- `bbq_race_ethnicity`: **10**
- `poetav2_logiqa`: **10**
- `poetav2_gsm8k`: **10**
- `poetav2_coqa`: **10**
- `poetav2_triviaqa`: **10**
- `poetav2_arithmetic`: **10**
- `poetav2_wikitext`: **1** (bloco para sinal de perplexidade/proxy)

Total: **111 amostras por modelo** (bom equilíbrio custo x confiança).

---

## Critério de aceitação sugerido

Para considerar uma quantização “aceitável para edge”, usar regra conjunta:

1. **Qualidade**
   - queda no score macro principal (`accuracy_enem_macro` e/ou `score_poetav2_macro`) de no máximo **3 a 5 pontos percentuais** vs FP16;
   - sem degradação crítica em tarefas sensíveis (ex.: `poetav2_logiqa`, `gsm8k`, `bbq_disambig`).

2. **Eficiência**
   - ganho de throughput (`avg_tps`) de pelo menos **1.5x** (ideal **2x**);
   - redução relevante de memória (`ram_peak_gb` / `vram_peak_gb`) compatível com o hardware edge alvo.

3. **Estabilidade operacional**
   - `inference_success_rate` alto (ideal próximo de 1.0);
   - sem aumento relevante de `timeouts`, `loop_aborts`, `error_count`.

Se a quantização passar nos 3 blocos acima, ela é candidata forte para produção edge.

---

## Execução gradual por etapas (modelos no driver)

Como os `.gguf` serão disponibilizados gradualmente, recomenda-se:

1. Rodar primeiro `FP16` (baseline).
2. Rodar `Q8_0`.
3. Rodar `Q5_1` e `Q5_0`.
4. Rodar `Q4_K_M`.
5. Rodar `Q4_1` por último.

Em cada etapa, comparar com o baseline FP16 já registrado no mesmo dataset/corte.

---

## Modificações no projeto para cortes por dataset via `.env`

O projeto foi preparado para aceitar cortes específicos por dataset, além dos cortes globais.

### Variáveis globais (fallback)

- `SAMPLE_SIZE_ENEM`
- `SAMPLE_SIZE_BBQ`
- `SAMPLE_SIZE_POETAV2`

### Variáveis específicas por dataset (recomendado para este estudo)

- `SAMPLE_SIZE_ENEM_2022`
- `SAMPLE_SIZE_ENEM_2023`
- `SAMPLE_SIZE_BBQ_GENDER_IDENTITY`
- `SAMPLE_SIZE_BBQ_PHYSICAL_APPEARANCE`
- `SAMPLE_SIZE_BBQ_RACE_ETHNICITY`
- `SAMPLE_SIZE_POETAV2_LOGIQA`
- `SAMPLE_SIZE_POETAV2_GSM8K`
- `SAMPLE_SIZE_POETAV2_COQA`
- `SAMPLE_SIZE_POETAV2_TRIVIAQA`
- `SAMPLE_SIZE_POETAV2_ARITHMETIC`
- `SAMPLE_SIZE_POETAV2_WIKITEXT`

Quando uma variável específica está definida, ela sobrescreve o corte global da família.

---

## Exemplo recomendado de `.env` (corte ideal não exaustivo)

```env
# fallback globais
SAMPLE_SIZE_ENEM=15
SAMPLE_SIZE_BBQ=10
SAMPLE_SIZE_POETAV2=10

# overrides por dataset (recomendado)
SAMPLE_SIZE_ENEM_2022=15
SAMPLE_SIZE_ENEM_2023=15
SAMPLE_SIZE_BBQ_GENDER_IDENTITY=10
SAMPLE_SIZE_BBQ_PHYSICAL_APPEARANCE=10
SAMPLE_SIZE_BBQ_RACE_ETHNICITY=10
SAMPLE_SIZE_POETAV2_LOGIQA=10
SAMPLE_SIZE_POETAV2_GSM8K=10
SAMPLE_SIZE_POETAV2_COQA=10
SAMPLE_SIZE_POETAV2_TRIVIAQA=10
SAMPLE_SIZE_POETAV2_ARITHMETIC=10
SAMPLE_SIZE_POETAV2_WIKITEXT=1
```

---

## Observação importante sobre “resposta correta no llama-bench”

`llama-bench` fornece métricas de desempenho (TPS/latência), mas **não** valida correção semântica de respostas de dataset.

A qualidade (acurácia/F1/EM etc.) continua sendo medida no runner a partir das respostas inferidas por modelo. Isso é esperado e necessário para avaliar o trade-off real.
