# PoETaV2: funcionamento do benchmark e metricas geradas

Este documento descreve, com base no codigo atual do projeto, como o benchmark PoETaV2 funciona de ponta a ponta e quais metricas sao analisadas/geradas.

Tambem considera o contexto dos datasets convertidos em `py_convert_dataset/dataset_jsonl`.

## 1) Visao geral da execucao

No fluxo padrao, o benchmark segue este caminho:

1. `main.py` recebe argumentos de execucao (modelo, tarefas, few-shot, prompt mode, output etc.).
2. `main.py` chama `lm_eval.evaluator.simple_evaluate(...)`.
3. O evaluator instancia o modelo (`lm_eval.models`) e as tarefas (`lm_eval.tasks`).
4. Para cada tarefa:
   - monta contexto few-shot (`fewshot_context`);
   - gera requests para o modelo (`loglikelihood`, `greedy_until` ou `loglikelihood_rolling`);
   - calcula metricas por exemplo (`process_results`).
5. O evaluator agrega metricas por tarefa/prompt (`aggregation`) e inclui `num_examples`.
6. O resultado final e salvo em JSON (`--output_path`) e impresso em formato tabela.

Arquivos principais desse fluxo:

- `main.py`
- `lm_eval/evaluator.py`
- `lm_eval/base.py`
- `lm_eval/tasks/*.py`
- `lm_eval/metrics.py`

## 2) Modos de execucao

### Execucao de tarefa unica

Comando tipico:

```bash
python main.py \
  --model gpt \
  --model_args pretrained=SEU_MODELO \
  --tasks assin_rte_greedy \
  --num_fewshot 2 \
  --prompt_modes dynamic-random \
  --output_path resultados.json \
  --description_dict_path description.json \
  --no_cache
```

### Execucao em lote (benchmark completo)

Script:

```bash
python scripts/bulk_evaluation.py \
  --model_config configs/hf_model_template.json \
  --task_configs configs/poeta_v2_full.json \
  --experiment_name meu_experimento
```

Esse fluxo gera resultados por tarefa e depois calcula o NPM (Normalized Preferred Metric) com `scripts/calculate_npm.py`.

## 3) Estrutura do resultado gerado

O JSON final segue, em alto nivel:

```json
{
  "results": {
    "nome_tarefa": {
      "prompt_mode": {
        "metrica_1": ...,
        "metrica_1_stderr": ...,
        "metrica_2": ...,
        "num_examples": ...
      }
    }
  },
  "versions": { "nome_tarefa": ... },
  "config": {
    "model": "...",
    "model_args": "...",
    "num_fewshot": ...,
    "batch_size": ...,
    "no_cache": ...,
    "limit": ...
  }
}
```

Pontos importantes:

- `num_examples` e adicionado por tarefa/prompt no fim da avaliacao.
- Para algumas metricas, o evaluator calcula `*_stderr` via bootstrap.
- Se `output_dir` for passado, amostras de geracao podem ser gravadas em `*_samples.txt`.

## 4) Como as metricas sao calculadas

Cada tarefa define:

- `process_results(doc, results)`: metrica por exemplo.
- `aggregation()`: como agregar a lista de exemplos.
- `higher_is_better()`: direcao da metrica.

Ou seja, as metricas do PoETaV2 nao estao em um unico ponto: elas nascem nas tarefas e sao agregadas no evaluator.

## 5) Principais familias de metricas no PoETaV2

Com base em `configs/poeta_v2_full.json` (44 tarefas):

- `acc` (predominante)
- `f1`, `f1-macro`, `f1-weighted`
- `exact` (QA estilo SQuAD/FaQuAD)
- `mse`, `pearson` (STS)
- metricas especificas de MKQA: `best_em`, `best_f1`, `best_answerable_em`, `best_answerable_f1`, `best_unanswerable_em`, `best_f1_threshold`
- metricas por subconjunto em algumas tarefas (ex.: anos do ENEM, subconjuntos do BLUEX, BK etc.)

Resumo observado no config:

- 44 tarefas
- 28 nomes de metricas no total
- `acc` aparece em 41 tarefas

## 6) Exemplos concretos de metricas por tarefa

### Classificacao (assin_rte, agnews_pt, tweetsentbr, massive etc.)

- `acc` (media da acuracia)
- `f1-macro` e/ou `f1-weighted` (quando aplicavel)

### Similaridade semantica (assin_sts)

- `mse` (quanto menor, melhor)
- `pearson` (quanto maior, melhor)

### QA extrativo (faquad)

- `exact` (exact match normalizado)
- `f1` (sobreposicao de tokens)

### QA com resposta/sem resposta (mkqa)

- conjunto de metricas `best_*` calculadas por agregador custom.

### Multiplas escolhas com recortes por grupo

- ENEM e BLUEX geram `acc` global e metricas por ano/subconjunto.

## 7) Prompt modes e impacto na avaliacao

O projeto suporta prompt modes como:

- `dynamic-random`
- `fixed`
- `manual`

As metricas sao armazenadas por prompt mode. Exemplo:

`results["assin_rte_greedy"]["dynamic-random"]["acc"]`

## 8) NPM (Normalized Preferred Metric)

Depois da avaliacao por tarefa, o projeto calcula NPM em `scripts/calculate_npm.py`.

Formula por tarefa:

```text
normalized_metric = 100 * (raw_metric - random_score) / (max_score - random_score)
```

Depois disso:

- `NPM.All`: media de todas as tarefas
- `NPM.Translated`: media das tarefas marcadas como translated
- `NPM.Native`: media das nao traduzidas

Saida escrita em:

- `results_folder/npm.json`

Com campos:

- `NPM.All`
- `NPM.Translated`
- `NPM.Native`
- `NPMPerTask`

## 9) Relacao com os datasets convertidos em JSONL

Voce ja gerou datasets em:

- `py_convert_dataset/dataset_jsonl`

Eles cobrem:

- `arithmetic`, `coqa`, `gsm8k`, `logiqa`, `triviaqa`, `wikitext`

Esses arquivos sao base de dados. Para entrarem no benchmark PoETaV2 "oficial" com metricas no mesmo pipeline, o passo natural e criar/ajustar tarefas em `lm_eval/tasks/` que:

1. leiam esses JSONL;
2. implementem `process_results`;
3. definam `aggregation` e `higher_is_better`;
4. sejam registradas em `lm_eval/tasks/__init__.py`;
5. sejam adicionadas ao config (ex.: `configs/poeta_v2_full.json` ou um config custom).

## 10) Artefatos finais esperados em uma execucao completa

- JSON de resultado por tarefa (`*.json`)
- Tabela impressa no terminal (markdown)
- Arquivo `npm.json` com NPM global/segmentado
- (opcional) logs em Weights & Biases, quando habilitado
- (opcional) arquivos `*_samples.txt` com prompts/predicoes para auditoria

## Referencias de codigo

- `main.py`
- `lm_eval/evaluator.py`
- `lm_eval/base.py`
- `lm_eval/metrics.py`
- `lm_eval/tasks/assin.py`
- `lm_eval/tasks/faquad.py`
- `lm_eval/tasks/mkqa.py`
- `lm_eval/tasks/enem.py`
- `lm_eval/tasks/bluex.py`
- `scripts/bulk_evaluation.py`
- `scripts/calculate_npm.py`
- `configs/poeta_v2_full.json`
