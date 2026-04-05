# PoETaV2 — Resumo executivo (1 pagina)

## O que e o PoETaV2

O PoETaV2 e um benchmark para avaliar modelos de linguagem em portugues em um conjunto amplo de tarefas (mais de 40), cobrindo classificacao, inferencia, similaridade semantica, QA e raciocinio de multipla escolha.

O objetivo e medir desempenho de forma comparavel entre modelos, com metricas por tarefa e um indicador agregado de benchmark (NPM).

## Como o benchmark funciona (visao de negocio)

O fluxo operacional e direto:

1. Define-se o modelo e o conjunto de tarefas (configuracoes em JSON).
2. O sistema executa cada tarefa com um esquema de prompt (few-shot).
3. Para cada exemplo, o modelo gera uma resposta ou pontuacao.
4. O benchmark calcula metricas especificas de cada tarefa.
5. Ao final, consolida os resultados em:
   - relatorio por tarefa;
   - score agregado NPM.

No projeto, isso acontece principalmente via:

- `main.py` (entrypoint)
- `lm_eval/evaluator.py` (orquestracao)
- `lm_eval/tasks/*` (logica de cada tarefa)
- `scripts/calculate_npm.py` (agregacao final do benchmark)

## Quais metricas sao analisadas

As metricas variam por tipo de tarefa. Principais grupos:

- **Classificacao**: `acc`, `f1`, `f1-macro`, `f1-weighted`
- **Similaridade semantica (STS)**: `mse`, `pearson`
- **QA extrativo**: `exact`, `f1`
- **QA com resposta/não-resposta (MKQA)**: `best_em`, `best_f1` e variantes `best_*`
- **Recortes por subconjunto** (em tarefas como ENEM/BLUEX): metricas por ano, prova ou bloco

No conjunto oficial atual (`configs/poeta_v2_full.json`):

- 44 tarefas
- 28 nomes de metricas
- `acc` e a metrica dominante (presente na maioria das tarefas)

## Como o score global e gerado (NPM)

O PoETaV2 nao usa apenas media bruta de metricas heterogeneas. Ele normaliza a metrica preferida de cada tarefa usando piso aleatorio e teto esperado:

`NPM_tarefa = 100 * (raw - random_score) / (max_score - random_score)`

Depois agrega em:

- `NPM.All` (todas as tarefas)
- `NPM.Translated` (tarefas traduzidas)
- `NPM.Native` (tarefas nativas)

Isso permite comparacao mais justa entre tarefas com escalas diferentes.

## Entregaveis da execucao

Uma execucao completa gera:

- JSON de resultados por tarefa/prompt
- tabela consolidada no terminal
- `npm.json` com score agregado
- opcionalmente logs em Weights & Biases

## Relacao com seus datasets convertidos

Voce ja possui datasets em JSONL em:

- `py_convert_dataset/dataset_jsonl`

Eles podem ser incorporados ao benchmark criando tarefas em `lm_eval/tasks/` com as mesmas interfaces de metrica/aggregacao usadas hoje. Assim, seus datasets entram no mesmo pipeline de comparacao e score agregado.

## Mensagem-chave

O PoETaV2 combina **avaliacao detalhada por tarefa** com **visao executiva consolidada (NPM)**. Isso permite tomar decisao de modelo com equilibrio entre profundidade tecnica e visao global de performance.
