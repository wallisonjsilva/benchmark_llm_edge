"""
convert_json_to_jsonl.py
------------------------
Converte os arquivos JSON dos datasets de PoETaV2/lm_eval/datasets
para o formato JSONL (uma linha por objeto JSON).

Estrutura esperada:
  PoETaV2/lm_eval/datasets/<nome_dataset>/dataset_infos.json

Saída gerada em:
  dataset/poetav2/<nome_dataset>_dataset_infos.jsonl

Uso:
  python utils/convert_json_to_jsonl.py

  # Converter apenas datasets específicos:
  python utils/convert_json_to_jsonl.py --datasets gsm8k lambada logiqa

  # Especificar diretórios customizados:
  python utils/convert_json_to_jsonl.py \
      --source PoETaV2/lm_eval/datasets \
      --output dataset/poetav2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Raiz do projeto (pasta acima de utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE = PROJECT_ROOT / "PoETaV2" / "lm_eval" / "datasets"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "poetav2"


def convert_json_to_jsonl(
    json_path: Path,
    jsonl_path: Path,
) -> int:
    """
    Converte um arquivo JSON para JSONL.

    - Se o JSON for um dict, cada valor de chave top-level vira uma linha,
      com a chave adicionada como campo ``"_config_name"``.
    - Se o JSON for uma lista, cada elemento vira uma linha.
    - Qualquer outro tipo é emitido como uma única linha.

    Retorna o número de linhas escritas.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    lines_written = 0
    with open(jsonl_path, "w", encoding="utf-8") as out:
        if isinstance(data, dict):
            for config_name, record in data.items():
                if isinstance(record, dict):
                    record = {"_config_name": config_name, **record}
                else:
                    record = {"_config_name": config_name, "value": record}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                lines_written += 1
        elif isinstance(data, list):
            for record in data:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                lines_written += 1
        else:
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
            lines_written = 1

    return lines_written


def convert_all(
    source_dir: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
) -> dict[str, int]:
    """
    Itera sobre as subpastas de ``source_dir``, localiza cada
    ``dataset_infos.json`` e converte para JSONL em ``output_dir``.

    Parâmetros
    ----------
    source_dir:
        Diretório raiz com as subpastas de datasets.
    output_dir:
        Diretório de saída para os arquivos ``.jsonl``.
    datasets:
        Lista de nomes de datasets a processar. ``None`` processa todos.

    Retorna um dict {nome_dataset: linhas_escritas}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    subdirs = sorted(
        d for d in source_dir.iterdir() if d.is_dir()
    )

    if datasets:
        subdirs = [d for d in subdirs if d.name in datasets]
        missing = set(datasets) - {d.name for d in subdirs}
        for name in missing:
            log.warning("Dataset '%s' não encontrado em %s", name, source_dir)

    if not subdirs:
        log.error("Nenhum dataset encontrado em: %s", source_dir)
        return results

    for dataset_dir in subdirs:
        json_file = dataset_dir / "dataset_infos.json"
        if not json_file.exists():
            log.warning("Skipping '%s': dataset_infos.json não encontrado", dataset_dir.name)
            continue

        jsonl_file = output_dir / f"{dataset_dir.name}_dataset_infos.jsonl"

        try:
            n = convert_json_to_jsonl(json_file, jsonl_file)
            log.info("✓ %-30s → %s  (%d linhas)", dataset_dir.name, jsonl_file.name, n)
            results[dataset_dir.name] = n
        except Exception as exc:  # noqa: BLE001
            log.error("✗ %-30s → ERRO: %s", dataset_dir.name, exc)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte dataset_infos.json de PoETaV2 para JSONL.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Diretório fonte com subpastas de datasets (padrão: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Diretório de saída para arquivos JSONL (padrão: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        metavar="DATASET",
        help="Nomes dos datasets a converter (padrão: todos)",
    )
    args = parser.parse_args()

    log.info("Fonte  : %s", args.source)
    log.info("Saída  : %s", args.output)

    if not args.source.exists():
        log.error("Diretório fonte não existe: %s", args.source)
        sys.exit(1)

    results = convert_all(args.source, args.output, datasets=args.datasets)

    total = sum(results.values())
    log.info("─" * 50)
    log.info("Concluído: %d datasets convertidos, %d linhas totais", len(results), total)


if __name__ == "__main__":
    main()
