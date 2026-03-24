from __future__ import annotations

import json
import random  # Adicionado para o shuffle
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

# Semente padrão para garantir reprodutibilidade em todos os hardware/softwares
DEFAULT_SEED = 42

def read_jsonl(path: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if not cleaned:
                continue
            records.append(json.loads(cleaned))
    return records

def _slice_records(records: list[JsonDict], sample_size: int, seed: int = DEFAULT_SEED) -> list[JsonDict]:
    """
    Embaralha os registros usando uma semente fixa e retorna o slice.
    Isso garante que o 'Sample 5' seja sempre o mesmo grupo de 5 itens.
    """
    if sample_size <= 0:
        return records
    
    # Criamos uma instância local do Random para não afetar o estado global do script
    rng = random.Random(seed)
    
    # Fazemos uma cópia para não alterar a lista original por referência
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    
    return shuffled_records[:sample_size]

def _dataset_sample_size(
    dataset_name: str,
    default_sample_size: int,
    sample_size_overrides: dict[str, int] | None,
) -> int:
    if sample_size_overrides and dataset_name in sample_size_overrides:
        return sample_size_overrides[dataset_name]
    return default_sample_size

def load_enem_datasets(
    dataset_root: Path,
    sample_size: int,
    sample_size_overrides: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED
) -> dict[str, list[JsonDict]]:
    enem_dir = dataset_root / "poetav2" / "enem"
    datasets: dict[str, list[JsonDict]] = {}
    if not enem_dir.exists(): return datasets

    for year_file in sorted(enem_dir.glob("[0-9][0-9][0-9][0-9].jsonl")):
        year = year_file.stem
        dataset_name = f"enem_{year}"
        datasets[dataset_name] = _slice_records(
            read_jsonl(year_file),
            _dataset_sample_size(dataset_name, sample_size, sample_size_overrides),
            seed=seed
        )
    return datasets

def load_bbq_datasets(
    dataset_root: Path,
    sample_size: int,
    sample_size_overrides: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED
) -> dict[str, list[JsonDict]]:
    bbq_dir = dataset_root / "bbq"
    return {
        "bbq_gender_identity": _slice_records(
            read_jsonl(bbq_dir / "Gender_identity.jsonl"),
            _dataset_sample_size("bbq_gender_identity", sample_size, sample_size_overrides),
            seed=seed
        ),
        "bbq_physical_appearance": _slice_records(
            read_jsonl(bbq_dir / "Physical_appearance.jsonl"),
            _dataset_sample_size("bbq_physical_appearance", sample_size, sample_size_overrides),
            seed=seed
        ),
        "bbq_race_ethnicity": _slice_records(
            read_jsonl(bbq_dir / "Race_ethnicity.jsonl"),
            _dataset_sample_size("bbq_race_ethnicity", sample_size, sample_size_overrides),
            seed=seed
        ),
    }


def _flatten_coqa_examples(records: list[JsonDict]) -> list[JsonDict]:
    flattened: list[JsonDict] = []
    for entry in records:
        story = str(entry.get("story", ""))
        questions = entry.get("questions", [])
        answers = entry.get("answers", [])
        additional_answers = entry.get("additional_answers", {})

        for idx, question in enumerate(questions):
            question_text = str(question.get("input_text", "")).strip()
            references: list[str] = []

            if idx < len(answers):
                primary = str(answers[idx].get("input_text", "")).strip()
                if primary:
                    references.append(primary)

            for annotator_answers in additional_answers.values():
                if not isinstance(annotator_answers, list):
                    continue
                if idx >= len(annotator_answers):
                    continue
                candidate = str(annotator_answers[idx].get("input_text", "")).strip()
                if candidate:
                    references.append(candidate)

            unique_references: list[str] = []
            seen = set()
            for ref in references:
                normalized = ref.casefold()
                if normalized in seen:
                    continue
                seen.add(normalized)
                unique_references.append(ref)

            if not question_text or not unique_references:
                continue

            flattened.append(
                {
                    "story": story,
                    "question": question_text,
                    "references": unique_references,
                }
            )
    return flattened


def _load_arithmetic_records(arithmetic_dir: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    for jsonl_path in sorted(arithmetic_dir.glob("*.validation.jsonl")):
        for row in read_jsonl(jsonl_path):
            records.append(
                {
                    "context": row.get("context", ""),
                    "completion": row.get("completion", ""),
                    "source_file": jsonl_path.name,
                }
            )
    return records


def load_poetav2_datasets(
    dataset_root: Path,
    sample_size: int,
    sample_size_overrides: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED
) -> dict[str, list[JsonDict]]:
    poetav2_root = dataset_root / "poetav2"
    coqa_rows = read_jsonl(poetav2_root / "coqa" / "validation.jsonl")

    return {
        "poetav2_logiqa": _slice_records(
            read_jsonl(poetav2_root / "logiqa" / "validation.jsonl"),
            _dataset_sample_size("poetav2_logiqa", sample_size, sample_size_overrides),
            seed=seed
        ),
        "poetav2_gsm8k": _slice_records(
            read_jsonl(poetav2_root / "gsm8k" / "test.jsonl"),
            _dataset_sample_size("poetav2_gsm8k", sample_size, sample_size_overrides),
            seed=seed
        ),
        "poetav2_coqa": _slice_records(
            _flatten_coqa_examples(coqa_rows),
            _dataset_sample_size("poetav2_coqa", sample_size, sample_size_overrides),
            seed=seed
        ),
        "poetav2_triviaqa": _slice_records(
            read_jsonl(poetav2_root / "triviaqa" / "validation.jsonl"),
            _dataset_sample_size("poetav2_triviaqa", sample_size, sample_size_overrides),
            seed=seed
        ),
        "poetav2_arithmetic": _slice_records(
            _load_arithmetic_records(poetav2_root / "arithmetic"),
            _dataset_sample_size("poetav2_arithmetic", sample_size, sample_size_overrides),
            seed=seed
        ),
        "poetav2_wikitext": _slice_records(
            read_jsonl(poetav2_root / "wikitext" / "wikitext-2-raw-v1" / "validation.jsonl"),
            _dataset_sample_size("poetav2_wikitext", sample_size, sample_size_overrides),
            seed=seed
        ),
    }

def load_all_datasets(
    dataset_root: Path,
    sample_size_enem: int,
    sample_size_bbq: int,
    sample_size_poetav2: int,
    sample_size_overrides: dict[str, int] | None = None,
    seed: int = DEFAULT_SEED  # Passamos o seed aqui
) -> dict[str, list[JsonDict]]:
    datasets: dict[str, list[JsonDict]] = {}
    datasets.update(load_enem_datasets(dataset_root, sample_size_enem, sample_size_overrides, seed=seed))
    datasets.update(load_bbq_datasets(dataset_root, sample_size_bbq, sample_size_overrides, seed=seed))
    datasets.update(load_poetav2_datasets(dataset_root, sample_size_poetav2, sample_size_overrides, seed=seed))
    return datasets
