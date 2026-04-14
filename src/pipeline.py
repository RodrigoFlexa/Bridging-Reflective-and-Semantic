"""
Lógica central do pipeline: responder questões e avaliar modelos em datasets.
"""

import csv
import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import datasets as hf_datasets
from src.prompts import (
    CRITIQUE_TEMPLATE,
    FORMATTED_TEMPLATE,
    SIMPLE_TEMPLATE,
    STRUCTURED_CRITIQUE_TEMPLATE,
    answer_json_parser,
    critique_json_parser,
    str_parser,
)
from utils.dataset_utils import (
    format_question,
    get_correct_answer,
    get_correct_answer_text,
)
from utils.extractors_utils import extract_answer

# ---------------------------------------------------------------------------
# Resposta a uma única questão
# ---------------------------------------------------------------------------


def answer_question(model, question: str, structured: bool = False):
    """Invoca o modelo e retorna a resposta (dict se estruturada, str caso contrário)."""
    if structured:
        chain = FORMATTED_TEMPLATE | model | answer_json_parser
    else:
        chain = SIMPLE_TEMPLATE | model | str_parser
    return chain.invoke({"question": question})


def detect_answer(
    response,
    valid_labels=None,
    judge=None,
    structured: bool = False,
    question: str = None,
    debug: bool = False,
) -> str:
    """Extrai a letra da resposta a partir do output do modelo."""
    if valid_labels is None:
        valid_labels = ["A", "B", "C", "D"]

    if structured:
        if isinstance(response, dict) and "alternative" in response:
            return response["alternative"]
        return "N/A"

    return extract_answer(
        judge,
        response,
        valid_labels=valid_labels,
        question=question,
        debug=debug,
    )


def evaluate_answer(predicted: str, correct: str) -> bool:
    return predicted.strip().upper() == correct.strip().upper()


def generate_critique(
    model,
    question: str,
    answer: str,
    correct: str,
    structured: bool = False,
    feedback: str = None,
) -> str:
    if structured:
        chain = STRUCTURED_CRITIQUE_TEMPLATE | model | critique_json_parser
    else:
        chain = CRITIQUE_TEMPLATE | model | str_parser

    return chain.invoke(
        {
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "semantic_facts": "No facts provided",  # Placeholder, to be replaced with actual facts if needed
        }
    )


# ---------------------------------------------------------------------------
# Avaliação em dataset completo
# ---------------------------------------------------------------------------


def run_dataset(
    model,
    dataset,
    judge,
    structured: bool = False,
    debug: bool = False,
    generate_critique_flag: bool = False,
    generate_fact_flag: bool = False,
) -> dict:
    """
    Roda o modelo sobre todas as amostras do dataset.

    Retorna um dicionário indexado por posição com os campos:
      question, answer, correct_answer_alt, predicted_alt,
      correct_answer_text, flag_acerto
    """
    output = {}
    acertos = []
    loop = tqdm(range(len(dataset)), desc="Avaliando")

    for i in loop:
        question = format_question(dataset[i])
        full_answer = answer_question(model, question, structured)

        correct_alt = get_correct_answer(dataset[i])
        correct_text = get_correct_answer_text(dataset[i])
        predicted_alt = detect_answer(
            full_answer,
            judge=judge,
            structured=structured,
            question=question,
            debug=debug,
        )
        acerto = evaluate_answer(predicted_alt, correct_alt)
        acertos.append(acerto)

        output[i] = {
            "question": question,
            "answer": full_answer
            if isinstance(full_answer, str)
            else json.dumps(full_answer),
            "correct_answer_alt": correct_alt,
            "predicted_alt": predicted_alt,
            "correct_answer_text": correct_text,
            "flag_acerto": acerto,
        }

        acc = sum(acertos) / len(acertos)
        loop.set_postfix(acc=f"{acc:.1%}")

        if generate_critique_flag and not acerto:
            # feedback_type = "basic"
            # if feedback_type == "basic":
            #     feedback = "correta" if acerto else "incorreta"
            # elif feedback_type == "detailed":
            #     feedback = f"The correct answer was {correct_alt} ({correct_text}), but you answered {predicted_alt}."

            simple_feedback = "correta" if acerto else "incorreta"
            detailed_feedback = f"The correct answer was {correct_alt} ({correct_text}), but you answered {predicted_alt}."

            critique_basic = generate_critique(
                model, question, full_answer, correct_alt, structured, simple_feedback
            )

            critique_detailed = generate_critique(
                model, question, full_answer, correct_alt, structured, detailed_feedback
            )

            if structured:
                # critique_json_parser já retorna dict puro, não objeto Pydantic
                output[i]["critique"] = critique_detailed.get("critique", "")
                output[i]["scored_facts"] = critique_detailed.get("scored_facts", [])
            else:
                output[i]["critique"] = critique_detailed
    return output


# ---------------------------------------------------------------------------
# Carregamento de dataset
# ---------------------------------------------------------------------------
# Depois personalizaremos para termos datasets específicos, mas por ora só o arc é suficiente para testes.
BUILTIN_DATASETS = {
    "train": "datasets/arc_challenge_train.parquet",
    "validation": "datasets/arc_challenge_validation.parquet",
    "test": "datasets/arc_challenge_test.parquet",
}


def load_dataset(
    dataset_arg: str,
    samples: Optional[int] = None,
    pct: Optional[float] = None,
):
    """
    Carrega um dataset HuggingFace a partir de:
      - alias: 'train', 'validation', 'test'
      - caminho direto para arquivo .parquet

    Prioridade de slicing: --samples > --pct > dataset inteiro.

    pct aceita tanto fração (0.1 = 10%) quanto percentual (10 = 10%).
    """
    path = BUILTIN_DATASETS.get(dataset_arg, dataset_arg)
    ds = hf_datasets.Dataset.from_parquet(path)

    if samples is not None:
        n = min(samples, len(ds))
    elif pct is not None:
        ratio = pct / 100 if pct > 1 else pct
        n = max(1, round(len(ds) * ratio))
    else:
        return ds

    return ds.select(range(n))


# ---------------------------------------------------------------------------
# Salvamento de resultados
# ---------------------------------------------------------------------------


def save_results(results: dict, output_path: str, model_name: str, dataset_name: str):
    """Salva resultados em JSON e um CSV resumo."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    # Metadados
    acertos = [v["flag_acerto"] for v in results.values()]
    accuracy = sum(acertos) / len(acertos) if acertos else 0.0
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "total": len(acertos),
        "correct": sum(acertos),
        "accuracy": round(accuracy, 4),
    }

    # JSON completo
    json_file = path / f"{model_name.replace(':', '_')}_{dataset_name}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results}, f, ensure_ascii=False, indent=2
        )

    # CSV resumo por questão
    csv_file = path / f"{model_name.replace(':', '_')}_{dataset_name}.csv"
    fieldnames = ["index", "correct_answer_alt", "predicted_alt", "flag_acerto"]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in results.items():
            writer.writerow(
                {
                    "index": idx,
                    "correct_answer_alt": row["correct_answer_alt"],
                    "predicted_alt": row["predicted_alt"],
                    "flag_acerto": row["flag_acerto"],
                }
            )

    return summary, json_file, csv_file
