"""
Pipeline CLI para avaliação de LLMs em datasets de múltipla escolha.

Exemplos de uso
---------------
# Rodar llama3 no validation (100 amostras, saída não-estruturada):
python -m src.main --model llama3.1:latest --dataset validation --samples 100

# Rodar gpt-4o-mini com saída estruturada:
python -m src.main --model gpt-4o-mini --provider openai --dataset validation --samples 50 --structured

# Usar arquivo parquet direto:
python -m src.main --model phi:latest --dataset datasets/arc_challenge_test.parquet --samples 200

# Salvar em pasta específica:
python -m src.main --model llama3.1:latest --dataset train --output results/meus_resultados

# Usar modelo diferente como juiz:
python -m src.main --model deepseek-r1:8b --dataset validation --judge gpt-4o-mini --judge-provider openai
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Garante que a raiz do projeto está no sys.path, tanto ao rodar com
# `python src/main.py` quanto com `python -m src.main`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_model
from src.pipeline import load_dataset, run_dataset, save_results


def _bool(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia LLMs em datasets de múltipla escolha (ARC Challenge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Modelo principal
    parser.add_argument(
        "--model",
        default="llama3.1:latest",
        help="Nome do modelo a avaliar (padrão: llama3.1:latest).",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="Provedor do modelo principal (padrão: ollama).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de geração (padrão: 0.0).",
    )

    # Modelo juiz (para extração de resposta em modo não-estruturado)
    parser.add_argument(
        "--judge",
        default="llama3.1:latest",
        help="Modelo usado como juiz para extrair a resposta (padrão: llama3.1:latest).",
    )
    parser.add_argument(
        "--judge-provider",
        default="ollama",
        choices=["ollama", "openai"],
        help="Provedor do modelo juiz (padrão: ollama).",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="validation",
        help="Dataset a usar: 'train', 'validation', 'test' ou caminho para .parquet (padrão: validation).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Número absoluto de amostras a processar.",
    )
    parser.add_argument(
        "--pct",
        type=float,
        default=10.0,
        metavar="PERCENT",
        help="Porcentagem do dataset a usar, ex: 0.1 (10%%) ou 50 (50%%) (padrão: 10). "
        "Ignorado se --samples for fornecido.",
    )

    # Modo de resposta
    parser.add_argument(
        "--structured",
        type=_bool,
        default=True,
        metavar="BOOL",
        help="Saída estruturada JSON (true/false, padrão: true).",
    )
    parser.add_argument(
        "--critique",
        type=_bool,
        default=True,
        metavar="BOOL",
        help="Gera auto-crítica após cada resposta (true/false, padrão: true).",
    )

    # Saída
    parser.add_argument(
        "--output",
        default="results",
        help="Pasta onde salvar os resultados (padrão: results/).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa logs de debug na extração de respostas.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Modelo     : {args.model} [{args.provider}]")
    print(f"  Juiz       : {args.judge} [{args.judge_provider}]")
    if args.samples:
        size_label = f"{args.samples} amostras"
    elif args.pct is not None:
        pct_val = args.pct if args.pct > 1 else args.pct * 100
        size_label = f"{pct_val:.1f}%"
    else:
        size_label = "completo"
    print(f"  Dataset    : {args.dataset} ({size_label})")
    print(f"  Estruturado: {args.structured}")
    print(f"  Critique   : {args.critique}")
    print(f"  Saída      : {args.output}/")
    print(f"{'=' * 60}\n")

    # Carrega modelos
    print("Carregando modelos...")
    model = load_model(args.model, args.provider, args.temperature)
    judge = load_model(args.judge, args.judge_provider, temperature=0.0)

    # Carrega dataset
    print(f"Carregando dataset '{args.dataset}'...")
    dataset = load_dataset(args.dataset, samples=args.samples, pct=args.pct)
    print(f"  {len(dataset)} amostras carregadas.\n")

    # Roda o pipeline
    results = run_dataset(
        model=model,
        dataset=dataset,
        judge=judge,
        structured=args.structured,
        debug=args.debug,
        generate_critique_flag=args.critique,
    )

    # Salva e exibe resumo
    dataset_label = (
        args.dataset.replace("/", "_").replace("\\", "_").replace(".parquet", "")
    )
    summary, json_file, csv_file = save_results(
        results,
        output_path=args.output,
        model_name=args.model,
        dataset_name=dataset_label,
    )

    print(f"\n{'=' * 60}")
    print("  RESULTADOS")
    print(f"  Modelo   : {summary['model']}")
    print(f"  Dataset  : {summary['dataset']}")
    print(f"  Total    : {summary['total']}")
    print(f"  Corretos : {summary['correct']}")
    print(f"  Acurácia : {summary['accuracy']:.1%}")
    print(f"\n  JSON salvo em : {json_file}")
    print(f"  CSV  salvo em : {csv_file}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
