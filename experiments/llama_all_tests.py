import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    model = os.getenv("MODEL", "llama3.1:8b")

    # Lista de datasets
    datasets = ["agievalar-test", "aqua-test", "gsm8k-test", "mmlu-test"]

    # No bash, ${MODEL//:/\_} troca os ":" por "_". Em Python, fazemos assim:
    safe_model_name = model.replace(":", "_")
    output_dir = f"results/{safe_model_name}/all_tests"

    for dataset in datasets:
        print(f"==> {model} | {dataset}")

        # Monta o comando como uma lista de strings (é a forma mais segura no subprocess)
        command = [
            sys.executable,
            "-m",
            "src.main",
            "--model",
            model,
            "--provider",
            "ollama",
            "--dataset",
            dataset,
            "--question-limit",
            "100",
            "--structured",
            "false",
            "--critique",
            "false",
            "--output",
            output_dir,
        ]

        # Executa o comando e espera ele terminar antes de ir para o próximo loop
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao processar o dataset {dataset}: {e}")


if __name__ == "__main__":
    main()
