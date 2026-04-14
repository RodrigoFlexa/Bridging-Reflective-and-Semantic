# TODO
# Orientar a instalação do ambiente virtual e instalação das dependências
# Instruções para rodar o código

import os

from datasets import load_dataset

ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

# Criar pasta datasets
os.makedirs("datasets", exist_ok=True)

# Salvar cada split como Parquet (preserva dicionários e listas)
for split_name, split_data in ds.items():
    filepath = f"datasets/arc_challenge_{split_name}.parquet"
    split_data.to_parquet(filepath)
    print(f"Salvo: {filepath} ({len(split_data)} registros)")
