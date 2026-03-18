import pandas as pd

# 1. Carregar o arquivo JSONL
df = pd.read_json("./dataset/bbq/Race_ethnicity.jsonl", lines=True)

# 2. Limpeza (opcional, mas recomendada para benchmarks)
# Remove duplicatas e linhas vazias para não enviesar o benchmark
#df = df.dropna(subset=['text']).drop_duplicates()

# 3. Salvar como Parquet
# O motor 'pyarrow' é o padrão ouro para compatibilidade
df.to_parquet(
    "./dataset/bbq/Race_ethnicity.parquet", 
    engine="pyarrow", 
    compression="snappy", # Snappy oferece bom equilíbrio entre velocidade e tamanho
    index=False
)