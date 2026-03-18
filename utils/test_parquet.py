import pandas as pd

df = pd.read_parquet("./dataset/poetav2/coqa.parquet")

print(df.head())