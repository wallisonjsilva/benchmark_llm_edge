import pandas as pd

df = pd.read_parquet("./dataset/poetav2/arithmetic_dataset_infos.parquet")

print(df.head())
print(df.columns)
print(df.info())
print(df.describe())