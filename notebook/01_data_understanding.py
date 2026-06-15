import pandas as pd
# Dataset Load
df = pd.read_csv("dataset/European_Bank.csv")
print("Dataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nFirst 5 Rows:")
print(df.head())