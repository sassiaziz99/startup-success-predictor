import pandas as pd
df = pd.read_csv("data/labeled_dataset.csv")
print(df.isnull().sum())
print(df.head())