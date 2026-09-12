from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("tdf.csv")

print(df.info())

print(df["Team"].unique())

le = LabelEncoder()
df["Team"] = le.fit_transform(df["Team"])

print(df["Team"].unique())