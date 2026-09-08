from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv("audi.csv")

print(df.head(3))

df = df.drop(columns=['index', 'href', 'MileageRank', 'PriceRank', 'PPYRank', 'Score'])

df["Engine"] = df["Engine"].str.replace("L", "")
df["Engine"] = pd.to_numeric(df["Engine"])

df = pd.get_dummies(df, columns=["Type", "Transmission", "Fuel"], drop_first=True)

y = df[["Price(£)"]]
x = df.drop("Price(£)", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

l = LinearRegression()
model = l.fit(x_train, y_train)
model_score = model.score(x_test, y_test)

print("--------------------------------")
print("Model Score: ", model_score)

model_prediction = model.predict([[2020, 20000, 2.0, 1, 0, 1, 0, 1]])
print("Model Prediction: ", model_prediction)