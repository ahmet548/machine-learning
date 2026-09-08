from sklearn.linear_model import LinearRegression
import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('audi.csv')
print(df.head(3))

df = df.drop(columns=['index', 'href', 'MileageRank', 'PriceRank', 'PPYRank', 'Score'])
print("--------------------------------")
print(df.head(3))

df['Engine'] = df['Engine'].str.replace('L', '')
df['Engine'] = pd.to_numeric(df['Engine'])

df = pd.get_dummies(df, columns=['Type', 'Transmission', 'Fuel'], drop_first=True)


y = df[['Price(£)']]
x = df.drop('Price(£)', axis=1)

print("--------------------------------")
print(df.head(3))

l = LinearRegression()
model = l.fit(x, y)

print("--------------------------------")
print("Model Score: ", model.score(x, y))
print("Model Prediction: ", model.predict([[2020, 20000, 2.0, 1, 0, 1, 0, 1]]))

