from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("insurance.csv")

df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

y = df["charges"]
x = df.drop("charges", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

lr = LinearRegression()
lModel = lr.fit(x_train, y_train)

fr = RandomForestRegressor(n_estimators=200, max_depth=4)
frModel = fr.fit(x_train, y_train)

df.info()

print(f"LinearRegression Score = {lModel.score(x_test,y_test)}, \nRandomForestRegressor Score = {frModel.score(x_test,y_test)}")

example_value = x.head(1).values
print(f"data for linear regression: {example_value} --> {lModel.predict(example_value)}")
print(f"data for random forest regressor: {example_value} --> {frModel.predict(example_value)}")


