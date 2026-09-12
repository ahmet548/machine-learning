from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pandas as pd

df = pd.read_csv("cla.csv")

# Preprocessing
df = df.drop("sehir", axis=1)

le = LabelEncoder()
df["renk"] = le.fit_transform(df["renk"])
df["yakit"] = le.fit_transform(df["yakit"])
df["paket"] = le.fit_transform(df["paket"])

print(f"Colors = {df['renk'].unique()}, \nFuel Type = {df['yakit'].unique()} \nPackage = {df['paket'].unique()}")

##

y = df['fiyat']
x = df.drop('fiyat', axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.77, random_state=6
)

rf = RandomForestRegressor(n_estimators=200)
rfModel = rf.fit(x_train, y_train)

lr = LinearRegression()
lrModel = lr.fit(x_train, y_train)

xgb = XGBRegressor()
xgbModel = xgb.fit(x_train, y_train)


print(df.info())
print(df.head(3))

example_value = x.head()

print(f"Linear Regression Score: {lrModel.score(x_test,y_test)} --> {lrModel.predict(example_value)} \nRandom Forest Score: {rfModel.score(x_test,y_test)} --> {rfModel.predict(example_value)} \nXGBoost Score: {xgbModel.score(x_test,y_test)} --> {xgbModel.predict(example_value)}")
 