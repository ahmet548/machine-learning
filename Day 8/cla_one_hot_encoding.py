from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRFRegressor
import pandas as pd

df = pd.read_csv("cla.csv")

df = df.drop('sehir', axis=1)

ohe = OneHotEncoder()
xd = ohe.fit_transform(df[['renk', 'yakit', 'paket']]).toarray()

columns_name = ohe.get_feature_names_out()

df[columns_name] = xd

df = df.drop(columns=['renk', 'yakit', 'paket'])

y = df['fiyat']
x = df.drop('fiyat', axis=1)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=42,train_size=0.77
)

forest = RandomForestRegressor()
fModel = forest.fit(x_train,y_train)

xgb = XGBRFRegressor()
xgbModel = xgb.fit(x_train,y_train)

example_value = x.head(2).values

print(f"XGBRF Regressor Score: {xgbModel.score(x_test,y_test)} --> {xgbModel.predict(example_value)} \nRandom Forest Regressor Score: {fModel.score(x_test,y_test)} --> {fModel.predict(example_value)}")

print(df.info())