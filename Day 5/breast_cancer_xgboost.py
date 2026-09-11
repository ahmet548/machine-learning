from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd

df = pd.read_csv("breast_cancer.csv")

df = pd.get_dummies(df, "diagnosis", drop_first=True)
df = df.drop("id", axis=1)

y = df["diagnosis_M"]
x = df.drop("diagnosis_M", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=6,train_size=0.77)

rf = RandomForestClassifier(n_estimators=200, max_depth=4)
rfModel = rf.fit(x_train,y_train)

xgb = xgb.XGBClassifier()
xgbModel = xgb.fit(x_train,y_train)

df.info()

print(f"Random forest score: {rfModel.score(x_test,y_test)} \nXGBOOST score: {xgbModel.score(x_test,y_test)}")

example_value = x.head(1).values

print(f"Data for example: {example_value} \nfor Random Forest Classifier: {rfModel.predict(example_value)} \nfor XGBOOST: {xgbModel.predict(example_value)}")


