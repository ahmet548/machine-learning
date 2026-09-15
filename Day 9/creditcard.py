from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv("creditcard.csv")

df = df.drop('Time', axis=1)

y = df['Class']
x = df.drop('Class', axis=1)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=6,train_size=0.9
)

lgbm = LGBMClassifier()
lgbmModel = lgbm.fit(x_train,y_train)

xgb = XGBClassifier()
xgbModel = xgb.fit(x_train,y_train)

exmaple_data = x.head(5).values

print(f"LGBM Classifier: {lgbmModel.score(x_test,y_test)} --> {lgbmModel.predict(exmaple_data)}")
print(f"XGBoots Classifier: {xgbModel.score(x_test,y_test)} --> {xgbModel.predict(exmaple_data)}")


print(df.info())