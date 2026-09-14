from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv("Churn.csv")

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

ohe = OneHotEncoder()
xd = ohe.fit_transform(df[['Geography','Gender']]).toarray()

columns_name = ohe.get_feature_names_out()
df[columns_name] = xd

df = df.drop(columns=['Geography','Gender'])

y = df['Exited']
x = df.drop('Exited', axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=6,train_size=0.77
)

tree = DecisionTreeClassifier()
tModel = tree.fit(x_train,y_train)

forest = RandomForestClassifier()
fModel = forest.fit(x_train,y_train)

xgb = xgb.XGBClassifier()
xgbModel = xgb.fit(x_train,y_train)

print(df.info())

example_value = x.head(1).values

print(f"Decision Tree Classifier Score: {tModel.score(x_test,y_test)} --> {tModel.predict(example_value)} \nRandom Forest Classifier Score: {fModel.score(x_test,y_test)} --> {fModel.predict(example_value)} \nXGB Classifier Score: {xgbModel.score(x_test,y_test)} --> {xgbModel.predict(example_value)}")



