import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

df = pd.read_csv("hospital_readmission_risk_10000.csv")

df = df.drop('patient_id', axis=1)
df = df.dropna()

le = LabelEncoder()
df['readmission_risk_label'] = le.fit_transform(df['readmission_risk'])

df = df.drop('readmission_risk', axis=1)
y = df['readmission_risk_label']
x = df.drop('readmission_risk_label', axis=1)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=6,train_size=0.85
)

cat_features = []
for i in x.columns:
    if x[i].dtype == "str":
        cat_features.append(i)

cat = CatBoostClassifier(
    n_estimators = 200,
    loss_function = "MultiClass",
    cat_features = cat_features
)

model = cat.fit(x_train,y_train)

print(df.info())

print(model.score(x_test,y_test))