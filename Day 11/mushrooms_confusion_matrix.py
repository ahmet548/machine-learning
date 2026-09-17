import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("mushrooms.csv")

df = pd.get_dummies(df, columns=['class'], drop_first=True)

y = df['class_p']
x = df.drop('class_p', axis=1)

cat_features = []
for i in x.columns:
    if x[i].dtype == 'str':
        cat_features.append(i)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=42,train_size=0.77
)

cat = CatBoostClassifier(
    n_estimators = 400,
    loss_function = "MultiClass",
    learning_rate=0.02,
    depth=6,    
    cat_features=cat_features,
    random_seed=42,
    verbose=100,
)

model = cat.fit(x_train,y_train)

print(df.info())

y_pred = model.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(model.score(x_test,y_test))