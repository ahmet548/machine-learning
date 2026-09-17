import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train.drop('Id', axis=1)

print(train.info())

y = train['SalePrice']
x = train.drop('SalePrice', axis=1)

cat_features = []
for i in x.columns:
    if x[i].dtype == 'str':
        cat_features.append(i)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.88,random_state=6
)

x_train[cat_features] = x_train[cat_features].fillna('Missing')
x_test[cat_features] = x_test[cat_features].fillna('Missing')

cat = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    cat_features=cat_features,
    random_seed=42,
    verbose=100
)

model = cat.fit(x_train,y_train)

print(model.score(x_test,y_test))

xt = test.drop('Id', axis=1)

xt[cat_features] = xt[cat_features].fillna("Missing")

predictions = model.predict(xt)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': predictions
})

submission.to_csv('submission.csv', index=False)

print("File Created!")
