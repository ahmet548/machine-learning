import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ObesityDataSet.csv')

y = df['NObeyesdad']
x = df.drop('NObeyesdad', axis=1)

cat_features = []
for i in x.columns:
    if x[i].dtype == 'str':
        cat_features.append(i)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=21,train_size=0.77
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
print(model.score(x_test,y_test))

y_pred = model.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Karmaşıklık Matrisi')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.show()

plt.figure(figsize=(8,6))


print(df.info())