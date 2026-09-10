from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("breast_cancer.csv")

df = pd.get_dummies(df, "diagnosis", drop_first=True)

y = df["diagnosis_M"]
x = df.drop(columns=["diagnosis_M", "id"])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

forest = RandomForestClassifier(n_estimators=200, max_depth=4)
fModel = forest.fit(x_train, y_train)

print(df.info())
print(fModel.predict(x_test))
print(fModel.score(x_test, y_test))