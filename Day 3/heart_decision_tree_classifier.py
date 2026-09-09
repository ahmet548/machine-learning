from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import graphviz

pd.set_option('display.max_columns', None)

df = pd.read_csv('heart.csv')

print(df.info())

df = pd.get_dummies(df, columns=["Result"], drop_first=True)

print(df.head(3))

y = df["Result_positive"]
x = df.drop("Result_positive", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

tree = DecisionTreeClassifier()
model = tree.fit(x_train, y_train)

print("--------------------------------")
print("Model Score: ", model.score(x_test, y_test))
print(model.predict([[49, 0, 105, 138, 74, 112, 0.82, 0.0045]]))

dot = export_graphviz(tree)
image = graphviz.Source(dot)
print(image)
