from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

df = pd.read_csv("heart.csv")

df = pd.get_dummies(df, "Result", drop_first=True)

y = df["Result_positive"]
x = df.drop("Result_positive", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

rTree = RandomForestClassifier(n_estimators=200, max_depth=4)
rTreeModel = rTree.fit(x_train, y_train)

tree = DecisionTreeClassifier()
treeModel = tree.fit(x_train, y_train)

print(f"{treeModel.score(x_test, y_test)}")
print(f"{rTreeModel.score(x_test, y_test)}")
print(df)