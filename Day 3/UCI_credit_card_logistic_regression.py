from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
df = pd.read_csv('UCI_Credit_Card.csv')

print(df.head(3))

df = df.drop("ID", axis=1)

y = df[["default.payment.next.month"]]
x = df.drop("default.payment.next.month", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.77, random_state=6)

log = LogisticRegression()
model = log.fit(x_train, y_train)

print("--------------------------------")
print("Model Score: ", model.score(x_test, y_test))

print(x.iloc[1903])

example_x = np.array(x.iloc[1903])

print("Example X: ", model.predict([example_x]))