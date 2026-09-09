from sklearn.linear_model import LinearRegression, Ridge 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('student_scores.csv')
print(df.head(3))

y = df['Scores']
x = df[['Hours']]

plt.style.use('fivethirtyeight')
plt.figure(figsize=(7, 7))
plt.scatter(x, y)
plt.title("Hours vs Scores")
plt.show()

lr = LinearRegression()
model = lr.fit(x, y)
print("Linear Regression Score:", model.score(x, y))

alpha_arr = [0.01, 0.1, 1, 10, 100]

print("\n--- Ridge Regression Results ---")
for alpha in alpha_arr:
    r = Ridge(alpha=alpha) 
    modelr = r.fit(x, y)
    
    print(f"Alpha: {alpha}")
    print(f"Score: {modelr.score(x, y)}")
    print(f"Coefficient: {modelr.coef_}\n")