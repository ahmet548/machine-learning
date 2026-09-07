from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("student.csv")

df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

y = df[["Performance Index"]]
x = df[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]]

l = LinearRegression()
model = l.fit(x, y)

# Prediction
print("Student Achievement Model Score:", model.score(x, y))
print("Student Achievement Prediction: ", model.predict([[12, 70, 1, 10, 7]]))