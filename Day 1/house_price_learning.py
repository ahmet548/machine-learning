from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("house_price.csv")

y = df[["House_Price"]]
x = df[["Square_Footage", "Num_Bedrooms", "Num_Bathrooms", "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"]]

l = LinearRegression()
model = l.fit(x, y)

# Başarı Skoru (R² Score)
print("Model Success Score (R2):", model.score(x, y))

# Prediction 1
ornek_1 = pd.DataFrame([[1400, 3, 2, 2014, 0.5, 1, 10]], columns=x.columns)
print("1. Home Prediction:", model.predict(ornek_1))

# Prediction 2
ornek_2 = pd.DataFrame([[1360, 2, 1, 1981, 0.599637, 0, 5]], columns=x.columns)
print("2. Home Prediction:", model.predict(ornek_2))

# Coefficients and Intercept
print("Coef:", model.coef_)
print("Intercept:", model.intercept_)