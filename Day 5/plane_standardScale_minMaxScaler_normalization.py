from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

df = pd.read_csv('plane.csv')

df = df[["Rcmnd cruise Knots", "Stall Knots dirty", "Fuel gal/lbs", "Eng out rate of climb", "Takeoff over 50ft", "Price"]]

y = df["Price"]
x = df.drop("Price", axis=1)

ss = StandardScaler()
x2 = ss.fit_transform(x)

mm = MinMaxScaler()
x3 = mm.fit_transform(x)

print(f"Standard Xcaler: {pd.DataFrame(x2)} \nMin-Max Scaler {pd.DataFrame(x3)}")

print(df.info())