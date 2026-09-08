from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd

df = pd.read_csv('insurance.csv') 
print(df.head(3))

df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print(df.head(3))

y = df['charges']
X = df.drop('charges', axis=1)

l = LinearRegression()
model = l.fit(X, y)
print(model.score(X, y))

df_errors = pd.DataFrame()
df_errors['y_true'] = y
df_errors['y_pred'] = model.predict(X)
df_errors['error'] = df_errors['y_true'] - df_errors['y_pred']


df_errors['squared_error'] = df_errors['error'] ** 2
df_errors['abs_error'] = df_errors['error'].abs()
df_errors['percent_error'] = ((df_errors['y_true'] - df_errors['y_pred']) / df_errors['y_true']).abs() * 100
print(df_errors.head(3))

print(df_errors.mean())

##############################################################################################################

print('Mean Squared Error (MSE):', mean_squared_error(df_errors['y_true'], df_errors['y_pred']))
print('Mean Absolute Error (MAE):', mean_absolute_error(df_errors['y_true'], df_errors['y_pred']))
print('Mean Absolute Percentage Error (MAPE):', mean_absolute_percentage_error(df_errors['y_true'], df_errors['y_pred']))