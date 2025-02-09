import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
file_name = 'AQI_Data.csv'  
df = pd.read_csv(file_name)
print(df.isnull().sum()) 
df = df.dropna()  
X = df[['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM']]  # Features
y = df['PM 2.5']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Predicted AQI for the test set:", y_pred)
new_data = pd.DataFrame({
    'T': [20], 'TM': [24], 'Tm': [18], 'SLP': [1011], 'H': [40], 'VV': [1], 'V': [12], 'VM': [12]
})
predicted_aqi = model.predict(new_data)
print("Predicted AQI for new data:", predicted_aqi[0])
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test, color='blue', alpha=0.5, label='Actual AQI')
plt.scatter(y_test, y_pred, color='green', alpha=0.5, label='Predicted AQI')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Perfect Prediction Line")
plt.xlabel('Actual AQI (PM 2.5)')
plt.ylabel('Predicted AQI (PM 2.5)')
plt.title('Actual vs Predicted AQI')
plt.legend()
plt.grid(True)
plt.show()
