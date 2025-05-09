*******************Rental Trend Predictor**************
Description=================
Use mock dataset of monthly rent prices for 5 cities and predict next month's average rent.

Requirements===============
Use linear regression
Data preprocessing
Model evaluation metrics

Bonus======================
Visualize trends using matplotlib or seaborn

Project Overview===========
This project predicts the average monthly rent for five Indian cities — Chennai, Bangalore, Hyderabad, Coimbatore, and Trichy — using historical data from the past 12 months. It then forecasts rent for the 13th month using a simple Linear Regression model.


Explanation==============


1.Data Preparation
	Monthly rent data is collected for each city.
  An average rent is calculated for each month.
  The month number (1–12) is the input; average rent is the target.
  The month values are standardized to improve model performance.


2.Model Training
  A Linear Regression model learns the trend between months and average rent.
  Once trained, it can predict rent for future months (like month 13).

3.Prediction
   The model predicts the average rent for month 13 using the same scaling.
   This gives us an estimate of rent for the next month.

4.Evaluation
   We use Mean Squared Error (MSE) and R² Score to measure accuracy.
   Lower MSE and a higher R² mean better predictions.

5.Visualization
  Line graphs show rent trends for each city and the overall average.
  Helps visualize rent patterns and compare city trends over time.



code============
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Month': list(range(1, 13)),
    'Chennai': [1200, 1210, 1225, 1230, 1250, 1270, 1280, 1300, 1320, 1330, 1340, 1360],
    'Bangalore': [900, 920, 940, 950, 960, 970, 980, 1000, 1010, 1020, 1030, 1050],
    'Hyderabad': [1500, 1490, 1485, 1470, 1450, 1440, 1430, 1420, 1410, 1400, 1390, 1380],
    'Coimbatore': [800, 805, 810, 815, 820, 825, 830, 835, 840, 845, 850, 855],
    'Trichy': [1000, 1005, 1015, 1020, 1035, 1040, 1055, 1070, 1080, 1095, 1110, 1130],
}
df = pd.DataFrame(data)
df['Average_Rent'] = df[['Chennai', 'Bangalore', 'Hyderabad', 'Coimbatore', 'Trichy']].mean(axis=1)
X = df[['Month']]  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df['Average_Rent']
model = LinearRegression()
model.fit(X_scaled, y)
X_13_scaled = scaler.transform([[13]])
predicted_avg_rent = model.predict(X_13_scaled)[0]
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("************Rental Trend Predictor**************")
print(f"Predicted Average Rent for Month 13: ${predicted_avg_rent:.2f}")
print(f"\nModel Evaluation Metrics: \nMSE: {mse:.2f}, R²: {r2:.2f}")
plt.figure(figsize=(10, 6))
for city in df.columns[1:-1]:
    sns.lineplot(x='Month', y=city, data=df, label=city)
sns.lineplot(x='Month', y='Average_Rent', data=df, label='Average Rent', color='black', linestyle='--', marker='x')
plt.title("Monthly Rent Trends for 5 Cities and Average Rent")
plt.xlabel("Month")
plt.ylabel("Rent Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

