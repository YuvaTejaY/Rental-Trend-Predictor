*******************Rental Trend Predictor**************




Description=================




Use mock dataset of monthly rent prices for 5 cities and predict next month's average rent.

Requirements==============



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





