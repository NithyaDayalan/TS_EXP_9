## EX.NO.09 : A project on Time series analysis on weather forecasting using ARIMA model
### Date : 
### AIM : 
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.

### ALGORITHM : 
1. Explore the dataset of weather
2. Check for stationarity of time series time series plot ACF plot and PACF plot ADF test Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions

### PROGRAM :
```
Developed By : NITHYA D
Reg.No : 212223240110
```
#### Import necessary packages
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
```
#### Load the dataset
```
data = pd.read_csv("Gold Price Prediction.csv")
```
#### Display the first few rows and column names
```
print("Dataset Preview:")
print(data.head())
print("\nColumn Names:")
print(data.columns)
```
#### Try to infer the target column automatically
```
date_col = None
target_col = None
```
#### Detect date column
```
for col in data.columns:
    if 'date' in col.lower():
        date_col = col
        break
```
#### Detect numeric column (gold price)
```
for col in data.columns:
    if data[col].dtype in ['float64', 'int64'] and col.lower() != date_col:
        target_col = col
        break
```
#### Validate column detection
```
if not date_col or not target_col:
    raise ValueError("Could not automatically detect 'Date' or 'Price' column. Please check column names manually.")

print(f"\nUsing '{date_col}' as date column and '{target_col}' as price column.")
```
#### Convert 'Date' column to datetime format
```
data[date_col] = pd.to_datetime(data[date_col])
```
#### Set 'Date' column as index
```
data.set_index(date_col, inplace=True)
```
#### Sort and drop missing values
```
data.sort_index(inplace=True)
data.dropna(inplace=True)
```
#### ARIMA model function
```
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)
```
#### Run the model
```
arima_model(data, target_col, order=(5, 1, 0))
```

### OUTPUT :
![image](https://github.com/user-attachments/assets/c0f82813-a86a-46de-a359-b058e7af7530)

![image](https://github.com/user-attachments/assets/b2410272-a092-4c4e-b57f-448b965addd0)

### RESULT :
Thus the program run successfully based on the ARIMA model using python.
