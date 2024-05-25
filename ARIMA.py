import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import auto_arima

df = pd.read_csv('combined_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(df)

df['Price'].plot(title='Crude Oil WTI Price')
plt.xlabel('Date')
plt.ylabel('Price (in USD)')
plt.show()

# ADF daily data 
result = adfuller(df['Price'])
print('Results for Price:')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
        print(f'    {key}: {value}')
print('\n')

# ADF weekly data
# Extract only Friday
df['weekday'] = df.index.weekday
friday_data = df[df['weekday'] == 3]  

# ADF test function
def adf_test(series, title=''):
    result = adfuller(series.dropna())  # Ensure to drop NA values
    print(f'Results for {title}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value}')

# Perform ADF test on the 'Price' for thursday
adf_test(friday_data['Price'], 'Weekly Price (thursdays)')

friday_data['Price'].plot(title='Crude Oil WTI Price')
plt.xlabel('Date')
plt.ylabel('Price (in USD)')
plt.show()
print(friday_data)

# ADF monthly data

# Find the index of January 3, 1986
start_index = df.index.get_loc('1986-01-02')

# Select every 28th day starting from January 3, 1986
df_every_28_days = df.iloc[start_index::28]

# ADF test function
def adf_test(series, title=''):
    result = adfuller(series.dropna())  # Ensure to drop NA values
    print(f'Results for {title}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value}')

# Perform ADF test on the data selected every 28 days
adf_test(df_every_28_days['Price'], 'Price Every 28 Days')

plt.plot(df_every_28_days['Price'])

# ADF quarterly data

# Find the index of January 3, 1986
start_index = df.index.get_loc('1986-01-02')

# Select every 28th day starting from January 3, 1986
df_every_91_days = df.iloc[start_index::91]

# ADF test function
def adf_test(series, title=''):
    result = adfuller(series.dropna())  # Ensure to drop NA values
    print(f'Results for {title}:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value}')

# Perform ADF test on the data selected every 91 days
adf_test(df_every_91_days['Price'], 'Price Every 91 Days')

plt.plot(df_every_91_days['Price'])


# Function to perform the ADF test
def adf_test(series, title=''):
    print(f'Results for {title}:')
    result = adfuller(series.dropna())  # Dropping NA to ensure the test runs
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    print('\n')

# First differencing
df['Price_diff'] = df['Price'].diff()
friday_data.loc[:, 'Price_diff'] = friday_data['Price'].diff()
df_every_28_days.loc[:, 'Price_diff'] = df_every_28_days['Price'].diff()
df_every_91_days.loc[:, 'Price_diff'] = df_every_91_days['Price'].diff()

# ADF test on the differenced data
adf_test(df['Price_diff'].dropna(), 'Daily Price - First Difference')
adf_test(friday_data['Price_diff'].dropna(), 'Weekly Price (Thursdays) - First Difference')
adf_test(df_every_28_days['Price_diff'].dropna(), 'Price Every 28 Days - First Difference')
adf_test(df_every_91_days['Price_diff'].dropna(), 'Price Every 91 Days - First Difference')

# Define a dictionary of datasets
datasets = {
    'Daily': df['Price'],
    'Weekly (Thursdays)': friday_data['Price'],
    'Every 28 Days': df_every_28_days['Price'],
    'Every 91 Days': df_every_91_days['Price']
}

# Function to run auto_arima and display results
def run_auto_arima(data, title):
    print(f"\nRunning auto_arima for {title}:")
    model = auto_arima(data, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
    print("Best ARIMA model order:", model.order)
    print("AIC:", model.aic())
    return model

# Dictionary to store models
models = {}

# Loop through the datasets and apply auto_arima
for title, data in datasets.items():
    models[title] = run_auto_arima(data, title)


# Dictionary of datasets
datasets = {
    'Daily': df['Price_diff'],
    'Weekly (Thursdays)': friday_data['Price_diff'],
    'Every 28 Days': df_every_28_days['Price_diff'],
    'Every 91 Days': df_every_91_days['Price_diff']
}

# Define a function to dynamically split data based on a specified training percentage
def train_test_split(data, train_percent):
    split_point = int(len(data) * train_percent)  # Corrected to use the length of the data argument
    train = data.iloc[:split_point].dropna()  # Use data, not df
    test = data.iloc[split_point:].dropna()   # Use data, not df
    return train, test  # Ensure the function returns values

train_percent = 0.90  # Define the training set size as 70% of the data

# Dictionary to store training and test sets
train_test_data = {}

# Loop through the datasets, splitting each into training and test sets
for key, data in datasets.items():
    train, test = train_test_split(data, train_percent)
    train_test_data[key] = (train, test)
    print(f"{key} data: Training data points: {len(train)}, Test data points: {len(test)}")

# Loop through the train_test_data dictionary and print each dataset
'''
for key, (train, test) in train_test_data.items():
    print(f"{key} Training Data:")
    print(train)  # Print the training dataset
    print(f"\n{key} Testing Data:")
    print(test)  # Print the testing dataset
    print("\n" + "-"*50 + "\n")  
'''
train, test = train_test_split(df['Price_diff'], 0.9)  
print("Training Data:")
print(train)
print("\nTesting Data:")
print(test)

orders = {
    'Daily': (1, 1, 1),
    'Weekly (Thursdays)': (1, 1, 2),
    'Every 28 Days': (2, 1, 1),
    'Every 91 Days': (1, 1, 1)
}

# Assuming 'train_test_data' holds the training data for each series
models = {}
for key, order in orders.items():
    train_data = train_test_data[key][0]  # Assuming index 0 is the training dataset
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    models[key] = fitted_model
    print(f"Model summary for {key}:")
    print(fitted_model.summary())
    print("\n")

def check_residuals(model, title):
    residuals = pd.DataFrame(model.resid)
    residuals.plot(title=f"Residuals for {title}")
    plt.show()
    residuals.plot(kind='kde', title=f"Residual Density for {title}")
    plt.show()

for key, model in models.items():
    check_residuals(model, key)


for key, data in datasets.items():
    # Split the data
    train_data, test_data = train_test_split(data, 0.9)
    
    # Fit ARIMA model on the training data
    order = orders[key]
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    
   # Forecasting
    forecast = fitted_model.get_forecast(steps=len(test_data))
    test_forecast_series = pd.Series(forecast.predicted_mean)
    test_forecast_series.index = test_data.index

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data, label='True Values', color='grey', alpha=0.5)  # True values
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')  # In-sample predictions
    plt.plot(test_forecast_series.index, test_forecast_series, label='Out-of-Sample Predictions', color='orange')  # Out-of-sample predictions
    plt.title(f'ARIMA Forecasting - {key} ')
    plt.xlabel('Date')
    plt.ylabel('Price Difference')
    plt.legend()
    plt.show()


def evaluate_forecasts(test, predictions, title):
    mse = mean_squared_error(test, predictions)
    mae = mean_absolute_error(test, predictions)
    print(f"{title} - MSE: {mse:.2f}, MAE: {mae:.2f}")

for key in models.keys():
    forecast = models[key].get_forecast(steps=len(train_test_data[key][1]))
    predictions = forecast.predicted_mean
    evaluate_forecasts(train_test_data[key][1], predictions, f"Evaluation for {key}")