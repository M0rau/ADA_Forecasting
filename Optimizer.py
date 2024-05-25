import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, SimpleRNN, LSTM, Input
import matplotlib.pyplot as plt

# Load and prepare the data
file_path = 'combined_data.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Price', 'Price_t+1', 'Price_t+7', 'Price_t+28', 'Price_t+91']]

# Normalize the relevant columns
scaler = MinMaxScaler()
df[['Price', 'Price_t+1', 'Price_t+7', 'Price_t+28', 'Price_t+91']] = scaler.fit_transform(df[['Price', 'Price_t+1', 'Price_t+7', 'Price_t+28', 'Price_t+91']])

# Splitting the dataset
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# Extract dates for plotting
train_dates = train_df.index
val_dates = val_df.index
test_dates = test_df.index

# Convert DataFrames to numpy arrays for scaling
train_scaled = train_df.values
val_scaled = val_df.values
test_scaled = test_df.values

def create_sequences(data, window_size, target_idx):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size, 0])  # Use only the 'Price' column for sequences
        labels.append(data[i+window_size-1, target_idx])  # The target is the column at target_idx
    return np.array(sequences), np.array(labels)

window_size = 5

# Create sequences and labels for t+1, t+7, t+28, t+91
targets = {'t+1': 1, 't+7': 2, 't+28': 3, 't+91': 4}

data = {}

for target_name, target_idx in targets.items():
    X_train, y_train = create_sequences(train_scaled, window_size, target_idx)
    X_val, y_val = create_sequences(val_scaled, window_size, target_idx)
    X_test, y_test = create_sequences(test_scaled, window_size, target_idx)
    data[target_name] = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}


# Function to denormalize data
def denormalize(data, scaler, column_idx):
    dummy = np.zeros((data.shape[0], scaler.n_features_in_))
    dummy[:, column_idx] = data[:, 0]
    return scaler.inverse_transform(dummy)[:, column_idx]

# Model definitions
def create_cnn(input_shape, units):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=units, kernel_size=2, activation='relu'),
        Conv1D(filters=units, kernel_size=2, activation='relu'),  # Hidden layer
        Flatten(),  # Flatten the output of the Conv1D layer
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_rnn(input_shape, units):
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(units, activation='relu', return_sequences=True),
        SimpleRNN(units, activation='relu'),  # First hidden layer
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_lstm(input_shape, units):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, activation='relu', return_sequences=True),
        LSTM(units, activation='relu'),   # First hidden layer
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Hyperparameters to tune
units_list = [16, 32, 64]
epochs_list = [10, 20, 30, 40]
batch_sizes = [16, 32, 64]

# Create dictionaries to store RMSE for each hyperparameter configuration
cnn_rmse = {}
rnn_rmse = {}
lstm_rmse = {}

# Function to evaluate models
def evaluate_model(create_model_fn, input_shape, units, epochs, batch_size, X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx):
    model = create_model_fn(input_shape, units)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, verbose=0)
    y_test_pred = model.predict(X_test)
    y_test_pred_denorm = denormalize(y_test_pred, scaler, target_idx)
    y_test_denorm = denormalize(y_test.reshape(-1, 1), scaler, target_idx)
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_test_pred_denorm))
    return rmse

# Evaluate all configurations
for target_name, target_idx in targets.items():
    print(f"Evaluating models for target: {target_name}")
    X_train, y_train = data[target_name]['X_train'], data[target_name]['y_train']
    X_val, y_val = data[target_name]['X_val'], data[target_name]['y_val']
    X_test, y_test = data[target_name]['X_test'], data[target_name]['y_test']
    
    # Reshape the data to be 3D for the model input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    for units in units_list:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                print(f"Evaluating CNN with units={units}, epochs={epochs}, batch_size={batch_size}")
                cnn_rmse[(target_name, units, epochs, batch_size)] = evaluate_model(
                    create_cnn, input_shape, units, epochs, batch_size, X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx)
                
                print(f"Evaluating RNN with units={units}, epochs={epochs}, batch_size={batch_size}")
                rnn_rmse[(target_name, units, epochs, batch_size)] = evaluate_model(
                    create_rnn, input_shape, units, epochs, batch_size, X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx)
                
                print(f"Evaluating LSTM with units={units}, epochs={epochs}, batch_size={batch_size}")
                lstm_rmse[(target_name, units, epochs, batch_size)] = evaluate_model(
                    create_lstm, input_shape, units, epochs, batch_size, X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx)

# Find the optimal parameters for each model and each time frame
optimal_params = {'CNN': {}, 'RNN': {}, 'LSTM': {}}
for target_name in targets.keys():
    optimal_params['CNN'][target_name] = min(
        ((t, units, epochs, batch_size) for (t, units, epochs, batch_size) in cnn_rmse.keys() if t == target_name), 
        key=lambda x: cnn_rmse[x]
    )
    optimal_params['RNN'][target_name] = min(
        ((t, units, epochs, batch_size) for (t, units, epochs, batch_size) in rnn_rmse.keys() if t == target_name), 
        key=lambda x: rnn_rmse[x]
    )
    optimal_params['LSTM'][target_name] = min(
        ((t, units, epochs, batch_size) for (t, units, epochs, batch_size) in lstm_rmse.keys() if t == target_name), 
        key=lambda x: lstm_rmse[x]
    )

# Create a DataFrame to display the optimal parameters and their corresponding RMSE values
optimal_df = pd.DataFrame(columns=['Model', 'Target', 'Units', 'Epochs', 'Batch Size', 'RMSE'])
for model_name, params in optimal_params.items():
    for (target_name, units, epochs, batch_size) in params.values():
        rmse = eval(f"{model_name.lower()}_rmse")[(target_name, units, epochs, batch_size)]
        optimal_df = pd.concat([optimal_df, pd.DataFrame([{
            'Model': model_name,
            'Target': target_name,
            'Units': units,
            'Epochs': epochs,
            'Batch Size': batch_size,
            'RMSE': rmse
        }])], ignore_index=True)

os.makedirs('table', exist_ok=True)
optimal_df.to_csv('table/optimal_parameters.csv', index=False)

print(optimal_df)
