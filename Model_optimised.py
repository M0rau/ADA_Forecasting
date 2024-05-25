import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, SimpleRNN, LSTM
import os

# Create directories if they don't exist
os.makedirs('graph', exist_ok=True)
os.makedirs('table', exist_ok=True)

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

# Define the window sizes for each target
window_sizes = 5

# Create sequences and labels for t+1, t+7, t+28, t+91
targets = {'t+1': 1, 't+7': 2, 't+28': 3, 't+91': 4}
data = {}

for target_name, target_idx in targets.items():
    X_train, y_train = create_sequences(train_scaled, window_sizes, target_idx)
    X_val, y_val = create_sequences(val_scaled, window_sizes, target_idx)
    X_test, y_test = create_sequences(test_scaled, window_sizes, target_idx)
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

# Define the optimal parameters
optimal_params = {
    'CNN': {
        't+1': (32, 40, 32),
        't+7': (64, 30, 16),
        't+28': (64, 40, 16),
        't+91': (16, 20, 32)
    },
    'RNN': {
        't+1': (32, 30, 16),
        't+7': (16, 10, 64),
        't+28': (16, 40, 32),
        't+91': (64, 10, 64)
    },
    'LSTM': {
        't+1': (64, 40, 16),
        't+7': (64, 40, 16),
        't+28': (16, 20, 16),
        't+91': (32, 10, 32)
    }
}


# Create a 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# Flatten axes for easier iteration
axes = axes.flatten()

# Initialize results dictionary
results = {'Model': ['CNN', 'RNN', 'LSTM']}
mse_mae_data = []

for idx, (target_name, target_idx) in enumerate(targets.items()):
    X_train, y_train = data[target_name]['X_train'], data[target_name]['y_train']
    X_val, y_val = data[target_name]['X_val'], data[target_name]['y_val']
    X_test, y_test = data[target_name]['X_test'], data[target_name]['y_test']
    
    # Reshape the data to be 3D for the model input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Get optimal parameters for CNN
    units, epochs, batch_size = optimal_params['CNN'][target_name]
    print(f"Evaluating CNN for {target_name} with units={units}, epochs={epochs}, batch_size={batch_size}")
    
    # CNN
    cnn_model = create_cnn(input_shape, units)
    cnn_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, verbose=1)
    y_test_pred_cnn = cnn_model.predict(X_test)
    
    # Denormalize the predictions and true values
    y_test_pred_cnn_denorm = denormalize(y_test_pred_cnn, scaler, target_idx)
    y_test_denorm = denormalize(y_test.reshape(-1, 1), scaler, target_idx)
    
    mse_cnn = mean_squared_error(y_test_denorm, y_test_pred_cnn_denorm)
    mae_cnn = mean_absolute_error(y_test_denorm, y_test_pred_cnn_denorm)
    
    # Get optimal parameters for RNN
    units, epochs, batch_size = optimal_params['RNN'][target_name]
    print(f"Evaluating RNN for {target_name} with units={units}, epochs={epochs}, batch_size={batch_size}")
    
    # RNN
    rnn_model = create_rnn(input_shape, units)
    rnn_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, verbose=1)
    y_test_pred_rnn = rnn_model.predict(X_test)
    
    # Denormalize the predictions
    y_test_pred_rnn_denorm = denormalize(y_test_pred_rnn, scaler, target_idx)
    
    mse_rnn = mean_squared_error(y_test_denorm, y_test_pred_rnn_denorm)
    mae_rnn = mean_absolute_error(y_test_denorm, y_test_pred_rnn_denorm)
    
    # Get optimal parameters for LSTM
    units, epochs, batch_size = optimal_params['LSTM'][target_name]
    print(f"Evaluating LSTM for {target_name} with units={units}, epochs={epochs}, batch_size={batch_size}")
    
    # LSTM
    lstm_model = create_lstm(input_shape, units)
    lstm_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=batch_size, verbose=1)
    y_test_pred_lstm = lstm_model.predict(X_test)
    
    # Denormalize the predictions
    y_test_pred_lstm_denorm = denormalize(y_test_pred_lstm, scaler, target_idx)
    
    mse_lstm = mean_squared_error(y_test_denorm, y_test_pred_lstm_denorm)
    mae_lstm = mean_absolute_error(y_test_denorm, y_test_pred_lstm_denorm)
    
    mse_mae_data.append({'Time_Frame': target_name, 'Model': 'CNN', 'MSE': mse_cnn, 'MAE': mae_cnn})
    mse_mae_data.append({'Time_Frame': target_name, 'Model': 'RNN', 'MSE': mse_rnn, 'MAE': mae_rnn})
    mse_mae_data.append({'Time_Frame': target_name, 'Model': 'LSTM', 'MSE': mse_lstm, 'MAE': mae_lstm})
    
    print(f"Results for {target_name}:")
    print(f"CNN - MSE: {mse_cnn}, MAE: {mae_cnn}")
    print(f"RNN - MSE: {mse_rnn}, MAE: {mae_rnn}")
    print(f"LSTM - MSE: {mse_lstm}, MAE: {mae_lstm}")
    
    # Plotting the actual vs predicted values for test data (out-of-sample) for each model
    ax = axes[idx]
    ax.plot(test_dates[window_sizes:], y_test_denorm, label='Actual (Test)')
    ax.plot(test_dates[window_sizes:], y_test_pred_cnn_denorm, label='Predicted - CNN')
    ax.plot(test_dates[window_sizes:], y_test_pred_rnn_denorm, label='Predicted - RNN')
    ax.plot(test_dates[window_sizes:], y_test_pred_lstm_denorm, label='Predicted - LSTM')
    ax.set_title(f'Out-of-Sample Predictions - {target_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

plt.tight_layout()
plt.savefig('graph/Graph_af_opti.png')
plt.show()

# Convert mse_mae_data to DataFrame and save
mse_mae_df = pd.DataFrame(mse_mae_data)
mse_mae_df.to_csv('table/Table_af_opti.csv', index=False)
print(mse_mae_df)
