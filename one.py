import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Load the dataset
data = pd.read_excel('a_data.xlsx')

# Split the data into training and testing sets
train_data = data[data['Year/Month'] <= '2005-12-31']
test_data = data[data['Year/Month'] > '2005-12-31']

# Extracting the 'Total Precipitation' column
train_precipitation = train_data['Total Precipitation'].values.reshape(-1, 1)
test_precipitation = test_data['Total Precipitation'].values.reshape(-1, 1)

# Scaling the data
scaler = MinMaxScaler()
train_precipitation_scaled = scaler.fit_transform(train_precipitation)
test_precipitation_scaled = scaler.transform(test_precipitation)  # Scale the test data using the same scaler

# Function to create sequences from the data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 12  # Assuming monthly data, 12 months make a year

# Create sequences for training and testing
X_train, y_train = create_sequences(train_precipitation_scaled, seq_length)
X_test, y_test = create_sequences(test_precipitation_scaled, seq_length)

# Reshape the data for different models
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Model definitions

# ANN Model
def build_ann_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(seq_length,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# CNN Model
def build_cnn_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(seq_length, 1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# RNN Model
def build_rnn_model():
    model = Sequential([
        SimpleRNN(64, input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# LSTM Model
def build_lstm_model():
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train and predict with KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_reshaped, y_train)
knn_pred_scaled = knn_model.predict(X_test_reshaped)
knn_pred = scaler.inverse_transform(knn_pred_scaled.reshape(-1, 1))

# Train and predict with SVR
svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_model.fit(X_train_reshaped, y_train.ravel())
svr_pred_scaled = svr_model.predict(X_test_reshaped)
svr_pred = scaler.inverse_transform(svr_pred_scaled.reshape(-1, 1))

# Helper function to train and predict with Keras models
def train_and_predict_keras_model(build_model_func, X_train, y_train, X_test):
    model = build_model_func()
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    pred_scaled = model.predict(X_test)
    pred = scaler.inverse_transform(pred_scaled)
    return pred

# ANN
ann_pred = train_and_predict_keras_model(build_ann_model, X_train_reshaped, y_train, X_test_reshaped)

# CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
cnn_pred = train_and_predict_keras_model(build_cnn_model, X_train_cnn, y_train, X_test_cnn)

# RNN
rnn_pred = train_and_predict_keras_model(build_rnn_model, X_train_cnn, y_train, X_test_cnn)

# LSTM
lstm_pred = train_and_predict_keras_model(build_lstm_model, X_train_cnn, y_train, X_test_cnn)

# Error calculation
def calculate_errors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# Calculate errors for each model
ann_mae, ann_mse, ann_rmse, ann_r2 = calculate_errors(test_precipitation[seq_length:], ann_pred)
cnn_mae, cnn_mse, cnn_rmse, cnn_r2 = calculate_errors(test_precipitation[seq_length:], cnn_pred)
rnn_mae, rnn_mse, rnn_rmse, rnn_r2 = calculate_errors(test_precipitation[seq_length:], rnn_pred)
lstm_mae, lstm_mse, lstm_rmse, lstm_r2 = calculate_errors(test_precipitation[seq_length:], lstm_pred)
knn_mae, knn_mse, knn_rmse, knn_r2 = calculate_errors(test_precipitation[seq_length:], knn_pred)
svr_mae, svr_mse, svr_rmse, svr_r2 = calculate_errors(test_precipitation[seq_length:], svr_pred)

# Plotting
plt.figure(figsize=(14, 8))
plt.plot(test_data['Year/Month'].iloc[seq_length:], test_precipitation[seq_length:], label='Original', marker='o')
plt.plot(test_data['Year/Month'].iloc[seq_length:], ann_pred, label='ANN Predicted', marker='x')
plt.plot(test_data['Year/Month'].iloc[seq_length:], cnn_pred, label='CNN Predicted', marker='x')
plt.plot(test_data['Year/Month'].iloc[seq_length:], rnn_pred, label='RNN Predicted', marker='x')
plt.plot(test_data['Year/Month'].iloc[seq_length:], lstm_pred, label='LSTM Predicted', marker='x')
plt.plot(test_data['Year/Month'].iloc[seq_length:], knn_pred, label='KNN Predicted', marker='x')
plt.plot(test_data['Year/Month'].iloc[seq_length:], svr_pred, label='SVR Predicted', marker='x')
plt.xlabel('Year/Month')
plt.ylabel('Total Precipitation')
plt.title('Total Precipitation Prediction Comparison')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print errors
print(f"ANN - MAE: {ann_mae}, MSE: {ann_mse}, RMSE: {ann_rmse}, R2: {ann_r2}")
print(f"CNN - MAE: {cnn_mae}, MSE: {cnn_mse}, RMSE: {cnn_rmse}, R2: {cnn_r2}")
print(f"RNN - MAE: {rnn_mae}, MSE: {rnn_mse}, RMSE: {rnn_rmse}, R2: {rnn_r2}")
print(f"LSTM - MAE: {lstm_mae}, MSE: {lstm_mse}, RMSE: {lstm_rmse}, R2: {lstm_r2}")
print(f"KNN - MAE: {knn_mae}, MSE: {knn_mse}, RMSE: {knn_rmse}, R2: {knn_r2}")
print(f"SVR - MAE: {svr_mae}, MSE: {svr_mse}, RMSE: {svr_rmse}, R2: {svr_r2}")
