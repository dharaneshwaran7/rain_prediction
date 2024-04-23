import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(100)
import random
random.seed(100)
import tensorflow as tf
tf.random.set_seed(10)

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

# Create sequences for training
X_train, y_train = create_sequences(train_precipitation_scaled, seq_length)

# Reshape the data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test, y_test = create_sequences(test_precipitation_scaled, seq_length)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Model Building
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)  # Set verbose=0 for no output

# Make predictions
predicted_precipitation_scaled = model.predict(X_test)

# Inverse scaling the predicted precipitation data
predicted_precipitation = scaler.inverse_transform(predicted_precipitation_scaled)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(test_data['Year/Month'].iloc[seq_length:], test_precipitation[seq_length:], label='Original', marker='o')
plt.plot(test_data['Year/Month'].iloc[seq_length:], predicted_precipitation, label='Predicted', marker='x')
plt.xlabel('Year/Month')
plt.ylabel('Total Precipitation')
plt.title('Total Precipitation Prediction from 2005 to 2010 using LSTM')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate error
error = np.mean(np.abs(predicted_precipitation - test_precipitation[seq_length:]))
print("Mean Absolute Error:", error)
