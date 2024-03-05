import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset from Excel
weather_data = pd.read_excel("a_data.xlsx")
weather_data = weather_data.fillna(0)

# Separate features and target variable
X = weather_data.drop(columns=["Total Precipitation","Year/Month","Avg Sunshine Hours "])  # Features
y = weather_data["Total Precipitation"]  # Target variable


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the degree of the polynomial
degree = 3

# Create polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Initialize the linear regression model
model = LinearRegression()

# Fit the polynomial regression model to the training data
model.fit(X_train_poly, y_train)

# Make predictions on the testing data
predictions = model.predict(X_test_poly)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

df = pd.DataFrame({'Numbers': range(1, 56)})
array = df['Numbers'].values
array

# Plot actual vs. predicted
plt.scatter(array,predictions,c='r')
plt.scatter(array,y_test,c='b')
plt.xlabel('timeline')
plt.ylabel('percipation')
plt.title('Actual vs. Predicted Precipitation')
plt.show()