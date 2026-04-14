# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset (replace with your file name)
df = pd.read_csv("ML/datasets/linearregressiondataset.CSV")

# Show first rows (optional)
print(df.head())

# Define input (X) and output (y)
X = df[["Population"]]   # input
y = df["Profit"]         # output

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print coefficient and intercept
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Calculate errors
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
