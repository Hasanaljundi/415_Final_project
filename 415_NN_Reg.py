import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('C:/Users/hasan/Desktop/Year 3/merged_dataset.csv')

# Convert the relevant columns to numeric, handling non-numeric values
data['age-adjusted_death_rate'] = pd.to_numeric(data['age-adjusted_death_rate'], errors='coerce')
data['age-adjusted_incidence_rate_-_cases_per_100,000'] = pd.to_numeric(data['age-adjusted_incidence_rate_-_cases_per_100,000'], errors='coerce')

# Drop rows with missing values in these columns
data_clean = data.dropna(subset=['age-adjusted_death_rate', 'age-adjusted_incidence_rate_-_cases_per_100,000'])

# Define features (X) and target (y)
X = data_clean[['age-adjusted_incidence_rate_-_cases_per_100,000']]
y = data_clean['age-adjusted_death_rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the training and testing set
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate the R2 and RMSE for the training and testing sets
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f'Training R^2: {train_r2:.4f}')
print(f'Testing R^2: {test_r2:.4f}')
print(f'Training RMSE: {train_rmse:.4f}')
print(f'Testing RMSE: {test_rmse:.4f}')

# Plotting actual vs. predicted values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()