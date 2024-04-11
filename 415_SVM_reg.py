import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/hasan/Desktop/Year 3/merged_dataset.csv' 
data = pd.read_csv(file_path)

# Preprocessing
data_clean = data.replace('**', np.nan).dropna(subset=['age-adjusted_death_rate', 'age-adjusted_incidence_rate_-_cases_per_100,000'])
data_clean['age-adjusted_death_rate'] = pd.to_numeric(data_clean['age-adjusted_death_rate'], errors='coerce')
data_clean['age-adjusted_incidence_rate_-_cases_per_100,000'] = pd.to_numeric(data_clean['age-adjusted_incidence_rate_-_cases_per_100,000'], errors='coerce')
data_clean.dropna(inplace=True)

# Feature and target selection
X = data_clean[['age-adjusted_incidence_rate_-_cases_per_100,000']] 
y = data_clean['age-adjusted_death_rate']  

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the SVR model
svr = SVR(kernel='linear')
svr.fit(X_train_scaled, y_train)

# Predicting the target values
y_train_pred = svr.predict(X_train_scaled)
y_test_pred = svr.predict(X_test_scaled)

# Evaluating the model
training_r2 = r2_score(y_train, y_train_pred)
testing_r2 = r2_score(y_test, y_test_pred)
training_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
testing_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print(f'Training R2: {training_r2}')
print(f'Testing R2: {testing_r2}')
print(f'Training RMSE: {training_rmse}')
print(f'Testing RMSE: {testing_rmse}')

# Plotting actual vs predicted values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values for the Test Set')
plt.show()