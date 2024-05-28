import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('data/advertising.csv')

# Feature engineering
data['log_TV'] = np.log(data['TV'] + 1)
data['log_Radio'] = np.log(data['Radio'] + 1)
data['log_Newspaper'] = np.log(data['Newspaper'] + 1)
data['total_ad_expenditure'] = data['TV'] + data['Radio'] + data['Newspaper']
data['log_total_ad_expenditure'] = np.log(data['total_ad_expenditure'] + 1)

# Features and target
X = data[['log_TV', 'log_Radio', 'log_Newspaper']]
y = data['Sales']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Save the trained model
model_path = 'models/rf_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf_reg, file)

print("Model trained and saved at 'models/rf_model.pkl'")
