import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("mushroom_d3_data.csv")

# Convert categorical features into numerical format using One-Hot Encoding
df = pd.get_dummies(df, columns=["Mushroom_type", "Extraction_method"], drop_first=True)

# Features (Independent Variables) & Target (Dependent Variable)
features = ["UV_exposure_time", "UV_intensity", "Temperature"] + list(df.columns[3:-1])
X = df[features]
y = df["Vitamin_D3_yield"]

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features (Optional for better performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Scatter Plot: Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.xlabel("Actual Vitamin D3 Yield")
plt.ylabel("Predicted Vitamin D3 Yield")
plt.title("Actual vs Predicted Vitamin D3 Yield (Random Forest)")
plt.show()
