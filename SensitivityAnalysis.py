import numpy as np
import matplotlib.pyplot as plt
import joblib  # Import joblib to load the model
import pandas as pd

# Load the trained Random Forest model
rf_model = joblib.load("random_forest_d3_model.pkl")
print("Model loaded successfully!")

# Load dataset to get feature names
df = pd.read_csv("synthetic_mushroom_d3_large.csv")
df = pd.get_dummies(df, columns=["Mushroom_type", "Extraction_method"], drop_first=True)

# Features
X = df.drop(columns=["Vitamin_D3_yield"])

# Define a base input sample
base_sample = X.iloc[0].copy()  # Start with a random sample

# Test effect of UV exposure time
uv_exposure_values = np.linspace(10, 180, 50)  # Vary UV exposure from 10 to 180 mins
predicted_yields = []

for uv_time in uv_exposure_values:
    sample = base_sample.copy()
    sample["UV_exposure_time"] = uv_time  # Modify only UV time
    predicted_yield = rf_model.predict([sample])[0]
    predicted_yields.append(predicted_yield)

# Plot the effect of UV exposure time
plt.figure(figsize=(8, 5))
plt.plot(uv_exposure_values, predicted_yields, marker="o", linestyle="-", color="blue")
plt.xlabel("UV Exposure Time (minutes)")
plt.ylabel("Predicted Vitamin D3 Yield (IU/g)")
plt.title("Effect of UV Exposure on Vitamin D3 Yield")
plt.show()
