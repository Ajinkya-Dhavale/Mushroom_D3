import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate synthetic data
data = {
    "Mushroom_type": np.random.choice(["Shiitake", "Button", "Oyster", "Portobello"], num_samples),
    "UV_exposure_time": np.random.uniform(10, 180, num_samples),  # in minutes
    "UV_intensity": np.random.uniform(1, 10, num_samples),  # Arbitrary unit
    "Temperature": np.random.uniform(15, 35, num_samples),  # Celsius
    "Humidity": np.random.uniform(50, 90, num_samples),  # Percentage
    "Extraction_method": np.random.choice(["Ethanol", "Water", "Supercritical CO2"], num_samples),
}

# Convert categorical variables into numerical labels
mushroom_d3_base = pd.DataFrame(data)

# Simulated function to calculate Vitamin D3 yield (IU/g)
def calculate_d3_yield(row):
    base_yield = 20  # Base yield in IU/g
    # UV exposure effect
    uv_effect = row["UV_exposure_time"] * (row["UV_intensity"] / 10) * 0.5  
    # Temperature impact (Optimal around 25°C)
    temp_effect = max(0, (30 - abs(row["Temperature"] - 25)) * 0.3)
    # Humidity impact (Optimal 70-80%)
    humidity_effect = max(0, (80 - abs(row["Humidity"] - 75)) * 0.2)

    # Extraction method impact
    method_bonus = {"Ethanol": 5, "Water": 3, "Supercritical CO2": 8}
    extraction_effect = method_bonus[row["Extraction_method"]]

    # Final yield calculation with some randomness
    return base_yield + uv_effect + temp_effect + humidity_effect + extraction_effect + np.random.uniform(-5, 5)

# Apply function to dataset
mushroom_d3_base["Vitamin_D3_yield"] = mushroom_d3_base.apply(calculate_d3_yield, axis=1)

# Save to CSV
mushroom_d3_base.to_csv("synthetic_mushroom_d3_large.csv", index=False)

print("✅ Synthetic dataset with 1000+ entries generated and saved as 'synthetic_mushroom_d3_large.csv'.")
