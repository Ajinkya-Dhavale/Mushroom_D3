import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of synthetic entries
num_entries = 200

# Possible values for categorical features
mushroom_types = ["Shiitake", "Button", "Oyster", "Portobello"]
extraction_methods = ["Solvent", "Heat Extraction", "UV Exposure"]

# Generate random synthetic data
data = {
    "UV_exposure_time": np.random.randint(10, 120, num_entries),  # Time in minutes (10-120 min)
    "UV_intensity": np.round(np.random.uniform(1.0, 5.0, num_entries), 2),  # Intensity in W/m²
    "Mushroom_type": [random.choice(mushroom_types) for _ in range(num_entries)],
    "Temperature": np.random.randint(25, 80, num_entries),  # Temperature in Celsius (25-80°C)
    "Extraction_method": [random.choice(extraction_methods) for _ in range(num_entries)],
}

# Function to simulate Vitamin D3 yield based on the above factors
def generate_vitamin_d3(uv_time, uv_intensity, temp, mushroom, method):
    base_yield = 50  # Base IU/g of Vitamin D3
    
    # Adjust yield based on UV exposure
    yield_d3 = base_yield + (uv_time * 0.8) + (uv_intensity * 10)
    
    # Mushroom type impact
    if mushroom == "Shiitake":
        yield_d3 *= 1.2
    elif mushroom == "Button":
        yield_d3 *= 1.1
    elif mushroom == "Oyster":
        yield_d3 *= 1.05
    else:  # Portobello
        yield_d3 *= 1.15

    # Extraction method impact
    if method == "Solvent":
        yield_d3 *= 1.3
    elif method == "Heat Extraction":
        yield_d3 *= 1.2

    # Temperature effect
    if 40 <= temp <= 60:
        yield_d3 *= 1.1  # Optimal range
    elif temp > 60:
        yield_d3 *= 0.9  # Too much heat reduces yield

    return round(yield_d3, 2)

# Generate Vitamin D3 Yield using the function
data["Vitamin_D3_yield"] = [
    generate_vitamin_d3(data["UV_exposure_time"][i], 
                         data["UV_intensity"][i], 
                         data["Temperature"][i], 
                         data["Mushroom_type"][i], 
                         data["Extraction_method"][i]) 
    for i in range(num_entries)
]

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset as CSV
df.to_csv("mushroom_d3_data.csv", index=False)

# Display first few rows
print(df.head())
