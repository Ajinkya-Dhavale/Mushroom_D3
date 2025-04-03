import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib  

# Load dataset
df = pd.read_csv("synthetic_mushroom_d3_large.csv")

# One-Hot Encode Categorical Variables
df = pd.get_dummies(df, columns=["Mushroom_type", "Extraction_method"], drop_first=True)

# Features & Target
X = df.drop(columns=["Vitamin_D3_yield"])
y = df["Vitamin_D3_yield"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get Feature Importance
feature_importance = rf_model.feature_importances_

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importance, color="skyblue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance for Vitamin D3 Yield")
plt.show()

# Save the trained model
joblib.dump(rf_model, "random_forest_d3_model.pkl")
print("Model saved successfully!")

# --- Check Shiitake Yield Compared to Other Mushrooms ---
if "Mushroom_type_Shiitake" in X.columns:
    # Create a base sample
    base_sample = X_train.iloc[0].copy()

    # Modify for Shiitake Mushroom
    sample_shiitake = base_sample.copy()
    sample_shiitake["Mushroom_type_Shiitake"] = 1  # Shiitake selected

    # Modify for Another Mushroom (e.g., Button)
    sample_other = base_sample.copy()
    sample_other["Mushroom_type_Shiitake"] = 0  # Assume another type (e.g., Button)

    # Predict Yield
    yield_shiitake = rf_model.predict([sample_shiitake])[0]
    yield_other = rf_model.predict([sample_other])[0]

    print(f"Predicted Yield for Shiitake: {yield_shiitake:.2f} IU/g")
    print(f"Predicted Yield for Other Mushroom: {yield_other:.2f} IU/g")

    if yield_shiitake > yield_other:
        print("Shiitake mushrooms have the highest yield!")
    else:
        print(" Model does not show Shiitake as highest. Check data or retrain!")

else:
    print("Shiitake Mushroom feature not found in dataset!")
