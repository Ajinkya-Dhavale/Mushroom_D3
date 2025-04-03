from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("random_forest_d3_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        uv_exposure_time = float(request.form["uv_exposure_time"])
        uv_intensity = float(request.form["uv_intensity"])
        mushroom_type = request.form["mushroom_type"]
        temperature = float(request.form["temperature"])
        extraction_method = request.form["extraction_method"]

        # Debugging: Print received values
        print(f"Received: UV Time={uv_exposure_time}, UV Intensity={uv_intensity}, Type={mushroom_type}, Temp={temperature}, Extraction={extraction_method}")

        # Convert categorical variables to match model input (One-Hot Encoding)
        mushroom_types = ["Shiitake", "White Button", "Oyster"]  # Update with actual types
        extraction_methods = ["Solvent", "Heat Extraction", "UV Exposure"]

        # Create input data with placeholders
        input_data = {f"Mushroom_type_{m}": 0 for m in mushroom_types}
        input_data[f"Mushroom_type_{mushroom_type}"] = 1  # Set selected type
        
        extraction_data = {f"Extraction_method_{e}": 0 for e in extraction_methods}
        extraction_data[f"Extraction_method_{extraction_method}"] = 1  # Set selected method

        # Prepare final input
        input_values = [uv_exposure_time, uv_intensity, temperature] + list(input_data.values()) + list(extraction_data.values())
        input_array = np.array(input_values).reshape(1, -1)

        # Make Prediction
        prediction = model.predict(input_array)[0]

        # Format prediction to two decimal places
        formatted_prediction = f"{prediction:.2f}"

        return render_template("result.html", prediction=formatted_prediction)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)