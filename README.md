# 🍄 Vitamin D3 Yield Prediction

This project predicts the **Vitamin D3 yield** in mushrooms based on various environmental and extraction factors. It uses a **Random Forest model** trained on a dataset with different mushroom types, UV exposure, and extraction methods.

---

## 📌 Features
- Train a **Random Forest Regressor** to predict Vitamin D3 yield.
- **Web interface** built using Flask for user-friendly input and predictions.
- **One-hot encoding** for categorical variables (Mushroom Type & Extraction Method).
- Model trained on a **synthetic dataset** and saved for inference.

---

## 📂 Project Structure
```
/flask_app
│── app.py                # Flask backend for predictions
│── model_training.py      # Script to train and save the model
│── templates/
│   ├── index.html         # Web interface for input
│── static/
│   ├── styles.css         # CSS for styling
│── synthetic_mushroom_d3_large.csv  # Dataset file
│── random_forest_d3_model.pkl       # Trained model
```

---

## 🛠️ Libraries Required
To run this project, install the following dependencies:
```bash
pip install flask pandas numpy scikit-learn joblib
```

---

## 🚀 How to Run
1. **Train the Model** (if not already trained):
   ```bash
   python model_training.py
   ```
   This will train the model and save it as `random_forest_d3_model.pkl`.

2. **Start the Flask Web App**:
   ```bash
   python app.py
   ```
   The app will be available at **http://127.0.0.1:5000/**.

3. **Use the Web Interface**:
   - Enter input parameters like UV exposure time, mushroom type, and extraction method.
   - Click "Predict" to get the estimated Vitamin D3 yield.

---

## 📊 Model Performance
The trained **Random Forest Regressor** achieved:
- **Mean Absolute Error (MAE):** ~X.XX
- **R² Score:** ~X.XX (Indicating model accuracy)

---

## 🛠 Future Improvements
- **Enhance Dataset** with real-world mushroom yield data.
- **Hyperparameter tuning** to improve prediction accuracy.
- **Deploy on cloud** for remote access and API integration.

---