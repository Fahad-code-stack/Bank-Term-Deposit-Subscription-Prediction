import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders once at startup
try:
    model = joblib.load("bank_model.pkl")
    encoders = joblib.load("encoders.pkl")
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model = None
    encoders = {}

@app.route("/")
def index():
    # Extract classes for dropdowns dynamically
    dropdown_options = {}
    if encoders:
        for key in ["job", "marital", "education", "default", "housing", "loan"]:
            if key in encoders:
                dropdown_options[key] = list(encoders[key].classes_)
            
    return render_template("index.html", options=dropdown_options)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # 1. Input Extraction and Validation
        try:
            age = int(data.get("age", 0))
            if age <= 0:
                return jsonify({"error": "Age must be a valid positive integer."}), 400
        except ValueError:
            return jsonify({"error": "Age must be a valid positive integer."}), 400
            
        try:
            balance = float(data.get("balance", 0.0))
        except ValueError:
            return jsonify({"error": "Balance must be a valid number."}), 400
            
        # 2. Gather inputs in exact feature order
        sample = {
            "age": age,
            "job": data.get("job"),
            "marital": data.get("marital"),
            "education": data.get("education"),
            "default": data.get("default"),
            "balance": balance,
            "housing": data.get("housing"),
            "loan": data.get("loan")
        }
        
        # 3. Transform categorical inputs using encoders
        for col in ["job", "marital", "education", "default", "housing", "loan"]:
            if col in encoders and sample.get(col) is not None:
                try:
                    sample[col] = encoders[col].transform([sample[col]])[0]
                except ValueError:
                    return jsonify({"error": f"Invalid value for {col}."}), 400
                
        # 4. Create DataFrame and Predict (enforce strict column order matching training)
        feature_order = ["age", "job", "marital", "education", "default", "balance", "housing", "loan"]
        sample_df = pd.DataFrame([sample])[feature_order]
        
        # Logging details for audit/debugging
        print("\n--- Flask Prediction Pipeline Debug Info ---")
        print(f"Raw user inputs: {data}")
        print(f"Encoded values: {sample}")
        print(f"Final DataFrame sent to model:\n{sample_df}")
        print("Feature types:\n", sample_df.dtypes)
        print("--------------------------------------------\n")
        
        prediction = model.predict(sample_df)[0]
        probabilities = model.predict_proba(sample_df)[0]
        
        # Prediction logic (1 = Subscribe, 0 = Not Subscribe)
        probability_sub = probabilities[1] * 100
        
        result_text = "Likely to Subscribe" if prediction == 1 else "Unlikely to Subscribe"
        
        return jsonify({
            "success": True,
            "result": result_text,
            "probability": round(probability_sub, 2),
            "model_used": "Random Forest",
            "accuracy": "87.76%"
        })
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/datasets")
def datasets():
    return render_template("datasets.html")

@app.route("/models")
def models():
    feature_names = ["Age", "Job", "Marital Status", "Education", "Credit Default", "Balance", "Housing Loan", "Personal Loan"]
    importances_raw = model.feature_importances_ if model is not None else [0] * 8
    importances = []
    for name, imp in zip(feature_names, importances_raw):
        importances.append({
            "name": name,
            "importance": round(imp * 100, 2)
        })
    importances.sort(key=lambda x: x["importance"], reverse=True)
    return render_template("models.html", importances=importances)

@app.route("/settings")
def settings():
    import sys
    import platform
    import sklearn
    import flask
    sys_info = {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "scikit_learn_version": sklearn.__version__,
        "flask_version": flask.__version__,
    }
    return render_template("settings.html", sys_info=sys_info)

@app.route("/presentation-3d")
def presentation_3d():
    return render_template("presentation_3d.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)

