# Bank Term Deposit Subscription Prediction System

## 1. Project Overview
The **Bank Term Deposit Subscription Prediction System** is a predictive machine learning web application built to determine whether a customer is likely to subscribe to a bank term deposit. 

By analyzing outbound direct marketing campaign parameters, customer demographic details, and financial indicators, the application utilizes a trained **Random Forest Classifier** model to output predictions and subscription probabilities. This system helps financial institutions optimize marketing strategies, reduce campaign costs, and improve conversion rates by prioritizing high-probability customer profiles.

---

## 2. Features
- **Interactive Customer Profiling**: Predict customer response using inputs like age, account balance, job, marital status, education, credit defaults, and loans.
- **Subscription Probability Meter**: Displays the prediction class ("Likely to Subscribe" vs. "Unlikely to Subscribe") alongside a prominent circular SVG gauge representing the exact probability percentage calculated via `predict_proba()`.
- **Session Prediction History Table**: Keeps a running log of the last 5 prediction runs in the current session.
- **Dataset Overview Dashboard**: Displays dataset specifications, features glossary, and class imbalance charts (45,211 records).
- **Model Comparison Suite**: Compares accuracy metrics across Naive Bayes, Decision Tree, Random Forest, and Neural Network (MLP) models, highlighting the production model selection.
- **Feature Importance Chart**: Dynamically parses and displays relative feature importances read straight from the Random Forest model.
- **Settings & Environment Specs**: Explains technologies used and prints real-time server operating environment versions.

---

## 3. Project Structure
| File / Directory | Description |
| :--- | :--- |
| `Bank_dataset.csv` | Original campaign CSV data containing 45,211 customer profiles and target responses. |
| `bank_model.pkl` | Serialized Random Forest Classification model (Accuracy: 87.76%) used for inference. |
| `encoders.pkl` | Serialized scikit-learn LabelEncoder objects for preprocessing categorical dropdown features. |
| `train.py` | Python training script demonstrating data preprocessing, splits, model training, evaluation, and serialization. |
| `test_model.py` | Verification script to load ML model files and test predictions on sample data profiles. |
| `flask_app.py` | Flask web application backend, handling page rendering, prediction pipeline routing, and dynamic data loading. |
| `requirements.txt` | Standard project dependencies file listing required packages for server execution. |
| `templates/` | Directory containing Jinja2 HTML pages (`base.html`, `index.html`, `datasets.html`, `models.html`, `settings.html`). |
| `static/` | Directory storing styling rules (`static/css/style.css`) and client-side interactions (`static/js/script.js`). |

---

## 4. Dependencies
Verify if each dependency is installed on your computer:
1. **Python** (version 3.8+ recommended)
2. **Flask** (Web framework)
3. **Pandas** (Data structure library)
4. **NumPy** (Numerical computing)
5. **Scikit-Learn** (ML algorithms)
6. **Joblib** (Object serialization)

To check for installed packages, run:
```bash
python -c "import flask, pandas, numpy, sklearn, joblib; print('All dependencies are installed!')"
```

---

## 5. Installation Guide (Windows Users)
Follow these step-by-step instructions:
1. **Open Command Prompt (CMD)**:
   - Press the Windows Key on your keyboard.
   - Type `cmd` and press **Enter**.
2. **Verify Python Installation**:
   - In the Command Prompt window, type:
     ```cmd
     python --version
     ```
   - If a version number is displayed, proceed.
   - If not recognized, download the latest version of Python from [python.org](https://www.python.org/downloads/) (make sure to check **"Add Python to PATH"** during installation).
3. **Navigate to the Project Folder**:
   - Locate your project directory path (e.g., `C:\Users\STC\Desktop\ML LAB TRYING`).
   - In Command Prompt, navigate to it:
     ```cmd
     cd "C:\Users\STC\Desktop\ML LAB TRYING"
     ```
4. **Install Required Packages**:
   - Run the following command to download and install packages via pip:
     ```cmd
     pip install -r requirements.txt
     ```

---

## 6. How to Run the Project
1. Open Command Prompt inside the project directory.
2. Run the web server using:
   ```bash
   python flask_app.py
   ```
3. Open your web browser and go to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 7. Classification Threshold
The system uses the following classification threshold based on the model's output subscription probability:
- **Probability $\ge$ 50%** $\rightarrow$ **Likely to Subscribe** (Green success badge)
- **Probability $<$ 50%** $\rightarrow$ **Unlikely to Subscribe** (Red danger badge)

---

## 8. Demo Test Cases
Below are verified customer profiles and their corresponding predicted subscription probabilities generated directly by the Random Forest model:

### Likely to Subscribe ($\ge$ 50% Probability)
1. **Case 1**:
   - **Inputs**: Age: `28`, Balance: `5000`, Job: `student`, Marital: `single`, Education: `tertiary`, Default: `no`, Housing: `no`, Loan: `no`
   - **Result**: `Likely to Subscribe` (Probability: **74.67%**)
2. **Case 2**:
   - **Inputs**: Age: `28`, Balance: `10000`, Job: `student`, Marital: `single`, Education: `tertiary`, Default: `no`, Housing: `no`, Loan: `no`
   - **Result**: `Likely to Subscribe` (Probability: **57.88%**)
3. **Case 3**:
   - **Inputs**: Age: `42`, Balance: `8000`, Job: `management`, Marital: `single`, Education: `tertiary`, Default: `no`, Housing: `no`, Loan: `no`
   - **Result**: `Likely to Subscribe` (Probability: **51.00%**)

### Unlikely to Subscribe ($<$ 50% Probability)
1. **Case 4**:
   - **Inputs**: Age: `35`, Balance: `2000`, Job: `management`, Marital: `married`, Education: `tertiary`, Default: `no`, Housing: `yes`, Loan: `no`
   - **Result**: `Unlikely to Subscribe` (Probability: **0.00%**)
2. **Case 5**:
   - **Inputs**: Age: `60`, Balance: `10000`, Job: `retired`, Marital: `married`, Education: `tertiary`, Default: `no`, Housing: `no`, Loan: `no`
   - **Result**: `Unlikely to Subscribe` (Probability: **13.00%**)

---

## 9. How the Prediction Works
1. **Inputs Collection**: The user enters customer features (demographics, balance, loans) in the frontend form.
2. **Categorical Preprocessing**: Flask retrieves the inputs and applies encoders from `encoders.pkl` to transform labels (e.g., "management", "yes") into numeric classifications.
3. **Model Evaluation**: The system loads the trained Random Forest model from `bank_model.pkl` and formats the features into the exact training column order:
   `[age, job, marital, education, default, balance, housing, loan]`
4. **Output Generation**:
   - `model.predict()` determines classification outcome (Likely/Unlikely).
   - `model.predict_proba()` calculates prediction probability (0-100%).

---

## 10. Model Information
- **Algorithm**: Random Forest Classifier
- **Dataset Size**: 45,211 records
- **Input Dimensions**: 8 Features
- **Testing Accuracy**: 87.76%

---

## 11. Project Screenshots
*Here are placeholders for application screen captures. To view these pages, run the application locally and navigate to the dashboard.*
- **Predictor Page**: `[Screenshot Placeholder: Predictor Dashboard]`
- **Dataset Page**: `[Screenshot Placeholder: Dataset Overview]`
- **Models Page**: `[Screenshot Placeholder: Model Comparison & Feature Importance]`

---

## 12. Troubleshooting
- **`python` is not recognized**: Add Python to your Windows system Environment Variables PATH, or reinstall Python checking the "Add to PATH" option.
- **`ModuleNotFoundError`**: Re-run the installation command `pip install -r requirements.txt`.
- **Port 5000 already in use**: Terminate any other running server tasks on port 5000, or edit `flask_app.py` to change `port=5000` to `port=8080` in `app.run()`.
- **Missing model files**: Ensure `bank_model.pkl` and `encoders.pkl` exist in the root of the project directory.

---

## 13. Team Information
- **Zaigham Rouf** (BS Artificial Intelligence)
- **Fahad Mumtaz** (BS Artificial Intelligence)
- **Course**: Machine Learning Lab
- **Project Scope**: Final Lab Project Submission

---

## 14. Conclusion
The Bank Term Deposit Subscription Prediction System successfully demonstrates the application of ensemble learning models in real-world retail banking scenarios. Developed for our Machine Learning Lab curriculum, it models data pipeline design, model serialization, and dashboard integration.
