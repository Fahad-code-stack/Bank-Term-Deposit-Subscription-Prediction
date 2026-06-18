import joblib
import pandas as pd

model = joblib.load("bank_model.pkl")
encoders = joblib.load("encoders.pkl")

sample = {
    "age": 55,
    "job": "retired",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 15000,
    "housing": "no",
    "loan": "no"
}

for col in encoders:
    if col != "y":
        sample[col] = encoders[col].transform([sample[col]])[0]

sample_df = pd.DataFrame([sample])

prediction = model.predict(sample_df)

probability = model.predict_proba(sample_df)

print("Prediction:", prediction)
print("Probabilities:", probability)
print("Subscribe Probability:", round(probability[0][1] * 100, 2), "%")

if prediction[0] == 1:
    print("Prediction: Customer will subscribe to the term deposit")
else:
    print("Prediction: Customer will NOT subscribe to the term deposit")

    print(sample_df)
    print(model.classes_)