from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Model
model = joblib.load(os.path.join(BASE_DIR, "decision_tree_model.pkl"))

# Load Label Encoders
encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))

print("Model Loaded Successfully")
print("Features Expected:", model.n_features_in_)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():

    # -----------------------------
    # Read Categorical Inputs
    # -----------------------------

    gender = request.form["gender"]
    car = request.form["car"]
    reality = request.form["reality"]

    income_type = request.form["income_type"]
    education = request.form["education"]
    family_status = request.form["family_status"]
    housing = request.form["housing"]

    # -----------------------------
    # Encode Categorical Inputs
    # -----------------------------

    gender = encoders["CODE_GENDER"].transform([gender])[0]
    car = encoders["FLAG_OWN_CAR"].transform([car])[0]
    reality = encoders["FLAG_OWN_REALTY"].transform([reality])[0]

    income_type = encoders["NAME_INCOME_TYPE"].transform([income_type])[0]
    education = encoders["NAME_EDUCATION_TYPE"].transform([education])[0]
    family_status = encoders["NAME_FAMILY_STATUS"].transform([family_status])[0]
    housing = encoders["NAME_HOUSING_TYPE"].transform([housing])[0]

    # -----------------------------
    # Read Numeric Inputs
    # -----------------------------

    children = int(request.form["children"])
    income = float(request.form["income"])

    age = int(request.form["age"])
    employed = int(request.form["employed"])

    work_phone = int(request.form["work_phone"])
    phone = int(request.form["phone"])
    email = int(request.form["email"])

    family_members = int(request.form["family_members"])

    # -----------------------------
    # Derived Features
    # -----------------------------

    days_birth = age * 365
    days_employed = employed * 365

    family_size = family_members

    flag_mobil = 1

    open_month = -15
    end_months = 0
    window = 16

    # -----------------------------
    # Create DataFrame
    # -----------------------------

    data = pd.DataFrame([{
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": car,
        "FLAG_OWN_REALTY": reality,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "FLAG_MOBIL": flag_mobil,
        "FLAG_WORK_PHONE": work_phone,
        "FLAG_PHONE": phone,
        "FLAG_EMAIL": email,
        "CNT_FAM_MEMBERS": family_members,
        "FAMILY_SIZE": family_size,
        "open_month": open_month,
        "end_months": end_months,
        "window": window
    }])

    print("\n================ INPUT ================")
    print(data)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    print("Prediction :", prediction)
    print("Probability:", probability)

    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        prediction_text = "✅ Credit Card Approved"
    else:
        prediction_text = "❌ Credit Card Rejected"

    return render_template(
        "result.html",
        prediction_text=prediction_text,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)