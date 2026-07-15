from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load("model/model.pkl")
encoders = joblib.load("model/label_encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Gender": int(request.form["Gender"]),
        "Age": float(request.form["Age"]),
        "Debt": float(request.form["Debt"]),
        "Married": int(request.form["Married"]),
        "BankCustomer": int(request.form["BankCustomer"]),
        "Industry": request.form["Industry"],
        "Ethnicity": request.form["Ethnicity"],
        "YearsEmployed": float(request.form["YearsEmployed"]),
        "PriorDefault": int(request.form["PriorDefault"]),
        "Employed": int(request.form["Employed"]),
        "CreditScore": int(request.form["CreditScore"]),
        "DriversLicense": int(request.form["DriversLicense"]),
        "Citizen": request.form["Citizen"],
        "ZipCode": int(request.form["ZipCode"]),
        "Income": int(request.form["Income"])
    }

    # Encode categorical columns
    data["Industry"] = encoders["Industry"].transform([data["Industry"]])[0]
    data["Ethnicity"] = encoders["Ethnicity"].transform([data["Ethnicity"]])[0]
    data["Citizen"] = encoders["Citizen"].transform([data["Citizen"]])[0]

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "Approved"
    else:
        result = "Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)