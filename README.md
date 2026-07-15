# Credit Card Approval Prediction using Machine Learning

## Overview

This project is a **Machine Learning-based web application** that predicts whether a credit card application is likely to be **Approved** or **Rejected** based on the applicant's information.

The application is built using **Python**, **Flask**, **Scikit-learn**, **Pandas**, **HTML**, and **CSS**.

---

## Features

- User-friendly web interface
- Machine Learning prediction using Random Forest
- Professional and responsive design
- Fast prediction results
- Easy to understand and maintain

---

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML
- CSS

---

## Project Structure

```text
Credit_Card_Approval_Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── credit_record.csv
│
├── model/
│   ├── model.pkl
│   └── label_encoders.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── style.css
```

---

## Installation

### 1. Clone or Download the Project

Download the project or clone the repository.

---

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

### 3. Activate the Virtual Environment

**Windows PowerShell**

```bash
.venv\Scripts\Activate
```

**Windows Command Prompt**

```bash
.venv\Scripts\activate.bat
```

---

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

---

### 5. Run the Application

```bash
python app.py
```

---

### 6. Open in Browser

Visit:

```
http://127.0.0.1:5000
```

---

## Input Features

The prediction model uses the following features:

- Gender
- Age
- Debt
- Married
- Bank Customer
- Industry
- Ethnicity
- Years Employed
- Prior Default
- Employed
- Credit Score
- Driver's License
- Citizen
- Zip Code
- Income

---

## Machine Learning Model

- Algorithm: **Random Forest Classifier**
- Data preprocessing includes label encoding for categorical features.
- The trained model is stored using Joblib.

---

## How It Works

1. The user enters applicant details.
2. Flask receives the form data.
3. Categorical values are encoded.
4. The trained Random Forest model predicts the result.
5. The application displays either **Approved** or **Rejected**.

---

## Future Improvements

- Display prediction confidence.
- Add input validation.
- Deploy the application online.
- Improve the user interface with charts and analytics.

---

## Developed By

Agola Arifulla

B.Tech CSE (Artificial Intelligence)

Mini Project – Credit Card Approval Prediction
