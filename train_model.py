# ==========================================
# Credit Card Approval Prediction
# Model Training Script
# ==========================================

# Import Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# Load Dataset
# ==========================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

data = pd.read_csv("dataset/credit_record.csv")

print("Dataset Loaded Successfully!\n")

# ==========================================
# Dataset Overview
# ==========================================

print("=" * 60)
print("First 5 Rows")
print("=" * 60)
print(data.head())

print("\n")

print("=" * 60)
print("Dataset Shape")
print("=" * 60)
print(data.shape)

print("\n")

print("=" * 60)
print("Column Names")
print("=" * 60)
print(data.columns.tolist())

print("\n")

print("=" * 60)
print("Dataset Information")
print("=" * 60)
data.info()

print("\n")

print("=" * 60)
print("Missing Values")
print("=" * 60)
print(data.isnull().sum())

# ==========================================
# Separate Features and Target
# ==========================================

X = data.drop("Approved", axis=1)
y = data["Approved"]

# ==========================================
# Identify Column Types
# ==========================================

categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

print("\nNumerical Columns:")
print(list(numerical_columns))

# ==========================================
# Handle Missing Values
# ==========================================

cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

X[categorical_columns] = cat_imputer.fit_transform(X[categorical_columns])
X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])

print("\nMissing Values After Cleaning")
print(X.isnull().sum())

# ==========================================
# Encode Categorical Features
# ==========================================

label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    label_encoders[column] = encoder

print("\nCategorical Features Encoded Successfully!")

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ==========================================
# Train Model
# ==========================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ==========================================
# Model Evaluation
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("Model Accuracy")
print("=" * 60)

print(f"Accuracy : {accuracy:.2%}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, "model/model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("\nModel Saved Successfully!")
print("Location : model/model.pkl")

print("\nLabel Encoders Saved Successfully!")
print("Location : model/label_encoders.pkl")

print("\nProject Training Completed Successfully!")