# credit_card_approval_decision_tree.py
# Comprehensive workflow for Credit Card Approval Prediction using Decision Tree
# Author: Puli
# Lines: ~700

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Load and Inspect Data
# -----------------------------
def load_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())
    return df

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
def preprocess_data(df):
    print("\nPreprocessing data...")
    # Handle missing values
    df = df.dropna()

    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and target
    X = df.drop("ApprovalStatus", axis=1)  # assuming target column
    y = df["ApprovalStatus"]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders

# -----------------------------
# 3. Train-Test Split
# -----------------------------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Decision Tree Model
# -----------------------------
def d_tree(xtrain, xtest, ytrain, ytest):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(xtrain, ytrain)
    ypred = dt.predict(xtest)

    print("\n*** DecisionTreeClassifier ***")
    print("Confusion Matrix:\n", confusion_matrix(ytest, ypred))
    print("Classification Report:\n", classification_report(ytest, ypred))
    print("Accuracy:", accuracy_score(ytest, ypred))

    return dt

# -----------------------------
# 5. Hyperparameter Tuning
# -----------------------------
def tune_model(xtrain, ytrain):
    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, scoring="accuracy")
    grid.fit(xtrain, ytrain)
    print("\nBest Parameters:", grid.best_params_)
    return grid.best_estimator_

# -----------------------------
# 6. Visualization
# -----------------------------
def visualize_tree(model, feature_names):
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=feature_names, filled=True, fontsize=10)
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

# -----------------------------
# 7. Cross Validation
# -----------------------------
def cross_validation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5)
    print("\nCross-validation scores:", scores)
    print("Mean CV Accuracy:", scores.mean())

# -----------------------------
# 8. Save and Load Model
# -----------------------------
def save_model(model, filename="decision_tree_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def load_model(filename="decision_tree_model.pkl"):
    return joblib.load(filename)

# -----------------------------
# 9. Deployment Simulation
# -----------------------------
def predict_new(model, scaler, label_encoders, new_data):
    # Preprocess new applicant data
    for col, le in label_encoders.items():
        if col in new_data:
            new_data[col] = le.transform([new_data[col]])[0]

    X_new = pd.DataFrame([new_data])
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return "Approved" if prediction[0] == 1 else "Not Approved"

# -----------------------------
# 10. Main Execution
# -----------------------------
def main():
    filepath = "cleaned_merged_data.csv"  # update with your dataset
    df = load_data(filepath)

    correlation_heatmap(df)

    X, y, scaler, label_encoders = preprocess_data(df)
    xtrain, xtest, ytrain, ytest = split_data(X, y)

    # Train baseline model
    model = d_tree(xtrain, xtest, ytrain, ytest)

    # Hyperparameter tuning
    best_model = tune_model(xtrain, ytrain)
    ypred_best = best_model.predict(xtest)
    print("\nTuned Model Accuracy:", accuracy_score(ytest, ypred_best))

    # Cross-validation
    cross_validation(best_model, X, y)

    # Visualization
    visualize_tree(best_model, df.drop("ApprovalStatus", axis=1).columns)

    # Save model
    save_model(best_model)

    # Deployment simulation
    new_applicant = {
        "Age": 35,
        "Income": 50000,
        "EmploymentStatus": "Employed",
        "CreditScore": 700,
        "MaritalStatus": "Single"
    }
    result = predict_new(best_model, scaler, label_encoders, new_applicant)
    print("\nNew Applicant Prediction:", result)

if __name__ == "__main__":
    main()
