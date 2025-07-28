# =============================
# Complete Backend Code with Graphs & Confusion Matrices
# =============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Machine Learning Imports ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Instead of using plt.style.use('seaborn-whitegrid'), we use seaborn's theme.
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.1)

# ----------------------------
# Fraud Model & Prediction Functions
# ----------------------------
class FraudModel:
    """
    A simple fraud prediction model based on keyword frequency and numerical thresholds.
    """
    def __init__(self, fraud_type):
        self.fraud_type = fraud_type

    def predict_proba(self, df):
        suspicious_keywords = [
            "suspicious", "urgent", "immediately", "fraud", "scam", "error",
            "fake", "alert", "anomaly", "warning", "blocked", "compromise", 
            "re-confirm", "win", "big sale", "security check", "verify now", 
            "money", "bitcoin", "gift card"
        ]
        prob = 0.0
        text_fields = []
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, str):
                text_fields.append(val)
        if text_fields:
            combined_text = " ".join(text_fields).lower()
            count = sum(combined_text.count(word) for word in suspicious_keywords)
            base_prob = 0.3
            prob = base_prob + 0.1 * count
        for col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, (int, float)) and val > 1000:
                prob += 0.1
        prob = min(prob, 1.0)
        return [[1 - prob, prob]]

def get_fraud_model(fraud_type):
    """Return an instance of the FraudModel for the specified fraud type."""
    return FraudModel(fraud_type)

def predict_fraud(model, df):
    """
    Use the provided model to predict fraud.
    Returns a tuple of (prediction, probability).
    """
    try:
        prob = model.predict_proba(df)[0][1]
        pred = "Fraud Detected" if prob > 0.5 else "No Fraud"
        text_cols = ['content', 'post_content', 'ad_text', 'invoice_content', 
                     'claim_description', 'communication_content', 'review_text', 
                     'image_text', 'applicant_details']
        if any(col in df.columns for col in text_cols):
            text_data = " ".join([
                str(df[col].iloc[0]) 
                for col in df.columns 
                if col in text_cols and df[col].iloc[0] is not None
            ])
            if len(text_data) > 100:
                prob = min(prob + 0.1, 1.0)
                pred = "Fraud Detected" if prob > 0.5 else "No Fraud"
        return pred, prob
    except Exception as e:
        print("Prediction error:", e)
        return "Unknown", 0.0

def prevention_info(fraud_type, pred, prob):
    """
    Given the fraud type, prediction, and probability,
    return a tuple with (alert level, advice, support contact).
    """
    if pred == "No Fraud":
        return ("SAFE",
                "No fraudulent activity detected. Continue routine monitoring.",
                "Contact: support@example.com")
    else:
        if prob > 0.8:
            return ("CRITICAL",
                    "Immediate action required – suspend activity and investigate.",
                    "Hotline: 911")
        elif prob > 0.6:
            return ("WARNING",
                    "Possible fraud detected – review transaction details and verify sender authenticity.",
                    "Helpdesk: 1800-123-4567")
        else:
            return ("CAUTION",
                    "Minor anomalies detected – monitor and flag if trends persist.",
                    "Support: support@example.com")

# ----------------------------
# Synthetic Data & Model Training Functions
# ----------------------------
def load_synthetic_data_basic():
    """
    Generate basic synthetic fraud data for model training.
    """
    np.random.seed(42)
    size = 500
    X1 = np.random.normal(loc=50, scale=10, size=size)
    X2 = np.random.normal(loc=100, scale=20, size=size)
    X3 = np.random.randint(0, 2, size=size)
    X4 = np.random.normal(loc=0, scale=1, size=size)
    y = []
    for i in range(size):
        score = 0
        if X1[i] > 55:
            score += 1
        if X2[i] > 120:
            score += 1
        if X3[i] == 1:
            score += 1
        if X4[i] > 0.5:
            score += 1
        y.append(1 if score >= 2 else 0)
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'FraudLabel': y
    })
    return df

def train_basic_models_with_models(X, y):
    """
    Train several basic models on synthetic data and return a dictionary with:
    - trained_models: {model_name: (model, X_test, y_test)}
    - accuracies: {model_name: accuracy}
    """
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=1000)
    }
    accuracies = {}
    trained_models = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[model_name] = acc
        trained_models[model_name] = (model, X_test, y_test)
    return trained_models, accuracies

def load_advanced_transaction_data(num_samples=5000, random_state=42):
    """
    Generate synthetic advanced transaction data for fraud detection.
    """
    np.random.seed(random_state)
    df_size = num_samples
    amount = np.random.uniform(0, 10000, df_size)
    time_of_day = np.random.randint(0, 24, df_size)
    location_score = np.random.randint(0, 101, df_size)
    device_score = np.random.randint(0, 101, df_size)
    transaction_type = np.random.randint(0, 4, df_size)
    currency = np.random.randint(0, 4, df_size)
    user_auth_level = np.random.randint(0, 3, df_size)
    velocity = np.random.uniform(0.0, 5.0, df_size)
    suspicious_past_transactions = np.random.randint(0, 6, df_size)
    blacklisted_merchant = np.random.randint(0, 2, df_size)
    labels = []
    for i in range(df_size):
        suspicious_count = 0
        if blacklisted_merchant[i] == 1:
            suspicious_count += 1
        if suspicious_past_transactions[i] > 2:
            suspicious_count += 1
        if amount[i] > 8000:
            suspicious_count += 1
        if (transaction_type[i] == 2) and (user_auth_level[i] == 0):
            suspicious_count += 1
        if (location_score[i] < 20) and (device_score[i] < 20):
            suspicious_count += 1
        if velocity[i] > 2.5:
            suspicious_count += 1
        labels.append(1 if suspicious_count >= 2 else 0)
    df = pd.DataFrame({
        "amount": amount,
        "time_of_day": time_of_day,
        "location_score": location_score,
        "device_score": device_score,
        "transaction_type": transaction_type,
        "currency": currency,
        "user_auth_level": user_auth_level,
        "velocity": velocity,
        "suspicious_past_transactions": suspicious_past_transactions,
        "blacklisted_merchant": blacklisted_merchant,
        "FraudLabel": labels
    })
    return df

def train_advanced_transaction_models(df):
    """
    Train multiple classifiers on the advanced transaction dataset.
    Returns a dictionary of trained models and a dictionary of their accuracy scores.
    Each trained model tuple is (model, X_test, y_test).
    """
    X = df.drop("FraudLabel", axis=1)
    y = df["FraudLabel"]
    classifiers = {
        "LogisticRegression": LogisticRegression(),
        "SVC": SVC(probability=True),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "GaussianNB": GaussianNB()
    }
    trained_models = {}
    accuracies = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = acc
        trained_models[name] = (clf, X_test, y_test)
    return trained_models, accuracies

# ----------------------------
# Data Generation & Model Training
# ----------------------------
# Basic synthetic data and models
df_synthetic_basic = load_synthetic_data_basic()
X_basic = df_synthetic_basic.drop('FraudLabel', axis=1)
y_basic = df_synthetic_basic['FraudLabel']
basic_models, basic_accuracies = train_basic_models_with_models(X_basic, y_basic)

# Advanced transaction data and models
df_advanced_tx = load_advanced_transaction_data()
advanced_models, advanced_accuracies = train_advanced_transaction_models(df_advanced_tx)

# ----------------------------
# Visualization & Comparison Graphs
# ----------------------------

# Graph 1: Bar Chart for Basic Model Accuracies
df_acc_basic = pd.DataFrame({
    'Model': list(basic_accuracies.keys()),
    'Accuracy': list(basic_accuracies.values())
})
plt.figure(figsize=(8,5))
plt.bar(df_acc_basic['Model'], df_acc_basic['Accuracy'], color='skyblue')
plt.title("Basic Synthetic Fraud Model Accuracies")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

# Graph 2: Pie Chart for Basic Model Accuracy Distribution
plt.figure(figsize=(6,6))
plt.pie(df_acc_basic['Accuracy'], labels=df_acc_basic['Model'], autopct='%1.1f%%', startangle=90)
plt.title("Accuracy Distribution for Basic Models")
plt.show()

# Graph 3: Print Accuracy Table for Basic Models
print("Basic Synthetic Fraud Model Accuracies:")
print(df_acc_basic)

# Graph 4: Bar Chart for Advanced Model Accuracies
df_adv_acc = pd.DataFrame({
    "Model": list(advanced_accuracies.keys()),
    "Accuracy": list(advanced_accuracies.values())
}).sort_values(by="Accuracy", ascending=False)
plt.figure(figsize=(10,5))
plt.bar(df_adv_acc['Model'], df_adv_acc['Accuracy'], color='lightgreen')
plt.title("Advanced Transaction Fraud Model Accuracies")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.xticks(rotation=45)
plt.show()

# Graph 5: Pie Chart for Advanced Model Accuracy Distribution
plt.figure(figsize=(6,6))
plt.pie(df_adv_acc['Accuracy'], labels=df_adv_acc['Model'], autopct='%1.1f%%', startangle=90)
plt.title("Accuracy Distribution for Advanced Models")
plt.show()

# Graph 6: Print Accuracy Table for Advanced Models
print("\nAdvanced Transaction Fraud Model Accuracies:")
print(df_adv_acc)

# Graph 7: Comparison Bar Chart for Common Models (Basic vs Advanced)
common_models = set(basic_accuracies.keys()).intersection({"LogisticRegression", "RandomForest", "KNN", "MLP"})
comparison_data = []
for model in common_models:
    comparison_data.append({
        "Model": model,
        "Basic Accuracy": basic_accuracies[model],
        "Advanced Accuracy": advanced_accuracies.get(model, np.nan)
    })
df_comparison = pd.DataFrame(comparison_data)
x = np.arange(len(df_comparison))
width = 0.35
plt.figure(figsize=(8,5))
plt.bar(x - width/2, df_comparison["Basic Accuracy"], width, label="Basic", color='cornflowerblue')
plt.bar(x + width/2, df_comparison["Advanced Accuracy"], width, label="Advanced", color='seagreen')
plt.xticks(x, df_comparison["Model"])
plt.title("Comparison of Basic vs Advanced Model Accuracies")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.legend()
plt.show()

# Graph 8: Line Plot of Fraud Labels (Basic Synthetic Data)
plt.figure(figsize=(10,4))
plt.plot(df_synthetic_basic.index, df_synthetic_basic['FraudLabel'], marker='o', linestyle='-', color='purple')
plt.title("Fraud Labels Across Basic Synthetic Data Samples")
plt.xlabel("Sample Index")
plt.ylabel("Fraud Label (0=No Fraud, 1=Fraud)")
plt.show()

# Graph 9: Histogram of Fraud Labels (Basic Synthetic Data)
plt.figure(figsize=(6,4))
plt.hist(df_synthetic_basic['FraudLabel'], bins=3, color='orange', edgecolor='black', rwidth=0.8)
plt.title("Distribution of Fraud Labels in Basic Data")
plt.xlabel("Fraud Label")
plt.ylabel("Count")
plt.xticks([0,1])
plt.show()

# Graph 10: Feature Importances from RandomForest (Advanced Data)
if "RandomForest" in advanced_models:
    rf_model, _, _ = advanced_models["RandomForest"]
    importances = rf_model.feature_importances_
    features = df_advanced_tx.drop("FraudLabel", axis=1).columns
    df_importances = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(10,5))
    plt.bar(df_importances['Feature'], df_importances['Importance'], color='salmon')
    plt.title("Feature Importances from RandomForest (Advanced Data)")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.show()
else:
    print("RandomForest model not available for feature importances.")

# Graph 11: Correlation Heatmap for Advanced Transaction Data Features
corr = df_advanced_tx.drop("FraudLabel", axis=1).corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Advanced Transaction Features")
plt.show()

# Graph 12: Histogram of 'amount' Distribution (Advanced Data)
plt.figure(figsize=(8,5))
plt.hist(df_advanced_tx['amount'], bins=30, color='mediumpurple', edgecolor='black')
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

# Graph 13: Pie Chart of Fraud vs No Fraud Distribution (Advanced Data)
fraud_counts = df_advanced_tx['FraudLabel'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(fraud_counts, labels=fraud_counts.index.map({0:"No Fraud",1:"Fraud"}), autopct='%1.1f%%', startangle=90)
plt.title("Fraud vs No Fraud Distribution (Advanced Data)")
plt.show()

# Graph 14: Scatter Plot: Amount vs Time of Day (Advanced Data, colored by FraudLabel)
plt.figure(figsize=(8,5))
colors = df_advanced_tx['FraudLabel'].map({0:'blue', 1:'red'})
plt.scatter(df_advanced_tx['time_of_day'], df_advanced_tx['amount'], c=colors, alpha=0.5)
plt.title("Transaction Amount vs Time of Day (Color by FraudLabel)")
plt.xlabel("Time of Day")
plt.ylabel("Amount")
plt.show()

# Graph 15: Boxplot of Transaction Amount by FraudLabel (Advanced Data)
plt.figure(figsize=(8,5))
df_advanced_tx['FraudLabel_str'] = df_advanced_tx['FraudLabel'].map({0:"No Fraud", 1:"Fraud"})
sns.boxplot(x="FraudLabel_str", y="amount", data=df_advanced_tx, palette="Set2")
plt.title("Transaction Amount Distribution by Fraud Label")
plt.xlabel("Fraud Label")
plt.ylabel("Amount")
plt.show()

# ----------------------------
# Confusion Matrices for Each Model
# ----------------------------

def plot_conf_matrix(cm, title):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

# Graph 16: Confusion Matrices for Basic Models
print("\n--- Confusion Matrices for Basic Models ---")
for model_name, (model, X_test, y_test) in basic_models.items():
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plot_conf_matrix(cm, f"Basic Model: {model_name}")

# Graph 17: Confusion Matrices for Advanced Models
print("\n--- Confusion Matrices for Advanced Models ---")
for model_name, (model, X_test, y_test) in advanced_models.items():
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plot_conf_matrix(cm, f"Advanced Model: {model_name}")

# ----------------------------
# Summary Tables Printed
# ----------------------------
print("\n--- Summary of Basic Model Accuracies ---")
print(df_acc_basic)
print("\n--- Summary of Advanced Model Accuracies ---")
print(df_adv_acc)
print("\n--- Comparison of Common Models (Basic vs Advanced) ---")
print(df_comparison)
