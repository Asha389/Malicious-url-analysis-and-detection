import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATASET_PATH = "dataset/phishing.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["label"])
y = df["label"]

print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {y.nunique()}")


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 4. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")
print("This may take some time...")

start_time = time.time()

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")


# ==========================================
# 5. PREDICTION
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ==========================================
# 6. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 9. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/malicious_url_model.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")

print("\nModel saved successfully!")
print("models/malicious_url_model.pkl")
print("models/feature_columns.pkl")

print("\nProject ML training completed.")