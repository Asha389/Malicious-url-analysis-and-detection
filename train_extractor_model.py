import pandas as pd
import joblib

from feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATASET_PATH = "dataset/phishing.csv"
MODEL_PATH = "models/final_malicious_url_model.pkl"
FEATURES_PATH = "models/final_feature_columns.pkl"

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

df = df.dropna(subset=["url", "type"]).copy()

# Binary label
df["binary_label"] = (df["type"] != "benign").astype(int)

# Take equal samples from both classes
safe = df[df["binary_label"] == 0].sample(
    n=min(30000, len(df[df["binary_label"] == 0])),
    random_state=42
)

malicious = df[df["binary_label"] == 1].sample(
    n=min(30000, len(df[df["binary_label"] == 1])),
    random_state=42
)

sample = pd.concat([safe, malicious]).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("URLs selected:", len(sample))
print("Extracting features...")

X_list = []
y = []

for i, row in enumerate(sample.itertuples(index=False), start=1):

    try:
        features = extract_features(row.url)

        X_list.append(features)
        y.append(row.binary_label)

    except Exception:
        continue

    if i % 5000 == 0:
        print(f"Processed {i}/{len(sample)} URLs")

X = pd.DataFrame(X_list)
X = X.fillna(0)

print("\nFeature matrix:", X.shape)

feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n======================================")
print("       FINAL MODEL RESULTS")
print("======================================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print("======================================")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["SAFE", "MALICIOUS"]
    )
)

joblib.dump(model, MODEL_PATH)
joblib.dump(feature_columns, FEATURES_PATH)

print("\nFinal model saved:")
print(MODEL_PATH)
print(FEATURES_PATH)