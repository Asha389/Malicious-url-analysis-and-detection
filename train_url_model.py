import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DATASET_PATH = "dataset/phishing.csv"
MODEL_PATH = "models/url_malicious_url_model.pkl"
FEATURES_PATH = "models/url_feature_columns.pkl"

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

# Remove rows without a type
df = df.dropna(subset=["type"]).copy()

# Binary target
df["binary_label"] = (df["type"] != "benign").astype(int)

# Features that can be extracted directly from a URL
FEATURES = [
    "url_len",
    "@", "?", "-", "=", ".", "#", "%", "+", "$", "!", "*", ",", "//",
    "digits", "letters",
    "abnormal_url",
    "https",
    "Shortining_Service",
    "having_ip_address",

    "defac_has_hacked_terms",
    "defac_has_suspicious_ext",
    "defac_path_depth",
    "defac_is_deep_path",
    "defac_path_underscores",
    "defac_is_gov_edu",
    "defac_has_index_php",
    "defac_has_option_param",

    "phish_has_brand",
    "phish_brand_in_subdomain",
    "phish_brand_in_path",
    "phish_hyphen_count",
    "phish_digit_count",
    "phish_long_domain",
    "phish_many_subdomains",
    "phish_suspicious_tld",
    "phish_keyword_count",
    "phish_has_redirect",
    "phish_param_count",
    "phish_encoded_chars",

    "enh_urgency_count",
    "enh_security_count",
    "enh_brand_count",
    "enh_brand_hijack",
    "enh_subdomain_count",
    "enh_long_path",
    "enh_many_params",
    "enh_suspicious_tld",

    "adv_domain_ngram_entropy",
    "adv_path_entropy",
    "adv_consonant_ratio",
    "adv_vowel_ratio",
    "adv_digit_ratio",
    "adv_subdomain_count",
    "adv_avg_subdomain_len",
    "adv_token_count",
    "adv_avg_token_length"
]

X = df[FEATURES].copy()
X = X.fillna(0)

y = df["binary_label"]

print("Training samples:", len(X))
print("Features:", len(FEATURES))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
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
print("       URL-ONLY MODEL RESULTS")
print("======================================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print("======================================")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["SAFE", "MALICIOUS"]
))

joblib.dump(model, MODEL_PATH)
joblib.dump(FEATURES, FEATURES_PATH)

print("\nModel saved:")
print(MODEL_PATH)
print(FEATURES_PATH)