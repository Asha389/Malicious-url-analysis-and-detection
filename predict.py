import joblib
import pandas as pd
from urllib.parse import urlparse

from feature_extraction import extract_features

MODEL_PATH = "models/final_malicious_url_model.pkl"
FEATURES_PATH = "models/final_feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)


# Known legitimate domains for demonstration/sanity checking.
TRUSTED_DOMAINS = {
    "google.com",
    "www.google.com",
    "microsoft.com",
    "www.microsoft.com",
    "apple.com",
    "www.apple.com",
    "amazon.com",
    "www.amazon.com",
    "github.com",
    "www.github.com",
}


def normalize_url(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def is_trusted_domain(url):
    try:
        parsed = urlparse(normalize_url(url))
        hostname = parsed.hostname

        if not hostname:
            return False

        hostname = hostname.lower()

        return (
            hostname in TRUSTED_DOMAINS
            or any(hostname.endswith("." + domain) for domain in TRUSTED_DOMAINS)
        )

    except Exception:
        return False


def predict_url(url):

    # Application-level trusted-domain safeguard
    if is_trusted_domain(url):
        return "SAFE", 99.0

    features = extract_features(url)

    X = pd.DataFrame([features])

    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    X = X.fillna(0)

    prediction = int(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]

    confidence = float(max(probabilities)) * 100

    if prediction == 0:
        result = "SAFE"
    else:
        result = "MALICIOUS"

    return result, confidence


if __name__ == "__main__":

    url = input("Enter URL: ").strip()

    if not url:
        print("Please enter a URL.")
    else:

        result, confidence = predict_url(url)

        print("\n===================================")
        print("       MALICIOUS URL DETECTOR")
        print("===================================")
        print(f"URL        : {url}")
        print(f"Prediction : {result}")
        print(f"Confidence : {confidence:.2f}%")
        print("===================================")