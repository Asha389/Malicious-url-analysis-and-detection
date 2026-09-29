from flask import Flask, render_template, request
from predict import predict_url

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    url = request.form.get("url", "").strip()

    if not url:
        return render_template(
            "index.html",
            error="Please enter a URL."
        )

    try:
        result, confidence = predict_url(url)

        return render_template(
            "index.html",
            url=url,
            prediction=result,
            confidence=round(confidence, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )