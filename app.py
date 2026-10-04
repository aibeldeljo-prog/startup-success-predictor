import joblib
from flask import Flask, render_template, request

app = Flask(__name__)


# ============================================================
# LOAD MODEL FILES
# ============================================================

model = joblib.load("model/startup_success_model.pkl")
imputer = joblib.load("model/imputer.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


# ============================================================
# FEATURES USED BY THE TRAINED MODEL
# ============================================================

FEATURES = [
    "funding_total_usd",
    "funding_rounds",
    "milestones",
    "relationships",
    "avg_participants",
]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        prediction=None,
        success_probability=None,
        unsuccessful_probability=None
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    form = request.form

    # --------------------------------------------------------
    # Get values from the HTML form
    # --------------------------------------------------------

    funding_total_usd = float(
        form.get("funding_total_usd", 0) or 0
    )

    funding_rounds = float(
        form.get("funding_rounds", 0) or 0
    )

    milestones = float(
        form.get("milestones", 0) or 0
    )

    relationships = float(
        form.get("relationships", 0) or 0
    )

    avg_participants = float(
        form.get("avg_participants", 0) or 0
    )


    # --------------------------------------------------------
    # Arrange features in the exact order used during training
    # --------------------------------------------------------

    ordered = [
        funding_total_usd,
        funding_rounds,
        milestones,
        relationships,
        avg_participants,
    ]


    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    X = imputer.transform([ordered])


    # --------------------------------------------------------
    # Get prediction probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[0]


    # Probability of class 1 = success
    success_probability = float(probabilities[1]) * 100

    # Probability of class 0 = unsuccessful
    unsuccessful_probability = float(probabilities[0]) * 100


    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    if success_probability >= 50:

        prediction = "Acquired (likely success)"

    else:

        prediction = "Closed (higher risk)"


    # --------------------------------------------------------
    # Render result page
    # --------------------------------------------------------

    return render_template(
        "index.html",

        prediction=prediction,

        success_probability=round(
            success_probability, 1
        ),

        unsuccessful_probability=round(
            unsuccessful_probability, 1
        )
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)