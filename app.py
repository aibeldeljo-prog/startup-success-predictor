from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


# Load trained model
model = joblib.load("model/startup_success_model.pkl")

# Load imputer
imputer = joblib.load("model/imputer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        # Get values from the website
        funding = float(request.form["funding"])
        rounds = float(request.form["rounds"])
        milestones = float(request.form["milestones"])
        relationships = float(request.form["relationships"])
        participants = float(request.form["participants"])


        # Create input data
        input_data = pd.DataFrame({
            "funding_total_usd": [funding],
            "funding_rounds": [rounds],
            "milestones": [milestones],
            "relationships": [relationships],
            "avg_participants": [participants]
        })


        # Handle missing values
        input_data = pd.DataFrame(
            imputer.transform(input_data),
            columns=input_data.columns
        )


        # Make prediction
        result = model.predict(input_data)[0]


        # Display result
        if result == 1:
            prediction = "Startup is predicted to be SUCCESSFUL"
        else:
            prediction = "Startup is predicted to be UNSUCCESSFUL"


    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)