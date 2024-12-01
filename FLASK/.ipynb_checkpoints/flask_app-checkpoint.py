from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model and preprocessor
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Extract data from the form
    input_data = {
        "ENGINESIZE": [float(request.form["enginesize"])],
        "CYLINDERS": [int(request.form["cylinders"])],
        "FUELCONSUMPTION_CITY": [float(request.form["fuel_city"])],
        "FUELCONSUMPTION_HWY": [float(request.form["fuel_hwy"])],
        "FUELCONSUMPTION_COMB": [float(request.form["fuel_comb"])],
        "FUELCONSUMPTION_COMB_MPG": [int(request.form["fuel_mpg"])],
        "VEHICLECLASS": [request.form["vehicleclass"]],
        "TRANSMISSION": [request.form["transmission"]],
        "FUELTYPE": [request.form["fueltype"]],
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Preprocess and predict
    preprocessed_data = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed_data)

    return render_template(
        "index.html", prediction=round(prediction[0], 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
