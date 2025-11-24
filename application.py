from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

application = Flask(__name__)

# Load models
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('model/scaler.pkl', 'rb'))

app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form data (ensure order matches training features)
        data = [float(x) for x in request.form.values()]
        final_input = scaler_model.transform([data])
        prediction = ridge_model.predict(final_input)[0]

        return render_template(
            "index.html",
            prediction_text=f"Fire Danger Prediction: {prediction:.2f}"
        )

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
