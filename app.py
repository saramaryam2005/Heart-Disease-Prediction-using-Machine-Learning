import os
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Model aur scaler load karein
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scalar = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = rf_model.predict(new_data)
    return jsonify(int(output[0]))

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = rf_model.predict(final_input)[0]
    
    return render_template(
        "home.html",
        prediction_text="The chances of Heart Disease is (0=no, 1=yes): {}".format(int(output))
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)