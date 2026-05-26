import pickle
from flask import Flask,request,app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
app= Flask (__name__)
rf_model=pickle.load(open("rf_model (1).pkl","rb"))
scalar = pickle.load(open("scaler.pkl", "rb"))
@app.route("/")
def home():
    return render_template("home html") 

@app.route("/predict_api", methods=["post"])

def predict_api():
    data=request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) 
    new_data= scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=rf_model.predict(new_data)
    print(output[0])
    return jsonify(int(output[0]))

if __name__ =="__main__":
    app.run(debug=True)