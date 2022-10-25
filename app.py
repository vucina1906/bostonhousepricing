# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:17:52 2022

@author: Vuk
"""

import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app = Flask(__name__,template_folder="templates")
#load pickle model
model = pickle.load(open("BHP.pickle","rb"))
scaler = pickle.load(open("scaler.pickle","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))   
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    result = model.predict(scaled_data)
    print(result[0])
    return jsonify(result[0])

@app.route("/predict",methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The predicted price is {}".format(output))
    
    
if __name__ == "__main__":
    app.run(debug=True)
    
