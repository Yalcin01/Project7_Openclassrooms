# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:21:46 2021

@author: ykele
"""

# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
import json
import requests
import pickle
import numpy as np


url = 'http://localhost:5000/results'



app = Flask(__name__)
model = pickle.load(open('Modele/model_LGBM_pickle.pkl', 'rb'))



@app.route('/api',methods=['POST'])
def predict():
    

    data = request.get_json(force=True)
    print(data)
    print(type(data))
    
    predict_request=[]
    
    for values in data.values():
        predict_request.append(values)
    #predict_request=[[data['sepal_length'],data['sepal_width'],data['petal_length'],data['petal_width']]]
    print(predict_request)
    print(type(predict_request))
    result=[np.array(predict_request)]
    
    print(result)
    
    #prediction = model.predict(predict_request)
    
    prediction = model.predict(result)
    pred = prediction[0]
    print(pred)
    return jsonify(int(pred))




if __name__ == "__main__":
    app.run(debug=True)
    
    


