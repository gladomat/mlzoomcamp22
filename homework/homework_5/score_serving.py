import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict(customer, model, dv):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

with open('model2.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('score')

@app.route('/predict', methods=['POST'])
def predict_view():
    customer = request.get_json()
    y_pred = predict(customer, model, dv)

    result = {
        'score_proba': y_pred,
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)