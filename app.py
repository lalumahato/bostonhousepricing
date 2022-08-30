import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Load the regression model and scaler
model = pickle.load(open('regression.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
  data = request.json['data']
  test_data = np.array(list(data.values())).reshape(1, -1)
  new_data = scaler.transform(test_data)
  output = model.predict(new_data)
  return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
  data = [float(i) for i in request.form.values()]
  input = scaler.transform(np.array(data).reshape(1, -1))
  output = model.predict(input)
  return render_template('home.html', prediction_text='The House price prediction is {}'.format(output))


if __name__ == '__main__':
  app.run(debug=True)


