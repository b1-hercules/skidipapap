from flask import Flask, request
from flask import Flask, request
from keras.models import load_model
from keras.models import model_from_json
import json
import tensorflow as tf
import numpy as np
import flask

app = Flask(__name__)

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights('model_weights.h5')

graph = tf.get_default_graph()

#request prediksi model
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        a = request.args['a']
        b = request.args['b']
    elif request.method == 'POST':
        a = request.form['a']
        b = request.form['b']

          # Required because of a bug in Keras when using tensorflow graph cross threads
    with graph.as_default():
        result = model.predict(np.array([[a,b]]))[0].tolist()
        data = {'result': result}
        return flask.jsonify(data)

# start the flask app, allow remote connections 
app.run(host='127.0.0.1', port=5000, debug=False)