import logging
import os
import json
import azure.functions as func
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from numpy import argmax


MODELS_DIR = 'models'


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    input_data = req.params.get('data')
    if not input_data:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            input_data = req_body.get('data')

    if input_data:
        # animals = ['cat', 'dog', 'unknown']
        # answer = random.choice(animals)
        class_names = ['cat', 'dog']
        model = load_model(os.path.join(MODELS_DIR, os.getenv('CURRENT_MODEL')))
        img_array = expand_dims(input_data, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return func.HttpResponse(json.dumps({'message': class_names[argmax(score)],
                                             'data': score.numpy(),
                                             'status': 'success'}),
                                 mimetype="application/json")
    else:
        return func.HttpResponse(json.dumps({'message': 'improper or no input data',
                                             'status': 'failure'}),
                                 mimetype="application/json", status_code=400)
        # "This HTTP triggered function executed successfully. "
        # "Pass a name in the query string or in the request body for a personalized response.",
