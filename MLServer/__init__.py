import logging
import os.path
import json
import sys
import azure.functions as func
import tensorflow as tf
import pathlib
from numpy import argmax, min, max
from PIL import Image
from datetime import datetime

MODELS_DIR = 'models'


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Entry point for ML functions.
    Requires image to be send in request body.
    Image is then saved locally, and predicted using pre-trained model.

    !!! IMPORTANT !!!
    File must have dimensions 160x160x3. Otherwise it won't work correctly!

    :param req: func.HttpRequest
    :return: func.HttpResponse
    """
    logging.info('Python HTTP trigger function processed a request.')
    base_64_image_bytes = req.get_body()
    if base_64_image_bytes:
        # Get current time data
        timestamp = datetime.now()
        timestamp = timestamp.strftime('%Y%m%d%H%M%S')

        # Check image
        file_size = sys.getsizeof(base_64_image_bytes)
        logging.info(f"Received image of size: {file_size} bytes")

        # Save image locally
        temp_filename = f"temp_image_{timestamp}.jpg"
        logging.info('Saving binary data...')
        with open(temp_filename, 'wb') as img:
            img.write(base_64_image_bytes)
        logging.info(f"File saved: {os.path.isfile(temp_filename)}")
        img = Image.open(temp_filename)

        # Check dimensions
        img2 = Image.open(temp_filename)
        (width, height) = img2.size
        logging.info(f"Saved file dimensions: {width} x {height}")
        if (width, height) != (160, 160):
            return func.HttpResponse(json.dumps({'message': 'input image has incorrect dimensions',
                                                 'data': 'none',
                                                 'status': 'error'}),
                                     mimetype="application/json", status_code=400)

        # Load with Tensorflow
        logging.info("Processing image with Tensorflow")
        img = tf.keras.preprocessing.image.load_img(temp_filename, target_size=(height, width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Load model
        logging.info("Loading Tensorflow model")
        model = tf.keras.models.load_model(os.path.join(pathlib.Path(__file__).parent, 'catordog_model_07_.h5'))

        # Make predictions
        logging.info("Making predictions")
        class_names = ['cat', 'dog']
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        score_diff = max(score) - min(score)

        # Remove temporary file
        logging.info("Removing temporary image file")
        os.remove(temp_filename)

        if score_diff >= 0.5:
            return func.HttpResponse(json.dumps({'message': str(class_names[argmax(score)]),
                                                 'data': str(score.numpy()),
                                                 'status': 'success'}),
                                     mimetype="application/json", status_code=200)
        else:
            return func.HttpResponse(json.dumps({'message': 'unknown',
                                                 'data': str(score.numpy()),
                                                 'status': 'success'}),
                                     mimetype="application/json", status_code=200)

    else:
        return func.HttpResponse(json.dumps({'message': 'improper or no input data',
                                             'data': 'none',
                                             'status': 'error'}),
                                 mimetype="application/json", status_code=400)
