import logging
import os.path
from typing import Union

import sys
import azure.functions as func
import tensorflow as tf
import pathlib
from numpy import argmax, min, max
from PIL import Image
from datetime import datetime

from .http_asgi import AsgiMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, File, HTTPException, Request
from fastapi.responses import JSONResponse

MODELS_DIR = 'models'


class ResponseData(BaseModel):
    message: str
    data: Union[str, dict]
    status: str


class NoDataException(Exception):
    def __init__(self, text: str):
        self.text = text


app = FastAPI()


@app.exception_handler(NoDataException)
async def unicorn_exception_handler(request: Request, exc: NoDataException):
    return JSONResponse(
        status_code=400,
        content={'message': f"{exc.text}",
                 'data': 'none',
                 'status': 'error'})


@app.post("/api/MLServer", status_code=200, response_model=ResponseData)
async def receive_image(file: bytes = File(...)):
    """
    Entry point for ML functions.
    Requires image to be send in request body.
    Image is then saved locally, and predicted using pre-trained model.

    !!! IMPORTANT !!!
    File must have dimensions 160x160x3. Otherwise it won't work correctly!

    :return: json
    """
    logging.info('Python HTTP trigger function processed a request.')
    base_64_image_bytes = file
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
            raise NoDataException(text='input image has incorrect dimensions')

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
        if max(score) > 0.8:
            if score_diff >= 0.4:
                return ResponseData(message=str(class_names[argmax(score)]),
                                    data={'scores': str(score.numpy()),
                                          'classes': str(class_names),
                                          'score_diff': str(score_diff)
                                          },
                                    status='success')
            else:
                return ResponseData(message='unknown',
                                    data={'scores': str(score.numpy()),
                                          'classes': str(class_names),
                                          'score_diff': str(score_diff)
                                          },
                                    status='success')
        else:
            return ResponseData(message='unknown',
                                data={'scores': str(score.numpy()),
                                      'classes': str(class_names),
                                      'score_diff': str(score_diff)
                                      },
                                status='success')

    else:
        raise NoDataException(text='improper or no input data')


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return AsgiMiddleware(app).handle(req, context)
