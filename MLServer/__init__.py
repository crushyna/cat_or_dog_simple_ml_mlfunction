import logging
import random
import json

import azure.functions as func

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
        animals = ['cat', 'dog', 'unknown']
        answer = random.choice(animals)

        return func.HttpResponse(json.dumps({'message': answer, 'status': 'success'}),
                                 mimetype="application/json")
    else:
        return func.HttpResponse(json.dumps({'message': 'improper or no input data', 'status': 'failure'}),
                                 mimetype="application/json", status_code=400)
        # "This HTTP triggered function executed successfully. "
        # "Pass a name in the query string or in the request body for a personalized response.",
