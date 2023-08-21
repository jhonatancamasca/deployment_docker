import json
import time
from uuid import uuid4

import redis
import settings

db = redis.Redis(host=settings.REDIS_IP,port=settings.REDIS_PORT,db=settings.REDIS_DB_ID)


def model_predict(image_name):
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    prediction, score : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    prediction = None
    score = None

    
    data_image = {"id": str(uuid4()), "image_name": image_name}

    job_data = json.dumps(data_image)
    job_id = data_image["id"]

    # Send the job to the model service using Redis
    db.lpush(settings.REDIS_QUEUE, job_data)

    while True:
        #Get model predictions
        output = db.get(job_id)

        #Checking
        if output is not None:
            output = json.loads(output.decode("utf-8"))
            prediction = output["prediction"]
            score = output["score"]

            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return prediction, score