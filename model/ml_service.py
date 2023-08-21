import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image


db = redis.Redis(host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID)

# Loading the model
model = ResNet50(include_top=True, weights="imagenet")


  
def predict(image_name):
    
    """
    Load image and run ML model for predictions.

    Parameters:

    image_name: Image filename.
    Returns:

    class_name: Predicted class as a string.
    pred_probability: Confidence score as a number

    """
    class_name = None
    pred_probability = None
    #Setting the path of the image
    path_image = settings.UPLOAD_FOLDER + "/" + image_name
    #Loading the image
    img = image.load_img(path_image, target_size=(224, 224))
    # Converting the image to an array
    x = image.img_to_array(img)
    #Process the image
    x_batch = np.expand_dims(x, axis=0)
   
    x_batch = preprocess_input(x_batch)
    #Predict the image
    preds = model.predict(x_batch)
    res_model = decode_predictions(preds, top=1)[0]
    # Get the class name and probabilities
    class_name = res_model[0][1]
    pred_probability = round(res_model[0][2],4)
   
    return class_name, pred_probability


def classify_process():
    """
    Continuously monitor Redis for new job requests. Upon receiving a new job, 
    retrieve it from the Redis queue. Utilize the loaded ML model to generate predictions and store the results back in Redis using the original job ID. This allows other services to identify the job as processed and access the corresponding results.

    Load image from the relevant folder based on the provided image name and utilize our ML 
    model to obtain predictions.
    """
    while True:
    
        try:
            
            queue_name, msg = db.brpop(settings.REDIS_QUEUE)
            print("msg",msg)
        
            newmsg = json.loads(msg)
        
            class_name, pred_probability = predict(newmsg["image_name"])

            
            res_dict = {
                "prediction": str(class_name),
                "score": np.float64(pred_probability),
            }


            res_id = newmsg["id"]
        
            db.set(res_id, json.dumps(res_dict))
        except:
            raise SystemExit("ERROR: Results Not Stored")
            

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    print("Launching ML service...")
    classify_process()
