from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Define the input schema
class InputData(BaseModel):
    age: float
    sex: float
    rbc: float
    hgb: float
    hct: float
    mcv: float
    mch: float
    mchc: float
    rdw_cv: float
    wbc: float
    neu: float
    lym: float
    mo: float
    eos: float
    ba: float


# Load the saved model
model = tf.keras.models.load_model('./model.h5')


# Create the FastAPI app
app = FastAPI()


# Define the default route
@app.get("/")
def hello():
    return {"message": "fastAPI CloudRun Tensorflow Deployment"}


# Define the prediction route
@app.post("/predict")
def predict(data: InputData):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Standardize the input features
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_df)

    # Make predictions
    predictions = model.predict(input_scaled)

    # Prepare the response
    interpretation = np.argmax(predictions[0][:2])
    class_ = np.argmax(predictions[0][2:])
    response = {
        "interpretation": interpretation,
        "class": class_
    }

    return response
