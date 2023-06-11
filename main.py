from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = FastAPI()

# Load the model
model = load_model('./model.h5')


# Define the request model
class InputData(BaseModel):
    age: float
    sex: int
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


# Define the response model
class Prediction(BaseModel):
    interpretation: str
    class_: int


# Define the default route
@app.get("/")
def hello():
    return {"message": "fastAPI CloudRun Tensorflow Deployment"}


# Endpoint for model prediction
@app.post("/predict", response_model=Prediction)
def predict(input_data: InputData):
    # Convert input data to a NumPy array
    input_array = np.array([[input_data.age, input_data.sex, input_data.rbc, input_data.hgb, input_data.hct,
                            input_data.mcv, input_data.mch, input_data.mchc, input_data.rdw_cv, input_data.wbc,
                            input_data.neu, input_data.lym, input_data.mo, input_data.eos, input_data.ba]])

    # Perform prediction using the model
    predictions = model.predict(input_array)

    # Decode the predictions
    interpretation_idx = np.argmax(predictions[:, :2])
    class_idx = np.argmax(predictions[:, 2:])
    interpretation = interpretation_idx
    class_ = class_idx

    # Create the response
    response = Prediction(interpretation=interpretation, class_=class_)
    return response
