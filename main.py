from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf


# Initialize the FastAPI app
app = FastAPI()


# Define the input data schema
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


class OutputData(BaseModel):
    prediction: int


# Load the ML model
try:
    model = tf.keras.models.load_model('elaborate_model.h5')
except:
    raise Exception("Failed to load the ML model.")


@app.get("/")
def hello():
    return {"message": "ML Model successfully deployed."}


# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to a numpy array
        input_array = np.array([
            [data.age, data.sex, data.rbc, data.hgb, data.hct, data.mcv, data.mch,
             data.mchc, data.rdw_cv, data.wbc, data.neu, data.lym, data.mo, data.eos, data.ba]
        ])

        # Perform the prediction
        prediction = model.predict(input_array)

        return {"prediction": prediction}
    except:
        raise Exception("Failed to make a prediction.")
