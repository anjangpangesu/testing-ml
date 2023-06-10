from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

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


# Load the ML model
model = tf.keras.models.load_model('./elaborate.h5')

# Define the get endpoint


@app.get("/")
def hello():
    return {"message": "ML Model Success to Deploy"}


# Define the prediction endpoint


@app.post("/predict", response_model=InputData)
def predict(data: InputData):
    input_array = np.array([
        data.age, data.sex, data.rbc, data.hgb, data.hct, data.mcv, data.mch,
        data.mchc, data.rdw_cv, data.wbc, data.neu, data.lym, data.mo, data.eos, data.ba
    ])

    predict = model.predict(input_array)
    res = predict.item()

    return {"Interpretation": res}
