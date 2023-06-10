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
    ]).reshape(1, -1)

    predict = model.predict(input_array)
    res = int(predict.item())

    interpretations = {
        0: "Normal health status: no immediate need for medical consultation",
        1: "It is advisable to seek a medical consultation in order to facilitate a comprehensive assessment for further diagnostic purposes.",
        2: "Medical consultation recommended for moderate condition",
        3: "Immediate medical consultation required"
    }

    interpretation = interpretations[res]
    return {"Interpretation": res, "Description": interpretation}
