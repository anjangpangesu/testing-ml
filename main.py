from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the input data schema


class InputData(BaseModel):
    age: int
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

# Define the output data schema


class OutputData(BaseModel):
    prediction: int


# Load the ML model
try:
    model = load_model('./elaborate.h5')
except:
    raise Exception("Failed to load the ML model.")

# Initialize the FastAPI app
app = FastAPI()

# Define the get endpoint


@app.get("/")
def hello():
    return {"message": "ML Model successfully deployed."}

# Define the prediction endpoint


@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    try:
        # Convert input data to a numpy array
        input_array = np.array([
            [data.age, data.sex, data.rbc, data.hgb, data.hct, data.mcv, data.mch,
             data.mchc, data.rdw_cv, data.wbc, data.neu, data.lym, data.mo, data.eos, data.ba]
        ])

        # Perform the prediction
        prediction = model.predict(input_array)

        # Convert the prediction to the corresponding class label
        class_label = np.argmax(prediction, axis=1)[0]

        # Create the output data
        output_data = OutputData(prediction=class_label)

        return output_data
    except:
        raise Exception("Failed to make a prediction.")
