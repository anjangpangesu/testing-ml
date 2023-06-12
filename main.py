from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase credentials and Firestore
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

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
model = load_model('./elaborate_model.h5')

# Initialize the FastAPI app
app = FastAPI()

# Define the GET endpoint to retrieve data


@app.get("/data/{data_id}")
def get_data(data_id: str):
    doc_ref = db.collection('data').document(data_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return {"message": "Data not found"}

# Define the GET endpoint for hello message


@app.get("/")
def hello():
    return {"message": "ML Model Success to Deploy"}

# Define the prediction endpoint


@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    # Convert input data to a numpy array
    input_array = np.array([[
        data.age, data.sex, data.rbc, data.hgb, data.hct, data.mcv, data.mch,
        data.mchc, data.rdw_cv, data.wbc, data.neu, data.lym, data.mo, data.eos, data.ba
    ]])

    # Perform the prediction
    prediction = model.predict(input_array)

    # Convert the prediction to the corresponding class label
    class_label = np.argmax(prediction, axis=1)[0]

    # Create the output data
    output_data = OutputData(prediction=class_label)

    # Save the input and output data to Firestore
    data_dict = {
        'input_data': data.dict(),
        'output_data': output_data.dict()
    }
    doc_ref = db.collection('data').document()
    doc_ref.set(data_dict)

    return output_data
