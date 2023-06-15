from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Define the input data structure


class PredictionRequest(BaseModel):
    age: float
    sex: int
    rbc: float
    rbc: float
    hct: float
    mcv: float
    mch: float
    mchc: float
    rdw_cv: float
    wbc: float
    wbc: float
    wbc: float
    mo: float
    eos: float
    ba: float


# Load the trained model
model = load_model('elaborate_model.h5')
scaler = StandardScaler()

# Define the prediction route


@app.post("/predict")
def predict(request: PredictionRequest):
    # Prepare the input data
    input_data = np.array(list(request.dict().values())).astype(np.float32)
    # Scale the input using the same scaler used during training
    input_data = scaler.transform([input_data])

    # Make the prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    # Define the class labels
    class_labels = {
        0: "Normal health status: no immediate need for medical consultation",
        1: "It is advisable to seek a medical consultation in order to facilitate a comprehensive assessment for further diagnostic purposes.",
        2: "Medical consultation recommended for moderate condition",
        3: "Immediate medical consultation required"
    }

    # Create the response
    response = {
        "prediction": class_labels[predicted_class]
    }

    return response
