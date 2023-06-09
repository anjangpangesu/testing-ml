from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf


# Inisialisasi FastAPI
app = FastAPI()

# Definisikan struktur data input menggunakan Pydantic BaseModel


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


# Muat model dari file H5
model = tf.keras.models.load_model('./elaborate_model.h5')


@app.get("/")
def hello():
    return {"message": "fastAPI CloudRun Tensorflow Deployment"}

# Definisikan endpoint untuk prediksi


@app.post('/predict')
def predict(data: InputData):
    # Ubah data input menjadi array numpy
    input_array = np.array([[data.age, data.sex, data.rbc, data.hgb, data.hct, data.mcv, data.mch, data.mchc,
                             data.rdw_cv, data.wbc, data.neu, data.lym, data.mo, data.eos, data.ba]])

    # Lakukan prediksi menggunakan model
    prediction = model.predict(input_array)

    # Kembalikan hasil prediksi
    return {'prediction': float(prediction[0])}
