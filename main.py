from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os

# Definisikan struktur data input untuk API


class InputData(BaseModel):
    age: int
    sex: str
    rbc: float
    hgb: float
    hct: float
    mcv: float
    mch: float
    mchc: float
    rdwcv: float
    wbc: float
    neutrophils: float
    lymphocytes: float
    monocytes: float
    eosinophils: float
    basophils: float


# Inisialisasi FastAPI
app = FastAPI()

# Muat model TensorFlow Lite (.tflite)
model_path = os.path.join(os.path.dirname(
    __file__), "./elaborate_model.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# Definisikan endpoint untuk melakukan prediksi


@app.post('/predict')
def predict(data: InputData):
    # Ubah data input menjadi array NumPy
    input_data = np.array([
        data.Age,
        data.Sex,
        data.RBC,
        data.HGB,
        data.HCT,
        data.MCV,
        data.MCH,
        data.MCHC,
        data.RDW,
        data.WBC,
        data.NEU,
        data.LYM,
        data.MO,
        data.Eos,
        data.Baso
    ], dtype=np.float32)

    # Masukkan data input ke model TensorFlow Lite
    interpreter.set_tensor(
        input_details[0]['index'], input_data.reshape((1, -1)))
    interpreter.invoke()

    # Dapatkan output dari model
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Ambil kelas prediksi
    predicted_class = np.argmax(output_data)

    # Return hasil prediksi
    return {'predicted_class': predicted_class}
