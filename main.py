from fastapi import FastAPI
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import numpy as np
import tensorflow as tf
import os

# Inisialisasi FastAPI
app = FastAPI()

# Konfigurasi Firestore
service_account_key_path = os.path.join(
    os.path.dirname(__file__), 'serviceAccountKey.json')
cred = credentials.Certificate(service_account_key_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Definisikan struktur data input


class PatientData(BaseModel):
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

# Endpoint untuk melakukan prediksi


@app.post("/predict")
def predict(patient_data: PatientData):

    # Mendapatkan path absolut dari file model.tflite
    modelpath = os.path.join(os.path.dirname(
        __file__), 'elaborate_model.tflite')

    # Load model .tflite
    interpreter = tf.lite.Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Mengambil input dan output tensor dari model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocessing data
    input_data = np.array([[patient_data.age, patient_data.sex, patient_data.rbc, patient_data.hgb,
                            patient_data.hct, patient_data.mcv, patient_data.mch, patient_data.mchc,
                            patient_data.rdwcv, patient_data.wbc, patient_data.neutrophils,
                            patient_data.lymphocytes, patient_data.monocytes, patient_data.eosinophils,
                            patient_data.basophils]], dtype=np.float32)

    # Menyimpan data input ke Firestore
    doc_ref = db.collection('patient_data').document()
    doc_ref.set(patient_data.dict())

    # Menjalankan prediksi
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data[0])

    # Menyimpan hasil prediksi ke Firestore
    doc_ref.update({'predicted_class': predicted_class})

    # Mengembalikan hasil prediksi
    return {'predicted_class': predicted_class}


# Menjalankan server FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
