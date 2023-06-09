from fastapi import FastAPI
from pydantic import BaseModel

# Import TensorFlow and TFLiteInterpreter
import tensorflow as tf
from tensorflow.lite.python import interpreter as tflite

# Define the input schema


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

# Define the output schema


class OutputData(BaseModel):
    prediction: int


# Initialize the FastAPI app
app = FastAPI()

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='elaborate_model.tflite')
interpreter.allocate_tensors()

# Define the prediction route


@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    # Preprocess the input data
    input_data = [data.age, data.sex, data.rbc, data.hgb, data.hct, data.mcv, data.mch, data.mchc, data.rdw_cv,
                  data.wbc, data.neu, data.lym, data.mo, data.eos, data.ba]
    input_data = tf.convert_to_tensor([input_data], dtype=tf.float32)

    # Set the input tensor to the interpreter
    interpreter.set_tensor(interpreter.get_input_details()[
                           0]['index'], input_data)

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor from the interpreter
    output_data = interpreter.get_tensor(
        interpreter.get_output_details()[0]['index'])

    # Create the output response
    response = OutputData(prediction=int(output_data[0]))

    return response
