from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

app = FastAPI()

class InputText(BaseModel):
    text: str

## Inference Session

tokenizer = AutoTokenizer("onnx_model")

session = ort.InferenceSession(
    "onnx_model/model.onnx",
    providers=["CPUExecutionProvider"]
)

def generate_onnex_inputs(session, tokens):
    onnx_inputs = {}
    for inp in session.get_inputs():
        name = inp.name
        onnx_inputs[name] = tokens[name]
    return onnx_inputs


@app.post("/model")
async def classification_model(request: InputText):

    text = request.text

    tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)

    ort_inputs = generate_onnex_inputs(session, tokens)

    output = session.run(None, ort_inputs)[0]

    logits = output[0]

    prediction = int(np.argmax(logits, axis=1)[0])

    label_map = {int(k): v for v, k in tokenizer.model_input_names_to_id.items()}

    label = label_map[prediction] if label_map else str(prediction)

    return {
        "user_input": text, 
        "prediction": label
    }