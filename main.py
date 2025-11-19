from fastapi import FastAPI
from model import tokenizer, model
from pydantic import BaseModel
import torch

app = FastAPI()


class InputText(BaseModel):
    text: str

@app.post("/model")
async def classification_model(request: InputText):

    text = request.text

    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)

    prediction = outputs.logits.argmax(dim=1).item()

    label = model.config.id2label[prediction]
    return {
        "user_input": text, 
        "prediction": label
    }