from transformers import AutoTokenizer, AutoConfig
from optimum.onnxruntime import ORTModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

labels = {0: "NEGATIVE", 1: "POSITIVE"}

config = AutoConfig.from_pretrained(model_name)
config.id2label = labels
config.label2id = {v: k for k, v in labels.items()}

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    export=True
)

model.save_pretrained("onnx_model")
tokenizer.save_pretrained("onnx_model")