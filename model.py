from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "textattack/bert-base-uncased-imdb"

labels = {0: "NEGATIVE", 1: "POSITIVE"}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    id2label=labels,
    label2id={v:k for k, v in labels.items()}
    )
model.eval()