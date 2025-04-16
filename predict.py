import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
sys.stdout.reconfigure(encoding='utf-8')


# 1. Load your trained model
model_path = "./malayalam_sentiment_model_gpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 3. Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return model.config.id2label[probs.argmax().item()]

# 4. Test a Malayalam sentence
text = "ഈ സിനിമ വളരെ നല്ലതായിരുന്നു"  # Replace with your text
sentiment = predict(text)
print(f"Text: {text}")
print(f"Sentiment: {sentiment}")