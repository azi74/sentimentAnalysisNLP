import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
sys.stdout.reconfigure(encoding='utf-8')

# Load model and tokenizer
model_path = "./malayalam_sentiment_model_gpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return model.config.id2label[probs.argmax().item()]

# Gradio interface
import gradio as gr

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter a Malayalam sentence"),
    outputs=gr.Label(label="Predicted Sentiment"),
    title="Malayalam Sentiment Analysis",
    description="Type a Malayalam sentence and get the sentiment (Positive, Negative, Neutral).",
)

iface.launch()

