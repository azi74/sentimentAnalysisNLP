import pandas as pd
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from flask import Flask, request, jsonify

# ======================
# 1. Configuration (GPU Optimized)
# ======================
CONFIG = {
    "data_path": "labeledMarked.csv",
    "text_column": "text",
    "label_column": "sentiment",
    "model_name": "ai4bharat/indic-bert",
    "max_length": 128,
    "batch_size": 16,  # Increased for GPU
    "epochs": 5,
    "output_dir": "./malayalam_sentiment_model_gpu",
    "test_size": 0.2,
    "random_state": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Malayalam synonym dictionary
MALAYALAM_SYNONYMS = {
    # Positive Sentiment
    'നല്ല': ['ഉത്തമം', 'മികച്ച', 'അതിമനോഹരം', 'പ്രശസ്തമായ', 'അദ്ഭുതം', 'അതുല്യം', 'ആകർഷണീയം', 'ഉന്നതമായ'],
    'മികച്ച': ['അത്യുത്തമം', 'വിശിഷ്ടം', 'പ്രതിഭാസമ്പന്നമായ', 'അസാധാരണമായ', 'ശ്രേഷ്ഠം'],
    'സുന്ദരം': ['മനോഹരം', 'അലങ്കാരം', 'ഭംഗിയുള്ള', 'ചന്തമുള്ള', 'രമണീയം'],
    'രസകരം': ['വിനോദാത്മകം', 'മനോരഞ്ജനം', 'ആനന്ദദായകം', 'ഹാസ്യാത്മകം', 'വിനോദം'],
    'ആകർഷണീയം': ['മോഹിപ്പിക്കുന്ന', 'ശ്രദ്ധേയം', 'ഗാംഭീര്യമുള്ള', 'വിസ്മയജനകം', 'അത്ഭുതകരം'],
    
    # Negative Sentiment
    'മോശം': ['ചീത്ത', 'ക്ഷീണം', 'അപര്യാപ്തം', 'നിസ്സാരം', 'ഹീനം', 'നിസ്സത്ത്', 'അപ്രിയം'],
    'വിഷമം': ['ദുഃഖം', 'ഖേദം', 'സങ്കടം', 'വേദന', 'ക്ലേശം', 'പീഡ', 'അസ്വസ്ഥത'],
    'ഭയങ്കരം': ['ദാരുണം', 'ഘോരം', 'വികടം', 'അമാനുഷികം', 'ക്രൂരം'],
    'വിരസം': ['ക്ലേശകരം', 'അരുചികരം', 'ശുഷ്കം', 'നീരസം', 'അസഹ്യം'],
    'അസംതൃപ്തി': ['അതൃപ്തി', 'അപ്രസന്നത', 'ക്ഷോഭം', 'അസന്തുഷ്ടി', 'അസ്വാസ്ഥ്യം'],
    
    # Neutral/Descriptive
    'സാധാരണ': ['പതിവ്', 'സ്വാഭാവികം', 'നിസ്സംശയം', 'പൊതുവായ', 'യഥാർത്ഥ'],
    'വ്യത്യസ്തം': ['പ്രത്യേകം', 'വിഭിന്നം', 'അസാമാന്യം', 'അപൂർവം', 'വൈവിധ്യം'],
    'സങ്കീർണ്ണം': ['ജടിലം', 'ഗൂഢാര്ത്ഥമുള്ള', 'ദുര്ഗ്രഹം', 'പെരുമാറ്റസങ്കീർണ്ണത', 'അസ്പഷ്ടം'],
    'വേഗം': ['ശീഘ്രം', 'ത്വരിതം', 'ദ്രുതഗതി', 'ചുറുക്ക്', 'ആവേഗം'],
    'വലുത്': ['വിശാലം', 'ഭീമാകാരം', 'പ്രപഞ്ചം', 'വിസ്തൃതം', 'വിശാലമായ'],
    
    # Movie-specific terms
    'അഭിനയം': ['നടനം', 'പ്രതിപാദനം', 'അഭിനിവേശം', 'പാത്രധാരണം', 'സാക്ഷാത്കാരം'],
    'കഥ': ['കഥാവസ്തു', 'പ്ലോട്ട്', 'കാവ്യം', 'ആഖ്യാനം', 'വിവരണം'],
    'സംവിധാനം': ['ദിശാനിർദേശം', 'നിർമ്മാണം', 'ക്രമീകരണം', 'അധിനിർദേശം', 'സംഘടന'],
    'സംഗീതം': ['ഗാനം', 'ലയം', 'സ്വരം', 'താളം', 'ശബ്ദസംയോജനം'],
    'ഛായാഗ്രഹണം': ['സിനിമാട്ടോഗ്രഫി', 'ഫോട്ടോഗ്രഫി', 'ചിത്രീകരണം', 'ദൃശ്യഗ്രഹണം', 'ഇമേജറി'],
    
    # Intensity modifiers
    'വളരെ': ['അത്യധികം', 'അമിതമായി', 'അതിശയിച്ച്', 'അസാധാരണമായി', 'അതിമാത്രം'],
    'കുറച്ച്': ['സ്വല്പം', 'അല്പം', 'അൽപമായ', 'ചെറുതായി', 'മാത്രമായ'],
    'പൂർണ്ണമായി': ['സമ്പൂർണ്ണമായി', 'മുഴുവനായി', 'പരിപൂർണ്ണമായി', 'അശേഷം', 'നിറവേറ്റി'],
    'അൽപ്പം': ['അണുമാത്രം', 'സ്വല്പമായ', 'ഇഷ്ടമായ', 'ചെറിയ അളവിൽ', 'കുറഞ്ഞ'],
    
    # Common verbs in reviews
    'ആസ്വദിക്കുക': ['രുചികാണുക', 'അനുഭവിക്കുക', 'സ്വീകരിക്കുക', 'ആനന്ദിക്കുക', 'അഭിരമിക്കുക'],
    'വിമര്ശിക്കുക': ['ശകാരിക്കുക', 'ആക്ഷേപിക്കുക', 'തിരസ്കരിക്കുക', 'എതിർക്കുക', 'ദൂഷണം'],
    'പ്രശംസിക്കുക': ['സ്തുതിക്കുക', 'അഭിനന്ദിക്കുക', 'പുകഴ്ത്തുക', 'ആദരിക്കുക', 'മാനിക്കുക'],
    'ശുപാർശചെയ്യുക': ['സുഹൃത്തായി നിർദ്ദേശിക്കുക', 'പ്രോത്സാഹിപ്പിക്കുക', 'ഉപദേശിക്കുക', 'അനുമോദിക്കുക', 'പിന്തുണയ്ക്കുക']
}

# ======================
# 2. Data Preparation
# ======================
def clean_malayalam(text):
    text = str(text)
    text = re.sub(r'[^\u0D00-\u0D7F\s.,!?0-9]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def augment_text(text, n=2):
    words = text.split()
    augmented = []
    for _ in range(n):
        new_words = []
        for word in words:
            if word in MALAYALAM_SYNONYMS and random.random() > 0.7:
                new_words.append(random.choice(MALAYALAM_SYNONYMS[word]))
            else:
                new_words.append(word)
        augmented.append(' '.join(new_words))
    return augmented

def load_and_preprocess_data():
    df = pd.read_csv(CONFIG["data_path"])
    df['cleaned_text'] = df[CONFIG["text_column"]].apply(clean_malayalam)
    
    augmented = []
    for _, row in df.iterrows():
        for aug_text in augment_text(row['cleaned_text']):
            augmented.append({
                'text': aug_text,
                'sentiment': row[CONFIG["label_column"]],
                'cleaned_text': aug_text
            })
    
    combined_df = pd.concat([df, pd.DataFrame(augmented)])
    combined_df['label'] = combined_df[CONFIG["label_column"]].map(
        {'negative': 0, 'neutral': 1, 'positive': 2}
    )
    return combined_df

# ======================
# 3. Model Training (GPU Accelerated)
# ======================
class MalayalamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        encoding = self.tokenizer(
            text,
            max_length=CONFIG["max_length"],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

def train_model(combined_df):
    try:
        # Initialize accelerator with GPU
        accelerator = Accelerator()
        device = accelerator.device
        print(f"\nTraining on device: {device}")

        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            num_labels=3,
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            label2id={"negative": 0, "neutral": 1, "positive": 2}
        ).to(device)

        # Data splits
        X_train, X_test, y_train, y_test = train_test_split(
            combined_df['cleaned_text'],
            combined_df['label'],
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"]
        )
        
        # Datasets
        train_dataset = MalayalamDataset(X_train, y_train, tokenizer)
        test_dataset = MalayalamDataset(X_test, y_test, tokenizer)

        # Training setup with GPU optimizations
        training_args = TrainingArguments(
            output_dir=CONFIG["output_dir"],
            num_train_epochs=CONFIG["epochs"],
            per_device_train_batch_size=CONFIG["batch_size"],
            per_device_eval_batch_size=32,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir='./logs',
            logging_steps=10,
            fp16=True,  # Enable mixed precision
            gradient_accumulation_steps=2,
            optim="adamw_torch_fused"  # Optimized for CUDA
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # Accelerate preparation
        trainer = accelerator.prepare(trainer)
        
        # Train and save
        trainer.train()
        trainer.save_model(CONFIG["output_dir"])
        tokenizer.save_pretrained(CONFIG["output_dir"])

        # Evaluate
        results = trainer.evaluate()
        print("\nTraining Results:")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\nTraining Error: {str(e)}")
        return None, None

# ======================
# 4. Flask API (GPU Optimized)
# ======================
def create_flask_app(model, tokenizer):
    app = Flask(__name__)
    
    # Ensure model is on GPU and in eval mode
    device = torch.device(CONFIG["device"])
    model = model.to(device).eval()
    print(f"\nAPI Model loaded on: {device}")

    # Warmup GPU
    with torch.no_grad():
        dummy_input = tokenizer("GPU warmup", return_tensors="pt").to(device)
        _ = model(**dummy_input)

    @app.route('/')
    def home():
        return f"""
        <h1>Malayalam Sentiment Analysis (RTX 4060 GPU)</h1>
        <p>Device: <strong>{device}</strong></p>
        <p>Send POST request to /predict with JSON body:</p>
        <pre>{{ "text": "മലയാളം വാക്യം" }}</pre>
        """

    @app.route('/predict', methods=['POST'])
    def predict():
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        text = request.json.get('text')
        if not text:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        try:
            # Tokenize and move to GPU
            inputs = tokenizer(
                text,
                max_length=CONFIG["max_length"],
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # GPU-accelerated prediction
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get results
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs).item()
            
            return jsonify({
                "text": text,
                "sentiment": model.config.id2label[pred],
                "confidence": float(probs[0][pred].item()),
                "device": str(device),
                "performance": "gpu_accelerated"
            })
            
        except Exception as e:
            return jsonify({
                "error": str(e),
                "device": str(device)
            }), 500
    
    return app

# ======================
# 5. Main Execution
# ======================
if __name__ == "__main__":
    # Verify CUDA
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Using Device: {CONFIG['device']}\n")
    
    # Load data
    print("1. Loading and preprocessing data...")
    data = load_and_preprocess_data()
    
    # Train model
    print("\n2. Training model with GPU acceleration...")
    model, tokenizer = train_model(data)
    
    if model and tokenizer:
        # Start Flask API
        print("\n3. Starting GPU-accelerated Flask API...")
        app = create_flask_app(model, tokenizer)
        app.run(host='0.0.0.0', port=5000, threaded=True)
    else:
        print("\nFailed to initialize model. Check errors above.")