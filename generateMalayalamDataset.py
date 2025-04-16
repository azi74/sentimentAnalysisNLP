import pandas as pd
import random

# Sentiment categories
sentiments = ["positive", "neutral", "negative"]

# Predefined meaningful Malayalam sentence sets
positive_sentences = [
    "ഈ സിനിമ മനോഹരമായ അനുഭവം ആയിരുന്നു.",
    "അഭിനയം വളരെ മികച്ചതാണ്.",
    "കഥ വളരെ ആകർഷണീയമായിരുന്നു.",
    "സംഗീതം മനോഹരമായി ചേര്‍ന്നിരിക്കുന്നു.",
    "ദൃശ്യം വളരെ സുന്ദരമായിരുന്നു.",
    "ഈ അനുഭവം എനിക്ക് സന്തോഷം നല്‍കി.",
    "നിർമ്മാണം ഉന്നത നിലവാരത്തിലായിരുന്നു.",
    "ഞാൻ ഈ സിനിമ മറ്റുള്ളവർക്ക് ശുപാർശ ചെയ്യും."
]

neutral_sentences = [
    "സിനിമ 2 മണിക്കൂർ നീളമുള്ളതാണ്.",
    "പ്രധാന കഥാപാത്രം ഒരു പോലീസ് ഓഫീസറാണ്.",
    "കഥ ഗ്രാമത്തിലായിരുന്നു.",
    "ഇത് ഒരു കുടുംബdrama ആണ്.",
    "ഇത് 2022-ൽ പുറത്തിറങ്ങിയ സിനിമയാണ്.",
    "നടി നയൻതാര പ്രധാനവേഷം അവതരിപ്പിക്കുന്നു.",
    "ഫിലിം ഷൂട്ടിംഗ് കേരളത്തിൽ നടന്നു.",
    "സിനിമയുടെ ഭാഷ മലയാളമാണ്."
]

negative_sentences = [
    "കഥ വളരെ ബോധമില്ലാത്തതായിരുന്നു.",
    "അഭിനയം തികച്ചും മോശമായിരുന്നു.",
    "സംഗീതം സാമാന്യമായിരുന്നു.",
    "ഞാൻ ഏറെ നിരാശനായി.",
    "ഈ സിനിമ സമയം നഷ്‍ടമാക്കി.",
    "ദൃശ്യങ്ങൾ അനാവശ്യമായിരുന്നു.",
    "സംവിധാനം തികച്ചും പിടിച്ചുപറിഞ്ഞതായിരുന്നു.",
    "ഞാൻ ഈ സിനിമ ആരെയും ശുപാർശ ചെയ്യില്ല."
]

# Generate dataset
def generate_dataset(num_samples=100):
    data = []
    for _ in range(num_samples):
        sentiment = random.choice(sentiments)
        if sentiment == "positive":
            text = random.choice(positive_sentences)
        elif sentiment == "neutral":
            text = random.choice(neutral_sentences)
        else:
            text = random.choice(negative_sentences)
        data.append({"text": text, "sentiment": sentiment})
    return pd.DataFrame(data)

# Create and save dataset
df = generate_dataset(num_samples=200)
df.to_csv("generated_dataset.csv", index=False, encoding='utf-8-sig')

print("✅ Malayalam dataset generated and saved as 'generated_dataset.csv'")
