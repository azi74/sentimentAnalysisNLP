import requestsToTest

response = requestsToTest.post(
    "http://127.0.0.1:5000/predict",
    json={"text": "ഈ സിനിമ വളരെ നല്ലതായിരുന്നു"}
)
print(response.json())