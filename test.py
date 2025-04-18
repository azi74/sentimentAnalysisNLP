import gradio as gr

# Dummy function that just returns a fixed value or empty string
def dummy_predict(text):
    return "🔮 This is just a demo. No prediction done."

# Gradio interface
iface = gr.Interface(
    fn=dummy_predict,
    inputs=gr.Textbox(label="Enter a Malayalam sentence"),
    outputs=gr.Label(label="Predicted Sentiment"),
    title="Malayalam Sentiment Analysis (Demo)",
    description="Type a Malayalam sentence and get the sentiment (This is a demo without prediction logic)."
)

iface.launch()
