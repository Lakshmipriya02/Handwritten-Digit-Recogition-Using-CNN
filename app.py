import os

import gradio as gr
import numpy as np
import torch

from model import DigitsCNN
from preprocess import extract_canvas, to_mnist

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digits_cnn.pt")

model = DigitsCNN()
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()


def predict(drawing):
    img = to_mnist(extract_canvas(drawing))
    if img is None:
        return None, None
    # Same normalisation as training: (x - 0.5) / 0.5
    x = torch.from_numpy((img - 0.5) / 0.5).float().reshape(1, 1, 28, 28)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    scores = {str(i): float(probs[i]) for i in range(10)}
    # Show what the model actually sees (28x28, upscaled for visibility)
    seen = np.kron((img * 255).astype(np.uint8), np.ones((8, 8), dtype=np.uint8))
    return scores, seen


with gr.Blocks(title="Handwritten Digit Recognizer (CNN)") as demo:
    gr.Markdown(
        "# ✍️ Handwritten Digit Recognizer\n"
        "Draw a single digit (0-9) and the convolutional neural network predicts it live. "
        "PyTorch CNN trained on MNIST."
    )
    with gr.Row():
        with gr.Column():
            pad = gr.Sketchpad(
                label="Draw here",
                canvas_size=(280, 280),
                brush=gr.Brush(default_size=14, colors=["#000000"], default_color="#000000", color_mode="fixed"),
                image_mode="RGBA",
                type="numpy",
                layers=False,
                transforms=(),
            )
            clear = gr.ClearButton([pad])
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Prediction")
            seen = gr.Image(label="What the model sees (28×28)", image_mode="L", height=200)

    pad.change(predict, pad, [label, seen], show_progress="hidden")
    clear.click(lambda: (None, None), None, [label, seen])

if __name__ == "__main__":
    demo.launch()
