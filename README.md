# ✍️ Handwritten Digit Recognizer (CNN)

Draw a digit in the browser and a PyTorch convolutional neural network predicts it live.

**Model:** Conv(32) → MaxPool → Conv(64) → MaxPool → Dropout → Linear(128) → Linear(10), trained on MNIST
with light data augmentation (small rotations, shifts, zoom).

## Run locally (Python 3.10+)

```bash
pip install -r requirements.txt
python train.py      # downloads MNIST, trains, prints test accuracy, creates digits_cnn.pt
python app.py        # opens http://127.0.0.1:7860
```

`train.py` also prints per-digit accuracy and saves `confusion_matrix.png`.

## Files
- `model.py`       CNN definition
- `train.py`       trains on MNIST, evaluates, saves `digits_cnn.pt`
- `preprocess.py`  converts a canvas drawing into an MNIST-style 28x28 input (crop, scale to 20px, centre by centre of mass)
- `app.py`         Gradio app with a drawing canvas

## Hosting
Gradio Spaces on Hugging Face now need a paid plan for new Spaces. Free options: record a demo video from the
local app, or host a static (in-browser) version. Colab can also give a temporary public link with `demo.launch(share=True)`.
