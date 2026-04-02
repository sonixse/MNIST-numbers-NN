# 🔢 MNIST CNN Streamlit App

An interactive MNIST digit classifier built with PyTorch and Streamlit. You can train a small CNN, inspect learned filters and feature maps, and run predictions on sample or uploaded images.

## ✨ What You Can Do

- Train a CNN on MNIST from the command line or in the Streamlit UI
- Explore learned filters and intermediate activation maps
- Upload a handwritten digit image and predict the number
- Review confidence scores and basic evaluation metrics

## 🧰 Requirements

- Python 3.10+ is recommended
- A virtual environment is strongly recommended
- Internet access is needed the first time Streamlit downloads its frontend assets and MNIST is downloaded by PyTorch

## 🚀 Quick Start

1. Open the project folder.

2. Install the dependencies.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

3. Start the Streamlit app.

```bash
python3 -m streamlit run app/ui.py
```

4. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 🧭 Using the App

The app is organized into three modes in the sidebar:

- Train: adjust batch size, epochs, and other training options, then run training and review metrics.
- Explore: inspect convolution filters and activation maps for selected samples and layers.
- Predict: use a built-in sample or upload your own digit image, then press Predict to see the result.

### 🔎 Predict Mode Tips

- Use the sample selector for a quick smoke test.
- Upload a PNG or JPG if you want to test your own digit image.
- The model input appears below the Predict button, while the confidence plot and predicted digit appear on the right.

## 💻 Command-Line Training

If you want to train outside the UI, run the main entry point:

```bash
python3 main.py --epochs 3 --batch-size 64
```

Useful options:

- `--epochs`: number of training epochs
- `--batch-size`: training batch size
- `--limit-train`: optionally cap the training set for quick experiments
- `--limit-test`: optionally cap the test set for quick experiments
- `--lr`: learning rate
- `--save-path`: output filename for the saved model

The trained weights are saved in `output/`.

## ✅ Simple Testing And Verification

This project does not ship with a large automated test suite, but you can still verify the app quickly:

1. Check that the UI file compiles.

```bash
python3 -m compileall app/ui.py
```

2. Launch the Streamlit app and confirm it opens without errors.

```bash
python3 -m streamlit run app/ui.py
```

3. In Predict mode, try one of the sample images from `test/` such as `number.jpg` or upload your own image.

4. In Train mode, run a short training session and make sure metrics and plots appear.

If you want a very fast check on a slower machine, use `main.py` with `--limit-train` and `--limit-test`.

## 🗂️ Project Layout

- `app/ui.py`: Streamlit interface
- `main.py`: command-line training entry point
- `model/`: CNN model, training, evaluation, and prediction helpers
- `gpu/`: device selection helpers
- `utils/plotting.py`: plots for filters, feature maps, predictions, and metrics
- `test/`: sample images for prediction
- `data/`: MNIST dataset downloaded automatically on first run
- `output/`: saved model and generated artifacts created after execution via terminal command

## 🛠️ Troubleshooting

If Streamlit cannot find packages:

```bash
python3 -m pip install -r requirements.txt
```

If the page looks stale or blank:

- Refresh the browser tab
- Stop Streamlit and run it again
- Restart VS Code if the editor is holding an old environment

If you want to verify the same Python environment used by the app, make sure your terminal and VS Code are using the same interpreter.

## 📝 Notes

- The first run may take a little longer because MNIST needs to be downloaded.
- If you are using a GPU, the app will try to use it automatically when available.
