# Bengali Handwriting Recognition Using CNN

## Project Overview

This project is a Bengali Handwriting Recognition
Application developed using a Convolutional Neural
Network (CNN).

The application recognizes 60 classes of handwritten
Bengali characters, including letters, signs, and numbers.

## Features

1. Handwriting Drawing Pad
   - Draw Bengali characters using a mouse or touchscreen.
   - Recognize handwritten characters using CNN.

2. Image Upload
   - Upload handwritten Bengali character images.
   - Automatically preprocess and recognize characters.

3. Camera Recognition
   - Capture handwritten characters using a camera.
   - Recognize characters directly from photographs.

4. Bengali Character Prediction
   - Display the recognized Bengali character.
   - Display the predicted class and model score.
   - Show the top three predictions.

## Dataset

Dataset: BanglaLekha-Isolated

The original dataset contains 84 classes.

This project uses only the first 60 classes:
- Classes 1-50: Bengali letters and signs
- Classes 51-60: Bengali numbers

## Technologies Used

- Python
- TensorFlow
- Keras
- Convolutional Neural Network (CNN)
- Gradio
- OpenCV
- NumPy
- Pillow
- Google Colab

## Model Information

- Model type: CNN
- Input image size: 64 x 64
- Image format: Grayscale
- Output classes: 60
- Model file: bangla_60_model.keras
- Character labels: bangla_labels_60.json

## How to Run

1. Install Python and the required libraries.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the application:

   python app.py

4. Open the Gradio application URL in your browser.

## Project Status

The application has been developed and tested
in Google Colab using Gradio.

Recognition performance may vary with handwriting
style, lighting, and image quality.
