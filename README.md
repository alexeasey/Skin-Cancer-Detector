# CSS2-SVM — Skin Lesion Classifier

A deep learning application for classifying skin lesions into 7 diagnostic categories using a fine-tuned DenseNet201 model, with a desktop GUI for real-time image inference.

[Project Report](https://alexeasey.github.io/reports/SkinCancerDetection.pdf)

---

## Overview

This project trains and deploys a convolutional neural network to classify dermoscopic images of skin lesions. The model is trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) and exposed via a desktop GUI built with CustomTkinter.

## Skin Lesion Classes

| Class | Description |
|---|---|
| Actinic Keratoses & Intraepithelial Carcinoma | Precancerous lesion; can develop into squamous cell carcinoma |
| Basal Cell Carcinoma | Most common type of skin cancer |
| Benign Keratosis Lesion | Non-cancerous skin growth |
| Dermatofibroma | Benign, usually harmless skin growth |
| Melanoma | Potentially life-threatening if not caught early |
| Melanocytic Nevi | Benign moles; monitor for changes |
| Vascular Lesions | Abnormalities of blood vessels in the skin |

## Model

- **Architecture:** DenseNet201 (pretrained on ImageNet, fine-tuned)
- **Dataset:** HAM10000 (~10,015 dermoscopic images)
- **Split:** 70% train / 15% validation / 15% test
- **Input size:** 224 × 224 × 3
- **Optimizer:** Adam with AMSGrad, learning rate scheduling via `ReduceLROnPlateau`
- **Test accuracy:** ~83.8%

### Training

1. **Phase 1** — Base layers frozen, top layers trained for 5 epochs
2. **Phase 2** — Full fine-tuning for 20 epochs with learning rate reduction on plateau

## Project Structure

```
SkinCancerDetection/
├── CSS2 GUI/
│   ├── main.py          # Desktop GUI 
│   └── denseNet3.h5     # Trained model 
└── DenseNet201.ipynb    # Model training notebook (Google Colab)
```

## Requirements

```
tensorflow
customtkinter
opencv-python
Pillow
numpy
pandas
scikit-learn
```

## Usage

### Running the GUI

```bash
cd "SkinCancerDetection/CSS2 GUI"
python main.py
```

1. Click **Open Image** to load a dermoscopic `.jpg` image
2. Click **Process Image** to classify the lesion
3. The predicted class and description are displayed below

### Training the Model

Open `DenseNet201.ipynb` in Google Colab and run all cells. The notebook mounts Google Drive to access the HAM10000 (available from kaggle.com) dataset and saves the trained model as `denseNet3.h5`.

---

> **Disclaimer:** This tool is intended for educational purposes only and should not be used as a substitute for professional medical advice or diagnosis.
