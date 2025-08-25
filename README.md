
# -Visualize-Activation-Maps
Visualize activation maps to understand which image regions activate CNN filters for emotion detection.
https://drive.google.com/file/d/1ti-wua5xpbY7FapK04xxvtXqsV4tKeHE/view?usp=sharing ( for saved model)
https://drive.google.com/file/d/1xagVwvMuDh3fEZ6BcDnINrNTe8Lea-j9/view?usp=sharing (weights for model)
https://drive.google.com/file/d/1aaRffs3d8I1nTx-952dr6Cfl8W5FTecG/view?usp=sharing (dataset) 

## Overview
This project trains a **ResNet50** model on the **FER-2013 dataset** for **emotion recognition**. It supports:

- 7 emotion classes: `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`
- Data augmentation and label smoothing
- Visualizing intermediate feature maps
- Grad-CAM heatmaps to highlight important regions

---

## Dataset
- **Dataset:** FER-2013
- **Classes:** 7 emotions
- **Train samples:** 28,709  
- **Test samples:** 7,178

> Dataset is unzipped from Drive to `/content/data`.

---

## Environment
- Python 3.12  
- PyTorch 2.8  
- Torchvision 0.23  
- Grad-CAM 1.5.5  
- PIL, Matplotlib, NumPy, OpenCV  
- Google Colab (CUDA GPU)

---

## Installation
```bash
pip install torch torchvision grad-cam matplotlib opencv-python
````

---

## Usage

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Unzip Dataset

```python
import zipfile, os
zip_path = "/content/drive/MyDrive/dataset/archive.zip"
extract_path = "/content/data"
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

## Notes

* Grayscale images are converted to 3 channels for ResNet.
* Last convolutional layer `layer4[-1].conv3` used for Grad-CAM.
* Fully compatible with Google Colab GPU runtime.







