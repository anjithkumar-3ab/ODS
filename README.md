
# 🐶🐱 Image Classifier

This is an image classification project that uses a deep learning model to classify images of cats and dogs.

## 📁 Project Structure

```
image_classifier/
│
├── yolov5l.pt                # Pre-trained YOLOv5 model (optional, if used)
├── data/                     # Dataset folder
├── models/                   # Trained model files
├── src/                      # Source code (training, evaluation, prediction scripts)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🚀 Features

- Classifies images into categories: **Cats** and **Dogs**
- Built with PyTorch and YOLOv5 (or your chosen model)
- Supports training, validation, and inference

## 📦 Installation

```bash
git clone https://github.com/anjithkumar-3ab/OD.git
cd image_classifier
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### ➤ Train the Model

```bash
python src/train.py --epochs 10 --batch-size 32
```

### ➤ Predict on New Image

```bash
python src/predict.py --image path/to/image.jpg
```

## 🧠 Model

- Model: YOLOv5 / CNN / Custom CNN
- Framework: PyTorch
- Accuracy: _e.g., 92% on validation set_

## 📊 Results

| Metric     | Value |
|------------|-------|
| Accuracy   | 92%   |
| Precision  | 90%   |
| Recall     | 93%   |

## 📷 Example Outputs

> (Insert sample images showing predictions here)

## ✍️ Author

- **Anjith Kumar**  
  [GitHub](https://github.com/anjithkumar-3ab)

## 📜 License

This project is licensed under the MIT License.

---
