
# Deep Face Detection Model with Python and TensorFlow

This project implements a deep learning-based face detection model using Python, TensorFlow, and OpenCV. It leverages Convolutional Neural Networks (CNNs) to detect faces in images by predicting bounding box coordinates. The model is trained and evaluated on a custom or benchmark dataset.

## 📌 Features

- Deep learning model for face detection
- Custom CNN architecture built with TensorFlow and Keras
- Bounding box prediction and visualization using OpenCV
- Jupyter Notebook for step-by-step training and evaluation
- Data preprocessing, augmentation, and visualization
- Metrics tracking: loss and accuracy during training

---

## 🧠 Model Architecture

- Custom CNN with multiple Conv2D, MaxPooling, Dropout, and Dense layers
- Output layer predicts bounding box coordinates: `[x_min, y_min, x_max, y_max]`
- Optimized with `Adam` optimizer and `mean squared error` loss

---

## 📂 Project Structure
```
.
├── model.ipynb                # Jupyter notebook with full implementation
├── README.md                  # Project overview and usage instructions
└── data                       # Folder containing images and annotations (not included)
      └── train
            ├── images         # Put the images of the object you are trying to detect as I have used my own face for this one 
            └── labels         # I have labeled my images using labelImg library and saved in create ML format
      └── test
            ├── images
            └── labels
      └── val
            ├── images
            └── labels             
└── aug_data                    # This folder contains the data created by augmentation which will be used later for training the model
      └── train
            ├── images
            └── labels
      └── test
            ├── images
            └── labels
      └── val
            ├── images
            └── labels                      
```
## Outcome

![Model Working ](https://github.com/user-attachments/assets/6fd28e59-e68e-43aa-8467-c453737c7211)

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/deep-face-detection.git
   cd deep-face-detection


2. **Create and activate a virtual environment (optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not provided, install the core libraries:

   ```bash
   pip install tensorflow opencv-python matplotlib numpy
   ```

---

## 🚀 Usage

Run the notebook:

```bash
jupyter notebook model.ipynb
```

Or convert it to a Python script:

```bash
jupyter nbconvert --to script model.ipynb
python model.py
```

---

## 🧪 Example Results

The notebook includes examples where predicted bounding boxes are drawn over input images using OpenCV.

---

## 📊 Training

* Model is trained on image inputs and normalized bounding box labels
* Loss function: `Mean Squared Error`
* Optimizer: `Adam`
* Batch size and number of epochs can be customized in the notebook

---

## 📈 Evaluation

* Visual loss tracking across epochs
* Image-level evaluation with bounding box overlay

---
