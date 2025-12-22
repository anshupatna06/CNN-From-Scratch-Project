# CNN-From-Scratch-Project
# 🧠 Convolutional Neural Network (CNN) From Scratch — NumPy Only

This project implements a **full Convolutional Neural Network (CNN) from scratch** using only **NumPy**, without relying on deep learning frameworks like TensorFlow or PyTorch.

The goal of this project is to deeply understand:
- How CNNs work mathematically
- How forward and backward propagation are implemented
- How filters are learned
- How spatial dimensions flow through layers
- How gradients are computed manually

This repository is designed for **learning, research, and interview preparation**.

---

## 🚀 Key Features

-  Convolution forward & backward pass (from scratch)
-  `im2col` and `col2im` optimization
-  ReLU activation (forward & backward)
-  Max Pooling layer (forward & backward)
-  Flatten layer
-  Fully Connected (Dense) layer
-  Softmax + Cross Entropy loss
-  Complete CNN pipeline
-  Shape-safe implementation
-  Jupyter notebook for visualizations
-  Feature map & filter visualizations

---

## 🧱 Project Structure
CNN-From-Scratch-Project/
│
├── src/ # Core CNN implementation
│ ├── conv_forward.py
│ ├── conv_backward.py
│ ├── im2col.py
│ ├── col2im.py
│ ├── relu.py
│ ├── maxpool.py
│ ├── flatten.py
│ ├── dense.py
│ ├── softmax_loss.py
│ └── cnn_model.py
│
├── notebooks/
│ └── CNN_From_Scratch.ipynb # Training & visualizations
│
├── visuals/ # Generated visual outputs
│ ├── training_loss_curve.png
│ ├── feature_maps.png
│ ├── learned_filters.png
│
├── README.md
└── requirements.txt


---

## 🧠 CNN Architecture

Input Image (N, 1, 28, 28)
↓
Convolution (3×3 filters)
↓
ReLU
↓
Max Pooling (2×2)
↓
Flatten
↓
Fully Connected Layer
↓
Softmax
↓
Loss

---

## 📐 Mathematical Concepts Used

- Convolution operation
- Sliding window & receptive fields
- `im2col` matrix transformation
- Backpropagation through convolution
- Chain rule
- Gradient descent
- Softmax probability distribution
- Cross-entropy loss

---

## 📊 Visualizations Included

The Jupyter notebook generates:
- 📉 Training loss curve
- 🧩 Feature maps after convolution
- 🎯 Learned convolution filters
- 📊 Softmax output probabilities

These visualizations help in **interpreting what the CNN learns internally**.

---

## ▶️ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/anshupatna06/CNN-From-Scratch-Project.git
cd CNN-From-Scratch-Project
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Jupyter Notebook
jupyter notebook


Open:

notebooks/CNN_From_Scratch.ipynb


Run all cells top-to-bottom.

##🧪 Notes

The CNN is trained on dummy / synthetic data for demonstration.

The goal is conceptual clarity, not accuracy benchmarking.

The implementation is fully extensible to real datasets like MNIST.

🔮 Future Improvements

Train on MNIST dataset

Add Adam optimizer

Add batch normalization

Add multiple convolution layers

Compare with PyTorch implementation

Add unit tests

🎯 Learning Outcome

By completing this project, you will:

Understand CNNs at a mathematical level

Be confident implementing deep learning models from scratch

Gain strong debugging intuition

Be well-prepared for ML/DL interviews
🙌 Acknowledgement

Inspired by:

CS231n (Stanford)

Deep Learning Specialization

Research-oriented learning approach

⭐ If you find this project helpful, feel free to star the repository!


-----------------------------------------



