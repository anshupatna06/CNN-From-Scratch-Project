# 🧠 Convolutional Neural Network — From Scratch & PyTorch

This repository demonstrates two complete implementations of a Convolutional Neural Network (CNN):

CNN from Scratch (NumPy only) — manual forward & backward propagation

CNN using PyTorch — framework-based, production-style implementation

The goal is deep conceptual understanding + practical engineering skills.

Most projects show how to use CNNs.
This project shows how CNNs actually work internally.

# 🔍 Why This Project Matters

Builds CNNs mathematically from first principles

Implements manual backpropagation through convolution

Demonstrates ability to translate theory → code

Shows framework independence (NumPy → PyTorch)

Strong signal for ML Intern / SWE / Research roles

This repository is designed for:

ML / DL interview preparation

Research-oriented learning

Systems-level understanding of deep learning

# 🗂️ Repository Structure
CNN-From-Scratch-Project/
│
├── scratch_cnn/                 # NumPy-only implementation
│   ├── conv_forward.py
│   ├── conv_backward.py
│   ├── im2col.py
│   ├── col2im.py
│   ├── relu.py
│   ├── maxpool.py
│   ├── flatten.py
│   ├── dense.py
│   ├── softmax_loss.py
│   └── cnn_model.py
│
├── pytorch_cnn/                 # PyTorch implementation
│   ├── model.py
│   ├── train.py
│   ├── dataset.py
│   └── utils.py
│
├── notebooks/
│   ├── CNN_From_Scratch.ipynb
│   └── CNN_PyTorch.ipynb
│
├── visuals/
│   ├── training_loss_curve.png
│   ├── feature_maps.png
│   └── learned_filters.png
│
├── requirements.txt
└── README.md

# 🧠 CNN Architecture (Both Versions)
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
Cross-Entropy Loss

# 🧪 Implementation 1: CNN From Scratch (NumPy)
✅ What’s implemented manually

Convolution (forward & backward)

im2col / col2im optimizations

ReLU activation (forward & backward)

Max Pooling (forward & backward)

Dense layer

Softmax + Cross Entropy loss

Gradient computation using chain rule

Shape-safe tensor handling

# 🧠 Concepts Covered

Sliding window convolution

Receptive fields

Parameter sharing

Gradient flow through convolution

Numerical stability

Manual backpropagation

This implementation does not use TensorFlow or PyTorch — only NumPy.

# ⚡ Implementation 2: CNN Using PyTorch
✅ What’s included

Modular CNN model (nn.Module)

Clean training loop

Dataset abstraction

Loss & optimizer handling

GPU-ready architecture

Comparison with scratch implementation

# 🎯 Purpose

Show production-style ML engineering

Validate scratch implementation correctness

Bridge theory → real-world ML pipelines

# 📊 Visualizations

Generated through notebooks:

📉 Training loss curves

🧩 Feature maps after convolution

🎯 Learned filters visualization

📊 Softmax probability outputs

These help interpret what the CNN is learning internally, not just final accuracy.

▶️ How to Run
1️⃣ Clone the repository
git clone https://github.com/anshupatna06/CNN-From-Scratch-Project.git
cd CNN-From-Scratch-Project

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run notebooks
jupyter notebook


Open:

notebooks/CNN_From_Scratch.ipynb

notebooks/CNN_PyTorch.ipynb

Run cells top-to-bottom.

# 🧪 Notes

Scratch CNN uses synthetic / small-scale data for clarity

Focus is understanding, not benchmark accuracy

Code is extensible to datasets like MNIST

# 🔮 Future Improvements

Train both versions on MNIST

Add Adam optimizer

Batch Normalization

Multiple convolution blocks

Unit tests for gradients

Performance comparison (NumPy vs PyTorch)

# 🎯 Learning Outcomes

By completing this project, you will:

Understand CNNs mathematically and programmatically

Gain confidence in debugging deep learning models

Be able to explain CNN internals in interviews

Demonstrate framework-agnostic ML thinking


⭐ If this repository helped you, feel free to star it!
