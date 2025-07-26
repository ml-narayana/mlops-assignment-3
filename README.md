# 📦 MLOps Assignment 3 – Quantized Linear Regression with PyTorch

## 🎯 Objective
Implement quantization of a linear regression model trained using `sklearn`, convert the model to PyTorch using quantized weights and bias, and evaluate its performance.

---

## 🛠️ Technologies Used
- Python 3.x
- Scikit-learn
- PyTorch
- NumPy
- Joblib

---

## 📂 Directory Structure

mlops-assignment-3/
│
├── model/ # Stores trained and quantized weights
│ ├── model.joblib # Original sklearn model
│ ├── unquant_params.joblib# Coefficients & intercept
│ └── quant_params.joblib # Quantized weights
│
│── train.py # Train sklearn LinearRegression model
│── quantize.py # Quantize weights and run PyTorch inference
│
└── README.md

Results

| Metric         | Sklearn (Original) | PyTorch (Quantized) |
| -------------- | ------------------ | ------------------- |
| R² Score       | 0.5758             | -0.1799             |
| Quantization   | No                 | 8-bit linear        |
| Model Size     | 413B               | 526B                |



## Authors
- M L Lakshminarayana : g24ai1086@iitj.ac.in
