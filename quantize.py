import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

os.makedirs("model", exist_ok=True)

sk_model = joblib.load("model/model.joblib")
coef = sk_model.coef_
intercept = sk_model.intercept_

unquant = {"coef": coef, "intercept": intercept}
joblib.dump(unquant, "model/unquant_params.joblib")

def quantize(arr, scale=255.0):
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        q = np.zeros_like(arr, dtype=np.uint8)
    else:
        q = np.round((arr - min_val) / (max_val - min_val) * scale).astype(np.uint8)
    return q, min_val, max_val

def dequantize(q, min_val, max_val, scale=255.0):
    if max_val == min_val:
        return np.full_like(q, min_val, dtype=np.float32)
    return (q.astype(np.float32) / scale) * (max_val - min_val) + min_val

q_coef, coef_min, coef_max = quantize(coef)
q_intercept, i_min, i_max = quantize(np.array([intercept]))

quant = {
    "coef": q_coef,
    "intercept": q_intercept,
    "meta": {
        "coef_min": coef_min, "coef_max": coef_max,
        "i_min": i_min, "i_max": i_max
    }
}
joblib.dump(quant, "model/quant_params.joblib")

d_coef = dequantize(q_coef, coef_min, coef_max)
d_intercept = dequantize(q_intercept, i_min, i_max)[0]

class QuantizedModel(nn.Module):
    def __init__(self, weights, bias):
        super().__init__()
        self.linear = nn.Linear(len(weights), 1)
        self.linear.weight.data = torch.tensor(np.array([weights]), dtype=torch.float32)
        self.linear.bias.data = torch.tensor([bias], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = QuantizedModel(d_coef, d_intercept)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_pred = model(X_test_tensor).detach().numpy().flatten()
r2 = r2_score(y_test, y_pred)

print(f"Quantized PyTorch Model R² Score: {r2:.4f}")

sk_preds = sk_model.predict(X_test)


torch.save(model.state_dict(), "model/quantized_model.joblib")
