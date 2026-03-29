import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(steps):
        logits = X @ w + b
        preds = _sigmoid(logits)
        error = preds - y

        dw = (X.T @ error) / n
        db = error.mean()

        w -= lr * dw
        b -= lr * db

    return w, b