import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    probs = y_pred[np.arange(len(y_true)), y_true]
    loss = -np.mean(np.log(probs))
    return loss
    