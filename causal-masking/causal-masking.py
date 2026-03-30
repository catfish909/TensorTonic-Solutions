import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores = np.array(scores, dtype=float)
    T = scores.shape[-1]

    rows = np.arange(T)[:, None] # (T, 1)
    cols = np.arange(T)[None, :] # (1, T)  
    mask = cols > rows           # (T, T): True where j > i  

    # or using np.triu
    # mask = np.triu(np.ones((T, T), dtype=bool), k=1) 
    # keep strictly above main diagonal as True, others are False
    #  False True  True                                                                       
    #  False False True                                                                       
    #  False False False                                                                      
    # True where j > i — the future positions that should be masked.     

    return np.where(mask, mask_value, scores)