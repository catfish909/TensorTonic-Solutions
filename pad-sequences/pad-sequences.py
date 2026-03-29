import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max_len if max_len is not None else (max(len(s) for s in seqs) if seqs else 0)

    out = np.full((N,L), fill_value=pad_value)
    for i, seq in enumerate(seqs):
        length = min(len(seq), L) # truncate if seq exceeds max_len
        out[i, :length] = seq[:length]

    return out
    pass