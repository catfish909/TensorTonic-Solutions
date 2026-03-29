import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    PE = np.zeros((seq_len, d_model))
                                                                                    
    pos = np.arange(seq_len)[:, None]          # (seq_len, 1)                         
    i   = np.repeat(np.arange((d_model + 1) // 2), 2)[:d_model]    # (d_model,)
    # example: d_model = 7
    # np.arange((7+1)//2)          # [0, 1, 2, 3]  — 4 elements                             
    # np.repeat(..., 2)             # [0, 0, 1, 1, 2, 2, 3, 3]  — 8 elements                
    # [:7]                          # [0, 0, 1, 1, 2, 2, 3]  — trimmed to 7   
                                                                                    
    angles = pos / base ** (2 * i / d_model)   # (seq_len, d_model//2)
                                                                                    
    PE[:, 0::2] = np.sin(angles[:, 0::2])   # even indices                                     
    PE[:, 1::2] = np.cos(angles[:, 1::2])   # odd indices
                                                                                    
    return PE