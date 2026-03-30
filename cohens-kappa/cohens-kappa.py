import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)

    p_o = np.mean(rater1 == rater2)

    classes = np.unique(np.concatenate([rater1, rater2]))

    p_e = sum(
        (np.mean(rater1 == c) * np.mean(rater2 == c))
        for c in classes
    )

    '''
    The degenerate case is when all labels are the same → p_e = 1.0 → division by zero.
    Convention is to return 1.0 since there's perfect agreement with no chance variation possible.    
    '''
    if p_e == 1.0:
        return 1.0

    return (p_o - p_e) / (1 - p_e)
    