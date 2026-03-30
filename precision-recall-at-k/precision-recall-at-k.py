def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = set(recommended[:k])
    relevant_set = set(relevant)
    num_hit = len(top_k & relevant_set)
    
    precision = num_hit / k
    recall = num_hit / len(relevant_set) if relevant_set else 0.0

    return  [precision, recall]