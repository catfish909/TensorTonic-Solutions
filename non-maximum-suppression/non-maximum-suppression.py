import numpy as np

def _iou_one_to_many(box, boxes):                                                
    """IoU between one box (4,) and an array of boxes (N, 4) → (N,)"""           
    # box: (4,)    scalar index → scalar                                         
    # boxes: (N, 4)   column slice → (N,)                                        
    # np.maximum(scalar, (N,)) → (N,)                                            
    x1 = np.maximum(box[0], boxes[:, 0])  # (N,) intersection left               
    y1 = np.maximum(box[1], boxes[:, 1])  # (N,) intersection top                
    x2 = np.minimum(box[2], boxes[:, 2])  # (N,) intersection right              
    y2 = np.minimum(box[3], boxes[:, 3])  # (N,) intersection bottom             
                                                                               
    # (N,) - (N,) → (N,), clip negatives to 0 (no overlap)                       
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # (N,)              
                                                                               
    area     = (box[2] - box[0]) * (box[3] - box[1])              # scalar       
    area_all = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # (N,) 
                                                                               
    # scalar + (N,) - (N,) → (N,)                                                
    union = area + area_all - inter  # (N,)                                      
                                                                               
    # (N,) / (N,) → (N,), guard division by zero                                 
    return np.where(union > 0, inter / union, 0.0)  # (N,)                       
                                                                                   
                                                                                   
def nms(boxes, scores, iou_threshold):                                           
    if len(boxes) == 0:                                                          
        return []                                                                
    
    boxes  = np.array(boxes, dtype=float)   # (B, 4)                             
    scores = np.array(scores, dtype=float)  # (B,)
                                                                               
    # negate for descending; stable preserves index order for tied scores        
    order = np.argsort(-scores, kind='stable')  # (B,) indices sorted by score  desc                                                                             
    kept  = []  
                                                                               
    while len(order) > 0:                                                        
        i = order[0]          # scalar — index of highest-score remaining box
        kept.append(int(i))   # int() converts np.int64 for JSON serializability 
                                                                               
        rest = order[1:]      # (N,) remaining candidate indices, N =            
        len(order)-1                                                                     
        if len(rest) == 0:                                                       
            break                                                                
                                                                               
        iou = _iou_one_to_many(boxes[i], boxes[rest])  # (N,)                    
                                                                               
        # boolean mask (N,) → filters rest down to non-suppressed boxes          
        order = rest[iou < iou_threshold]  # (M,) where M <= N
                                                                               
    return kept 
    