def _iou(a, b):                                                                  
      """IoU between two boxes, each a list/tuple [x1, y1, x2, y2]."""             
      # intersection corners — clamp to overlap region                             
      x1 = max(a[0], b[0])  # left edge of intersection                            
      y1 = max(a[1], b[1])  # top edge of intersection                             
      x2 = min(a[2], b[2])  # right edge of intersection                           
      y2 = min(a[3], b[3])  # bottom edge of intersection                          
                                                                                   
      # clip to 0: negative width/height means no overlap                          
      inter = max(0, x2 - x1) * max(0, y2 - y1)                                    
                                                                                   
      area_a = (a[2] - a[0]) * (a[3] - a[1])                                       
      area_b = (b[2] - b[0]) * (b[3] - b[1])                                       
      union = area_a + area_b - inter  # subtract inter once (counted twice in sum)
                                                                                   
      return inter / union if union > 0 else 0.0                                   
                                                                                   
                                                                                   
def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:                                                          
      return []                                                                
                                                                               
    # sort indices by score descending; ties broken by lower index first         
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))            
                                                                               
    kept = []                                                                    
                                                                               
    while order:                                                                 
        i = order[0]   # index of highest-score remaining box
        kept.append(i)                                                           
                                                                               
        # keep only boxes whose IoU with box i is below threshold                
        order = [j for j in order[1:] if _iou(boxes[i], boxes[j]) < iou_threshold]                                                                   
          
    return kept 