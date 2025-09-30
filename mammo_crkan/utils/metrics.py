
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score

def compute_metrics(y_true, y_prob, num_classes=3):
    """
    y_true: (N,) int labels
    y_prob: (N, C) probabilities/logits
    """
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception:
        auc = float('nan')
    return dict(acc=acc, precision=prec, f1=f1, auc=auc)
