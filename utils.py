import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def compute_metrics(outputs, labels):
    """
    outputs: raw logits from the model, shape [batch_size]
    labels: tensor of 0/1, shape [batch_size]
    """
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    labels_np = labels.detach().cpu().numpy()

    acc = accuracy_score(labels_np, preds)
    prec = precision_score(labels_np, preds, zero_division=0)
    rec = recall_score(labels_np, preds, zero_division=0)

    # ROC AUC can fail if only one class present in labels
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = 0.0

    return acc, prec, rec, auc
