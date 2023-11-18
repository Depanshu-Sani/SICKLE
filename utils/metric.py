from torchmetrics import F1Score, Accuracy, JaccardIndex, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, ignore_index=-999, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.ignore_index = ignore_index
        
    def forward(self,yhat,y, plot_mask=None):
        if plot_mask is not None:
            yhat, y = get_combined_yield(yhat, y, plot_mask)
        yhat = yhat.squeeze()
        yhat = yhat[y != self.ignore_index]
        y = y[y != self.ignore_index]
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics = {
    "f1": F1Score(average="none", task='multiclass', num_classes=2, ignore_index=-999, threshold=0.5).to(device=device),
    "acc_cls": Accuracy(average="none", task='multiclass', num_classes=2, ignore_index=-999, threshold=0.5).to(device=device),
    "iou_cls": JaccardIndex(average="none", task="multiclass", num_classes=2, ignore_index=-999, threshold=0.5).to(device=device),
    "f1_macro": F1Score(average="macro", task='multiclass', num_classes=2, ignore_index=-999, threshold=0.5).to(device=device),
    "acc": Accuracy(task='multiclass', num_classes=2, ignore_index=-999, threshold=0.5).to(device=device),
    "iou": JaccardIndex(task="multiclass", num_classes=2, ignore_index=-999, threshold=0.5).to(device=device),
    "rmse" : RMSELoss(ignore_index =-999).to(device=device),
    "mae": torch.nn.L1Loss(),
    "mape": MeanAbsolutePercentageError().to(device=device),
}

def get_metrics(y_pred, y_true, pid_masks, ignore_index=-999, task="crop_type"):
    if task != "crop_type":
        return get_regression_metrics(y_pred, y_true,pid_masks, ignore_index, task)
    paddy_f1, non_paddy_f1 = metrics["f1"](y_pred, y_true)
    paddy_acc, non_paddy_acc = metrics["acc_cls"](y_pred, y_true)
    paddy_iou, non_paddy_iou = metrics["iou_cls"](y_pred, y_true)
    f1_macro = metrics["f1_macro"](y_pred, y_true)
    acc = metrics["acc"](y_pred, y_true)
    iou = metrics["iou"](y_pred, y_true)

    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    y_pred = y_pred[y_true != ignore_index]
    y_true = y_true[y_true != ignore_index]

    return f1_macro, acc, iou, paddy_f1, non_paddy_f1, paddy_acc, non_paddy_acc, paddy_iou, non_paddy_iou, (y_pred, y_true)

def get_regression_metrics(y_pred, y_true, pid_masks, ignore_index=-999, task=None):
    if task == "crop_yield":
        y_pred, y_true = get_combined_yield(y_pred, y_true, pid_masks)
        mape = metrics["mape"](y_pred, y_true)
    else:
        mape = MAPE(y_pred, y_true, ignore_index=-999)
    # rmse already handles ignore index
    rmse = metrics["rmse"](y_pred, y_true)
    # handle ignore index for others

    y_pred = y_pred.squeeze()
    y_pred = y_pred[y_true != -999]
    y_true = y_true[y_true != -999]
    mae = metrics["mae"](y_pred, y_true)
    
    return rmse, mae, mape

def get_combined_yield(y_pred, y_true, pid_masks):
    y_pred, y_true, pid_masks = y_pred.squeeze(), y_true.squeeze(), pid_masks.squeeze()
    y_pred_temp, y_true_temp = None, None
    
    # combine plot yield 
    for i in range(pid_masks.size(0)):
        pids = torch.unique(pid_masks[i])
        for pid in pids:
            if pid != -999 and torch.sum(y_true[i,pid_masks[i]==pid]) > 0:
                if y_true_temp is None:
                    y_pred_temp = torch.sum(y_pred[i,pid_masks[i]==pid]).reshape(1)
                    y_true_temp = torch.sum(y_true[i,pid_masks[i]==pid]).reshape(1)
                else:
                    y_pred_temp = torch.cat((y_pred_temp,torch.sum(y_pred[i,pid_masks[i]==pid]).reshape(1)), dim=0)
                    y_true_temp = torch.cat((y_true_temp,torch.sum(y_true[i,pid_masks[i]==pid]).reshape(1)), dim=0)
    return y_pred_temp, y_true_temp

def MAPE(y_pred, y_true, ignore_index):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    y_pred = y_pred[y_true != ignore_index]
    y_true = y_true[y_true != ignore_index]
    mae_metric = torch.nn.L1Loss(reduction="none")
    mae = mae_metric(y_pred, y_true)
    return torch.mean(mae / 183.)