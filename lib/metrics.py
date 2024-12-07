import numpy as np

from typing import Optional
import torch
import torch.nn.functional as F


def MSE(y_pred, y_true):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def MSE_new(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)    
    
    
def RMSE(y_pred, y_true):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_pred, y_true):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae

    
def MAE_new(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


def MAPE(y_pred, y_true, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def MAPE_new(prediction: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """From BasicTS
    
    Masked mean absolute percentage error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    target = np.where(np.abs(target) < 1e-4, np.zeros_like(target), target)
    if np.isnan(null_val):
        mask = ~np.isnan(target)
    else:
        eps = 5e-5
        mask = ~np.isclose(target, np.full_like(target, null_val), atol=eps, rtol=0.)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(np.abs(prediction - target) / target)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)

    
def RMSE_MAE_MAPE(y_pred, y_true):
    return (
        RMSE(y_pred, y_true),
        MAE(y_pred, y_true),
        MAPE(y_pred, y_true),
    )


def MSE_RMSE_MAE_MAPE(y_pred, y_true):
    return (
        MSE(y_pred, y_true),
        RMSE(y_pred, y_true),
        MAE(y_pred, y_true),
        MAPE(y_pred, y_true),
    )
