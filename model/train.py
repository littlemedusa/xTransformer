import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy
from collections import namedtuple
from collections import defaultdict

sys.path.append("..")
from lib.utils import (
    WeightedMSELoss,
    MaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import MSE_RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.STAEformer.STAEformer import STAEformer
from model.FEDformer.FEDformer import FEDformer
from model.PatchTST.PatchTST import PatchTST
from model.PatchTST.PatchTST_norm import PatchTST_norm
from model.PatchTST.PatchTST_std import PatchTST_std
from model.PatchTST.PatchTST_test import PatchTST_test
from model.PatchTST.PatchTST_variant import PatchTST_variant
from model.iTransformer.iTransformer import iTransformer
from model.iTransformer.iTransformer_STnorm import iTransformer_test
from model.t_variant.t_variant_1 import t_variant_1
from model.t_variant.t_variant_2 import t_variant_2
from model.t_variant.t_variant_3 import t_variant_3
from model.t_variant.t_variant_4 import t_variant_4
from model.t_variant.t_variant_5 import t_variant_5
from model.DLinear.DLinear import DLinear
from model.xTransformer.xTransformer import xTransformer
from model.xTransformer_v1.xTransformer_v1 import xTransformer_v1

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    if args.addition:
        cond_num_origin_list, cond_num_latent_list = [], []
    for x_batch, x_batch_mark, y_batch, y_batch_mark in valset_loader:
        x_batch = x_batch.to(DEVICE)
        x_batch_mark = x_batch_mark.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        y_batch_mark = y_batch_mark.to(DEVICE)
        y_batch_zero = torch.zeros_like(y_batch)
        
        if args.addition:
            out_batch, cond_num_batch = model(x_batch, x_batch_mark, y_batch_zero, y_batch_mark)
            cond_num_origin_list.append(cond_num_batch[0].item())
            cond_num_latent_list.append(cond_num_batch[1].item())
        else:
            out_batch = model(x_batch, x_batch_mark, y_batch_zero, y_batch_mark)
        
        if cfg["inverse"]:
            out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    if args.addition:
        cond_num_origin = np.mean(cond_num_origin_list)
        cond_num_latent = np.mean(cond_num_latent_list)
        print_log("Val Cond_num_origin = %.5f" % cond_num_origin,
                "Cond_num_latent = %.5f \t" % cond_num_latent)
    
    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []
    if args.addition:
        cond_num_origin_list, cond_num_latent_list = [], []
    for x_batch, x_batch_mark, y_batch, y_batch_mark in loader:
        x_batch = x_batch.to(DEVICE)
        x_batch_mark = x_batch_mark.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        y_batch_mark = y_batch_mark.to(DEVICE)
        y_batch_zero = torch.zeros_like(y_batch)
        
        if args.addition:
            out_batch, cond_num_batch = model(x_batch, x_batch_mark, y_batch_zero, y_batch_mark)
            cond_num_origin_list.append(cond_num_batch[0].item())
            cond_num_latent_list.append(cond_num_batch[1].item())
        else:
            out_batch = model(x_batch, x_batch_mark, y_batch_zero, y_batch_mark)
        
        if cfg["inverse"]:
            out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    if args.addition:
        cond_num_origin = np.mean(cond_num_origin_list)
        cond_num_latent = np.mean(cond_num_latent_list)
        print_log("Test Cond_num_origin = %.5f" % cond_num_origin,
                "Cond_num_latent = %.5f \t" % cond_num_latent)
    
    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return out, y


def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    if args.addition:
        cond_num_origin_list, cond_num_latent_list = [], []
    for x_batch, x_batch_mark, y_batch, y_batch_mark in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        x_batch_mark = x_batch_mark.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        y_batch_mark = y_batch_mark.to(DEVICE)
        y_batch_zero = torch.zeros_like(y_batch)
        if args.addition:
            out_batch, cond_num_batch = model(x_batch, x_batch_mark, y_batch_zero, y_batch_mark)
            cond_num_origin_list.append(cond_num_batch[0].item())
            cond_num_latent_list.append(cond_num_batch[1].item())
        else:
            out_batch = model(x_batch, x_batch_mark, y_batch_zero, y_batch_mark)
        if cfg["inverse"]:
            out_batch = SCALER.inverse_transform(out_batch)
        
        loss = criterion(out_batch, y_batch)    # [B, T, N]
        
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)

    if args.addition:
        cond_num_origin = np.mean(cond_num_origin_list)
        cond_num_latent = np.mean(cond_num_latent_list)
        print_log("Train Cond_num_origin = %.5f" % cond_num_origin,
                "Cond_num_latent = %.5f" % cond_num_latent)
    
    scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=20,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1
            )
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                "Train Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_mse, train_rmse, train_mae, train_mape = MSE_RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_mse, val_rmse, val_mae, val_mape = MSE_RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train MSE = %.5f, RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_mse,
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val MSE = %.5f, RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_mse,
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_pred, y_true = predict(model, testset_loader)
    end = time.time()

    mse_all, rmse_all, mae_all, mape_all = MSE_RMSE_MAE_MAPE(y_pred, y_true)
    out_str = "All Steps MSE = %.5f, RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        mse_all,
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        mse_all, rmse_all, mae_all, mape_all = MSE_RMSE_MAE_MAPE(y_pred, y_true)
        out_str = "All Steps MSE = %.5f, RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            mse_all,
            rmse_all,
            mae_all,
            mape_all,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    
    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-m", "--model", type=str, default="STAEformer")
    parser.add_argument("-seq", "--seq_len", type=int, default=96)
    parser.add_argument("-pred", "--pred_len", type=int, default=336)
    parser.add_argument("-tag", "--add_tag", type=str, default=None)
    parser.add_argument('-ad', "--addition", action='store_true', help='additional output')
    parser.add_argument("--d_control", action='store_true', help='setting d_model and d_ff')
    parser.add_argument("-dm", "--d_model", type=int, default=16)
    parser.add_argument("-df", "--d_ff", type=int, default=128)
    parser.add_argument("--selector_settings", action='store_true', help='setting selector axis and compress')
    parser.add_argument("--axis", type=str, default="n")
    parser.add_argument("--compress", type=int, default=1)
    args = parser.parse_args()

    seed = torch.randint(1000, (1,)) # set random seed here
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    print("GPU_ID: ", GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:%s"%(GPU_ID) if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = args.model

    with open(f"{model_name}/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
        model_select_args
    ) = get_dataloaders_from_index_data(
        data_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        dom=cfg.get("day_of_month"),
        doy=cfg.get("day_of_year"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # -------------------------------- modify cfg -------------------------------- #
    
    cfg["model_args"]["seq_len"] = args.seq_len
    cfg["model_args"]["pred_len"] = args.pred_len
    cfg["model_args"]["batch_size"] = cfg["batch_size"]

    if args.d_control:
        cfg["model_args"]["d_model"] = args.d_model
        cfg["model_args"]["d_ff"] = args.d_ff
    
    if args.selector_settings:
        cfg["model_args"]["axis"] = args.axis
        cfg["model_args"]["compress"] = args.compress
    else:
        cfg["model_args"]["axis"] = model_select_args["axis"]
        cfg["model_args"]["compress"] = model_select_args["compress"]

    # -------------------------------- load model -------------------------------- #
    
    model_dict = {
            'FEDformer': FEDformer,
            'STAEformer': STAEformer,
            # 'PatchTST': globals()["PatchTST_%s"%args.add_tag] if args.add_tag is not None else PatchTST,
            'PatchTST': PatchTST_variant,
            # 'iTransformer': iTransformer_test if args.add_tag == "test" else iTransformer,
            'iTransformer': iTransformer,
            'DLinear': DLinear,
            't_variant': globals()["t_variant_%s"%args.add_tag] if args.add_tag is not None else None,
            # 't_variant': t_variant_2
            'xTransformer': xTransformer, 
            'xTransformer_v1': xTransformer_v1, 
        }
    if model_name == 'STAEformer':
        model = model_dict[model_name](**cfg["model_args"])
    else:
        print(cfg["model_args"])
        cfg_obj = namedtuple('cfg_obj', cfg["model_args"].keys())
        _cfg = cfg_obj(*cfg["model_args"].values())
        model = model_dict[model_name](_cfg)
    
    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    elif dataset in ("ETTH1", "ETTH2", "ETTM1", "ETTM2", "ECL", "TRAFFIC", "RATE", "ILI", "WEATHER", "SOLAR"):
        # criterion = WeightedMSELoss() if args.add_tag == "test" else nn.MSELoss() 
        criterion = nn.MSELoss() 
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model.to(DEVICE),
            [
                [cfg["batch_size"],
                args.seq_len,
                cfg["num_nodes"]],
                [cfg["batch_size"],
                args.seq_len,
                 cfg["num_marks"]],
                [cfg["batch_size"],
                args.pred_len,
                cfg["num_nodes"]],
                [cfg["batch_size"],
                args.pred_len,
                 cfg["num_marks"]],
                
            ],
            verbose=0,  # avoid print twice
            device=DEVICE,
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 20),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
    )

    print_log(f"Saved Model: {save}", log=log)

    test_model(model, testset_loader, log=log)

    log.close()
