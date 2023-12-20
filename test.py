# 
"""
Main script for semantic experiments
Built upon Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import argparse
import json
import os
import copy

import matplotlib.pyplot as plt

import wandb
import pprint

import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

# Custom import
from utils.dataset import SICKLE_Dataset
from utils import utae_utils, model_utils
from utils.weight_init import weight_init
from utils.metric import get_metrics, RMSELoss
# torch
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchnet as tnt

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,128]", type=str)
parser.add_argument("--out_conv", default="[32, 16]")
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)
parser.add_argument("--best_path", default=None, type=str)

# Set-up parameters
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument("--seed", default=0, type=int, help="Random seed")
# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--lr", default=1e-1, type=float, help="Learning rate")
# parser.add_argument("--wd", default=1e-2, type=float, help="weight decay")
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--ignore_index", default=-999, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument("--resume", default="", type=str, help="enter run path to resume")
parser.add_argument("--run_id", default="", type=str, help="enter run id to resume")
parser.add_argument("--wandb", action='store_true', help="debug?")
parser.add_argument('--satellites', type=str, default="[S2]")
parser.add_argument('--run_name', type=str, default="trial")
parser.add_argument('--exp_name', type=str, default="utae")
parser.add_argument('--task', type=str, default="crop_type",
                    help="Available Tasks are crop_type, sowing_date, transplanting_date, harvesting_date, crop_yield")
parser.add_argument('--actual_season', action='store_true', help="whether to consider actual season or not.")
parser.add_argument('--data_dir', type=str, default="../sickle/data")
parser.add_argument('--use_augmentation', type=bool, default=False)

list_args = ["encoder_widths", "decoder_widths", "out_conv", "satellites"]
parser.set_defaults(cache=False)


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(CFG):
    if CFG.wandb:
        if not os.path.exists(CFG.run_path):
            os.makedirs(CFG.run_path)
        elif CFG.resume:
            pass
        else:
            CFG.run_path = CFG.run_path + f"_{time.time()}"
            print("Run path already exist changed run path to ", CFG.run_path)
            os.makedirs(CFG.run_path)
    else:
        CFG.run_path += "_debug"
        os.makedirs(CFG.run_path, exist_ok=True)


def checkpoint(log, config):
    with open(
            os.path.join(config.run_path, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def set_seed(seed=42):
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(f"> SEEDING DONE {seed}")


def log_wandb(loss, metrics, phase="train"):
    f1_macro, acc, iou, f1_paddy, f1_non_paddy, \
    acc_paddy, acc_non_paddy, iou_paddy, iou_non_paddy, (y_pred, y_true) = metrics
    y_pred, y_true = y_pred.tolist(), y_true.tolist()
    if CFG.wandb:
        wandb.log(
            {
                f"{phase}_loss": loss,
                f"{phase}_f1_macro": f1_macro,
                f"{phase}_acc": acc,
                f"{phase}_iou": iou,
                f"{phase}_f1_paddy": f1_paddy,
                f"{phase}_f1_non_paddy": f1_non_paddy,
                f"{phase}_acc": acc,
                f"{phase}_acc_paddy": acc_paddy,
                f"{phase}_acc_non_paddy": acc_non_paddy,
                f"{phase}_iou_paddy": iou_paddy,
                f"{phase}_iou_non_paddy": iou_non_paddy,
            })
        if phase == "test":
            wandb.log({f"{phase}_conf_mat": wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred, probs=None,
                                                                        class_names=["Paddy", "Non Paddy"])})


def iterate(
        model, data_loader, criterion, optimizer=None, scheduler=None, mode="train", epoch=1, task="crop_type",
        device=None, log=False
):
    loss_meter = tnt.meter.AverageValueMeter()
    predictions = None
    targets = None
    pid_masks = None
    if log:
        columns = ["image_l8", "image_s2", "image_s1", "gt_mask", "pred_filtered", "pred_whole"]
        wandb_table = wandb.Table(columns=columns)

    t_start = time.time()
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"{mode}_{task}".title())
    for i, batch in pbar:
        if device is not None:
            batch = recursive_todevice(batch, device)
        data, masks = batch
        plot_mask = masks["plot_mask"]
        masks = masks[task]
        if task == "crop_type":
            masks = masks.long()
        else:
            masks = masks.float()
        if mode != "train":
            with torch.no_grad():
                y_pred = model(data)
        else:
            optimizer.zero_grad()
            y_pred = model(data)
        if task == "crop_yield":
            loss = criterion(y_pred, masks, plot_mask)
        else:
            loss = criterion(y_pred, masks)

        if mode == "train":
            loss.backward()
            optimizer.step()

        # Compute Metric
        if task == "crop_type":
            y_pred = nn.Softmax(dim=1)(y_pred)

        if predictions is None:
            predictions = y_pred
            targets = masks
            pid_masks = plot_mask
        else:
            predictions = torch.cat([predictions, y_pred], dim=0)
            targets = torch.cat([targets, masks], dim=0)
            pid_masks = torch.cat([pid_masks, plot_mask], dim=0)

        if log:
            if len(data.keys()) == 3:
                (l8_images, l8_dates) = data["L8"]
                (s2_images, s2_dates) = data["S2"]
                (s1_images, s1_dates) = data["S1"]
            else:
                (l8_images, l8_dates) = data[CFG.primary_sat]
                (s2_images, s2_dates) = data[CFG.primary_sat]
                (s1_images, s1_dates) = data[CFG.primary_sat]
            if task == "crop_type":
                # y_pred = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1)
                y_pred = nn.Softmax(dim=1)(y_pred)[:, 0, :, :]
                # log image of primary satellite
                l8_images, s2_images, s1_images, l8_dates, s2_dates, s1_dates, y_pred, masks = \
                    l8_images.cpu().numpy(), s2_images.cpu().numpy(), s1_images.cpu().numpy(), \
                    l8_dates.cpu().numpy(), s2_dates.cpu().numpy(), s1_dates.cpu().numpy(), \
                    y_pred.cpu().numpy(), masks.cpu().numpy()
            else:
                # log image of primary satellite
                y_pred = y_pred[:, 0, :, :]
                l8_images, s2_images, s1_images, l8_dates, s2_dates, s1_dates, y_pred, masks = \
                    l8_images.cpu().numpy(), s2_images.cpu().numpy(), s1_images.cpu().numpy(), \
                    l8_dates.cpu().numpy(), s2_dates.cpu().numpy(), s1_dates.cpu().numpy(), \
                    y_pred.cpu().numpy(), masks.cpu().numpy()
            _task = task
            if CFG.actual_season:
                _task = task + "_season"
            log_test_predictions(l8_images, s2_images, s1_images, l8_dates, s2_dates, s1_dates, masks, y_pred,
                                 wandb_table, CFG.seed, batch_id=i, task=_task)

        loss_meter.add(loss.item())

        # Just for Monitoring
        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            Loss=f"{loss.item():0.4f}",
            gpu_mem=f"{mem:0.2f} GB",
        )
    # take scheduler step
    if scheduler is not None and epoch < 3 * CFG.epochs // 4:
        scheduler.step()

    t_end = time.time()
    total_time = t_end - t_start
    # print("Epoch time : {:.1f}s".format(total_time))
    metrics = get_metrics(predictions, targets, pid_masks, ignore_index=CFG.ignore_index, task=task)
    if log:
        return loss_meter.value()[0], metrics, wandb_table
    return loss_meter.value()[0], metrics


n_log = 10  # no of samples to log


def generate_heatmap(mask):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure()
    hm = sns.heatmap(data=mask, vmin=-1, vmax=1 if np.max(mask) <= 1 else np.max(mask),
                     cmap='RdYlGn')
    plt.axis('off')
    fig.canvas.draw()
    mask = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    mask = mask.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return mask


def log_test_predictions(l8_images, s2_images, s1_images, l8_dates, s2_dates, s1_dates, gt_masks, pred_masks,
                         test_table, seed, batch_id=None, task="crop_type"):
    _id = 0
    # print(gt_masks.shape,pred_masks.shape)
    # pred_masks[pred_masks == 1] = 128
    gt_masks[gt_masks == -999] = -1
    gt_masks[gt_masks < -1] = 0


    # print(np.unique(pred_masks))
    i = 0
    for l8_sample, s2_sample, s1_sample, l8_sample_dates, s2_sample_dates, s1_sample_dates, gt_mask, pred_mask in \
            zip(l8_images, s2_images, s1_images, l8_dates, s2_dates, s1_dates, gt_masks, pred_masks):
        # if i == 9:
        #     for x, (l8_image, s1_image, s2_image) in enumerate(zip(l8_sample, s1_sample, s2_sample)):
        #         l8_image = l8_image[
        #             CFG.satellites["L8" if len(CFG.satellites) == 3 else CFG.primary_sat]["rgb_bands"]].transpose(1, 2,
        #                                                                                                           0)
        #         l8_image = ((l8_image - np.min(l8_image)) / (np.max(l8_image) - np.min(l8_image)))
        #         s2_image = s2_image[
        #             CFG.satellites["S2" if len(CFG.satellites) == 3 else CFG.primary_sat]["rgb_bands"]].transpose(1, 2,
        #                                                                                                           0)
        #         s2_image = ((s2_image - np.min(s2_image)) / (np.max(s2_image) - np.min(s2_image)))
        #         s1_image = s1_image[
        #             CFG.satellites["S1" if len(CFG.satellites) == 3 else CFG.primary_sat]["rgb_bands"]].transpose(1, 2,
        #                                                                                                           0)
        #         s1_image = ((s1_image - np.min(s1_image)) / (np.max(s1_image) - np.min(s1_image)))
        #         plt.imsave(f"test_results/seed2/{task}/{batch_id}_{i}_{x}_L8.png", l8_image)
        #         plt.imsave(f"test_results/seed2/{task}/{batch_id}_{i}_{x}_S2.png", s2_image)
        #         plt.imsave(f"test_results/seed2/{task}/{batch_id}_{i}_{x}_S1.png", s1_image)

        # get last available image
        l8_image = l8_sample[len(l8_sample_dates[l8_sample_dates != 0]) - 1]
        # reshape and normalize image
        l8_image = l8_image[CFG.satellites["L8" if len(CFG.satellites) == 3 else CFG.primary_sat]["rgb_bands"]].transpose(1, 2, 0)
        l8_image = ((l8_image - np.min(l8_image)) / (np.max(l8_image) - np.min(l8_image)))

        s2_image = s2_sample[len(s2_sample_dates[s2_sample_dates != 0]) - 1]
        # reshape and normalize image
        s2_image = s2_image[CFG.satellites["S2" if len(CFG.satellites) == 3 else CFG.primary_sat]["rgb_bands"]].transpose(1, 2, 0)
        s2_image = ((s2_image - np.min(s2_image)) / (np.max(s2_image) - np.min(s2_image)))

        s1_image = s1_sample[len(s1_sample_dates[s1_sample_dates != 0]) - 1]
        # reshape and normalize image
        s1_image = s1_image[CFG.satellites["S1" if len(CFG.satellites) == 3 else CFG.primary_sat]["rgb_bands"]].transpose(1, 2, 0)
        s1_image = ((s1_image - np.min(s1_image)) / (np.max(s1_image) - np.min(s1_image)))

        # log whole prediction mask
        os.makedirs(f"test_results/seed{seed}/test/{task}", exist_ok=True)
        if task == "crop_type":
            np.save(f"test_results/seed{seed}/test/{task}/{batch_id}_{i}.npy", pred_mask)
        else:
            crop_type_mask = np.load(f"test_results/seed{seed}/test/crop_type/{batch_id}_{i}.npy")
            pred_mask[crop_type_mask <= 0.5] = -1
        pred_mask_whole = generate_heatmap(copy.deepcopy(pred_mask))
        pred_mask[gt_mask == -1] = -1
        pred_mask = generate_heatmap(copy.deepcopy(pred_mask))
        if task == "crop_type":
            gt_mask[gt_mask == 0] = 2
            gt_mask[gt_mask == 1] = 0
            gt_mask[gt_mask == 2] = 1
        gt_mask = generate_heatmap(copy.deepcopy(gt_mask))
        plt.imsave(f"test_results/seed{seed}/test/{task}/{batch_id}_{i}_L8.png", l8_image)
        plt.imsave(f"test_results/seed{seed}/test/{task}/{batch_id}_{i}_S2.png", s2_image)
        plt.imsave(f"test_results/seed{seed}/test/{task}/{batch_id}_{i}_S1.png", s1_image)
        plt.imsave(f"test_results/seed{seed}/test/{task}/{batch_id}_{i}_ground_truth.png", gt_mask)
        plt.imsave(f"test_results/seed{seed}/test/{task}/{batch_id}_{i}_pred_mask_whole.png", pred_mask_whole)
        plt.imsave(f"test_results/seed{seed}/test/{task}/{batch_id}_{i}_pred_mask.png", pred_mask)
        i += 1

        test_table.add_data(wandb.Image(l8_image), wandb.Image(s2_image), wandb.Image(s1_image), wandb.Image(gt_mask),
                            wandb.Image(pred_mask), wandb.Image(pred_mask_whole))
        _id += 1
        if _id == n_log:
            break


def main(CFG):
    prepare_output(CFG)
    device = torch.device(CFG.device)

    # Dataset definition
    data_dir = CFG.data_dir
    df = pd.read_csv(os.path.join(data_dir, "sickle_dataset_tabular.csv"))
    # if "S2" in CFG.satellites.keys():
    #     df = df[df[f"S2_available"] == True].reset_index(drop=True)
    # else:
    #     df = df[df[f"{CFG.primary_sat}_available"] == True].reset_index(drop=True)
    if CFG.task != "crop_type":
        df = df[df.YIELD > 0].reset_index(drop=True)

    test_df = df[df.SPLIT == "test"].reset_index(drop=True)

    dt_args = dict(
        data_dir=data_dir,
        satellites=CFG.satellites,
        ignore_index=CFG.ignore_index,
        # transform=CFG.use_augmentation,
        actual_season=CFG.actual_season
    )
    dt_test = SICKLE_Dataset(df=test_df, **dt_args)

    collate_fn = lambda x: utae_utils.pad_collate(x, pad_value=CFG.pad_value)
    test_loader = data.DataLoader(
        dt_test,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )
    batch_data, masks = next(iter(test_loader))
    for sat in CFG.satellites.keys():
        (samples, dates) = batch_data[sat]

    # Model definition
    model = model_utils.Fusion_model(CFG)
    model = model.to(device)
    CFG.N_params = utae_utils.get_ntrainparams(model)
    # print("TOTAL TRAINABLE PARAMETERS :", CFG.N_params)
    with open(os.path.join(CFG.run_path, "conf.json"), "w") as file:
        file.write(json.dumps(vars(CFG), indent=4))

    # Optimizer, Loss and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    if CFG.task == "crop_type":
        criterion = nn.CrossEntropyLoss(ignore_index=CFG.ignore_index,
                                        weight=torch.tensor([0.62013, 0.37987])).to(device=CFG.device,
                                                                                    dtype=torch.float32)
    else:
        criterion = RMSELoss(ignore_index=CFG.ignore_index)
    best_checkpoint = torch.load(
        os.path.join(
            CFG.best_path, "checkpoint_best.pth.tar"
        )
    )
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    test_loss, test_metrics, _ = iterate(
        model,
        data_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        mode="test",
        device=device,
        task=CFG.task,
        log=True
    )
    print(f"Test Result {CFG.task}")
    if CFG.task == "crop_type":
        # test metric
        test_f1_macro, test_acc, test_iou, test_f1_paddy, test_f1_non_paddy, \
        test_acc_paddy, test_acc_non_paddy, test_iou_paddy, test_iou_non_paddy, _ = test_metrics
        deciding_metric = test_f1_macro
        # log and print metrics
        print(
            f"F1: {test_f1_macro:0.4f} | Paddy F1: {test_f1_paddy:0.4f} | Non-Paddy F1: {test_f1_non_paddy:0.4f} \nAcc:{test_acc:0.4f} | Paddy Acc: {test_acc_paddy:0.4f} | Non-Paddy Acc: {test_acc_non_paddy:0.4f}\niou:{test_iou:0.4f} | Paddy iou: {test_iou_paddy:0.4f} | Non-Paddy iou: {test_iou_non_paddy:0.4f}")
        log_wandb(test_loss, test_metrics, phase="test")

    else:
        # test metrics
        test_rmse, test_mae, test_mape = test_metrics
        print(f"Test RMSE: {test_rmse:0.4f} | Test MAE: {test_mae:0.4f} | Test MAPE: {test_mape:0.4f}")
        testlog = {
            "test_loss": test_loss,
            "test_rmse": test_rmse.item(),
            "test_mae": test_mae.item(),
            "test_mape": test_mape.item(),
        }
        if CFG.wandb:
            wandb.log(testlog)
    # log model to wandb
    # if CFG.wandb:
    #     best = wandb.Artifact('checkpoint_best', type='model')
    #     best.add_file(os.path.join(CFG.run_path, "checkpoint_best.pth.tar"))
    #     last = wandb.Artifact('checkpoint_last', type='model')
    #     last.add_file(os.path.join(CFG.run_path, "checkpoint_last.pth.tar"))
    #     wandb.log_artifact(best)
    #     wandb.log_artifact(last)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    CFG = parser.parse_args()
    set_seed(CFG.seed)
    for k, v in vars(CFG).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            try:
                CFG.__setattr__(k, list(map(int, v.split(","))))
            except:
                CFG.__setattr__(k, list(map(str, v.split(","))))

    CFG.exp_name = CFG.task

    # if task type is regression. Increase lr and change output channel to 1
    if CFG.task != "crop_type":
        # CFG.lr = 1e-1
        CFG.num_classes = 1
        # CFG.out_conv[-1] = 1

    # change out_conv incase of fusion
    # if len(CFG.satellites) >1:
    #     CFG.out_conv[-1] =  16
    # else:
    #     assert CFG.num_classes == CFG.out_conv[-1]

    CFG.run_path = f"runs/wacv_sickle_test/{CFG.exp_name}/{CFG.run_name}"
    satellite_metadata = {
        "S2": {
            "bands": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
            "rgb_bands": [3, 2, 1],
            "mask_res": 10,
            "img_size": (32, 32),
        },
        "S1": {
            "bands": ['VV', 'VH'],
            "rgb_bands": [0, 1, 0],
            "mask_res": 10,
            "img_size": (32, 32),
        },
        "L8": {
            "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"],
            "rgb_bands": [3, 2, 1],
            "mask_res": 30,
            "img_size": (32, 32),
        },
    }
    required_sat_data = {}
    for satellite in CFG.satellites:
        required_sat_data[satellite] = satellite_metadata[satellite]
    CFG.satellites = required_sat_data
    # first satellie is primary, img_size and mask_res is decided by it
    CFG.primary_sat = list(required_sat_data.keys())[0]
    CFG.img_size = required_sat_data[CFG.primary_sat]["img_size"]

    # WandB
    if CFG.wandb:
        wandb.login()
        run = wandb.init(
            project="temp_sickle_wacv_test",
            entity="agrifieldnet",
            config={k: v for k, v in dict(vars(CFG)).items() if "__" not in k},
            name=CFG.run_name,
            group=CFG.exp_name,
        )
    main(CFG)
