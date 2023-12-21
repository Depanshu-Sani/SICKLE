import os
import rasterio
import glob
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from datetime import date
import pandas as pd
from utils import transforms
import torch

# torch
from torch.utils.data import Dataset

# Albumentation for augmentation
import albumentations as A

mon_to_int = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
              "nov": 11, "dec": 12}


class UNET3d_Dataset(Dataset):
    def __init__(
            self,
            df,
            data_dir,
            satellite="S2",
            bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
            img_size=(32, 32),
            mask_res=10,
            ignore_index=0,
            transform=True,
            test=False,
            preload=False,
            phase="train"
    ):
        super(UNET3d_Dataset, self).__init__()

        self.df = df
        self.plot_ids = set(df.PLOT_ID)
        self.satellite = satellite
        self.mask_res = mask_res
        self.bands = bands
        self.data_dir = data_dir
        self.transform = transform
        self.test = test
        self.ignore_index = ignore_index
        self.img_size = img_size
        self.resize = A.Resize(height=img_size[0], width=img_size[1])
        self.preload = preload
        self.phase = phase

        #  Preloading data
        if preload:
            UIDs = list(df.UNIQUE_ID)
            with Pool(32) as p:
                self.samples = list(tqdm(p.imap(self._read_sample, UIDs), total=len(UIDs), desc='Preloading Samples: '))
            data = {}
            for sample in self.samples: data.update(sample)
            self.samples = data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = int(self.df.iloc[idx]["UNIQUE_ID"])
        if self.preload:
            sample, mask = self.samples[uid]
        else:
            sample, mask = self._read_sample(uid)[uid]

        plot_mask = mask[0]

        # Remove plots that are not in this split
        unmatched_plots = set(np.unique(plot_mask)[1:]) - self.plot_ids
        for unmatched_plot in unmatched_plots:
            mask[:, plot_mask == unmatched_plot] = self.ignore_index

        # Transform data
        if self.phase == "train" and self.transform:
            sample, mask = transforms.transform(sample, mask)
        else:
            sample = torch.tensor(sample)
            mask = torch.tensor(mask)

        # change shape to CxTxHxW
        sample = sample.permute(1, 0, 2, 3)
        mask = {
            "crop_type": mask[1],
            "sowing_date": mask[2],
            "transplanting_date": mask[3],
            "harvesting_date": mask[4],
            "crop_yield": mask[5],
        }
        return uid, sample, mask

    def check_zero_percentage(self, image):
        all_pixels = np.prod(self.img_size)
        zero_pixels = len(image[image == 0])
        return zero_pixels / all_pixels

    def _read_masks(self, uid):
        path = f"{self.data_dir}/masks/{self.mask_res}m/{uid}.tif"
        with rasterio.open(path, 'r') as fp:
            mask = fp.read()
        # only make crop mask binary and change ignore_index
        mask[0][mask[0]==0]=-1
        mask[1] -= 1
        # Make Cropt type binary i.e. paddy vs non-paddy
        mask[1][mask[1] > 1] = 1
        mask[2] -= 1
        # mask[2][mask[2] == -1] = self.ignore_index
        mask[3] -= 1
        # mask[3][mask[3] == -1] = self.ignore_index
        mask[4] -= 1
        # mask[4][mask[4] == -1] = self.ignore_index
        mask[5][mask[5] == 0] = -1
        mask[mask < -1] = 0
        mask[mask < 0] = self.ignore_index
        return mask

    def _read_sample(self, uid):
        #  get metadata
        season, year = self.df[self.df.UNIQUE_ID == int(uid)].iloc[0][["STANDARD_SEASON", "YEAR"]].values
        start_date = date(int(year), mon_to_int[season.split("-")[0]], 1)
        size = self.img_size

        # get all files associated this UID
        path = f"{self.data_dir}/images/{self.satellite}/npy/{uid}/*.npz"
        files = glob.glob(path)

        mask = self._read_masks(uid).transpose(1, 2, 0)
        mask = self.resize(image=mask)["image"].transpose(2, 0, 1)

        # define empty sample
        sample = np.zeros((184, len(self.bands), *size))
        missing_count = 0
        for file in files:
            print(file)
            try:
                data, index, zero_percentage = self._read_data(file, start_date)
            except Exception as e:
                missing_count+=1
                print("\n", e,"\n")
            # only consider data_file if no of zeros is less than 25%
            if zero_percentage < 0.25:
                sample[index] = data
    
        print(uid, missing_count)
        return {uid: (sample, mask)}

    def _read_data(self, file, start_date):
        # Read File
        data_file = np.load(file)
        # Get index of file
        index = self._get_data_index(file, start_date)
        # find zero percentage
        zero_percentage = self.check_zero_percentage(data_file[self.bands[0]])
        # get channels data
        try:
            all_channels = [self.resize(image=data_file[band])["image"] for band in self.bands]
        except Exception as e:
            # print("file ",file,"error ",e)
            all_channels = [self.resize(image=data_file[band])["image"] for band in self.bands[:-1]]
            all_channels = all_channels + [np.zeros(self.img_size, dtype=np.float32)]
        data = np.stack(all_channels, axis=0)
        return data, index, zero_percentage

    def _get_data_index(self, file, start_date):
        if self.satellite == "S2":
            # for S2_2018 data
            if os.path.basename(file)[0]=="T": index_date = os.path.basename(file).split("_")[1][:8]
            else: index_date = os.path.basename(file).split("_")[0][:8]
            
        elif self.satellite == "S1":
            index_date = os.path.basename(file).split("_")[4][:8]
        else:
            index_date = os.path.basename(file).split("_")[2][:8]
        index_date = date(int(index_date[:4]), int(index_date[4:6]), int(index_date[6:]))
        index = (index_date - start_date).days
        return index


class SICKLE_Dataset(UNET3d_Dataset):
    def __init__(
            self,
            df,
            data_dir,
            satellites={"S2": {
                "bands": ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
                "rgb_bands": [3, 2, 1],
                "mask_res": 10,
                "img_size": (32, 32),
            }},
            ignore_index=-999,
            transform=None,
            actual_season=False,
            phase="eval"
    ):
        self.df = df
        self.plot_ids = set(df.PLOT_ID)
        self.satellites = satellites
        self.data_dir = data_dir
        self.transform = transform
        self.ignore_index = ignore_index
        primary_sat = list(self.satellites.keys())[0]
        self.img_size = self.satellites[primary_sat]["img_size"]
        self.mask_res = self.satellites[primary_sat]["mask_res"]
        self.resize = A.Resize(height=self.img_size[0], width=self.img_size[1], interpolation=cv2.INTER_NEAREST)
        self.actual_season = actual_season
        self.phase = phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = int(self.df.iloc[idx]["UNIQUE_ID"])
        data = {}
        for satellite, satellite_info in self.satellites.items():
            # update satellite info
            self.satellite = satellite
            self.bands = satellite_info["bands"]
            # read sample
            sample, dates = self._read_sample(uid)[uid]
            data[satellite] = (sample, dates)

        # read and prepare mask
        mask = self._read_masks(uid).transpose(1, 2, 0)
        mask = self.resize(image=mask)["image"].transpose(2, 0, 1)

        # Remove plots that are not in this split
        plot_mask = mask[0]
        unmatched_plots = set(np.unique(plot_mask)[1:]) - self.plot_ids
        for unmatched_plot in unmatched_plots:
            mask[:, plot_mask == unmatched_plot] = self.ignore_index
        #transform/augment data
        if self.phase == "train" and self.transform:
            data, mask = transforms.transform(data, mask)

        mask = {
            "plot_mask": mask[0],
            "crop_type": mask[1],
            "sowing_date": mask[2],
            "transplanting_date": mask[3],
            "harvesting_date": mask[4],
            "crop_yield": mask[5],
        }
        return data, mask

    def _read_sample(self, uid):
        #  get metadata
        season, year = self.df[self.df.UNIQUE_ID == int(uid)].iloc[0][["STANDARD_SEASON", "YEAR"]].values
        start_date = date(int(year), mon_to_int[season.split("-")[0]], 1)

        # get all files associated this UID
        path = f"{self.data_dir}/images/{self.satellite}/npy/{uid}/*.npz"
        files = glob.glob(path)

        # define empty sample and dates
        sample = []
        dates = []

        if self.actual_season:
            sowing_day, harvesting_day = self.df[self.df.UNIQUE_ID == int(uid)].iloc[0][
                ["SOWING_DAY", "HARVESTING_DAY"]].values
        else:
            sowing_day, harvesting_day = 0, 183
        # print(files)
        missing_count = 0
        for file in files:
            try:
                data, index, zero_percentage = self._read_data(file, start_date)
                if sowing_day <= index <= harvesting_day:
                # only consider data_file if no of zeros is less than 25%
                    if zero_percentage < 0.25:
                        if index not in dates:
                            dates.append(index)
                            sample.append(data)
            except Exception as e:
                missing_count+=1
                
        # correct order
        dates, sample = zip(*sorted(zip(dates, sample)))
        sample = np.stack(sample)
        dates = torch.tensor(dates)
        return {uid: (sample, dates)}

    def _get_data_index(self, file, start_date):
        index = UNET3d_Dataset._get_data_index(self, file, start_date)
        return index + 1




