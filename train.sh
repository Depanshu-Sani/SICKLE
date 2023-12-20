#!/bin/bash
python train.py --data_dir $1 --satellite $2 --task crop_type --run_name "${2}_${3}" --model $3 &&
python train.py --data_dir $1 --satellite $2 --task sowing_date --run_name "${2}_${3}" --model $3 &&
python train.py --data_dir $1 --satellite $2 --task transplanting_date --run_name "${2}_${3}" --model $3 &&
python train.py --data_dir $1 --satellite $2 --task harvesting_date --run_name "${2}_${3}" --model $3 &&
python train.py --data_dir $1 --satellite $2 --task crop_yield --run_name "${2}_${3}" --model $3 &&
python train.py --data_dir $1 --satellite $2 --task crop_yield --run_name "${2}_${3}_season" --model $3 --actual_season