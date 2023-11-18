#!/bin/bash
python train.py --satellite $1 --task crop_type --run_name "${1}_${2}" --model $2 &&
python train.py --satellite $1 --task sowing_date --run_name "${1}_${2}" --model $2 &&
python train.py --satellite $1 --task transplanting_date --run_name "${1}_${2}" --model $2 &&
python train.py --satellite $1 --task harvesting_date --run_name "${1}_${2}" --model $2 &&
python train.py --satellite $1 --task crop_yield --run_name "${1}_${2}" --model $2 &&
python train.py --satellite $1 --task crop_yield --run_name "${1}_${2}_season" --model $2 --actual_season
