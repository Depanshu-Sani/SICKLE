#!/bin/bash
python evaluate.py --data_dir $1 --satellite $2 --task crop_type --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/crop_type/${2}_${3}" &&
python evaluate.py --data_dir $1 --satellite $2 --task sowing_date --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/sowing_date/${2}_${3}" &&
python evaluate.py --data_dir $1 --satellite $2 --task transplanting_date --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/transplanting_date/${2}_${3}" &&
python evaluate.py --data_dir $1 --satellite $2 --task harvesting_date --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/harvesting_date/${2}_${3}" &&
python evaluate.py --data_dir $1 --satellite $2 --task crop_yield --run_name "${2}_${3}" --model $3 --best_path "runs/wacv_2024/crop_yield/${2}_${3}" &&
python evaluate.py --data_dir $1 --satellite $2 --task crop_yield --run_name "${2}_${3}_season" --model $3 --best_path "runs/wacv_2024/crop_yield/${2}_${3}_season" --actual_season