#!/bin/bash
python test.py --satellite $1 --task crop_type --run_name "${1}_${2}" --model $2 --best_path "runs/wacv_2024/crop_type/${1}_${2}" &&
python test.py --satellite $1 --task sowing_date --run_name "${1}_${2}" --model $2 --best_path "runs/wacv_2024/sowing_date/${1}_${2}" &&
python test.py --satellite $1 --task transplanting_date --run_name "${1}_${2}" --model $2 --best_path "runs/wacv_2024/transplanting_date/${1}_${2}" &&
python test.py --satellite $1 --task harvesting_date --run_name "${1}_${2}" --model $2 --best_path "runs/wacv_2024/harvesting_date/${1}_${2}" &&
python test.py --satellite $1 --task crop_yield --run_name "${1}_${2}" --model $2 --best_path "runs/wacv_2024/crop_yield/${1}_${2}" &&
python test.py --satellite $1 --task crop_yield --run_name "${1}_${2}_season" --model $2 --best_path "runs/wacv_2024/crop_yield/${1}_${2}_season" --actual_season