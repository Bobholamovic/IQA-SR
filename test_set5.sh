#!/bin/bash

source activate gdf
python main.py val --list-dir '' --data-dir /home/gdf/Codes/SR/benchmark/0_test_dataset/Set5/ --out-dir Set5 --exp-config $1 --resume $2
