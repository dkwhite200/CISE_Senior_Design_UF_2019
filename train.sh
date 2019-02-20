#!/bin/bash

python3 retrain.py --output_graph=$1.pb --output_labels=./$1_labels.txt --image_dir=$2
