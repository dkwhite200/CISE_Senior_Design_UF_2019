#!/bin/bash

python label_image.py --graph=./$1.pb --labels=./$1_labels.txt --input_layer=Placeholder --output_layer=final_result --imgdir=$2
