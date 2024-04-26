#!/bin/bash

python run.py LRASPP_rgb_seg.yml human_seg_rgb_v1.yml --gpu_override [0,1]

python run.py DLv3_rn50_rgb_seg.yml human_seg_rgb_v1.yml --gpu_override [0,1]


