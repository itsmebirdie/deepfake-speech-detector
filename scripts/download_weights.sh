#!/usr/bin/env bash
# Step A2: released pretrained baseline weights.
set -e
mkdir -p weights
gdown --fuzzy "https://drive.google.com/file/d/15qOi26czvZddIbKP_SOR8SLQFZK8cf8E/view?usp=sharing" -O weights/baseline_pretrained.pth
