#!/usr/bin/env bash
# Step A2: released pretrained baseline weights.
set -e
# Current gdown has no --fuzzy option, so pass the bare Drive file ID.
# If Drive says "Too many users have viewed or downloaded this file", download it in a browser instead.
mkdir -p weights
gdown "15qOi26czvZddIbKP_SOR8SLQFZK8cf8E" -O weights/baseline_pretrained.pth
