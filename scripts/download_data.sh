#!/usr/bin/env bash
# Step A2: LibriSeVoc archive (Google Drive; very large). Run from repo root.
set -e
mkdir -p data/raw
gdown --fuzzy "https://drive.google.com/file/d/1Zh6b51S1WIsFjdCDRTQhYW61CQ0Ue1lk/view?usp=sharing" -O data/raw/LibriSeVoc.zip
echo "Unzip into data/raw/LibriSeVoc and record the folder layout in docs/DATASET_NOTES.md"
