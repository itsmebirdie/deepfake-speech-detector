#!/usr/bin/env bash
# Step A2: LibriSeVoc archive (Google Drive; very large). Run from repo root.
set -e
# Current gdown has no --fuzzy option, so pass the bare Drive file ID.
# If Drive says "Too many users have viewed or downloaded this file", download it in a browser instead.
mkdir -p data/raw
gdown "1Zh6b51S1WIsFjdCDRTQhYW61CQ0Ue1lk" -O data/raw/LibriSeVoc.zip
echo "Unzip into data/raw/LibriSeVoc and record the folder layout in docs/DATASET_NOTES.md"
