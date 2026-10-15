#!/usr/bin/env bash
# Step A1: add the authors' repo as a submodule (pinned commit; record the hash in docs/DATASET_NOTES.md)
set -e
git submodule add https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts \
  third_party/Synthetic-Voice-Detection-Vocoder-Artifacts
git submodule update --init --recursive
echo "Copy train.txt / dev.txt / test.txt from the submodule into data/splits/ (step A2)."
