# Baseline: CNN Model

## Overview
This is a baseline convolutional neural network for multi-label movie genre classification.

## Data
- Input: movie poster images
- Labels: multiple genres per movie

## Model
- Custom CNN (PosterCNN)
- Loss: BCEWithLogitsLoss

## Training
python train.py

## Evaluation
python evaluate.py

## Metrics
- Micro F1
- Macro F1
- Top-3 accuracy

## Notes
- Dataset and images are not included in the repository
- Place CSV files in `datasets/`
- Place images in `posters/`
