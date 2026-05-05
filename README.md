# Movie Poster Genre Classification (Baseline)

## Overview
This project implements a baseline CNN model for multi-label movie genre classification using poster images.

## Setup
pip install -r requirements.txt

## Train
python train.py

## Evaluate
python evaluate.py

## Model
- Custom CNN (PosterCNN)
- Loss: BCEWithLogitsLoss

## Metrics
- Micro F1
- Macro F1
- Top-3 accuracy

## Note
Dataset and poster images are not included due to size.

To run the project, place:
- CSV files in `datasets/`
- Images in `posters/`