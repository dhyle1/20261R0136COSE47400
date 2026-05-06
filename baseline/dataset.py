import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PosterDataset(Dataset):
    """
    PyTorch Dataset for movie posters.

    Loads samples on demand:
    - reads CSV for filenames + labels
    - loads image from disk
    - applies optional transform
    - return (image, multi-label tensor)
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # filter out rows with missing image files to avoid runtime errors
        self.df = self.df[
            self.df["img_filename"].apply(self.image_exists)
        ].reset_index(drop=True)

        self.genre_cols = ["Drama", 
                           "Comedy", 
                           "Romance", 
                           "Crime", 
                           "Short", 
                           "Adventure", 
                           "Mystery", 
                           "Horror", 
                           "Musical", 
                           "Fantasy", 
                           "Family", 
                           "Action", 
                           "Western", 
                           "History", 
                           "War", 
                           "Thriller", 
                           "Animation", 
                           "Biography", 
                           "Documentary", 
                           "Sport"
                           ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # return one sample (image, labels) for given index
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row["img_filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row[self.genre_cols].values.astype("float32")
        )

        return image, labels
    
    def image_exists(self, filename):
        path = os.path.join(self.img_dir, filename)
        return os.path.exists(path)