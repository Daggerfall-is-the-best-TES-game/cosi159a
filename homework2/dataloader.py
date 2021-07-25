from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
import os.path


class LFWDataset(Dataset):
    """Labeled Faces in the Wild Dataset"""
    def __init__(self, tsv_file, root_dir, transform=None):
        """
        Args:
            tsv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(tsv_file) as f:
            N = int(f.readline())
        matching_pairs = pd.read_csv(tsv_file, delim_whitespace=True, nrows=N, names=["name1", 1, 2], header=0)
        nonmatching_pairs = pd.read_csv(tsv_file, delim_whitespace=True, skiprows=N, names=["name1", 1, "name2", 2], header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.N = N
        self.pairs = pd.concat((matching_pairs, nonmatching_pairs), ignore_index=True,  sort=False)[nonmatching_pairs.columns]
        self.pairs.loc[0:N - 1, "name2"] = self.pairs.loc[:N - 1, "name1"]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.loc[idx]
        image_names = (f"{row.iloc[col_1]}/{row.iloc[col_1]}_{row.iloc[col_2]:04d}.jpg" for (col_1, col_2) in [(0, 1), (2, 3)])
        img_paths = (os.path.join(self.root_dir, name) for name in image_names)
        images = [imread(path) for path in img_paths]

        if self.transform:
            images = self.transform(images)

        return images, idx < self.N





if __name__ == "__main__":
    data = LFWDataset("../a-softmax_pytorch/data/lfw/pairsDevTrain.txt", "../a-softmax_pytorch/data/lfw/")

    print(data[0])

