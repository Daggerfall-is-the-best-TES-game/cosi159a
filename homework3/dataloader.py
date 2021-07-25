# David Fried
# 4/1/2021
# custom Wider Facial Landmarks in-the-wild dataset and loader for pytorch
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html?highlight=landmark
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import transform
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
from itertools import chain
import matplotlib.pyplot as plt
from skimage import io


class WFLWDataset(Dataset):
    """Wider Facial Landmarks in-the-wild Dataset"""
    def __init__(self, tsv_file, root_dir, train=False, transform=None):
        """
        Args:
            tsv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        tsv_file = Path(root_dir) / Path(tsv_file)
        cols = list(chain(chain.from_iterable(zip((f"x{crd}" for crd in range(98)), (f"y{crd}" for crd in range(98)))),
                     "x_min_rect y_min_rect x_max_rect y_max_rect pose expression illumination make-up occlusion blur image_name".split()))
        self.data = pd.read_csv(tsv_file, names=cols, delim_whitespace=True, header=0)
        self.n_landmarks = 98
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        path = self.root_dir / "WFLW_images" / row["image_name"]

        image = io.imread(path)
        landmarks = np.array(row["x0":"y97"]).astype("float")
        landmarks = landmarks.reshape(-1, 2)

        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        if self.train:
            return sample
        else:
            return {"image": sample["image"]}


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = F.normalize(image, self.mean, self.std)

        return {'image': image, 'landmarks': landmarks}






train_t = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    ToTensor(),
])
test_t = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    ToTensor(),
])


def show_landmarks(image, landmarks):
    """Show image with landmarks"""

    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=1, marker='.', c='r')



if __name__ == "__main__":
    data = WFLWDataset("WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt", "data", train=True, transform=train_t)
    print(data.__getitem__(4)["image"].dtype)


