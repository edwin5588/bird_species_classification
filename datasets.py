import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torchvision import transforms, datasets
from PIL import Image

# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class

class bird_dataset(Dataset):
# You can read the train_list.txt and test_list.txt files here.
    def __init__(self, root, file_path):
        self.root = root
        self.file_path = file_path

        self.datafile = os.path.join(self.root, self.file_path)

        #self.train = np.loadtxt("train_list.txt")
        #self.test = np.loadtxt("test_list.txt")
        #self.label = np.loadtxt("classes.txt")
    def pil_loader(self, path):
        # https://stackoverflow.com/questions/59218671/runtimeerror-output-with-shape-1-224-224-doesnt-match-the-broadcast-shape
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return sum(1 for line in open(self.datafile))

    # Reshape image to (224,224).
    # Try normalizing with imagenet mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] or
    # any standard normalization
    # You can other image transformation techniques too
    def __getitem__(self, item):
        with open(self.datafile) as f:
            all_fps = f.readlines()

        # image filename, label
        all_fps = [x.split() for x in all_fps]

        # read the image with given item
        img_path = os.path.join(self.root, 'images', all_fps[item][0])
        img = self.pil_loader(img_path)

        data_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        trans_img = data_transform(img)

        # image as tensor, label
        return trans_img.float(), float(all_fps[item][1])
