import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import cv2
import random
import math

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# Borrow from https://github.com/tudelft-iv/CrossViewMetricLocalization/blob/main/readdata_Oxford.py #L194
class RobotCar(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode    # model = 'train', 'val', 'test'
        self.grd_image_root = '/media/dongyuan/DATA/Oxford_processed_grd'
        # read the full satellite image into memory
        self.full_satellite_map = cv2.imread('/media/dongyuan/DATA/Oxford_processed_grd/satellite_map_new.png')
        self.sat_size = [256, 256]
        self.grd_size = [256, 384]
        # check use correct mode
        assert self.mode in ['train', 'val', 'test']
        # load ground image list for training, validation or testing
        if self.mode == 'train':
            self.train_grd_img_list = []
            with open('Oxford_split/training.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    self.train_grd_img_list.append(content.split(" "))
            with open('Oxford_split/train_yaw.npy', 'rb') as f:
                self.train_yaw = np.load(f)

            self.trainNum = len(self.train_grd_img_list)
            trainarray = np.array(self.train_grd_img_list)
            self.trainUTM = np.transpose(trainarray[:, 2:].astype(np.float64))
            print(f'number of ground images in training set : {self.trainNum}', )

        elif self.mode == 'val':
            self.val_grd_img_list = []
            with open('Oxford_split/validation.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    self.val_grd_img_list.append(content.split(" "))
            with open('Oxford_split/val_yaw.npy', 'rb') as f:
                self.val_yaw = np.load(f)

            self.valNum = len(self.val_grd_img_list)
            valarray = np.array(self.val_grd_img_list)
            self.valUTM = np.transpose(valarray[:, 2:].astype(np.float64))
            print(f"number of ground images in validation set : {self.valNum}")

        else:
            # 3 test traversals
            test_2015_08_14_14_54_57 = []
            with open('Oxford_split/test1_j.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    test_2015_08_14_14_54_57.append(content.split(" "))
            test_2015_08_12_15_04_18 = []
            with open('Oxford_split/test2_j.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    test_2015_08_12_15_04_18.append(content.split(" "))
            test_2015_02_10_11_58_05 = []
            with open('Oxford_split/test3_j.txt', 'r') as filehandle:
                filecontents = filehandle.readlines()
                for line in filecontents:
                    content = line[:-1]
                    test_2015_02_10_11_58_05.append(content.split(" "))
            with open('Oxford_split/test_yaw.npy', 'rb') as f:
                self.test_yaw = np.load(f)

            self.test_grd_img_list = test_2015_08_14_14_54_57 + test_2015_08_12_15_04_18 + test_2015_02_10_11_58_05
            self.testNum = len(self.test_grd_img_list)
            testarray = np.array(self.test_grd_img_list)
            self.testUTM = np.transpose(testarray[:, 2:].astype(np.float64))
            print(f"number of ground images in test set : {self.testNum}")

            # calculate the transformation from easting, northing to satellite image col, row
            # transformation for the satellite image
            primary = np.array([[619400., 5736195.],
                                [619400., 5734600.],
                                [620795., 5736195.],
                                [620795., 5734600.],
                                [620100., 5735400.]])
            secondary = np.array([[900., 900.],  # tl
                                  [492., 18168.],  # bl
                                  [15966., 1260.],  # tr
                                  [15553., 18528.],  # br
                                  [8255., 9688.]])  # c

            n = primary.shape[0]
            pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
            unpad = lambda x: x[:, :-1]
            X = pad(primary)
            Y = pad(secondary)

            # Solve the least squares problem X * A = Y
            # to find our transformation matrix A
            A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
            self.transform = lambda x: unpad(np.dot(pad(x), A))

            self.grd_transform = input_transform(self.grd_size)
            self.sat_transform = input_transform(self.sat_size)

            self.stride = 8  # total CNN down-sampling stride
            feature_len = int(self.sat_size[0] / self.stride) ** 2
            self.grid = torch.arange(0, feature_len, 1)
            self.grid = torch.reshape(self.grid, (int(self.sat_size[0] / self.stride),
                                                  int(self.sat_size[0] / self.stride)))  # 32 x 32

    def __len__(self):
        if self.mode == 'train':
            return self.trainNum
        if self.mode == 'val':
            return self.valNum
        if self.mode == 'test':
            return self.testNum

    def __getitem__(self, index):
        if self.mode == 'train':
            # load ground image
            grd_img = Image.open(os.path.join(self.grd_image_root, self.train_grd_img_list[index][0])).convert('RGB')
            grd_img = self.grd_transform(grd_img)
            # load satellite image from the full satellite map
            image_coord = self.transform(np.array([[self.trainUTM[0, index], self.trainUTM[1, index]]]))[0]

            # generate a random offset for the ground image
            alpha = 2 * math.pi * random.random()
            r = 200 * np.sqrt(2) * random.random()
            row_offset = int(r * math.cos(alpha))
            col_offset = int(r * math.sin(alpha))


        if self.mode == 'val':
            pass
        if self.mode == 'test':
            pass
