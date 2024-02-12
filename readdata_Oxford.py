import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import cv2
import random
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        full_satellite_map = cv2.imread('/media/dongyuan/DATA/Oxford_processed_grd/satellite_map_new.png')
        self.full_satellite_map = cv2.cvtColor(full_satellite_map, cv2.COLOR_BGR2RGB)
        self.sat_size = (512, 512)
        self.grd_size = (160, 240)
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
                print(self.test_yaw[0:1673].shape)

            print(len(test_2015_08_14_14_54_57))
            print(len(test_2015_08_12_15_04_18))
            print(len(test_2015_02_10_11_58_05))
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

        # set up ground/satellite image transforms
        self.grd_transform = transforms.Compose([
            transforms.Resize(size=tuple(self.grd_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.sat_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        feature_len = 1024
        grid_size = (32, 32)
        self.stride = self.sat_size[0] / grid_size[0]
        self.grid = torch.arange(0, feature_len, 1)
        self.grid = torch.reshape(self.grid, grid_size)  # 32 x 32

    def __len__(self):
        if self.mode == 'train':
            return self.trainNum
        if self.mode == 'val':
            return self.valNum
        if self.mode == 'test':
            return self.testNum

    def __getitem__(self, img_index):
        if self.mode == 'train':
            # load ground image
            grd_img = Image.open(os.path.join(self.grd_image_root, self.train_grd_img_list[img_index][0])).convert('RGB')
            grd_img = self.grd_transform(grd_img)
            # load satellite image from the full satellite map
            image_coord = self.transform(np.array([[self.trainUTM[0, img_index], self.trainUTM[1, img_index]]]))[0]

            # generate a random offset for the ground image
            alpha = 2 * math.pi * random.random()
            r = 200 * np.sqrt(2) * random.random()
            row_offset = int(r * math.cos(alpha))
            col_offset = int(r * math.sin(alpha))

            sat_coord_row = int(image_coord[1] + row_offset)
            sat_coord_col = int(image_coord[0] + col_offset)

            # crop a satellite patch centered at the location of the ground image offseted by a randomly generated amount
            img = self.full_satellite_map[sat_coord_row-400-200:sat_coord_row+400+200, sat_coord_col-400-200:sat_coord_col+400+200, :]
            rotate_angle = self.train_yaw[img_index]/np.pi*180-90
            rot_matrix = cv2.getRotationMatrix2D((600, 600), rotate_angle, 1)
            img = cv2.warpAffine(img, rot_matrix, (1200, 1200))
            img = img[200:1000, 200:1000, :]    # 800, 800, 3

            # get satellite image input [3, 256, 256] tensor
            img = cv2.resize(img, self.sat_size, interpolation=cv2.INTER_AREA)
            sat_img = self.sat_transform(img)

            # get original ground position on (1200,1200) cropped satellite image
            grd_col, grd_raw = 600 - col_offset, 600 - row_offset
            # get rotated ground position on (1200, 1200) rotated satellite image
            grd_rot_col, grd_rot_raw = np.dot(rot_matrix, np.array([[grd_col], [grd_raw], [1]]))
            # get rotated ground position on (800, 800) rotated satellite image
            grd_rot_col, grd_rot_raw = int(grd_rot_col.item() - 200), int(grd_rot_raw.item() - 200)

            # get ground position (Gx, Gy) related to sat_size:q
            ground_y, ground_x = int(grd_rot_raw / 800 * self.sat_size[0]), int(grd_rot_col / 800 * self.sat_size[1])
            # compute grid index
            grid_y, grid_x = int(ground_y // self.stride), int(ground_x // self.stride)
            idx = self.grid[grid_y, grid_x].item()   # [0, 1023]
            ty, tx = ground_y / self.stride - grid_y, ground_x / self.stride - grid_x

            return sat_img, grd_img, (idx, ty, tx)

        if self.mode == 'val':
            # load ground image
            grd_img = Image.open(os.path.join(self.grd_image_root, self.val_grd_img_list[img_index][0])).convert('RGB')
            grd_img = self.grd_transform(grd_img)

            # load satellite image from the full satellite map
            image_coord = self.transform(np.array([[self.valUTM[0, img_index], self.valUTM[1, img_index]]]))[0]
            col_split = int((image_coord[0]) // 400)
            if np.round(image_coord[0] - 400*col_split) < 200:
                col_split -= 1
            col_pixel = int(np.round(image_coord[0] - 400*col_split))  # related to 800x800

            row_split = int((image_coord[1]) // 400)
            if np.round(image_coord[1] - 400*row_split) < 200:
                row_split -= 1
            row_pixel = int(np.round(image_coord[1] - 400*row_split))  # related to 800x800

            img = self.full_satellite_map[row_split*400-200:row_split*400+800+200, col_split*400-200:col_split*400+800+200, :]  # read extra 200 pixels at each side to avoid blank after rotation
            rotate_angle = self.val_yaw[img_index]/np.pi*180-90
            rot_matrix = cv2.getRotationMatrix2D((600, 600), rotate_angle, 1)  # rotate satellite image
            img = cv2.warpAffine(img, rot_matrix, (1200, 1200))
            img = img[200:1000, 200:1000, :]  # 800, 800, 3

            # get satellite image input [3, 256, 256] tensor
            img = cv2.resize(img, self.sat_size, interpolation=cv2.INTER_AREA)
            sat_img = self.sat_transform(img)

            # get original ground position on [1200, 1200]
            col_pixel_orig = col_pixel + 200
            row_pixel_orig = row_pixel + 200
            # after rotated
            grd_rot_col, grd_rot_raw = np.dot(rot_matrix, np.array([[col_pixel_orig], [row_pixel_orig], [1]]))
            # get rotated ground position on (800, 800) rotated satellite image
            grd_rot_col, grd_rot_raw = int(grd_rot_col.item() - 200), int(grd_rot_raw.item() - 200)

            # get ground position (Gx, Gy) related to sat_size (256, 256)
            ground_y, ground_x = int(grd_rot_raw / 800 * self.sat_size[0]), int(grd_rot_col / 800 * self.sat_size[1])
            # compute grid index
            grid_y, grid_x = int(ground_y // self.stride), int(ground_x // self.stride)
            idx = self.grid[grid_y, grid_x].item()   # [0, 1023]
            ty, tx = ground_y / self.stride - grid_y, ground_x / self.stride - grid_x

            return sat_img, grd_img, (idx, ty, tx), (ground_y, ground_x)

        if self.mode == 'test':
            # load ground image
            grd_img = Image.open(os.path.join(self.grd_image_root, self.test_grd_img_list[img_index][0])).convert('RGB')
            grd_img = self.grd_transform(grd_img)
            # load satellite image from the full satellite map
            image_coord = self.transform(np.array([[self.testUTM[0, img_index], self.testUTM[1, img_index]]]))[0]

            col_split = int((image_coord[0]) // 400)
            if np.round(image_coord[0] - 400 * col_split) < 200:
                col_split -= 1
            col_pixel = int(np.round(image_coord[0] - 400 * col_split))  # related to 800x800

            row_split = int((image_coord[1]) // 400)
            if np.round(image_coord[1] - 400 * row_split) < 200:
                row_split -= 1
            row_pixel = int(np.round(image_coord[1] - 400 * row_split))  # related to 800x800

            img = self.full_satellite_map[row_split*400-200:row_split*400+800+200, col_split*400-200:col_split*400+800+200, :]  # read extra 200 pixels at each side to avoid blank after rotation

            rotate_angle = self.test_yaw[img_index] / np.pi * 180 - 90
            rot_matrix = cv2.getRotationMatrix2D((600, 600), rotate_angle, 1)  # rotate satellite image
            img = cv2.warpAffine(img, rot_matrix, (1200, 1200))
            img = img[200:1000, 200:1000, :]  # 800, 800, 3

            # get satellite image input [3, 256, 256] tensor
            img = cv2.resize(img, self.sat_size, interpolation=cv2.INTER_AREA)
            sat_img = self.sat_transform(img)

            # get original ground position on [1200, 1200]
            col_pixel_orig = col_pixel + 200
            row_pixel_orig = row_pixel + 200
            # after rotated
            grd_rot_col, grd_rot_raw = np.dot(rot_matrix, np.array([[col_pixel_orig], [row_pixel_orig], [1]]))
            # get rotated ground position on (800, 800) rotated satellite image
            grd_rot_col, grd_rot_raw = int(grd_rot_col.item() - 200), int(grd_rot_raw.item() - 200)

            # get ground position (Gx, Gy) related to sat_size (256, 256)
            ground_y, ground_x = int(grd_rot_raw / 800 * self.sat_size[0]), int(grd_rot_col / 800 * self.sat_size[1])
            # compute grid index
            grid_y, grid_x = int(ground_y // self.stride), int(ground_x // self.stride)
            idx = self.grid[grid_y, grid_x].item()  # [0, 1023]
            ty, tx = ground_y / self.stride - grid_y, ground_x / self.stride - grid_x

            return sat_img, grd_img, (idx, ty, tx), (ground_y, ground_x)

def train_data_collect(batch):
    sat_imgs = []
    grd_imgs = []
    labels = []
    for sample in batch:
        sat_imgs.append(sample[0])
        grd_imgs.append(sample[1])
        labels.append(sample[2])
    return torch.stack(sat_imgs, 0), torch.stack(grd_imgs, 0), labels

def val_data_collect(batch):
    sat_imgs = []
    grd_imgs = []
    labels = []
    ground_yx = []
    for sample in batch:
        sat_imgs.append(sample[0])
        grd_imgs.append(sample[1])
        labels.append(sample[2])
        ground_yx.append(sample[3])
    return torch.stack(sat_imgs, 0), torch.stack(grd_imgs, 0), labels, ground_yx

if __name__ == '__main__':
    dataset = RobotCar(mode='test')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=val_data_collect)
    for i, (sat, grd, label, ground_yx) in enumerate(dataloader):
        print(sat.shape)
        print(grd.shape)
        print(label)
        print(ground_yx)
        break