import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import random
def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# Borrow from https://github.com/tudelft-iv/CrossViewMetricLocalization/blob/main/readdata_VIGOR.py#L194
class VIGOR(Dataset):
    def __init__(self, area, train_test, semi_index):
        super(VIGOR, self).__init__()
        self.root = '/work/datasets/vigor/VIGOR'
        self.area = area
        self.train_test = train_test  # for load training, validation or testing data list
        self.semi_index = semi_index   # [0, 1, 2, 3]
        self.sat_size = [256, 256]  # [320, 320] or [512, 512]
        self.grd_size = [256, 512]  # [320, 640]  # [224, 1232]
        label_root = 'splits'

        if self.area == 'same':
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        elif self.area == 'cross':
            self.train_city_list = ['NewYork', 'Seattle']
            if self.train_test == 'train':
                self.test_city_list = ['NewYork', 'Seattle']
            elif self.train_test == 'test':
                self.test_city_list = ['SanFrancisco', 'Chicago']

        # load sat list, the training and test set both contain all satellite images
        self.train_sat_list = []
        self.train_sat_index_dict = {}
        idx = 0
        for city in self.train_city_list:
            train_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(train_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.train_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.train_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', train_sat_list_fname, idx)
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        self.__cur_sat_id = 0  # for test
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.test_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            print('InputData::__init__: load', test_sat_list_fname, idx)
        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))

        # load grd training list and test list.
        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0
        for city in self.train_city_list:
            # load train panorama list
            if self.area == 'same':
                train_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_train.txt')
            if self.area == 'cross':
                train_label_fname = os.path.join(self.root, label_root, city, 'pano_label_balanced.txt')
            with open(train_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.train_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.train_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.train_label.append(label)
                    self.train_delta.append(delta)
                    if not label[0] in self.train_sat_cover_dict:
                        self.train_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.train_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', train_label_fname, idx)

        # split the original training set into training and validation sets
        # if self.train_test == 'train':
        #     self.train_list, self.val_list, self.train_label, self.val_label, self.train_delta, self.val_delta = \
        #         train_test_split(self.train_list, self.train_label, self.train_delta, test_size=0.2, random_state=42)

        self.val_list = []
        self.val_label = []
        self.test_sat_cover_dict = {}
        self.val_delta = []
        idx = 0
        for city in self.test_city_list:
            # load test panorama list
            if self.area == 'same':
                test_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt')
            if self.area == 'cross':
                test_label_fname = os.path.join(self.root, label_root, city, 'pano_label_balanced.txt')
            with open(test_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.test_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.val_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.val_label.append(label)
                    self.val_delta.append(delta)
                    if not label[0] in self.test_sat_cover_dict:
                        self.test_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.test_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print('InputData::__init__: load ', test_label_fname, idx)

        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        self.val_label = np.array(self.val_label)
        self.val_delta = np.array(self.val_delta)
        self.train_data_size = len(self.train_list)
        self.val_data_size = len(self.val_list)

        self.grd_transform = input_transform(self.grd_size)
        self.sat_transform = input_transform(self.sat_size)

        self.stride = 8      # total CNN down-sampling stride
        feature_len = int(self.sat_size[0] / self.stride)**2
        self.grid = torch.arange(0, feature_len, 1)
        self.grid = torch.reshape(self.grid, (int(self.sat_size[0] / self.stride),
                                              int(self.sat_size[0] / self.stride)))  # 32 x 32

    def __len__(self):
        if self.train_test == 'test':
            return self.val_data_size
        elif self.train_test == 'train':
            return self.train_data_size

    def __getitem__(self, index):
        if self.train_test == 'train':
            # load train ground image
            grd_img = Image.open(self.train_list[index]).convert('RGB')
            grd_img = self.grd_transform(grd_img)
            # load train satellite image
            row_offset = 320
            col_offset = 320
            while (np.abs(col_offset) >= 320 or np.abs(row_offset) >= 320):
                pos_index = random.randint(0, 3)  # each ground image is covered by 4 satellite images, randomly pick one
                sat_img = Image.open(self.train_sat_list[self.train_label[index][pos_index]])
                [row_offset, col_offset] = self.train_delta[index, pos_index]  # delta = [delta_lat, delta_lon]
            sat_img = self.sat_transform(sat_img.convert('RGB'))
            row_offset_resized = np.round(row_offset / 640 * self.sat_size[0])
            col_offset_resized = np.round(col_offset / 640 * self.sat_size[0])
            # grd position (Gy, Gx) related to sat_size (256, 256)
            ground_y, ground_x = (self.sat_size[0] / 2) - 1 + row_offset_resized, (self.sat_size[1] / 2) - 1 - col_offset_resized
            # compute grid index
            grid_y, grid_x = int(ground_y // self.stride), int(ground_x // self.stride)
            index = self.grid[grid_y, grid_x].item()   # [0, 1023]
            ty, tx = ground_y / self.stride - grid_y, ground_x / self.stride - grid_x

            return sat_img, grd_img, (index, ty, tx)

        if self.train_test == 'test':
            # load validation ground image
            grd_img = Image.open(self.val_list[index]).convert('RGB')
            grd_img = self.grd_transform(grd_img)

            pos_index = self.semi_index  # choose one of the positive or semi-positive satellite patch [0, 1, 2, 3]
            sat_img = Image.open(self.test_sat_list[self.val_label[index][pos_index]])
            [row_offset, col_offset] = self.val_delta[index, pos_index]  # delta = [delta_lat, delta_lon]

            sat_img = self.sat_transform(sat_img.convert('RGB'))
            row_offset_resized = np.round(row_offset / 640 * self.sat_size[0])
            col_offset_resized = np.round(col_offset / 640 * self.sat_size[0])
            # grd position (Gy, Gx) related to sat_size (256, 256)
            ground_y, ground_x = (self.sat_size[0] / 2) - 1 + row_offset_resized, (self.sat_size[1] / 2) - 1 - col_offset_resized
            # compute grid index
            grid_y, grid_x = int(ground_y // self.stride), int(ground_x // self.stride)
            index = self.grid[grid_y, grid_x].item()   # [0, 1023]
            ty, tx = ground_y / self.stride - grid_y, ground_x / self.stride - grid_x

            return sat_img, grd_img, (index, ty, tx), (ground_y, ground_x)

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
    dataset = VIGOR(area='cross', train_test='test')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=val_data_collect)
    for i, (sat, grd, label, ground_yx) in enumerate(dataloader):
        print(sat.shape)
        print(grd.shape)
        print(label)
        print(ground_yx)
        break
