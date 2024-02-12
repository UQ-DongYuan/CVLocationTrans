from model import CVLocationTrans
from readdata_Oxford import RobotCar, train_data_collect, val_data_collect
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import torch
from loss import cross_entropy_loss, regression_loss


# config
backbone_lr = 1e-5
otherlayers_lr = 1e-4
weight_decay = 1e-4
start_epoch = 0
end_epoch = 100
batch_size = 4
lambda_cross_entropy = 1
lambda_regression = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 512
grid_size = (32, 32)
stride = input_size / grid_size[0]
resolution = 0.144375   # 800 / 512 x 0.0924


def main():
    # setup train/val dataset
    test_dataset = RobotCar(mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4,
                                  shuffle=False, drop_last=True, collate_fn=val_data_collect)
    torch.cuda.empty_cache()
    model = CVLocationTrans(d_model=256).to(device)
    checkpoint_path = 'checkpoint/CVLocationTrans_RobotCar_SAM_512_max.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    distances = []
    with torch.set_grad_enabled(False):
        for i, (val_sat, val_grd, val_labels, val_ground_yx) in tqdm(enumerate(test_dataloader)):
            val_sat_img = val_sat.to(device)
            val_grd_img = val_grd.to(device)
            val_targets = val_labels

            val_pred_location, val_coordinate_reg = model(val_sat_img, val_grd_img)
            for batch_idx in range(batch_size):
                ground_y, ground_x = val_ground_yx[batch_idx]
                cur_pred_location = val_pred_location[batch_idx].cpu().detach().numpy()
                cur_pred_index = cur_pred_location.argmax()
                pred_grid_y, pred_grid_x = np.unravel_index(cur_pred_index, grid_size)
                pred_ty, pred_tx = val_coordinate_reg[batch_idx].cpu().detach().numpy()[cur_pred_index]
                pred_ground_y, pred_groundx = (pred_grid_y + pred_ty) * stride, (pred_grid_x + pred_tx) * stride
                distances.append(np.sqrt((ground_y - pred_ground_y) ** 2 + (ground_x - pred_groundx) ** 2) * resolution)

        distance_mean_error = np.mean(distances)
        distance_median_error = np.median(distances)
        print(f'distance mean error: {distance_mean_error}')
        print(f'distance median error: {distance_median_error}')


if __name__ == '__main__':
    main()
