from model import CVLocationTrans
from readdata_VIGOR import VIGOR, val_data_collect
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
stride = 8
grid_size = (32, 32)
resolution = 0.285   # 640 / 256 x 0.114
def main():

    val_dataset = VIGOR(area='same', train_test='test')
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=4, shuffle=False, drop_last=True, collate_fn=val_data_collect)
    torch.cuda.empty_cache()
    model = CVLocationTrans(d_model=256).to('cuda')
    checkpoint_path = 'checkpoint/CVLocationTrans.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    distances = []
    with torch.set_grad_enabled(False):
        for i, (val_sat, val_grd, val_labels, val_ground_yx) in tqdm(enumerate(val_dataloader)):
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
        print(f'saturn same area positive distance mean error: {distance_mean_error}')
        print(f'saturn same area positive distance median error: {distance_median_error}')


if __name__ == '__main__':
    main()
