from model import CVLocationTrans
from readdata_Oxford import RobotCar, train_data_collect, val_data_collect
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import torch
from loss import cross_entropy_loss, regression_loss
from sam import SAM
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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
stride = 8
grid_size = (32, 32)
resolution = 0.28875   # 800 / 256 x 0.0924

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def main():
    # setup train/val dataset
    train_dataset = RobotCar(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                  shuffle=True, drop_last=True, collate_fn=train_data_collect)
    val_dataset = RobotCar(mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,
                                shuffle=False, drop_last=True, collate_fn=val_data_collect)
    torch.cuda.empty_cache()
    model = CVLocationTrans(d_model=256).to(device)
    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "extractor" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "extractor" in n and p.requires_grad],
            "lr": backbone_lr,
        },
    ]
    base_optimizer = optim.AdamW
    optimizer = SAM(param_dicts, base_optimizer, rho=2, adaptive=True, lr=otherlayers_lr, weight_decay=weight_decay)

    # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)
    best_mean_distance_error = 9999

    for epoch_idx in range(start_epoch, end_epoch):
        model.train()
        epoch_loss = []
        iteration = 0
        for iteration, (sat, grd, labels) in tqdm(enumerate(train_dataloader)):
            sat_img = sat.to(device)
            grd_img = grd.to(device)
            targets = labels
            # Use SAM method to optimize model
            optimizer.zero_grad()
            pred_location, coordinate_reg = model(sat_img, grd_img)
            location_loss = cross_entropy_loss(pred_location, targets)
            txy_loss = regression_loss(coordinate_reg, targets)
            loss = lambda_cross_entropy * location_loss + lambda_regression * txy_loss
            epoch_loss.append(loss.item())
            loss.backward()

            optimizer.first_step(zero_grad=True)
            pred_location, coordinate_reg = model(sat_img, grd_img)
            location_loss = cross_entropy_loss(pred_location, targets)
            txy_loss = regression_loss(coordinate_reg, targets)
            loss = lambda_cross_entropy * location_loss + lambda_regression * txy_loss
            loss.backward()
            optimizer.second_step(zero_grad=True)

        curr_training_loss = sum(epoch_loss) / (iteration + 1)
        train_file = 'training_robotcar_loss_SAM.txt'
        with open(train_file, 'a') as file:
            file.write(f"Epoch {epoch_idx} Training Loss: {curr_training_loss}" + '\n')

        print(f"Epoch {epoch_idx} Training Loss: {curr_training_loss}")

        model.eval()
        val_epoch_loss = []
        distances = []
        with torch.set_grad_enabled(False):
            for i, (val_sat, val_grd, val_labels, val_ground_yx) in tqdm(enumerate(val_dataloader)):
                val_sat_img = val_sat.to(device)
                val_grd_img = val_grd.to(device)
                val_targets = val_labels

                val_pred_location, val_coordinate_reg = model(val_sat_img, val_grd_img)
                val_location_loss = cross_entropy_loss(val_pred_location, val_targets)
                val_txy_loss = regression_loss(val_coordinate_reg, val_targets)
                val_loss = lambda_cross_entropy * val_location_loss + lambda_regression * val_txy_loss
                val_epoch_loss.append(val_loss.item())

                for batch_idx in range(batch_size):
                    ground_y, ground_x = val_ground_yx[batch_idx]
                    cur_pred_location = val_pred_location[batch_idx].cpu().detach().numpy()
                    cur_pred_index = cur_pred_location.argmax()
                    pred_grid_y, pred_grid_x = np.unravel_index(cur_pred_index, grid_size)
                    pred_ty, pred_tx = val_coordinate_reg[batch_idx].cpu().detach().numpy()[cur_pred_index]
                    pred_ground_y, pred_groundx = (pred_grid_y + pred_ty) * stride, (pred_grid_x + pred_tx) * stride
                    distances.append(np.sqrt((ground_y - pred_ground_y)**2 + (ground_x - pred_groundx)**2) * resolution)

            curr_val_loss = sum(val_epoch_loss) / (i+1)
            distance_mean_error = np.mean(distances)
            distance_median_error = np.median(distances)

            val_loss_file = 'robotcar_val_loss_SAM.txt'
            val_distance_error = 'robotcar_val_distance_error_SAM.txt'
            with open(val_loss_file, 'a') as file:
                file.write(f"Epoch {epoch_idx} val Loss: {curr_val_loss}" + '\n')
            print(f"Epoch {epoch_idx} validation Loss: {curr_val_loss}")
            with open(val_distance_error, 'a') as file:
                file.write(f"Epoch {epoch_idx} val distance error: {distance_mean_error}, {distance_median_error}" + '\n')
            print(f"Epoch {epoch_idx} validation distance mean error: {distance_mean_error}")
            print(f"Epoch {epoch_idx} validation distance median error: {distance_median_error}")

            if distance_mean_error < best_mean_distance_error:
                if not os.path.exists("checkpoint"):
                    os.mkdir("checkpoint")
                model_path = "checkpoint/CVLocationTrans_RobotCar_SAM.pth"
                save_checkpoint(
                    {"state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()
                     },
                    model_path
                )
                best_mean_distance_error = distance_mean_error
                print(f"Model saved at distance error: {best_mean_distance_error}")

if __name__ == '__main__':
    # RobotCar SAM training
    main()
