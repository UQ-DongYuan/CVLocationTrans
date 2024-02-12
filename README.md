# Cross-Attention Between Satellite and Ground Views for Enhanced Fine-Grained Robot Geo-Localization [WACV'2024]



[[`Paper`](https://openaccess.thecvf.com/content/WACV2024/html/Yuan_Cross-Attention_Between_Satellite_and_Ground_Views_for_Enhanced_Fine-Grained_Robot_WACV_2024_paper.html)] 
[[`BibTeX`](#citation-information)]



![](README_data/system.PNG)


### Paper Abstract
<p align="justify">
Cross-view image geo-localization aims to determine the locations of outdoor robots by mapping current street-view images with GPS-tagged satellite image patches. Recent works have attained a remarkable level of accuracy in identifying which satellite patches the robot is in, where the location of the central pixel within the matched satellite patch is used as the robot coarse location estimation. This work focuses on robot fine-grained localization within a known satellite patch. Existing fine-grain localization work utilizes correlation operation to obtain similarity between satellite image local descriptors and street-view global descriptors. The correlation operation based on liner matching simplifies the interaction process between two views, leading to a large distance error and affecting model generalization. To address this issue, we devise a cross-view feature fusion network with self-attention and cross-attention layers to replace correlation operation. Additionally, we combine classification and regression prediction to further decrease location distance error. Experiments show that our novel network architecture outperforms the state-of-the-art, exhibiting better generalization capabilities in unseen areas. Specifically, our method reduces the median localization distance error by 43% and 50% respectively in the same area and unseen areas on the VIGOR benchmark.
</p>

---
### 1. Environment 
<p align="justify">
please follow the requirements to setup your running environment
</p>

- Python >= 3.9
- Pytorch == 1.21.1
- torchvision == 0.13.1
- scikit-learn == 1.2.2
- numpy == 1.21.1
- einops == 0.6.1

### 2. Datasets
**VIGOR Dataset:** <br>
Please follow the guidelines from https://github.com/Jeff-Zilence/VIGOR to download and prepare the dataset. When using `readdata_VIGOR.py` to load images, please modify `self.root` to your dataset storing directory.<br> 
<br>
**Oxford RobotCar Dataset:** <br>
Please contact the authors from https://github.com/tudelft-iv/CrossViewMetricLocalization/tree/main to get the processed Oxford RobotCar ground view images and the corresponding satellite images.
Please download the `Oxford_split` directory for data splitting. When using `readdata_Oxford.py` to load training or testing data, please modify `self.grd_image_root` and `full_satellite_map` to your data storing directory.

### 3. Training and Evaluation
**Model training:** <br>
<br>
Training on VIGOR dataset:<br>
Run `python train_VIGOR.py` or `python train_VIGOR_SAM.py`. <br> 
Change the config parameter: `area` at the beginning of the training file to `same` or `cross` for different training setups. <br>
<br>
Training on Oxford RobotCar dataset: <br>
Run `python train_Oxford.py` or `python train_Oxford_SAM.py`. <br>
<br>
**Model evaluation**<br>

Our trained models can be found at: <br>
<br>

<br>
Evaluation of VIGOR Dataset: <br>
In `test_semi_positive_VIGOR.py`: <br>
1. Change the values of `semi_postive_index` for testing only *positive satellite images* or *positive + semi-positive* satellite images. <br>
2. Change the value of `area` to `same` or `cross` for different test settings. <br>
3. Change `checkpoint_path` to the correct path of the provided pre-trained model weights.
4. Run `python test_semi_positive_VIGOR.py`. <br>

<br>
Evaluation of Oxford RobotCar Dataset: <br>
1. Change the value of `self.test_grd_img_list` in `readdata_Oxford.py` to select one of three different testing traversals.<br>
2. In `test_Oxford.py`, Change `checkpoint_path` to the correct path of the provided pre-trained model weights.<br>
3. Run `test_Oxford.py` <br>

---

### Citation Information
<p align="justify">
If our work is useful to your research, please kindly recognize our contributions by citing our WACV paper:
</p>

```
@InProceedings{Yuan_2024_WACV,
    author    = {Yuan, Dong and Maire, Frederic and Dayoub, Feras},
    title     = {Cross-Attention Between Satellite and Ground Views for Enhanced Fine-Grained Robot Geo-Localization},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {1249-1256}
}
```




