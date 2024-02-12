# Cross-Attention Between Satellite and Ground Views for Enhanced Fine-Grained Robot Geo-Localization [WACV'2024]



[[`Paper`](https://openaccess.thecvf.com/content/WACV2024/html/Yuan_Cross-Attention_Between_Satellite_and_Ground_Views_for_Enhanced_Fine-Grained_Robot_WACV_2024_paper.html)] 



![](README_data/poster.png)



### Paper Abstract
<p align="justify">
Cross-view image geo-localization aims to determine the locations of outdoor robots by mapping current street-view images with GPS-tagged satellite image patches. Recent works have attained a remarkable level of accuracy in identifying which satellite patches the robot is in, where the location of the central pixel within the matched satellite patch is used as the robot coarse location estimation. This work focuses on robot fine-grained localization within a known satellite patch. Existing fine-grain localization work utilizes correlation operation to obtain similarity between satellite image local descriptors and street-view global descriptors. The correlation operation based on liner matching simplifies the interaction process between two views, leading to a large distance error and affecting model generalization. To address this issue, we devise a cross-view feature fusion network with self-attention and cross-attention layers to replace correlation operation. Additionally, we combine classification and regression prediction to further decrease location distance error. Experiments show that our novel network architecture outperforms the state-of-the-art, exhibiting better generalization capabilities in unseen areas. Specifically, our method reduces the median localization distance error by 43% and 50% respectively in the same area and unseen areas on the VIGOR benchmark.
</p>



