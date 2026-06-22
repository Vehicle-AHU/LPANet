# Large Language Model Guided Progressive Feature Alignment for Multimodal UAV Object Detection (LPANet)

**CM3AE: A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework**, 
Wentao Wu, Chenglong Li*, Xiao Wang, Bin Luo[[arXiv](https://arxiv.org/pdf/2503.06948),[IEEE](https://ieeexplore.ieee.org/abstract/document/11568942)] 


## News 

* [02-Jun-2026] LPANet is accepted by TIP




## Abstract 
Existing multimodal UAV object detection methods often overlook the impact of semantic gaps between modalities, which makes it difficult to achieve accurate semantic and spatial alignments and ultimately limits detection performance. To address this problem, we propose a Large Language Model (LLM) guided Progressive feature Alignment Network called LPANet, which leverages the semantic features extracted from a large language model to guide the progressive semantic and spatial alignment between modalities for multimodal UAV object detection. To employ the powerful semantic representation of LLM, we generate the fine-grained text descriptions of each object category by ChatGPT and then extract the semantic features using the large language model MPNet, providing high-level semantic priors to guide multimodal alignment. Based on the semantic features, we guide the semantic and spatial alignments in a progressive manner as follows. First, we design the Semantic Alignment Module (SAM) to pull the semantic features and multimodal visual features of each object closer, alleviating the semantic differences of objects between modalities. Second, we design the Explicit Spatial Alignment Module (ESM) by integrating the semantic relations into the estimation of feature-level offsets, alleviating the coarse spatial misalignment between modalities. Finally, we design the Implicit Spatial alignment Module (ISM), which leverages the cross-modal correlations to aggregate key features from neighboring regions to achieve implicit spatial alignment. Comprehensive experiments on two public multimodal UAV object detection datasets demonstrate that our approach outperforms state-of-the-art multimodal UAV object detectors.


<p align="center">
  <img width="50%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/first_image.jpg" alt="firstIMG"/>
</p> 

## Environment Setting 

Configure the environment according to the content of the requirements.txt file.

## Training

```bibtex
#Training
CUDA_VISIBLE_DEVICES=0 python train.py
#Test
python valtest.py --save-json --name 'obb_demo' --weight '/runs/train/exp/weights/best.pt'
python tools/TestJson2VocClassTxt.py --json_path 'runs/val/obb_demo/best_obb_predictions.json' --save_path 'runs/val/obb_demo/obb_predictions_Txt'
python DOTA_devkit-master/dota_evaluation_task1.py 
```


## Experimental Results 

Experimental results of our method and other detection algorithms on DroneVehicle object detection datasets.

<p align="center">
  <img width="100%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/result_dronvehicle.png" alt="result_1"/>
</p> 

Comparative experimental results of oriented object detection with other algorithms on the VEDAI dataset.

<p align="center">
  <img width="100%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/result_vedai.png" alt="result_2"/>
</p> 

## Visual Results 

Comparison of speed and accuracy on the DroneVehicle.

<p align="center">
  <img width="100%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/fig-5.png" alt="fps_image"/>
</p> 

Visualization of detection results on the DroneVehicle dataset, with different color boxes representing different categories.

<p align="center">
  <img width="100%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/detection_result.jpg" alt="detection_result_visualization"/>
</p> 

Visualization of similarity response maps between text descriptions of various categories and multimodal visual features.

<p align="center">
  <img width="100%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/vt.jpg" alt="similarity_response_maps_visualization"/>
</p> 

## Acknowledgement 
[[CALNet](https://github.com/hexiao0275/CALNet-Dronevehicle)] 

## Citation 

If you find this work helps your research, please cite the following paper and give us a star. 
```bibtex
@article{wu2026large,
  title={Large language model guided progressive feature alignment for multimodal UAV object detection},
  author={Wu, Wentao and Li, Chenglong and Wang, Xiao and Luo, Bin},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  publisher={IEEE}
}
```


if you have any problems with this work, please leave an issue. 
