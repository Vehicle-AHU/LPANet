# Large Language Model Guided Progressive Feature Alignment for Multimodal UAV Object Detection (LPANet)

**CM3AE: A Unified RGB Frame and Event-Voxel/-Frame Pre-training Framework**, 
Wentao Wu, Xiao Wang, Chenglong Li, Bo Jiang, Jin Tang, Bin Luo, Qi Liu [[arXiv](https://arxiv.org/pdf/2503.06948)] 


## News 

* [02-Jun-2026] LPANet is accepted by TIP




## Abstract 
Existing multimodal UAV object detection methods often overlook the impact of semantic gaps between modalities, which makes it difficult to achieve accurate semantic and spatial alignments and ultimately limits detection performance. To address this problem, we propose a Large Language Model (LLM) guided Progressive feature Alignment Network called LPANet, which leverages the semantic features extracted from a large language model to guide the progressive semantic and spatial alignment between modalities for multimodal UAV object detection. To employ the powerful semantic representation of LLM, we generate the fine-grained text descriptions of each object category by ChatGPT and then extract the semantic features using the large language model MPNet, providing high-level semantic priors to guide multimodal alignment. Based on the semantic features, we guide the semantic and spatial alignments in a progressive manner as follows. First, we design the Semantic Alignment Module (SAM) to pull the semantic features and multimodal visual features of each object closer, alleviating the semantic differences of objects between modalities. Second, we design the Explicit Spatial Alignment Module (ESM) by integrating the semantic relations into the estimation of feature-level offsets, alleviating the coarse spatial misalignment between modalities. Finally, we design the Implicit Spatial alignment Module (ISM), which leverages the cross-modal correlations to aggregate key features from neighboring regions to achieve implicit spatial alignment. Comprehensive experiments on two public multimodal UAV object detection datasets demonstrate that our approach outperforms state-of-the-art multimodal UAV object detectors.


<p align="center">
  <img width="100%" src="[https://github.com/Vehicle-AHU/LPANet/blob/main/figures/first_image.jpg" alt="firstIMG"/>
</p> 

## Environment Setting 

Configure the environment according to the content of the requirements.txt file.

## Training

```bibtex
#If you pre-training CM3AE using a single GPU, please run.
CUDA_VISIBLE_DEVICES=0 python main.py
#If you pre-training CM3AE using multiple GPUs, please run.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```


## Experimental Results 

Experimental results of our method and other detection algorithms on DroneVehicle object detection datasets.

<p align="center">
  <img width="100%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/result_dronevehicle.png" alt="result"/>
</p> 

Comparative experimental results of oriented object detection with other algorithms on the VEDAI dataset.

<p align="center">
  <img width="100%" src="https://github.com/Vehicle-AHU/LPANet/blob/main/figures/result_vedai.png" alt="result"/>
</p> 

## Visual Results 

Comparison of speed and accuracy on the DroneVehicle.

<p align="center">
  <img width="100%" src=""https://github.com/Vehicle-AHU/LPANet/blob/main/figures/fig-5.jpg" alt="fps"/>
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
