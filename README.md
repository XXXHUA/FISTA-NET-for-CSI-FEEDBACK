# FISTA-Net for CSI Feedback
## Introduction
 This is the Tensorflow implementation of the paper: Jianhua Guo, Lei Wang, Feng Li, and Jiang Xue, "**CSI Feedback with Model-Driven Deep Learning of Massive MIMO Systems**" .

## Requirements
* Python 3.6
* Tensorflow 1.15.0
* tflearn 0.3.2
* Numpy

## Data Preparation
In OFDM system, we use the channel state information (CSI) matirx is generated by the [COST2100 channel model](https://ieeexplore.ieee.org/document/6393523). Specifically, we use the pre-processed CSI dataset provided by Chao-Kai Wen, Wan-Ting Shih, and Shi Jin in the [repository](https://github.com/sydney222/Python_CsiNet#python-code-for-deep-learning-for-massive-mimo-csi-feedback) and you can download the dataset from [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing) and put it in ```DATA/``` folder.

## Train FISTA-Net
Train the FISTA-Net from the scratch in different CRs and scenarios with
```
python FISTA_Net.py
```


## Results and Reproduction
The CSI reconstruction results by FISTA-Net are presented as follows. We also provides the pre-trained model to reproduce this results with
```
python FISTA_Net_test.py
```

|  CR  | Methods | Indoor | Outdoor | Trainable Params |   MACC  |         |
|:----:| :----:  | :----: | :----:  |      :----:      |  :----: |  :----: |
|      |         |  NMSE  |   NMSE  |                  | Encoder | Decoder | 
| 1/4  |  CsiNet | -17.36 |  -8.75  |       2.10M      |  1.09M  |**4.39M**| 
|      | CsiNet+ | -27.37 |  -12.4  |       2.12M      |  1.45M  |  23.26M | 
|      |  FISTA  | -10.46 |  -6.35  |        -         |**1.05M**|  41.94M | 
|      |FISTA-Net|**-36.76**|**-22.4**|   **1.09M**    |**1.05M**|  74.71M | 
| 1/8  |  CsiNet | -12.7  |  -7.61  |       1.05M      |  0.56M  |**3.86M**| 
|      | CsiNet+ | -18.29 |  -8.72  |       1.07M      |  0.93M  |  22.73M | 
|      |  FISTA  | -6.39  |  -2.91  |        -         |**0.52M**|  20.97M | 
|      |FISTA-Net|**-26.5**|**-13.65**|   **0.56M**    |**0.52M**|  53.74M | 
| 1/16 |  CsiNet | -8.65  |  -4.51  |       0.53M      |  0.30M  |**3.60M**| 
|      | CsiNet+ | -14.14 |  -5.73  |       0.55M      |  0.67M  |  22.47M | 
|      |  FISTA  | -3.18  |  -1.15  |        -         |**0.26M**|  10.49M | 
|      |FISTA-Net|**-17.51**|**-7.57**|   **0.30M**    |**0.26M**|  43.26M | 
| 1/32 |  CsiNet | -6.24  |  -2.81  |       0.27M      |  0.17M  |**3.47M**| 
|      | CsiNet+ | -10.43 |  -3.4   |       0.29M      |  0.54M  |  22.34M | 
|      |  FISTA  | -1.11  |  -0.35  |        -         |**0.13M**|  5.24 M | 
|      |FISTA-Net|**-12.01**|**-4.41**|   **0.17M**    |**0.13M**|  38.01M | 
| 1/64 |  CsiNet | -5.84  |  -1.93  |       0.14M      |  0.11M  |  3.40M  | 
|      | CsiNet+ | -5.99  |  -2.22  |       0.16M      |  0.47M  |  22.27M | 
|      |  FISTA  | -0.29  |  -0.05  |        -         |**0.07M**|**2.62M**| 
|      |FISTA-Net|**-8.54**|**-2.60**|   **0.10M**     |**0.07M**|  35.39M | 

## TODO
The results and dataset with low-rank mmWave channel matrix by FISTA-Nets will be add this repository in the future.

## Citation
If you find our paper and code are helpful for your research or work, please cite our paper.
```
CSI Feedback with Model-Driven Deep Learning of Massive MIMO Systems
```

## Acknowledgment
* [Deep learning for massive MIMO CSI feedback](https://github.com/sydney222/Python_CsiNet#python-code-for-deep-learning-for-massive-mimo-csi-feedback)
* [ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing](https://github.com/jianzhangcs/ISTA-Net)



