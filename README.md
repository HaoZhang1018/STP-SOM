# STP-SOM IJCV
The code of STP-SOM: Scale-Transfer Learning for Pansharpening via Estimating Spectral Observation Model
````
@article{zhang2023stp,
  title={STP-SOM: Scale-Transfer Learning for Pansharpening via Estimating Spectral Observation Model},
  author={Zhang, Hao and Ma, Jiayi},
  journal={International Journal of Computer Vision},
  volume={131},
  number={12},
  pages={3226--3251},
  year={2023}
}
````

#### running environment :<br>
python=2.7, tensorflow-gpu=1.9.0.

#### Prepare data :<br>
Please first convert the satellite captured MS and PAN images to 8 bit, and put the training data in ".\Dataset\Training\MS\", ".\Dataset\Training\PAN\" following the provided examples.


you should construct the training data according to the Wald protocol, and put the training data in "\data\Train_data\......" following the provided examples.

#### To train :<br>
The training process is divided into two stages. In the first stage, please run "CUDA_VISIBLE_DEVICES=0 python train_T.py" to make TNet learn the gradient transformation prior. In the second stage, run "CUDA_VISIBLE_DEVICES=0 python train_P.py" to learn fusing multi-spectral and panchromatic images, in which the trained TNet is used to constrain the preservation of the spatial structures in pansharpening.


#### To test :<br>
Put test images in the "\data\Test_data\......" folders, and then run "CUDA_VISIBLE_DEVICES=0 python test.py" to test the trained P_model.
You can also directly use the trained P_model we provide (Quickbird &  GF-2).
