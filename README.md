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
Please first convert the satellite captured (full-resolution) MS and PAN images to 8 bit, and put the training data in ".\Dataset\Training\MS\", ".\Dataset\Training\PAN\" following the provided examples. The built-in code will automatically follow the Wald protocol to perform data degradation.


#### To train :<br>
The training process is divided into two stages. In the first stage, please run "CUDA_VISIBLE_DEVICES=X python Cycle_Trans_net_train.py" to obtain a good spectral degradation network. In the second stage, run "CUDA_VISIBLE_DEVICES=X python Sharpening_net_train.py" to obtain a well-trained pansharpening network.


#### To test :<br>
Put the satellite captured (full-resolution) MS and PAN images in the testing set in the ".\Dataset\Test\MS\", ".\Dataset\Test\PAN\" folders, and then run "CUDA_VISIBLE_DEVICES=X python evaluate_sharpening.py" to implement pansharpening. The built-in code will implement both full-resolution and reduced-resolution testing simultaneously. We provide a model weight pre-trained on QuickBird satellite, so you can use it directly to play with the test code.
