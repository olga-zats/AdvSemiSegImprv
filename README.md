# AdvSemiSegImprv
Computer Vision Laboratory Work: Adversarial Semi-supervised Semantic Segmentation.
This repository is the pytorch implementation of 5 experiments for improving the model, introduced in the following paper:

[Adversarial Learning for Semi-supervised segmentation](https://arxiv.org/abs/1802.07934)
Wei-Chih Hung, Yi-Hsuan Tsai,Yan-Ting Liou, Yen-Yu Lin, and Ming-Hsuan Yang 
Proceedings of the British Machine Vision Conference (BMVC), 2018.

Code is heavily borrowed from [here](https://github.com/hfslyc/AdvSemiSeg).

Contact: Olga Zatsarynna (s6olzats@uni-bonn.de)


## Prerequisites
  * CUDA/CUDNN
  * pytorch >= 0.2 (only 0.4 is supported for evaluation)
  * python-opencv >= 3.4.0  (3.3 will cause extra GPU memory on multithread data loader)


## Installation 
  * Clone this repository:
  ```
  git clone https://github.com/olga-zats/AdvSemiSegImprv.git
  ```
  
  * Place VOC2012 dataset in `AdvSemiSeg/dataset/VOC2012`. For training one needs augmented labels [(Download)](http://vllab1.ucmerced.edu/~whung/adv-semi-seg/SegmentationClassAug.zip). The folder structure should be like:
  ```
  AdvSemiSegImprv/dataset/VOC2012/JPEGImages
  AdvSemiSegImprv/dataset/VOC2012/SegmentationClassAug
  ```
  
  ## Training and Evaluation on VOC2012
  ### Experiment 1
  The first experiments consists in training the original model with different combinations of segmentator's loss function terms. Overall, we tried out 5 different combinations:
  
 1. Baseline (only cross-entropy loss for the labeled data)
 2. Baseline + Adversarial loss for **unlabeled** data 
 3. Baseline + Adversarial loss for both types of data
 4. Baseline + Adversarial loss + Semi-supervised loss
 
 To run the training for the desired experiment and evaluate it on the test data, run the following script, where **num - the number in name of the script** corresponds to the combination of loss terms you want to use for training (numbering as above):
  
  ```
  bash exp1/exp{num}.sh
  ```
  Evaluation results and trained model are going to be saved to directories `exp1/exp{num}` and `snapshots/exp1/exp{num}` respectively. To change, set arguments **--result-dir** and **--save-dir** in the script to the desired desination.
 
 
  ### Experiment 2
  The second experiment consists in removing the discriminator from the model and train segmentator with cross-entropy loss with ground-truth labels for labeled data and pseudo-ground-truth labels for unlabeled data. Cross entropy loss for unlabeled data is introduced into training after 5000 iterations of training using only labled data. To constuct pseudo-ground-truth labels, argmax over predicted label maps is taken.
  
  To run training and evaluation, run the following script:
  ```
  bash exp2/no_discr.sh
  ```
  Evaluation results and trained model are going to be saved to the directories `exp2/no_discr` and `snapshots/exp2/no_discr`
  respectively.
 
 
  ### Experiment 3
  #### Experiment 3.1
  The first part of the third experiment consist in training the model, where discriminator as input recieves a map, that has been average-pooled with a kernel of the predefined size. We tried out 2 kernel sizes: 107 and 321 (amounts to global average pooling).
  
  To run trainng and evaluation, run the following script:
  ``` 
  bash exp3/ave_pool.sh
  ```
  Kernel sizes, used for average pooling, are controlled by the input argument **--kernel-size**. By default, training is carried out for 2 models, with kernel sizes 107 and 321. To change thh default training procedure, please edit the script mentioned above. 
  Evaluation results and trained models are going to be saved to the directories `./exp3/ave_pool` and `./snapshots/exp3/ave_pool` respectively.
  
 #### Experiment 3.2
 The second part of the third experiment consist in training the model, where discriminator as input receives original label probability map concatenated with the global averaged pooled map, copied as many times, as it is needed to match the dimension of the original map.
 
 To run training and evaluation, run the following script
  ``` 
  bash exp3/gl_pyramid.sh
  ```
 Evaluation results and trained model are going to be saved do the directories `./exp3/gl_pyramid` and `./snapshots/exp3/gl_pyramid` respectively.
 
 
 ### Experiment 4
 The fourth experiment consisted in feeding discriminator with modified ground-truth maps instead of their original version. The experiment consisted of two variants: smoothing ground-truth with predictions and introducing an additional discriminator.
 
 #### Experiment 4.1
 In the first variant, the experiment consisted in smoothing the ground-truth maps for the unlabeled data with predictions, made on this data by the segmentation network in the following way:
 $ \alpha \cdot GT + (1 - \alpha) \cdot PRED $
 $ \alpha \in (0.5, 1] $
 We tried out 3 $\alpha$s: 0.51, 0.7 and 0.9.
 
 To run training and evaluation, run the following script:
 ```
 bash exp4/smooth_gt.sh
 ```
 
 Alphas, that are used for smoothing the ground-truth maps, are controlled by the input argument **-alpha**. By default, training is carried out for 3 models, with alphas equal to 0.51, 0.7 and 0.9. To change the default training procedure, please edit the script mentioned above.
 Evaluation results and trained models are going to be saved to the directories `./exp4/smooth_gt` and `./snapshots/exp4/smooth_gt` respectively.
 
 
 #### Experiment 4.2
 In the second variant, the experiment consisted training two separate discriminators: for computing adversarial losses for labeled and unlabeled data. For the first 5000 iterations, both discriminators are trained as in the original model, while for the remaining iterations discriminators are trained as follows:
   * First discriminator is trained using:
       * Ground-truth maps as ground-truth 
       * Predictions on **labeled** data as predictions
   
   This discriminator is used to compute adversarial loss of the segmentation network on the labeled data.
   
   * Second discriminator is trained using:
       * Predictions on **labeled data** as ground-truth 
       * Predictions on **unlabeled** data as predictions 
   
   This discriminator is used to compute adversarial loss of the segmentation network on the unlabeled data.
   
   To run training and evaluation, run the following script:
   ```
   bash exp4/two_discr.sh
   ```
   Evaluation results and trained models are going to be saved to the directories `./exp4/two_discr` and `./snapshots/exp4/two_discr` respectively.
   
   
   
   ### Experiment 5
   The fifth experiment consisted in separately training a classifier network, in addition to the original network. The trained classifier was then used at test time to remove segmentation predictions that are unlikely according to the classification scores. 
   
   To train the classifier and evaluate model's performance with it, run the following script:
   ``` 
   bash exp5/classifier.sh
   ```
   **Warning!**
   
   To be able to evaluate performance of the model, you need to run the final variant of the first experiment.
   Evaluation results and trained model is going to be saved to the directories `.exp5/classifier` and `./snapshots/exp5/classifier`
