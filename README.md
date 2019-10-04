# AdvSemiSegImprv
Computer Vision Laboratory Work: Adversarial Semi-supervised Semantic Segmentation.
This repository is the pytorch implementation of the 5 experiments aimed at improving the model, that was introduced in the following paper:

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
  The first experiment consists in training the original model with different combinations of segmentator's loss function terms. Overall, we tried out 5 different combinations:
  
 1. Baseline (only cross-entropy loss for the labeled data)
 2. Baseline + Adversarial loss for **unlabeled** data 
 3. Baseline + Adversarial loss for both types of data
 4. Baseline + Adversarial loss + Semi-supervised loss
 
 To perform training for the desired experiment and evaluate it on the test data, run the following script, where **num - the number in name of the script** represents the combination of loss terms you want to use for training (numbering as above):
  
  ```
  bash exp{num}.sh
  ```
  Evaluation results and trained model are going to be saved to directories `exp{num}` and `snapshots/exp{num}` respectively. To change, set arguments **--save-dir** and **--snapshot-dir** in the script to the desired desinations. Also, initially all models in this and further experiments are trained with one random seed equal to 0. To try out other seeds or change the initial one, change **--random-seed** input argument in the script responsible for training the model.
 
 
  ### Experiment 2
  The second experiment consists in removing the discriminator network from the model and training segmentator using cross-entropy loss with ground-truth maps for labeled data and pseudo-ground-truth maps for unlabeled data. Cross-entropy loss for the unlabeled data is introduced after 5000 iterations of training using only labled data. To construct pseudo-ground-truth maps for the unlabeled data, argmax over predicted maps is taken.
  
  To perform training and evaluation, run the following script:
  ```
  bash no_discr.sh
  ```
  Evaluation results and trained model are going to be saved to the directories `no_discr` and `snapshots/no_discr`
  respectively.
 
 
  ### Experiment 3
  #### Experiment 3.1
  The first part of the third experiment consist in training the model, where discriminator recieves as input a label probability map, that has been previously average-pooled with a kernel of the predefined size. We tried out 2 kernel sizes: 107 and 321 (amounts to global average pooling).
  
  To perform training and evaluation, run the following script:
  ``` 
  bash ave_pool.sh
  ```
  Kernel sizes, used for average pooling, are controlled by the input argument **--kernel-size**. By default, training is carried out for 2 models, with kernel sizes 107 and 321. To change the default training procedure, please edit the script mentioned above. 
  Evaluation results and trained models are going to be saved to the directories `ave_pool` and `snapshots/ave_pool` respectively.
  
 #### Experiment 3.2
 The second part of the third experiment consist in training the model, where discriminator receives as input the original label probability map concatenated with the global averaged pooled map, copied as many times, as it is needed to match the dimension of the original map.
 
 To perform training and evaluation, run the following script
  ``` 
  bash gl_pyramid.sh
  ```
 Evaluation results and trained model are going to be saved do the directories `gl_pyramid` and `snapshots/gl_pyramid` respectively.
 
 
 ### Experiment 4
 The fourth experiment consisted in feeding discriminator with the modified ground-truth maps instead of their original version. The experiment consisted of two variants: smoothing ground-truth with predictions and introducing an additional discriminator.
 
 #### Experiment 4.1
 In the first variant, the experiment consisted in smoothing the ground-truth maps with predictions made by the segmentation network using corresonding images in the following way:
 
 ```
 $ alpha * GT + (1 - alpha) * PRED $
 $ alpha in (0.5, 1] $
 ```
 
 We tried out 3 alphas: 0.51, 0.7 and 0.9.
 
 To perform training and evaluation, run the following script:
 ```
 bash smooth_gt.sh
 ```
 
 Alphas, that are used for smoothing the ground-truth maps, are controlled by the input argument **--interp-alpha**. By default, training is carried out for 3 models, with alphas equal to 0.51, 0.7 and 0.9. To change the default training procedure, please edit the script mentioned above.
 Evaluation results and trained models are going to be saved to the directories `smooth_gt` and `snapshots/smooth_gt` respectively.
 
 
 #### Experiment 4.2
 In the second variant, the experiment consisted training two separate discriminators that are used for computing adversarial losses for labeled and unlabeled data. For the first 5000 iterations, both discriminators are trained as in the original model, while for the remaining iterations discriminators are trained as follows:
   * First discriminator is trained using:
       * Ground-truth maps as ground-truth 
       * Predictions on **labeled** data as predictions
   
   This discriminator is used to compute adversarial loss of the segmentation network on the labeled data.
   
   * Second discriminator is trained using:
       * Predictions on **labeled data** as ground-truth 
       * Predictions on **unlabeled** data as predictions 
   
   This discriminator is used to compute adversarial loss of the segmentation network on the unlabeled data.
   
   To perform training and evaluation, run the following script:
   ```
   bash two_discr.sh
   ```
   Evaluation results and trained models are going to be saved to the directories `two_discr` and `snapshots/two_discr` respectively.
   
   
   
   ### Experiment 5
   The fifth experiment consisted in separately training a classifier network, in addition to the original network. The trained classifier was then used at test time to remove segmentation predictions that are unlikely according to the classification scores. 
   
   To train the classifier and evaluate model's performance with it, run the following script:
   ``` 
   bash classifier.sh
   ```
   **Warning!**
   
   To be able to evaluate performance of the model with the classifier, you need to have the original model trained in advance. To do that, you need to run the final (5-th) variant of the first experiment.
   Evaluation results and trained model are going to be saved to the directories `classifier` and `snapshots/classifier`
