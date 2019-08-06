---
layout: post
title: "Region-Based CNN RoadMap"
date: 2018-02-04 00:00:00 +0800
description: an overview of Region-Based CNN # Add post description (optional)
img:  detection.jpeg # Add image post (optional)
---
 
## Overview
> CVPR 14 [ Rich feature hierarchies for accurate object detection and semantic segmentation] introduce cnn to object detection => RCNN

> [Fast R-CNN ]
share feature + joint learn classification & regression

> [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks]
introduces shared-FCN RPN to accelerate region proposals

> [Mask R-CNN] add segmentation together with object detection in a multi-task style

|	Model 			|  VOC 2002 	|  VOC 2010 	| VOC 2012 	|  Train Speed | Test Speed   |  
| ---------------:	| -----------:  |--------------:|----------:| ----------:  | ----------:  |
|	rcnn			|	58.5%		|	53.7%		|	53.3%	|     × 1      |     × 1      |
|	fast rcnn		|	70.0%		|	68.8%		|	68.4%	|     × 9      |     × 213    |    
|	faster rcnn		|	73.2%		|	NULL    	|	70.4%	|	   NULL    |    × 2130 	  |
|	mask rcnn		|	73.2%		|	NULL    	|	70.4%	|	   NULL    |    × 1065 	  |


## RCNN
### Pipeline
![rcnn pipeline](../assets/img/rcnn_overview/rcnn-pipeline.png)
![rcnn pipeline](../assets/img/rcnn_overview/rcnn-pipeline2.png)
Selective Search => ROIs => warp to fixed size => feed into CNN for features => feed features into SVM for classification + feed features into a regressor to tune bbox (increase mAP by 3-4 %).

### Region Proposals
- author argues that the *RF of high-level layer in CNN is too large*, making it hard for CNN detector (can be viewed as sliding-window) to propose regions of various sizes(especially small ones), thus they apply SS. (!!LATER PROVED WRONG!!)
- after get the region, it is extended by p (16) pixels for more context information
- to get a fixed-length feature map of CNN, rois are resized into fixed sizes before feed them into CNN.

### Inference & Training
#### Inference(Test time)
ss => ~2000 proposals => cnn + svm => 2000 proposals, each with K class scores => class-wise NMS

#### Training
#### fine-tuning cnn
- Use all warped region proposals as training samples, with [label = i if iou with truth > 0.5 else 0]

#### train svm classifiers
- Use cnn features as input, with [label = i if iou with truth > 0.3 else 0]. (tiny changes in 0.3 can cause grate perfomance drop!!!)
- use hard negative mining

### Insights
- the author uses features from different layers in CNN. BEEORE fine-tuning, pool_5 achieves almost the same accuracy with fc_6 and fc_7, indicating **CONV LAYERS EXTRACT GERNERAL INFORMATIONS MOSTLY. ** AFTER fine-tuning, fc_7 is 8.0 % better, indicating **DOMAIN-SPECIFIC KNOWLEDGE IS LEARNT BY FC.**

## Fast RCNN
### Pipeline
![fastrcnn pipeline](../assets/img/rcnn_overview/fastrcnn-pipeline.png)
![fastrcnn pipeline](../assets/img/rcnn_overview/frcnn-pipeline2.png)

### Problems of RCNN
- training is multi-stage : ss don't need training(obviously), CNN fine-tune and SVM classifier has to be separately trained (slow & need store gigas of features) => Use FC tacked on CNN to jointly learn the network.
- object detection is slow : region proposals have large RF, thus they actually perceive the whole image. However, the computation is repeated as feeding them separately into CNN. => **share feature** : use CNN to calculate whole feature map, and crop corresponding ROI on it + use **SVD** to accelerate FC layer.

### ROI Pooling
A special case of spcial pyramid pooling used in SPP-Net. It divides the h × w RoI window into an H × W grid of sub-windows of approximate size h/H × w/W and then max-pooling the values in each sub-window into the corresponding output grid cell. This saves **image warp** step in rcnn.
#### Backwards Propagation of ROI Pooling
 $$ \frac{\partial L}{\partial x_i} = \sum_r \sum_j [i=i^{*}(r,j)]\frac{\partial L}{\partial y_{r,j}} $$ 

 $$ y_{r,j} $$ refers to the $$ j^{th} $$ sampling point from $$ r^{th} $$ region.

 $$ [i=i^{*}(r,j)] $$ indicates whether $$ y_{r,j} $$ comes from $$ x_i $$ from original roi.

### Joint Learning
 $$ L(p,u,t^u,v) = L_{cls}(p, u) + \lambda [u \ge 1] L_{loc}(t^u, v), \lambda =1 $$ 

 $$ u $$ is the truth label, while u=0 is background, whose regression loss don't count.

 $$ L_{loc} $$ is the smooth L1 function, which avoids gradient explosion of L2.
#### Details
- fine tuning does't include all layers, the conv_1 layers learns most general information and should be freezed for better performance (acc./speed.)
- each batch is composed of 128 rois from 2 images, each with 16 pos and 32 neg rois.
- hard negative : $$ iou \in [0.1, 0.5) $$ is labeled negative
- data augmentation :  images are horizontally flipped with probability 0.5.

### Insights
#### Multi-task / Two Stage?
Training with multi-task and remove bbox regression when testing shows better performance than merely training on $$ L_{cls} $$ ; Train separately also performs worse than multi-task.
#### Scale Invariance? 
Author argues that  deep ConvNets are adept at directly learning scale invariance. He sacles the inputs into 5 sclaes for training and testing, and mAP only imporves 0.5%. [??? but not clear test strategy given ???]
#### More training data / proposals?? 
Yes, more data more accuracy. But More **low-quality proposals** hurt mAP! 
#### SVM / Softmax?
SVM slightly better (+ 0.1 - 0.8)


## Faster RCNN
### Pipeline
![fasterrcnn](../assets/img/rcnn_overview/fasterrcnn-pipeline.png)

### RPN
- slide a n×n(n=3) window over feature map, for each window in feature-map propose K anchors (K=3*3)
- each anchor is centered at the sliding window.
- each anchor has TWO scores , indicating its objectness and 4 regression values, with 6 values in total.
- By experiment: RPN's proposals' RECALL curve is better than ss!, IT has accurate top rank proposals.
- ??? Authors claim this scale and aspect ratio method is translation invariant, while k-means used in Multi-Box is. ???
![rpn](../assets/img/rcnn_overview/rpn.png)

#### Loss for RPN
- $$ L(\{p_i\},\{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_
i p_i^* L_{reg}(t_i, t_i^*) $$ 
- $$ N_{cls} $$ is 256, mini-batchsize, $$ \lambda $$ is 10
- $$ N_{reg} $$ is ~2400, the size of the feature map.

**Two types of Positive Anchors**
-  the anchor[s] with the highest IoU overlap with a ground-truth box. (not necessarily > 0.7)
-  an anchor that has an IoU overlap higher than 0.7 with any ground-truth box

**One type of Negative Anchors**
- ious < 0.3

#### bbox regression in RPN
- different from the one in RCNN : uses fix-size window's feature as input rather than features pooled from carious sizes. use k bounding box regressor for every aspect and ratio combination, instead of sharing weights. 
- $$ t_x = (x - x_a)/w_a $$ , $$ t_x^* = (x^* - x_a) / w_a $$ , $$ x_a $$ , $$ w_a $$ are center x and width of anchor box
- $$ t_w = log(w / w_a), t_w^* = log(w^* / w_a) $$ 
- use $$ t_i $$ (relative style) to predict so that $$ t_i $$ is in a reasonable range, rather than aboslute size of anchors.

#### share net between RPN and Fast RCNN 's CNN
1.  initialize with an ImageNet pre-trained model and fine-tuned for the region proposal task.
2.  train a separate detection network by Fast R-CNN using the proposals generated by the step-1 RPN
3.  use the detector network to initialize RPN training, but only fine-tune the layers unique to RPN. 
4.  keeping the shared conv layers fixed, we fine-tune the fc layers of the Fast R-CNN. 

### Insights
- ACC May increase if samller stride is used in RPN. Powerful Net for RPN can improve mAP.
- anchor boxes that are larger than the underlying receptive field can be used.(one may still roughly infer the extent of an object if only the middle of the object is visible. ) 
- Without net sharing, mAP drops 1 %, the power of more tasks?
- Two stage is better than one stage detector.


## Mask RCNN
#### Pipeline
adding a branch for predicting an object mask inparallel with the existing branch for bounding box recognition.

