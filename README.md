# buddE
My own implementation of object detection using DeiT Encoder and trained Decoder (as proposed in Pix2Seq).
Train / Val set is PASCAL VOC.

This version does not predict pixel values as text. Instead, bounding boxes are reshaped to match the image patches in the Encoder. Each token represents a location in the image, there are:

num_tokens = (image_size / patch_size)**2

The prediction task is then defined as predicting the patch (token) of the upper left and lower right corner of the bbox.

## 30.11.2023
- added predict function for token generation
- utilizes recycling of encoded x (image) for fast inference
- here are the metrics for the validation set (10% of PASCAL VOC), mAP@IoU.50 0.34:


 'map': tensor(0.1914), <br>
 'map_50': tensor(0.3401),<br>
 'map_75': tensor(0.1849),<br>
 'map_large': tensor(0.2639),<br>
 'map_medium': tensor(0.0071),<br>
 'map_small': tensor(0.),<br>
 'mar_1': tensor(0.2492),<br>
 'mar_10': tensor(0.2685),<br>
 'mar_100': tensor(0.2685),<br>

Poor performance in small bboxes probably due to the patchwise tokenization (minimum bbox size is 4 patches - 32*32 pixels).<br>
Note: mAP calculated using also patchwise bounding boxes as ground truth (x / y dimensions are multiples of 16)

Next steps:
- calculate mAP with original ground truth


## 29.11.2023
FINALLY! Found the bug that kept my mAP metric down (softmax/argmax along wrong dim).
- mAP negatively correlates with val_loss now
- mAP@IoU.50 0.24

Next Steps:
- add predict method for unknown images
- scale up with MS COCO

<img src="./plots/map50.png" alt="mAP@IoU.50 after bugfix" title="mAP@IoU.50 after bugfix">

## 28.11.2023
- added augmentations for training
- overfitting delayed 1-2 epochs

<img src="./plots/augmentations.png" alt="Val loss, with vs. w/o augmentations" title="Val loss, with vs. w/o augmentations">

## 26.11.2023
- adjusted positional encodings: scaled by math.sqrt(embedding_dim)
- toml config replaced with Config class
- closed two issues

### Results:
- Overfitting relatively early (7-9 epochs), need albumentation transforms
- mAP still relatively low, should include proper prediction pipeline

<img src="./plots/train_loss.png" alt="Train loss, working pipeline (w/o transforms), encoder dims 224 vs 384" title="Train loss, working pipeline (w/o transforms), encoder dims 224 vs 384">
<img src="./plots/val_loss.png" alt="Val loss, working pipeline (w/o transforms), encoder dims 224 vs 384" title="Val loss, working pipeline (w/o transforms), encoder dims 224 vs 384">
<img src="./plots/mAP.png" alt="mAP, working pipeline (w/o transforms), encoder dims 224 vs 384" title="mAP, working pipeline (w/o transforms), encoder dims 224 vs 384">


## 23.11.2023
- re-train with warmup shows no signs of overfitting for 20 epochs, great
- investigation into poor mAP reveals some images have 99 predictions (max possible given max_seq_len=300)
- introduce probs for validation: torch.nn.functional.softmax passed to metric


## 21.11.2023 
- randomized object order in tokenizer does not improve map
- added learning rate warmup.
- added resume_training() function to resume training from checkpoint.
- added torchmetric: metric.reset() to avoid memory bleed (training stopped by Linux OOM-Killer after ~10 epochs.)
- resumed training, at epoch 16, with lr warmup (oops), results indicate overfiting.

#### Next steps:
- top-k sampling from predicted tokens? we get too many class indices, this should be learnable.
- re-run training from start with lr_warmup.


## 19.11.2023
Training pipeline works. mAP score still below expectation. Definitely overfitting.

#### Next steps: 
Investigate into randomizing object order in tokenizer. 


<img src="./wb_train_loss.png" alt="Alt text" title="First train run.">

Based on: 
@software{Shariatnia_Pix2Seq-pytorch_2022,
author = {Shariatnia, M. Moein},
doi = {10.5281/zenodo.7010778},
month = {8},
title = {{Pix2Seq-pytorch}},
version = {1.0.0},
year = {2022}
}

And the awesome tutorial (link below), thank you so much Moein Shariatnia:
https://pub.towardsai.net/easy-object-detection-with-transformers-simple-implementation-of-pix2seq-model-in-pytorch-fde3e7162ce7