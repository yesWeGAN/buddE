# buddE
My own implementation of object detection using DeiT Encoder and trained Decoder (as proposed in Pix2Seq).
This version does not predict pixel values as text. Instead, bounding boxes are reshaped to match the image patches in the Encoder.
The prediction task is then defined as predicting the patch (token) of the upper left and lower right corner of the bbox.


Based on: 
@software{Shariatnia_Pix2Seq-pytorch_2022,
author = {Shariatnia, M. Moein},
doi = {10.5281/zenodo.7010778},
month = {8},
title = {{Pix2Seq-pytorch}},
version = {1.0.0},
year = {2022}
}