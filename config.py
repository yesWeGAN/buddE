import albumentations as A
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    """Improve config accessibility."""

    annotation_path = "/home/frank/datasets/VOC2012/JSONAnnotation/annotation.json"
    label_path = "/home/frank/datasets/VOC2012/JSONAnnotation/labels.json"
    target_image_size = 384
    patch_size = 16
    batch_size = 32
    validation_batch_size = 256
    epochs = 25
    lr = 0.0001
    dropout = 0.05
    num_workers = 4
    weight_decay = 0.0001
    max_seq_len = 300
    pretrained_encoder = "facebook/deit-base-distilled-patch16-384"
    encoder_bottleneck = 256
    num_decoder_layers = 6
    decoder_layer_dim = 256
    num_heads = 2
    logging = False
    device = device
    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(
                width=target_image_size, height=target_image_size, p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.25, label_fields=["class_labels"]
        ),
    )

    assert (
        int(pretrained_encoder.split("-")[-1]) == target_image_size
    ), "Pretrained encoder and image size do not match."
    assert (
        encoder_bottleneck == decoder_layer_dim
    ), "Encoder bottleneck must match decoder layer dimension."
