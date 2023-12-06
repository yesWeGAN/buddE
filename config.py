import albumentations as A
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    """Improve config accessibility."""

    #annotation_path = "/home/frank/datasets/VOC2012/JSONAnnotation/annotation.json"
    #label_path = "/home/frank/datasets/VOC2012/JSONAnnotation/labels.json"
    label_path = "/home/frank/datasets/mscoco/annotations/budde_annotation_labels.json"
    annotation_path = "/home/frank/datasets/mscoco/annotations/budde_annotation_train2017.json"
    target_image_size = 224
    patch_size = 16
    batch_size = 96 if target_image_size==224 else 32
    validation_batch_size = 256
    epochs = 30
    lr = 0.00005
    dropout = 0.05
    num_workers = 4
    weight_decay = 0.0001
    max_seq_len = 300
    pretrained_encoder = f"facebook/deit-base-distilled-patch16-{target_image_size}"
    encoder_bottleneck = 256
    num_decoder_layers = 6
    decoder_layer_dim = 256
    num_heads = 2
    logging = True
    device = device
    train_transforms = A.Compose(
        [
            A.RandomResizedCrop(
                width=target_image_size, height=target_image_size, p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
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
