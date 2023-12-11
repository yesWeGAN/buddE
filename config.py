import albumentations as A
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set these paths to where the data is stored
DS_DIR = "/home/frank/datasets"
COCO_DIR = "mscoco/annotations"  # coco subdir
VOC_DIR = "VOC2012/JSONAnnotation"  # voc subdir


class Config:
    """Config to set filepaths and training params. 
    Setting target image size and dataset defines most other params."""

    dataset = "COCO"
    target_image_size = 224
    logging = True

    split_ratio = 0.9 if dataset == "VOC" else None
    train_annotation_path = (
        f"{DS_DIR}/{VOC_DIR}/annotation.json"
        if dataset == "VOC"
        else f"{DS_DIR}/{COCO_DIR}/train2017.json"
    )
    val_annotation_path = (
        train_annotation_path
        if dataset == "VOC"
        else f"{DS_DIR}/{COCO_DIR}/val2017.json"
    )
    label_path = (
        f"{DS_DIR}/{VOC_DIR}/labels.json"
        if dataset == "VOC"
        else f"{DS_DIR}/{COCO_DIR}/labels_80_classes.json"
    )

    patch_size = 16
    batch_size = 80 if target_image_size == 224 else 32
    validation_batch_size = 256
    epochs = 55
    lr = 0.0001
    dropout = 0.1
    num_workers = 4
    weight_decay = 0.0001
    max_seq_len = 300
    pretrained_encoder = f"facebook/deit-base-distilled-patch16-{target_image_size}"
    encoder_bottleneck = 512
    num_decoder_layers = 6
    decoder_layer_dim = 512
    num_heads = 8
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
