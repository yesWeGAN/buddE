class Config:
    """Improve config accessibility. """     
    annotation_path = "/home/frank/datasets/VOC2012/JSONAnnotation/annotation.json"
    label_path = "/home/frank/datasets/VOC2012/JSONAnnotation/labels.json"
    target_image_size = 224
    patch_size = 16
    batch_size = 96
    epochs = 25
    lr = 0.0001
    num_workers = 2
    weight_decay = 0.0001
    max_seq_len = 300
    pretrained_encoder = "facebook/deit-base-distilled-patch16-224"
    encoder_bottleneck = 256
    num_decoder_layers = 6
    decoder_layer_dim = 256
    num_heads = 2
    logging = True

    assert int(pretrained_encoder.split("-")[-1]) == target_image_size, "Pretrained encoder and image size do not match."
    assert encoder_bottleneck == decoder_layer_dim, "Encoder bottleneck must match decoder layer dimension."