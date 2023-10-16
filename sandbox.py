# initialize the dataset

import dataset
import toml
import importlib
importlib.reload(dataset)
from dataset import DatasetODT

config=toml.load("config.toml")
config
config["data"]["annotation_path"]
ds = DatasetODT(annotation_path=config["data"]["annotation_path"])
img, anno = ds.__getitem__(0)
for k in range(20):
    img, anno = ds.__getitem__(k)
    print(img.size)
    print(anno)


from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher, DeiTFeatureExtractor, DeiTModel
processor = DeiTImageProcessor()
preprocessed_image = processor(img, return_tensors="pt")
model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
outputs = model(**preprocessed_image)
outputs.last_hidden_state.shape # [1, 198, 768]. image size 224,224 gives 14 patches, 14*14 = 198 + BOS + EOS


from transformers.models.deit.feature_extraction_deit import DeiTFeatureExtractor, DeiTImageProcessor

feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-384')
model = DeiTForImageClassificationWithTeacher.from_pretrained('facebook/deit-base-distilled-patch16-384')
vars(model).keys()
type(feature_extractor)
inputs = feature_extractor(images=img, return_tensors="pt")
type(feature_extractor)
inputs.data["pixel_values"].shape
outputs = model(**inputs)
outputs
outputs.hidden_states
logits = outputs.logits
model._modules
processor = DeiTImageProcessor(do_rescale=False, do_normalize=False)    # does (3 ,224, 224)
a = processor.preprocess(img)["pixel_values"][0]
a.shape
from PIL import Image
import numpy as np
image = Image.fromarray(np.moveaxis(a, 0, 2), mode="RGB")
image.save("after_resize.jpg")
img.save("before_resize.jpg")
"""
Summary: 
model has DeiTEncoder + embedding layer
embeddinger layer: DeiTEmbeddings, which uses DeiTPatchEmbeddings:
    a Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16)) projecting as follows:
This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
`hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
Transformer.
which essentially means: 
the tokenization of the image is a learnt convolutional operation.
"""