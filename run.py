import torch
from cifar10_models.vgg import vgg11_bn
from torch.vison.models import vgg11, VGG11_Weights
weights = VGG11_Weights.DEFAULT
model1 = vgg11(weights=weights)
model2 = vgg11_bn(pretrained=True)

### preprocessing and model prediction for model1

preprocess1 = VGG11_Weights.transfrom()
img1 = read_image("test/assets/encode_jpeg/grace_hopper_517x606.jpg")
batch1 = preprocess1(img).unsqueeze(0)
prediction = model1(batch1)


### preprocessing for model2 is using mean = [0.4914, 0.4822, 0.4465] std = [0.2471, 0.2435, 0.2616] to normalize the image