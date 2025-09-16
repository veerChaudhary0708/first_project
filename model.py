import torch.nn as nn
import torchvision.models as models

CONTENT_LAYER = 'conv_4'
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

class StyleTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.model = nn.Sequential()
        i = 1
        j = 1
        for layer in vgg19.children():
            if isinstance(layer, nn.Conv2d):
                name = f"conv_{i}"
                self.model.add_module(name, layer)
                i += 1
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{j}"
                self.model.add_module(name, nn.ReLU(inplace=True))
                j += 1
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{j-1}"
                self.model.add_module(name, layer)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        content_features = []
        style_features = []
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in STYLE_LAYERS:
                style_features.append(x)
            if name == CONTENT_LAYER:
                content_features.append(x)
        return content_features, style_features