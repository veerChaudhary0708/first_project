import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# VGG19 normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_and_process_image(image_path, image_size=512):
    """Loads an image, resizes, converts to tensor, and normalizes it."""
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = loader(image).unsqueeze(0)
    return image_tensor

def denormalize_and_save_image(tensor, output_path):
    """Denormalizes a tensor and saves it as an image."""
    denorm = transforms.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )
    tensor = tensor.detach().squeeze(0)
    tensor = denorm(tensor).clamp_(0, 1)
    
    # Use torchvision's save_image utility
    from torchvision.utils import save_image
    save_image(tensor, output_path)

# --- Loss Functions ---

def content_loss(generated_content_features, original_content_features):
    return nn.functional.mse_loss(generated_content_features[0], original_content_features[0])

def gram_matrix(feature_map):
    batch_size, channels, height, width = feature_map.size()
    features = feature_map.view(batch_size * channels, height * width)
    G = torch.mm(features, features.t())
    return G.div(batch_size * channels * height * width)

def style_loss(generated_style_features, original_style_features):
    total_style_loss = 0
    for gen_feat, orig_feat in zip(generated_style_features, original_style_features):
        G_gen = gram_matrix(gen_feat)
        G_orig = gram_matrix(orig_feat)
        layer_loss = nn.functional.mse_loss(G_gen, G_orig)
        total_style_loss += layer_loss
    return total_style_loss

def total_variation_loss(img):
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_v = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return tv_h + tv_v